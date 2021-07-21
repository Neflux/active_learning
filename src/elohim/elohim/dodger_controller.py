import argparse
import math
import os
import time
from functools import partial

import numpy as np
import rclpy
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Point, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Float32MultiArray, Float32

from utils import fov_mask

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from service_utils import SyncServiceCaller
    from utils_ros import euler_to_quaternion, random_PIL, mypause, quaternion_to_euler
    import config
    from random_controller import RandomController
except ImportError:
    from elohim.service_utils import SyncServiceCaller
    from elohim.utils_ros import euler_to_quaternion, random_PIL, mypause, quaternion_to_euler
    import elohim.config as config
    from elohim.random_controller import RandomController


class FakePublisher:
    def publish(self, *args, **kwargs):
        pass


class DodgerControllerr(RandomController):
    def __init__(self, model_path=None, samples=None, rotation_factor=0.5, sensor_factor=10, drift_factor=1.0,
                 obstacle_dist_threshold=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_omega = 6

        if obstacle_dist_threshold is not None:
            self.obstacle_dist_threshold = obstacle_dist_threshold

        self.k = rotation_factor
        self.s = sensor_factor
        self.j = drift_factor

        best_effort = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
        qos = rclpy.qos.QoSProfile(depth=10, reliability=best_effort)
        self.odom_sub = self.create_subscription(Odometry, f'/{self.robot_name}/ground_truth/odom', self.update_theta,
                                                 qos)
        self.theta = 0
        self.obstacles_hit_pub = self.create_publisher(Int16, f'/{self.robot_name}/obstacles_hit', qos_profile=60)
        self.target_theta_pub = self.create_publisher(Float32, f'/{self.robot_name}/target_theta', qos_profile=60)

        self.output = np.empty(400)
        self.pred_angle = 0
        self.pred_center = 0

        self.get_readings = self.get_ordered_readings
        self.predicted_proximity_pub = FakePublisher()
        if model_path is not None:
            self.get_readings = self.get_ininfluential_readings

            print('This model will try to be smarter')
            import torch
            try:
                from utils import fov_mask
                from models import flexible_weights
                from models import initialize_model
                from dataset_ml import transform_function, to_pil, to_tensor
            except ImportError:
                from elohim.utils import fov_mask
                from elohim.models import initialize_model, flexible_weights
                from elohim.dataset_ml import transform_function, to_pil, to_tensor

            self.model, _, _, device = initialize_model(batch_size=1, samples=samples,
                                                        weights_path=os.path.join(model_path, 'checkpoint.pt'))
            self.model.eval()
            torch.autograd.set_grad_enabled(False)

            # TODO: increase 30 or create a timer

            qos = rclpy.qos.QoSProfile(depth=30, reliability=best_effort)
            self.bridge = CvBridge()
            self.camera_sub = self.create_subscription(Image, f'/{self.robot_name}/head_camera/image_raw',
                                                       self.update_prediction, qos)

            self.mask = fov_mask()
            self.setup_kernels()
            self.transform = lambda x, f=to_tensor, g=to_pil: f(g(x))

            self.timer = self.create_timer(1, self.unlock_prediction)
            self.can_predict = False

            self.output = np.full(400, fill_value=-1)
            self.predicted_proximity_pub = self.create_publisher(Float32MultiArray,
                                                                 f'/{self.robot_name}/predicted_map', qos_profile=30)

    @staticmethod
    def get_ininfluential_readings():
        return [0.12, 0.12, 0.12, 0.12, 0.12]

    def broadcast_info(self, i, tt):
        """ Broadcasts the unique id of the run,  """
        self.broadcast_run_id(id=i)
        self.obstacles_hit_pub.publish(Int16(data=self.obstacles_hit))
        self.target_theta_pub.publish(Float32(data=tt))
        self.predicted_proximity_pub.publish(Float32MultiArray(data=self.output))

    def unlock_prediction(self):
        self.can_predict = True

    def update_prediction(self, msg):
        if self.can_predict:
            output = self.model(
                self.transform(self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')).unsqueeze(0))
            self.output = output[0].detach().numpy()[..., 1]
            output = np.where(self.mask, self.output.reshape(20, 20), 0)
            #self.extended_range = output.sum(axis=0)
            self.pred_angle = np.sum(self.angleness * output)
            self.pred_center = np.sum(self.centerness * output)
            if self.pred_center > 0.:
                self.pred_angle += self.pred_center * np.sign(self.pred_angle)

            self.can_predict = False

    def update_theta(self, msg):
        self.theta = quaternion_to_euler(msg.pose.pose.orientation)[2]

    def smart_go(self, target_theta):
        """ Updates internal state and sets Thymio velocity to cruising speed on its forward axis (x) """
        self.update_state('running')

        # Target yaw
        angle_diff = min(target_theta - self.theta,
                         target_theta - self.theta + 2 * math.pi,
                         target_theta - self.theta - 2 * math.pi,
                         key=abs)

        # Naive controller
        sensor_readings = self.get_readings()
        angleness = np.dot(sensor_readings, np.array([-1, -2, 0, 2, 1])) / 3
        centerness = np.dot(sensor_readings, np.array([-1, -1, 4, -1, -1])) / 4
        angular = -angleness * 2 * np.pi
        if centerness > 0.2:
            centerness *= np.sign(angle_diff)
        else:
            centerness = 0

        #drift = self.extended_range[10:].sum() - self.extended_range[:10].sum()

        omega = np.clip(float(self.k * angle_diff +  # Target yaw
                              self.s * (angular + centerness)  # Proximity sensors
                              - self.j * (self.pred_angle)),  # Prediction
                        -self.max_omega, self.max_omega)

        # -0.50 -1.1e+01 -2.141593 2.156940
        print(f'\r{angle_diff:2.2f} {self.pred_angle:2.2f} {self.pred_center:2.2f}', end='')
        # print(f'\r{self.theta:2.2f}, {target_theta:2.2f}, {omega:2.2f}', end='')
        self.twist_publisher.publish(Twist(linear=Vector3(x=config.forward_velocity),
                                           angular=Vector3(z=omega)))

    def setup_kernels(self):
        import scipy.stats as st
        def gkern(kernlen=41, nsig:float=3):
            """Returns a 2D Gaussian kernel."""
            x = np.linspace(-nsig, nsig, kernlen + 1)
            kern1d = np.diff(st.norm.cdf(x))
            kern2d = np.outer(kern1d, kern1d)
            kern2d_upper_odd = (kern2d / kern2d.sum())[:int((kernlen - 1) // 2), ::2]
            kern2d_upper_even = np.delete(kern2d_upper_odd, int((kernlen - 1) // 4), 1)
            return kern2d_upper_even

        l = list(np.linspace(4, 1, 8))
        a = np.array(l[::-1] + [0, 0, 0, 0] + l)
        a[:10] *= -1
        a = np.tile(a, (20, 1))

        proximity_kernel_angleness = fov_mask() * a
        proximity_kernel_angleness /= proximity_kernel_angleness.max()
        self.angleness = proximity_kernel_angleness / (sum(l))

        l = list(-np.ones(7))
        a = l + [1, 2, 4, 4, 2, 1] + l
        a = np.tile(a, (20, 1))
        proximity_kernel_centerness = gkern(nsig=1.5) * fov_mask() * a
        proximity_kernel_centerness /= proximity_kernel_centerness.max()
        self.centerness = proximity_kernel_centerness / 14


def run(node, service_caller, x, y, t, tt):
    """
    Sets up the thymio for a new run and adds some logic for the obstacle scan

    @param node: the thymio controller node
    @param service_caller: SyncServiceCaller entity
    @param x: x coordinate of spawn position
    @param y: y coordinate of spawn position
    @param t: theta angle (yaw) of spawn position
    @return: True if the run completed successfully (obstacle or out of map), False if the user keyboard-interrupted
    """

    es = EntityState(name=node.robot_name)
    es.pose = Pose(position=Point(x=x, y=y), orientation=euler_to_quaternion(yaw=t))
    service_caller(srv=SetEntityState, srv_namespace="ros_state/set_entity_state", request_dict={"state": es})
    print(f"Thymio teleported to (x:{x:.2f}, y:{y:.2f}, t:{t:.2f}), target yaw: {tt:2.2f}")

    node.new_run()
    start = time.time()
    save_and_quit = False
    try:

        node.smart_go(tt)
        while rclpy.ok():
            id = 'auto'
            if node.time_elapsed(120., key=1):
                save_and_quit = True
                print("\nTimeout")

            if save_and_quit:
                id = -1
                if node.time_elapsed(3, key=2):
                    break

            state = node.state
            if state in 'running':
                node.smart_go(tt)
            if state == 'stopped':
                if node.time_elapsed(0.3):
                    angular_vel = 0.3
                    if 'left' in node.verbose:
                        angular_vel *= -1
                    # if random.random() > 0.5: # this is unfair
                    #     angular_vel *= -1
                    node.rotate(angular_vel)
            elif state == 'rotating':
                if not node.obstacle_nearby():
                    if node.time_elapsed(0.1):
                        node.stop(reset=True)
            elif state == 'reset':
                if 'out of map' in node.verbose:
                    id = -1
                    if node.time_elapsed(1):  # Wait some time so it saves
                        print(f'\nExit status: {node.verbose} - Time: {time.time() - start:.2f}s')
                        break
                else:
                    node.new_run(reset_obstacles=False)
                    node.smart_go(tt)

            node.broadcast_info(id, tt)
            rclpy.spin_once(node, timeout_sec=0.)

    except KeyboardInterrupt:
        node.stop()
        print("\nStopping thymio and signal broadcasting")
        return -1
    return 1


parser = argparse.ArgumentParser(description='Process some integers.', prefix_chars='@')
# parser.add_argument('@@spawn@coordinates', '@sc', nargs=3, help="x,y,t", dest='s', type=float, default=None)
parser.add_argument('@@yaws', '@y', metavar='path', dest='yaws_path', required=False)
parser.add_argument('@@model', '@m', metavar='path', dest='model', required=False,
                    help='Folder containing model checkpoint')
parser.add_argument('@@samples', '@s', dest='samples', type=int,
                    help='Number of samples for the bayesian network (ensemble)')


def main(args=None):
    rclpy.init(args=args)

    # @m history / final / red_hippo_t100
    # args = ['@sc', '0.0', '0.0', '0.0']
    args = ['@y', 'yaw_target.npy']#, '@m', 'history/final/red_hippo_t100']
    args = parser.parse_args(args)

    # Pre-compute all the spawn theta and target theta

    if args.yaws_path is not None:
        print('Loading cached spawn yaws and target yaws')
        angles = np.load(args.yaws_path)
    else:
        print('Creating random spawn yaws and target yaws')
        n = 1
        angles = np.random.uniform(-np.pi, np.pi, n * 2).reshape(-1, 2)
    # np.random.shuffle(angles)

    asc = SyncServiceCaller(rclpy)
    dodger_controller = DodgerControllerr(args.model, args.samples, obstacle_dist_threshold=None)
    r = partial(run, dodger_controller, asc)

    for theta, target_theta in angles[1:]:
        if r(0., 0., theta, target_theta) < 0:
            break
    dodger_controller.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
