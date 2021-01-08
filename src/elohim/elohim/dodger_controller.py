import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Point, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image

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


class DodgerControllerr(RandomController):
    def __init__(self, model_path=None, samples=None, rotation_factor=0.6, **kwargs):
        super().__init__(**kwargs)

        self.k = rotation_factor

        best_effort = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
        qos = rclpy.qos.QoSProfile(depth=10, reliability=best_effort)
        self.odom_sub = self.create_subscription(Odometry, f'/{self.robot_name}/ground_truth/odom', self.update_theta,
                                                 qos)
        self.theta = 0
        self.ref_theta = None

        if model_path is not None:
            print('This model will try to be smarter')
            import torch
            try:
                from utils import fov_mask
                from models import flexible_weights
                from models import initialize_model
                from dataset_ml import transform_function
            except ImportError:
                from elohim.utils import fov_mask
                from elohim.testing import flexible_weights
                from elohim.model import initialize_model
                from elohim.dataset_ml import transform_function

            self.model, _, _, device = initialize_model(samples=samples, weights_path=model_path)
            self.model.load_state_dict(flexible_weights(model_path, device))
            self.model.eval()
            torch.autograd.set_grad_enabled(False)

            # TODO: increase 30 or create a timer

            qos = rclpy.qos.QoSProfile(depth=30, reliability=best_effort)
            self.bridge = CvBridge()
            self.camera_sub = self.create_subscription(Image, f'/{self.robot_name}/head_camera/image_raw',
                                                       self.update_prediction, qos)

            self.mask = fov_mask()
            self.transform = partial(transform_function(resize=samples is not None), flip=False)
            self.extended_range = np.zeros(10)

    def update_prediction(self, msg):
        output = self.model(self.transform(self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')).unsqueeze(0))
        output = np.where(self.mask, output[0].detach().numpy()[..., 1].reshape(20, 20), 0)
        self.extended_range = output.sum(axis=0)

    def update_theta(self, msg):
        self.theta = quaternion_to_euler(msg.pose.pose.orientation)[2]
        # self.theta = np.pi*2+self.theta if self.theta < 0 else self.theta

    def theta_rotated(self, delta_theta):
        if self.ref_theta is None:
            self.ref_theta = self.theta
        if self.theta - self.ref_theta > delta_theta:
            self.ref_theta = None
            return True
        return False

    def smart_go(self, target_theta):
        """ Updates internal state and sets Thymio velocity to cruising speed on its forward axis (x) """
        self.update_state('running')

        angle_diff = (target_theta - self.theta)
        angle_diff = angle_diff + (np.pi * 2 * np.sign(angle_diff)) if abs(angle_diff) > np.pi else angle_diff


        omega = np.clip(float(self.k * angle_diff), -0.5, 0.5)
        # print(f'\r{self.theta:2.2f}, {target_theta:2.2f}, {omega:2.2f}', end='')
        self.twist_publisher.publish(Twist(linear=Vector3(x=config.forward_velocity),
                                           angular=Vector3(z=omega)))


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
    print(f"Thymio teleported to (x:{x:.2f}, y:{y:.2f}, t:{t:.2f})")

    node.new_run()
    start = time.time()
    try:
        node.smart_go(tt)
        while rclpy.ok():
            id = 'auto'
            state = node.state
            if state in 'running':
                node.smart_go(tt)
            if state == 'stopped':
                if node.time_elapsed(0.3):
                    angular_vel = 0.3
                    if random.random() > 0.5:
                        angular_vel *= -1
                    node.rotate(angular_vel)
            elif state == 'rotating':
                if not node.obstacle_nearby():
                    node.stop(reset=True)
            elif state == 'reset':
                id = -1
                if node.time_elapsed(0.5):
                    if 'out of map' in node.verbose:
                        print(f'\nExit status: {node.verbose} - Time: {time.time() - start:.2f}s')
                        break
                    node.new_run()
                    node.smart_go(tt)

            node.broadcast_run_id(id=id)
            rclpy.spin_once(node, timeout_sec=0.)

    except KeyboardInterrupt:
        node.stop()
        print("\nStopping thymio and signal broadcasting")
        return -1
    return 1


parser = argparse.ArgumentParser(description='Process some integers.', prefix_chars='@')
parser.add_argument('@@spawn', '@s', nargs=3, help="x,y,t", type=float, default=None)
parser.add_argument('@@model', '@m', metavar='path', dest='o_path', required=True,
                    help='Folder containing model checkpoint')
parser.add_argument('@@samples', '@s', dest='samples', type=int,
                    help='Number of samples for the bayesian network (ensemble)')


def main(args=None):
    rclpy.init(args=args)

    args = ['@s', '0.0', '0.0', '0.0']
    args = parser.parse_args(args)

    points_file = os.path.join(get_package_share_directory('elohim'), 'points.json')
    try:
        with open(points_file) as f:
            points = json.load(f)
            spawn_coords = np.array([[t["x"], t["y"]] for t in points["spawn_coords"]])
    except FileNotFoundError:
        print(f"Cannot find points.json file (at {os.path.dirname(points_file)})")
        print("Have you set up your environment at least once after your latest clean rebuild?"
              "\n\tros2 run elohim init")
        rclpy.shutdown()
        exit()

    # Pre-compute all the spawn points, shuffle them up
    angles = np.linspace(0, np.pi * 2, 360 // 2)
    spawn_poses = np.array(list(itertools.product(spawn_coords, angles)))
    ids = np.arange(len(spawn_poses))
    np.random.shuffle(ids)
    spawn_poses = spawn_poses[ids]

    asc = SyncServiceCaller(rclpy)
    random_controller = DodgerControllerr(args.model, args.sample)
    r = partial(run, random_controller, asc)

    np.random.shuffle(angles)
    if args.s is not None:
        for tt in angles:
            print((tt - np.pi) * 180 / np.pi)
            if r(*args.s, tt - np.pi) < 0:
                break
    else:
        for ((x, y), t), tt in zip(spawn_poses, angles):
            if r(x, y, t, tt - np.pi) < 0:
                break

    random_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
