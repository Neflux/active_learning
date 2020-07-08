import itertools
import json
import os
import time
from functools import partial
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from PIL import Image as Image
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from elohim.service_utils import SyncServiceCaller
from elohim.utils import euler_to_quaternion, random_PIL, maxcppfloat, mypause
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist, Pose, Vector3, Point
from matplotlib.widgets import Button
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from scipy.spatial.distance import cdist
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Int64


# asdasd noinspection PyUnresolvedReferences
class RandomController(Node):
    STOP_TWIST = Twist(linear=Vector3(x=0.))
    MAX_SPEED = 0.3
    SENSORS = ['left', 'center_left', 'center', 'center_right', 'right']

    def __init__(self, targets: np.ndarray,
                 plane_side: float = 10.1,
                 thymio_name: str = "thymioX",
                 rate: int = 10, camera_rate: int = 30) -> None:

        super().__init__('random_controller')

        self.targets = targets
        self.plane_side = plane_side
        self.robot_name = thymio_name
        self.raw_rate = rate
        self.go_twist = Twist(linear=Vector3(x=self.MAX_SPEED))
        self.run_counter = 0
        self.pending_reset = False

        best_effort = rclpy.qos.QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
        self.qos = rclpy.qos.QoSProfile(depth=self.raw_rate, reliability=best_effort)

        # Subscribers and publishers

        self.twist_publisher = self.create_publisher(Twist, f'/{thymio_name}/cmd_vel', qos_profile=self.raw_rate)
        self.sensor_subs, self.odom_sub = [], None
        self.subscribe()

        self.signal_pub = self.create_publisher(Bool, f'/{thymio_name}/virtual_sensor/signal', qos_profile=60)
        self.run_counter_pub = self.create_publisher(Int64, f'/{thymio_name}/run_counter', qos_profile=60)

        # Camera feed, 'mypause' trick disables AlwaysOnTop behaviour

        self.camera_period = 1. / camera_rate
        self.bridge = CvBridge()
        self.last_Image = ROSImage(data=np.random.randint(0, 256, 240 * 320 * 3).tolist(),
                                   height=240, width=320, encoding="rgb8", is_bigendian=0, step=960)
        camera_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(ROSImage, f'/{thymio_name}/head_camera/image_raw', self.save_image,
                                 qos_profile=rclpy.qos.QoSProfile(depth=30, reliability=best_effort),
                                 callback_group=camera_group)
        self.camera_feed = plt.imshow(random_PIL())
        self.create_timer(self.camera_period, self.update_image, callback_group=camera_group)
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Skip Run')
        bnext.on_clicked(self.set_pending_reset)
        axnext._button = bnext
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        plt.show(block=False)

    def update_image(self):
        cv2_img = self.bridge.imgmsg_to_cv2(self.last_Image, 'rgb8')
        data = Image.fromarray(cv2_img.astype('uint8'), 'RGB').convert('RGB')
        self.camera_feed.set_data(data)
        mypause(self.camera_period)

    def save_image(self, msg):
        self.last_Image = msg

    def sensor_callback(self, key, msg):
        r = msg.range  # if msg.range != maxcppfloat else np.inf
        signal = False
        if 0.01 < r:
            if r < 0.10:
                signal = True

            # TODO: Thymio has to go waaay slower for this to work. Any way to speed up the SIM?
            if r < 0.03:
                self.set_pending_reset(True)
                self.get_logger().info(f'{key}: too close to an object ({r:.6f})')

        if signal:
            print('Close obstacle!')
        self.signal_pub.publish(Bool(data=signal))

    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        euc_dist = cdist(np.array([[pos.x, pos.y]]), self.targets)[0]
        closer = min(euc_dist)

        if closer < 0.1:
            self.set_pending_reset(True)
            self.get_logger().info(f'{self.robot_name} should reset: too close to an object ({pos.x:.2f},{pos.y:.2f}) ')

        if abs(pos.x) > self.plane_side or abs(pos.y) > self.plane_side:
            self.set_pending_reset(True)
            self.get_logger().info(f'{self.robot_name} should reset: outside of map ({pos.x:.2f},{pos.y:.2f}) ')

    def go(self):
        self.run_counter_pub.publish(Int64(data=self.run_counter))
        self.twist_publisher.publish(self.go_twist)

    def stop(self):
        self.twist_publisher.publish(self.STOP_TWIST)
        self.get_logger().info(f'{self.robot_name}: stopped')

    def set_pending_reset(self, status=False):
        self.stop()
        self.subscribe() if status else self.unsubscribe()
        self.pending_reset = status

    def new_run(self):
        self.set_pending_reset(False)
        self.run_counter += 1
        self.go_twist = Twist(linear=Vector3(x=self.MAX_SPEED))

    def subscribe(self):
        sensor_group = MutuallyExclusiveCallbackGroup()
        for sensor in self.SENSORS:
            self.sensor_subs.append(self.create_subscription(Range, f'/{self.robot_name}/proximity/{sensor}',
                                                             partial(self.sensor_callback, sensor), self.qos,
                                                             callback_group=sensor_group))
        self.odom_sub = self.create_subscription(Odometry, f'/{self.robot_name}/ground_truth/odom', self.pose_callback,
                                                 self.qos)

    def unsubscribe(self):
        for sub in self.sensor_subs:
            self.destroy_subscription(sub)
        self.sensor_subs.clear()

        self.destroy_subscription(self.odom_sub)
        del self.odom_sub


def run(node, service_caller, x, y, t):
    node.new_run()

    es = EntityState(name=node.robot_name)
    es.pose = Pose(position=Point(x=x, y=y), orientation=euler_to_quaternion(yaw=t))
    service_caller(srv=SetEntityState, srv_namespace="ros_state/set_entity_state", request_dict={"state": es})
    print(f"Spawning thymio in (x:{x:.2f}, y:{y:.2f}, t:{t:.2f})")
    # time.sleep(1)

    try:
        node.get_logger().info(f'{node.robot_name}: to infinity and beyond')
        while rclpy.ok():
            node.go()
            rclpy.spin_once(node)

            if node.pending_reset:
                break

    except KeyboardInterrupt:
        print(" Stopping thymio and signal broadcasting")
        return False
    return True


def main(args=None):
    rclpy.init(args=args)

    points_file = os.path.join(get_package_share_directory('elohim'), 'points.json')
    try:
        with open(points_file) as f:
            points = json.load(f)
            spawn_coords = np.array([[t["x"], t["y"]] for t in points["spawn_coords"]])
            targets = np.array([[t["x"], t["y"]] for t in points["targets"]])

        # Pre-compute all the spawn points, shuffle them
        angles = np.linspace(0, np.pi * 2, 360 // 2)
        spawn_poses = np.array(list(itertools.product(spawn_coords, angles)))
        ids = np.arange(len(spawn_poses))
        np.random.shuffle(ids)

        asc = SyncServiceCaller(rclpy)
        random_controller = RandomController(targets=targets)

        r = partial(run, random_controller, asc)
        for i, ((x, y), t) in enumerate(spawn_poses[ids]):
            print(f'[{str(ids[i]).rjust(5)}] ', end='')
            if not r(x, y, t):
                break

        random_controller.destroy_node()

    except FileNotFoundError:
        print(f"Cannot find points.json file (at {os.path.dirname(points_file)})")
        print("Have you set up your environment at least once after your latest clean rebuild?"
              "\n\tros2 run elohim init")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
