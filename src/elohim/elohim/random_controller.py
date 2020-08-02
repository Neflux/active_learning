import argparse
import itertools
import json
import os
import time
from functools import partial
from logging import DEBUG, INFO

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from PIL import Image as Image
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist, Pose, Vector3, Point
from matplotlib.widgets import Button
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.logging import LoggingSeverity
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Int64

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from service_utils import SyncServiceCaller
    from utils import euler_to_quaternion, random_PIL, maxcppfloat, mypause
except ImportError:
    from elohim.service_utils import SyncServiceCaller
    from elohim.utils import euler_to_quaternion, random_PIL, maxcppfloat, mypause


class IllegalPosition(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# asdasd noinspection PyUnresolvedReferences
class RandomController(Node):
    STOP_TWIST = Twist(linear=Vector3(x=0.))
    TURN_RIGHT = Twist(angular=Vector3(z=-1.))
    TURN_LEFT = Twist(angular=Vector3(z=-1.))
    GO_TWIST = Twist(linear=Vector3(x=0.15))
    SENSORS = ['left', 'center_left', 'center', 'center_right', 'right']

    def __init__(self, targets: np.ndarray,
                 plane_side: float = 10.1,
                 thymio_name: str = "thymioX",
                 rate: int = 10) -> None:

        super().__init__('random_controller')
        self.get_logger().set_level(LoggingSeverity.INFO)

        self.targets = targets
        self.plane_side = plane_side
        self.robot_name = thymio_name
        self.raw_rate = rate
        self.run_counter = 0
        self.ref_time = None
        self.state = 'idle'

        # self.debug_radar = {k: np.inf for k in self.SENSORS}

        best_effort = rclpy.qos.QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
        self.qos = rclpy.qos.QoSProfile(depth=self.raw_rate, reliability=best_effort)

        # Subscribers and publishers

        self.twist_publisher = self.create_publisher(Twist, f'/{thymio_name}/cmd_vel', qos_profile=self.raw_rate)
        self.signal_pub = self.create_publisher(Bool, f'/{thymio_name}/virtual_sensor/signal', qos_profile=60)
        self.run_counter_pub = self.create_publisher(Int64, f'/{thymio_name}/run_counter', qos_profile=60)

        self.sensor_subs, self.odom_sub = [], None
        self.get_logger().debug("Subscribing (sensor and odometry)")

    def subscribe(self):
        """ Subscribes to sensors' topics and ground truth odometry to check for out of maps """
        sensor_group = None  # MutuallyExclusiveCallbackGroup()
        for sensor in self.SENSORS:
            self.sensor_subs.append(self.create_subscription(Range, f'/{self.robot_name}/proximity/{sensor}',
                                                             partial(self.sensor_callback, sensor), self.qos,
                                                             callback_group=sensor_group))
        self.odom_sub = self.create_subscription(Odometry, f'/{self.robot_name}/ground_truth/odom', self.pose_callback,
                                                 self.qos)

    def unsubscribe(self):
        """ Unsubscribes from all topics """
        self.get_logger().log("Unsubscribing (sensor and odometry)", LoggingSeverity.DEBUG)
        for sub in self.sensor_subs:
            self.destroy_subscription(sub)
        self.sensor_subs.clear()

        self.destroy_subscription(self.odom_sub)
        del self.odom_sub

    def go(self):
        """ Updates internal state and sets Thymio velocity to cruising speed on its forward axis (x) """
        self.update_state('running')
        self.twist_publisher.publish(self.GO_TWIST)

    def stop(self, reset=False):
        """ Updates internal state and sets Thymio velocity to cruising speed on its forward axis (x)
        @param reset: flag to differentiate obstacle-to-scan (stopped) and out-of-map/obs.scanned scenario (reset)
            """
        self.update_state('reset' if reset else 'stopped')
        if reset:
            self.unsubscribe()
        self.twist_publisher.publish(self.STOP_TWIST)

    def rotate(self, rotation):
        """
            Updates internal state and sets Thymio twist to the specified rotation value
        @param rotation: the yaw (z-axis) rotation speed in radians per second
        """
        self.update_state('rotating')
        # self.twist_publisher.publish(self.TURN_LEFT if rotation > 0 else self.TURN_RIGHT)
        self.twist_publisher.publish(Twist(angular=Vector3(z=rotation)))

    def update_state(self, state):
        """ Updates internal state with specified str """
        self.state = state

    def broadcast_run_id(self, id='auto'):
        """ Broadcasts the unique id of the run,  """
        self.run_counter_pub.publish(Int64(data=(self.run_counter if id == 'auto' else id)))

    def time_elapsed(self, seconds):
        """ Starts a simple home-made timer and checks whether it elapsed or not """
        if self.ref_time is None:
            self.ref_time = self.get_clock().now()
        diff = self.get_clock().now() - self.ref_time
        remaining = seconds - diff.nanoseconds * 1e-9
        #print(f'\r timer {remaining if remaining > 0 else 0:.2f}s', end='')
        result = remaining < 0
        if result:
            self.ref_time = None
            #print()
            return True
        return False

    def reset(self, msg, reset=False):
        """ Unsubscribes from all topics to prevent false flags """
        self.stop(reset)
        self.verbose = msg

    def new_run(self):
        self.update_state('ready')
        self.subscribe()
        self.run_counter += 1
        # self.debug_radar = {k: np.inf for k in self.SENSORS}

    def sensor_callback(self, key, msg):
        r = msg.range if msg.range != maxcppfloat else np.inf
        signal = False
        if 0.01 < r:
            # self.debug_radar[key] = r
            # print(' '.join([f'{v:.2f}' for v in self.debug_radar.values()]))
            if r < 0.10:
                signal = True

            if r < 0.03:
                self.reset(f'{key} sensor, close object ({r:.6f})')

        self.signal_pub.publish(Bool(data=signal))

    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        # print(f'{pos.x:.2f} {pos.y:.2f}')
        if abs(pos.x) > self.plane_side or abs(pos.y) > self.plane_side:
            self.reset(f'{self.robot_name} out of map ({pos.x:.2f},{pos.y:.2f})', reset=True)


def run(node, service_caller, x, y, t):
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
    print(f"[Run {str(node.run_counter).rjust(2)}]: Thymio teleported to (x:{x:.2f}, y:{y:.2f}, t:{t:.2f})")

    node.new_run()
    start = time.time()
    try:
        rotations = [1., -1., 1., -0.5]

        # idle -> ready -> running -> stopped -> ready -> ..

        while rclpy.ok():
            id = 'auto'

            state = node.state
            if state == 'ready':
                node.go()
            elif state == 'stopped':
                if node.time_elapsed(seconds=0.1):
                    node.rotate(rotations.pop())
            elif state == 'rotating':
                if node.time_elapsed(seconds=3):
                    if len(rotations):
                        node.rotate(rotations.pop())
                    else:
                        node.stop(reset=True)
            elif state == 'reset':
                id = -1
                if node.time_elapsed(1):
                    print(f'{node.verbose} - Time: {time.time() - start:.2f}s')
                    break

            node.broadcast_run_id(id=id)
            rclpy.spin_once(node, timeout_sec=0.)

    except KeyboardInterrupt:
        node.stop()
        print(" Stopping thymio and signal broadcasting")
        return False
    return True


parser = argparse.ArgumentParser(description='Process some integers.', prefix_chars='@')
parser.add_argument('@@s', nargs=3, help="x,y,t", type=float, default=None)


def main(args=None):
    rclpy.init(args=args)

    # args = ['@@s', '7.50', '-7.50', '4.10']
    args = parser.parse_args(args)

    points_file = os.path.join(get_package_share_directory('elohim'), 'points.json')
    try:
        with open(points_file) as f:
            points = json.load(f)
            spawn_coords = np.array([[t["x"], t["y"]] for t in points["spawn_coords"]])
            targets = np.array([[t["x"], t["y"]] for t in points["targets"]])

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
    random_controller = RandomController(targets=targets)
    r = partial(run, random_controller, asc)

    if args.s is not None:
        r(*args.s)
    else:
        for (x, y), t in spawn_poses:
            if not r(x, y, t):
                break

    random_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
