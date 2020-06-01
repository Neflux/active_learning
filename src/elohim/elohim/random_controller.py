import json
import os

import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from elohim.service_utils import AsyncServiceCaller
from elohim.utils import euler_to_quaternion
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist, Pose, Vector3, Point
from nav_msgs.msg import Odometry

from rclpy.node import Node
from scipy.spatial.distance import cdist
from tutorial_interfaces.msg import Num


class RandomController(Node):
    def __init__(self, rclpy_, targets, plane_side=10.1, signal_range=0.5, collision_range=0.3, robot_name="thymioX",
                 rate=10):
        assert (signal_range > collision_range)

        super().__init__('random_controller')

        self.targets = targets

        self.plane_side = plane_side
        self.signal_threshold = signal_range
        self.collision_threshold = collision_range

        self.robot_name = robot_name
        self.raw_rate = rate
        self.qos = rclpy_.qos.QoSProfile(depth=self.raw_rate,
                                         reliability=rclpy_.qos.QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
        #self.rate = self.create_rate(self.raw_rate)

        self.twist_publisher = self.create_publisher(Twist, f'/{robot_name}/cmd_vel', qos_profile=self.raw_rate)

        self.go_twist = Twist(linear=Vector3(x=0.3))
        self.stop_twist = Twist(linear=Vector3(x=0.))

        self.should_reset = False

        self.ground_truth_sub = self.create_subscription(Odometry, f'/{self.robot_name}/ground_truth/odom',
                                                         self.pose_update, qos_profile=self.qos)

        self.signal_publisher = self.create_publisher(Num, f'/{robot_name}/virtual_sensor/signal',
                                                      qos_profile=self.raw_rate)

    def pose_update(self, msg):
        pos = msg.pose.pose.position

        euc_dist = cdist([[pos.x, pos.y]], self.targets)[0]
        signal = int(any(euc_dist < self.signal_threshold))
        self.signal_publisher.publish(Num(num=signal))
        #self.get_logger().info(f'{self.robot_name} sensor: {signal}')

        if any(abs(x) > self.plane_side for x in [pos.x, pos.y]):
            self.should_reset = True
            self.get_logger().info(f'{self.robot_name} should reset: outside of map ({pos.x:.2f},{pos.y:.2f}) ')

        if any(euc_dist < self.collision_threshold):
            self.should_reset = True
            self.get_logger().info(f'{self.robot_name} should reset: too close to an object ({pos.x:.2f},{pos.y:.2f})')

    def go(self):
        self.twist_publisher.publish(self.go_twist)
        self.get_logger().info(f'{self.robot_name}: to infinity and beyond {self.go_twist.linear.x}')

    def stop(self, verbose=True):
        self.twist_publisher.publish(self.stop_twist)
        if verbose:
            self.get_logger().info(f'{self.robot_name}: stopped')


def main(args=None):
    rclpy.init(args=args)
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
        exit(-1)

    asc = AsyncServiceCaller(rclpy)

    random_controller = RandomController(rclpy_=rclpy, targets=targets)
    random_controller.stop(verbose=False)

    es = EntityState()
    es.name = random_controller.robot_name
    angles = np.linspace(0, np.pi * 2, 360 // 2)
    keyboard_interrupt = False
    for spawn in spawn_coords:
        for t in angles:
            es.pose = Pose(position=Point(x=spawn[0], y=spawn[1], z=0.), orientation=euler_to_quaternion(t, 0., 0.))
            response = asc(srv=SetEntityState,
                           srv_namespace="ros_state/set_entity_state",  # namespace set in empty_libstate.world
                           request_dict={"state": es})
            print(f"Spawning thymio in ({spawn[0]:.2f}, {spawn[1]:.2f}, {t:.2f}), status: {response.success})")

            random_controller.should_reset = False
            random_controller.go()
            try:
                while rclpy.ok():
                    rclpy.spin_once(random_controller)

                    if random_controller.should_reset:
                        break

                    #random_controller.rate.sleep()
            except KeyboardInterrupt:
                print(" Stopping thymio and signal broadcasting")
                keyboard_interrupt = True
                break
            finally:
                random_controller.stop()
        if keyboard_interrupt:
            break

    random_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
