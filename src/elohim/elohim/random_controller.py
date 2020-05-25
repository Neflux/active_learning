
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist

from ament_index_python import get_package_share_directory

import os
import json
import numpy as np
from scipy.spatial.distance import cdist

def HandyTwist(x=0.):
    t = Twist()
    t.linear.x = x
    return t

class RandomController(Node):
    def __init__(self):
        super().__init__('random_controller')
        self.publisher_ = self.create_publisher(Twist, '/thymioX/cmd_vel', 20)
        self.go_twist = HandyTwist(0.1)
        self.stop_twist = HandyTwist(0.)

    def go(self):
        self.publisher_.publish(self.go_twist)
        self.get_logger().info(f'Thymio: to infinity and beyond {self.go_twist.linear.x}')

    def stop(self):
        self.publisher_.publish(self.stop_twist)
        self.get_logger().info(f'Thymio: stopping')


def main(args=None):

    rclpy.init(args=args)

    with open(os.path.join(get_package_share_directory('elohim'),'targets.json')) as f:
        points = json.load(f)

    spawn_coords = np.array([[t["x"],t["y"]] for t in points["spawn_coords"]])

    random_controller = RandomController()

    for spawn_location in spawn_coords:
        pass

    while rclpy.ok():
        rclpy.spin_once(random_controller)


    rclpy.shutdown()


if __name__ == '__main__':
    main()
