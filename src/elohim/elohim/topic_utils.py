import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.distance import cdist
from tutorial_interfaces.msg import Bool

class VirtualSensor(Node):

    def __init__(self, targets, threshold):
        self.threshold = threshold
        self.targets = np.array([[t["x"],t["y"]] for t in targets["targets"]])

        super().__init__('virtual_sensor')
        self.publisher_ = self.create_publisher(Bool, '/thymioX/virtual_sensor', 20)
        self.subscription = self.create_subscription(Odometry,'/thymioX/odom', self.listener_callback, 20)

    def listener_callback(self, msg):
        p = msg.pose.pose.position
        euc_dist = cdist([[p.x, p.y]], self.targets)[0]
        signal = any(euc_dist < self.threshold)
        self.get_logger().info(f'I heard: ({p.x},{p.y}), I broadcast: {signal}')
        #self.get_logger().info(f'{int(signal)}')

        response = Bool()
        response.is_in_range = signal
        self.publisher_.publish(response)