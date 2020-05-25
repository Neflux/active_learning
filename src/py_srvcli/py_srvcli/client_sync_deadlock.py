import sys
from threading import Thread

from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientSync(Node):

    def __init__(self):
        super().__init__('minimal_client_sync')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self):
        self.req.a = 41
        self.req.b = 1
        return self.cli.call(self.req)

def main():
    rclpy.init()

    for i in range(3):
        print(i)
        minimal_client = MinimalClientSync()
        minimal_client.send_request()

        spin_thread = Thread(target=rclpy.spin_once, args=(minimal_client,))
        spin_thread.start()

        minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()