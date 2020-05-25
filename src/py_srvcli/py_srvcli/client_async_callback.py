# Copyright 2018 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from threading import Thread

from example_interfaces.srv import AddTwoInts

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup


class MinimalClientAsync(Node):

    def __init__(self, i, cb_group):
        self.i = i

        super().__init__(f'minimal_client_async_{i}')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints', callback_group=cb_group)

        self.did_run = False
        self.did_get_result = False

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

    async def call_service(self):
        self.did_run = True
        try:
            req = AddTwoInts.Request()
            req.a = 0
            req.b = self.i
            future = self.cli.call_async(req)
            try:
                result = await future
            except Exception as e:
                self.get_logger().info('Service call failed %r' % (e,))
            else:
                self.get_logger().info(
                    'Result of add_two_ints: for %d + %d = %d' %
                    (req.a, req.b, result.sum))
        finally:
            self.did_get_result = True


def main(args=None):
    rclpy.init(args=args)
    # Node's default callback group is mutually exclusive. This would prevent the client response
    # from being processed until the timer callback finished, but the timer callback in this
    # example is waiting for the client response
    cb_group = ReentrantCallbackGroup()
    mca = MinimalClientAsync(cb_group=cb_group)
    timer = mca.create_timer(0.5, mca.call_service, callback_group=cb_group)
    for i in range(10):

        while rclpy.ok() and not mca.did_run:
            rclpy.spin_once(mca)

        if mca.did_run:
            timer.cancel()

        while rclpy.ok() and not mca.did_get_result:
            rclpy.spin_once(mca)

        mca.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
