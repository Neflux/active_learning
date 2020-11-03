import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from PIL import Image as Image
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image as ROSImage


def random_PIL():
    a = np.random.rand(240, 320, 3) * 255
    return Image.fromarray(a.astype('uint8')).convert('RGB')


class Monitor(Node):
    def __init__(self, camera_rate=10):
        super().__init__('monitor_node')
        self.camera_rate = camera_rate
        self.camera_period = 1. / camera_rate
        self.bridge = CvBridge()
        self.create_subscription(ROSImage, '/thymioX/head_camera/image_raw', self.update_image,
                                 qos_profile=rclpy.qos.QoSProfile(depth=60,
                                                                  reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT))

        self.camera_feed = plt.imshow(random_PIL())
        plt.axis('off')  # this rows the rectangular frame
        plt.gca().get_xaxis().set_visible(False)  # this removes the ticks and numbers for x axis
        plt.gca().get_yaxis().set_visible(False)
        plt.tight_layout(pad=0.)
        self.fps = plt.annotate('', xy=(2., 8.), color='green', weight='bold')

        plt.gcf().canvas.mpl_connect('close_event', self.close)

        self.locked = True
        self.timer = self.create_timer(self.camera_period, self.unlock)

        self.start_time = time.time()
        self.x = 0.3  # displays the frame rate every 1 second
        self.counter = 0

    def close(self, evt):
        self.locked = True
        # self.timer.cancel()
        # self.timer.destroy()

    def unlock(self):
        self.locked = False

    def update_image(self, msg):
        if not self.locked:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            data = Image.fromarray(cv2_img.astype('uint8'), 'RGB').convert('RGB')
            self.camera_feed.set_data(data)

            self.counter += 1
            if (time.time() - self.start_time) > self.x:
                self.fps.set_text(f'FPS: {self.counter / (time.time() - self.start_time):.0f}')
                self.counter = 0
                self.start_time = time.time()

            plt.draw()
            plt.pause(self.camera_period)

            self.locked = True


def main(args=None):
    rclpy.init(args=args)

    monitor = Monitor()
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        print("\n\tCamera node stopped")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
