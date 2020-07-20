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
    def __init__(self, camera_rate=1 / 30.):
        super().__init__('monitor_node')
        self.camera_rate = camera_rate
        self.bridge = CvBridge()
        self.create_subscription(ROSImage, '/thymioX/head_camera/image_raw', self.update_image,
                                 qos_profile=rclpy.qos.QoSProfile(depth=30,
                                                                  reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT))

        self.camera_feed = plt.imshow(random_PIL())
        #cv2.imshow("window", np.random.randint(255, size=(240,320,3),dtype=np.uint8))
        #cv2.waitKey(1)

        self.locked = True
        self.create_timer(camera_rate, self.unlock)

    def unlock(self):
        self.locked = False

    def update_image(self, msg):
        if not self.locked:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

            #cv2.imshow("window", cv2_img)
            #cv2.waitKey(1)

            data = Image.fromarray(cv2_img.astype('uint8'),'RGB').convert('RGB')
            self.camera_feed.set_data(data)
            plt.draw()
            plt.pause(self.camera_rate)

            self.locked = True

def main(args=None):
    rclpy.init(args=args)

    monitor = Monitor()
    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        print("\nNode stopped")
    finally:
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
