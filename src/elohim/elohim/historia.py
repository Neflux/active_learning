from functools import partial

import rclpy
from nav_msgs.msg import Odometry

from rclpy.node import Node
from sensor_msgs.msg import Image
from tutorial_interfaces.msg import Num

import pandas as pd

CACHE = {}
STORE = 'store.h5'


class Recorder(Node):
    def __init__(self, rclpy, topics, namespace="thymioX", log_rate=4):
        super().__init__('recorder_node')

        self.last_value = {}
        self.subs_handlers = []
        for k, v in topics.items():

            self.last_value[k] = v["type"]()

            qos = v["qos"]
            if "reliability" in v:
                qos = rclpy.qos.QoSProfile(depth=v["qos"], reliability=v["reliability"])

            self.subs_handlers.append(self.create_subscription(
                msg_type=v['type'],
                topic=f"/{namespace.strip().strip('/')}/{k.strip().strip('/')}",
                callback=partial(self.updater, k),
                qos_profile=qos))

        self.timer = self.create_timer(1. / log_rate, self.timer_callback)

    def updater(self, k, msg):
        self.last_value[k] = msg

    def timer_callback(self):



    def process_row(self, d, key, max_len=5000, _cache=CACHE):
        lst = _cache.setdefault(key, [])
        if len(lst) >= max_len:
            self.store_and_clear(lst, key)
        lst.append(d)

    def store_and_clear(self, lst, key):
        """
        Convert key's cache list to a DataFrame and append that to HDF5.
        """
        df = pd.DataFrame(lst)
        with pd.HDFStore(STORE) as store:
            store.append(key, df)
        lst.clear()


def main(args=None):
    rclpy.init(args=args)

    import platform
    print(platform.python_version())

    from rclpy.qos import QoSReliabilityPolicy
    best_effort = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT

    recorder = Recorder(rclpy, topics={'odom': {'type': Odometry, 'qos': 20},
                                       'ground_truth/odom': {'type': Odometry, 'qos': 10, 'reliability': best_effort},
                                       'head_camera/image_raw': {'type': Image, 'qos': 30, 'reliability': best_effort},
                                       'virtual_sensor/signal': {'type': Num, 'qos': 10}})

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print(" Recording stopped")
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
