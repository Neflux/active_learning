import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import rclpy
from cv_bridge import CvBridge
from elohim.utils import quaternion_to_euler
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from elohim.utils import binary_from_cv

class Recorder(Node):
    def __init__(self, rclpy, topics, namespace="thymioX"):
        super().__init__('recorder_node')
        self.rlcpy = rclpy

        timestr = time.strftime("%d%m-%H%M%S")
        Path(os.path.join('history', timestr)).mkdir(parents=True, exist_ok=True)

        self.cache = {}
        self.cache_count = {}
        self.subs_handlers = []
        for k, v in topics.items():
            if 'use_pandas' not in v:
                v['use_pandas'] = True

            qos = v["qos"]
            if "reliability" in v:
                qos = rclpy.qos.QoSProfile(depth=v["qos"], reliability=v["reliability"])

            clean_topic = k.strip().strip('/')
            topic = f"/{namespace.strip().strip('/')}/{clean_topic}"
            file_path = os.path.join("history", timestr, clean_topic.replace('/', '_') + '.h5')

            self.subs_handlers.append(self.create_subscription(
                msg_type=v['type'],
                topic=topic,
                callback=partial(self.process_row, v["focus"], k, 10.*v["qos"], file_path),
                qos_profile=qos))

    def process_row(self, focus, key, limit, file_name, msg):
        msg = focus(msg)
        lst = self.cache.setdefault(key, [])
        if len(lst) >= limit:
            self.store_and_clear(lst, key, file_name)
        lst.append((self.get_clock().now().nanoseconds, msg))

    def store_and_clear(self, lst, key, file_name):
        self.cache_count[key] = self.cache_count.setdefault(key, 0) + 1
        #print(self.cache_count)
        arr = np.array(lst)
        df = pd.DataFrame(arr[:, 1].tolist())
        df.set_index(arr[:, 0], inplace=True)
        with pd.HDFStore(file_name) as store:
            store.append(key, df)
        print(f'{key}: {os.path.getsize(file_name)/1000000.:.2f} MB')
        lst.clear()

def main(args=None):
    rclpy.init(args=args)

    best_effort = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
    getattrs = lambda x, attrs: [getattr(x, a) for a in attrs]

    odom_focus = (lambda msg: getattrs(msg.pose.pose.position, ['x', 'y']) +  # x,y
                              [quaternion_to_euler(msg.pose.pose.orientation)[2]])  # yaw

    bridge = CvBridge()
    compress = lambda msg: [binary_from_cv(bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))]

    recorder = Recorder(rclpy, topics={'odom': {'type': Odometry, 'qos': 20, 'focus': odom_focus},
                                       'ground_truth/odom': {'type': Odometry, 'qos': 10, 'reliability': best_effort,
                                                             'focus': odom_focus},
                                       'head_camera/image_raw': {'type': Image, 'qos': 10, 'reliability': best_effort,
                                                                 'focus': compress},
                                       'virtual_sensor/signal': {'type': Bool, 'qos': 10,
                                                                 'focus': lambda msg: msg.data}})
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print(" Recording stopped")
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
