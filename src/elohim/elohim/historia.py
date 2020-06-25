import os
import time
from functools import partial
from pathlib import Path

import pandas as pd
import rclpy
from elohim.utils import quaternion_to_euler
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from elohim.utils import binary_from_ndarray


def store_and_clear(lst, key, file_name):
    df = pd.DataFrame(lst)
    with pd.HDFStore(file_name) as store:
        store.append(key, df)
    lst.clear()


def process_row(focus, key, file_name, _cache, msg):
    msg = focus(msg)
    lst = _cache.setdefault(key, [])
    if len(lst) >= 10:
        store_and_clear(lst, key, file_name)
    lst.append(msg)


class Recorder(Node):
    def __init__(self, rclpy, topics, namespace="thymioX"):
        super().__init__('recorder_node')

        timestr = time.strftime("%d%m-%H%M%S")
        Path(os.path.join('history', timestr)).mkdir(parents=True, exist_ok=True)

        self.cache = {}
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
                callback=partial(process_row, v["focus"], k, file_path, self.cache),
                qos_profile=qos))


def main(args=None):
    rclpy.init(args=args)

    best_effort = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
    getattrs = lambda x, attrs: [getattr(x, a) for a in attrs]

    odom_focus = (lambda msg: getattrs(msg.pose.pose.position, ['x', 'y']) +  # x,y
                              [quaternion_to_euler(msg.pose.pose.orientation)[2]])  # yaw

    recorder = Recorder(rclpy, topics={'odom': {'type': Odometry, 'qos': 20, 'focus': odom_focus},
                                       'ground_truth/odom': {'type': Odometry, 'qos': 10, 'reliability': best_effort,
                                                             'focus': odom_focus},
                                       'head_camera/image_raw': {'type': Image, 'qos': 30, 'reliability': best_effort,
                                                                 'focus': lambda msg: [binary_from_ndarray(msg.data)]},
                                       'virtual_sensor/signal': {'type': Bool, 'qos': 10, 'focus': lambda msg: msg.num}})
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print(" Recording stopped")
    finally:
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
