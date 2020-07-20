import os
import time
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from threading import Timer

import pandas as pd
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int64
from tables import PerformanceWarning

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from utils import binary_from_cv, quaternion_to_euler
except ImportError:
    from elohim.utils import binary_from_cv, quaternion_to_euler

# get_time = lambda node: time.time()
get_time = lambda node: node.get_clock().now().nanoseconds


class Recorder(Node):
    def __init__(self, rclpy, topics, namespace="thymioX"):
        """ Initialises the recording node. Sets up the subscriptions and the
        Args:
            @param topics: a dictionary that specifies the topics to record and all the necessary parameters.
                Each key should be the full string of a broadcasting topic and it should contain a dictionary with:
                    'type':  the python type of msg interface
                    'simplify': a custom function that returns only the bare minimum that is necessary from a message
                And optionally it should contain, if necessary:
                    'qos': the rate of broadcasting of said topic
                    'reliability': the reliability policy of the qos profile
                    'min_itemsize': a dictionary that specifies the block size for the HDF5, for each column label
                        """
        super().__init__('recorder_node')
        self.rlcpy = rclpy
        self.topics = topics

        self.last_value = {}
        self.cache = defaultdict(list)
        self.cache_count = defaultdict(lambda: 0)
        self.subs_handlers = []

        # self.group = MutuallyExclusiveCallbackGroup()

        for k, v in self.topics.items():
            self.topics[k]['group'] = group = None  # MutuallyExclusiveCallbackGroup()

            qos = v["qos"]
            if "reliability" in v:
                qos = rclpy.qos.QoSProfile(depth=v["qos"], reliability=v["reliability"])

            clean_topic = k.strip().strip('/')
            self.topics[k]['clean_topic'] = clean_topic
            topic = f"/{namespace.strip().strip('/')}/{clean_topic}"
            self.create_subscription(v['type'], topic, partial(self.updater, k), qos, callback_group=group)
        print("Subscription setup:\t\n" + ','.join(k for k in self.topics.keys()))
        timer = Timer(1, self.throttle_setup)
        timer.start()

    def throttle_setup(self, max_rate=10):
        """ The second part of __init__, split in a different function. It's automatically run after a short time.
          In this way, the cache can fill up with some values and we do not need to create any initial object.

          It creates a timer that launches a custom store function for every topic

            Args:
                @param max_rate: rate of the store operation. Should be lower than all topics broadcast rate """

        print(f"Throttling timer setup (period: {1. / max_rate:.2f})", end='\n\t')
        timestr = time.strftime("%d%m-%H%M%S")
        Path(os.path.join('history', timestr)).mkdir(parents=True, exist_ok=True)

        for k, v in self.topics.items():
            print(f"{k}, ", end='')
            self.topics[k]['file_path'] = file_path = os.path.join("history", timestr,
                                                                   v['clean_topic'].replace('/', '_') + '.h5')
            min_itemsize = v.pop('min_itemsize', {})
            self.create_timer(1. / max_rate, partial(self.process_row, v["simplify"], k, file_path, min_itemsize),
                              callback_group=v['group'])
        print(','.join(k for k in self.topics.keys()))
        print("Recording.. ")
        self.create_timer(30, self.display_summary)

    def updater(self, key, msg):
        """ Saves whatever object it receives to a small cache, overriding the previous one """
        self.last_value[key] = (self.get_clock().now().nanoseconds, msg)

    def process_row(self, simplify, key, file_name, min_itemsize):
        """ Simplifies the last received messsage of a topic and appends it to a local cache.
         When the list is big enough, its content are stored to file system and the list is then emptied.
         Args:
            @param simplify: a custom function that returns only the bare minimum that is necessary from a message
            @param key: the topic name, which serves as key to obtain the last read value on the small cache
            @param file_name: the file path of the hdf5 storage for this topic
            @param min_itemsize: the minimum itemsize for the appended block in HDF5. Useful for images.
                    """

        timestamp, msg = self.last_value[key]
        msg = simplify(msg)

        lst = self.cache[key]
        if len(lst) >= 100:
            df = pd.DataFrame(lst).drop_duplicates('time').set_index('time')  # TODO: no race condition, no artifacts?
            with pd.HDFStore(file_name) as store:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=PerformanceWarning)
                    store.append(key, df, min_itemsize=min_itemsize)
            lst.clear()

        msg['time'] = timestamp
        lst.append(msg)

    def display_summary(self):
        """ Displays a quick summaries of the files that have been created so far and their sizes """
        longest_topic_name = max([len(k) for k, _ in self.topics.items()])
        file_size_summary = [f'\t{str(k).ljust(longest_topic_name + 1)}: ' \
                             f'{os.path.getsize(v["file_path"]) / 1000000.:.2f} MB'
                             for k, v in self.topics.items() if os.path.exists(v["file_path"])]
        if len(file_size_summary) > 0:
            print('Files:\n' + '\n'.join(file_size_summary))


def main(args=None):
    rclpy.init(args=args)
    best_effort = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT

    odom = (lambda msg: {'x': msg.pose.pose.position.x,
                         'y': msg.pose.pose.position.y,
                         'theta': quaternion_to_euler(msg.pose.pose.orientation)[2]})

    bridge = CvBridge()
    compress = lambda msg: {'image': binary_from_cv(bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'),
                                                    jpeg_quality=50)}
    recorder = Recorder(rclpy, topics={'odom': {'type': Odometry, 'qos': 20, 'simplify': odom},
                                       'ground_truth/odom': {'type': Odometry, 'qos': 10, 'reliability': best_effort,
                                                             'simplify': odom},
                                       'head_camera/image_raw': {'type': Image, 'qos': 30, 'reliability': best_effort,
                                                                 'simplify': compress,
                                                                 'min_itemsize': {'image': 20000}},
                                       'virtual_sensor/signal': {'type': Bool, 'qos': 30,
                                                                 'simplify': lambda msg: {'sensor': msg.data}},
                                       'run_counter': {'type': Int64, 'qos': 30,
                                                       'simplify': lambda msg: {'run': msg.data}}}
                        )

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print("\nRecording stopped")
    except KeyError as ex:
        print(f"\n{ex} topic is not broadcasting")
    finally:
        recorder.display_summary()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
