import os
import time
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from shutil import copyfile

import pandas as pd
import rclpy
from ament_index_python import get_package_share_directory
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
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
    def __init__(self, rclpy, topics, target_node=None, save_topic=None, rate=None, namespace="thymioX"):
        """ Initialises the recording node. Sets up the subscriptions and the
        Args:
            @param topics: a dictionary that specifies the topics to record and all the necessary parameters.
                Each key should be the full string of a broadcasting topic and it should contain a dictionary with:
                    'type':  the python type of msg interface
                    'simplify': a function that returns only the bare minimum that is necessary from a message
                And optionally it contains:
                    'qos': the rate of broadcasting of said topic
                    'reliability': the reliability policy of the qos profile
                    'min_itemsize': a dictionary that specifies the block size for the HDF5, for each column label
                        """
        super().__init__('recorder_node')
        self.rlcpy = rclpy
        self.topics = topics
        r = min([v['qos'] for v in self.topics.values()])
        if rate is not None:
            if rate > r:
                self.get_logger().warn(
                    f'Your recording rate ({rate}) is bigger that the minimum rate of the topics ({r}).'
                    f' You will be losing a lot of data in the subsequent merging operation.')
        else:
            rate = r
            self.get_logger().debug(f'Recording rate automatically set to {rate}')
        self.period = Duration(seconds=1. / rate)

        self.last_time = {}
        self.cache = defaultdict(list)
        self.cache_count = defaultdict(lambda: 0)
        self.subs_handlers = []

        # self.group = MutuallyExclusiveCallbackGroup()
        if target_node is not None:
            while target_node not in self.get_node_names():
                print(f'Target topic "{target_node}" is not running. Re-checking in 3 seconds.. ')
                time.sleep(3)

        timestr = time.strftime("%d%m-%H%M%S")
        Path(os.path.join('history', timestr)).mkdir(parents=True, exist_ok=True)
        # TODO: copy here the points file
        copyfile(os.path.join(get_package_share_directory('elohim'), 'points.json'),
                 os.path.join("history", timestr, 'points.json'))

        self.last_value = None
        self.save_topic = save_topic['name']

        self.should_save = {}
        start = self.get_clock().now()
        for k, v in self.topics.items():

            self.last_time[k] = start
            self.topics[k]['group'] = group = None  # MutuallyExclusiveCallbackGroup()

            qos = v["qos"]
            if "reliability" in v:
                qos = rclpy.qos.QoSProfile(depth=v["qos"], reliability=v["reliability"])

            self.topics[k]['clean_topic'] = clean_topic = k.strip().strip('/')
            topic = f"/{namespace.strip().strip('/')}/{clean_topic}"

            self.topics[k]['file_path'] = file_path = os.path.join("history", timestr,
                                                                   v['clean_topic'].replace('/', '_') + '.h5')
            min_itemsize = v.pop('min_itemsize', {})

            self.topics[k]['store'] = store = pd.HDFStore(file_path)
            self.create_subscription(v['type'], topic, partial(self.process_row, v["simplify"], k, store, min_itemsize),
                                     qos, callback_group=group)

            self.should_save[k] = False
            if k == self.save_topic:
                max_rate = max([v['qos'] for v in self.topics.values()])
                print(f'Status of topic "{k}" will be used to schedule saving operations (rate: {max_rate})')
                self.create_subscription(v['type'], topic,
                                         partial(self.change_listener, v['simplify'], save_topic['save_value']),
                                         max_rate, callback_group=group)



        print("Subscription setup:\t\n" + ','.join(k for k in self.topics.keys()))

        self.create_timer(30, partial(self.display_summary, simple=True))

    def change_listener(self, simplify, target, msg):
        current_value = simplify(msg)['run']

        if current_value != self.last_value and current_value == target:
            #print(f'SHOULD NOW SAVE ({self.last_value} -> {current_value})')
            self.should_save.update({k: True for k in self.topics.keys()})

        self.last_value = current_value

    def process_row(self, simplify, key, store, min_itemsize, msg):
        """ Simplifies the last received messsage of a topic and appends it to a local cache.
         When the list is big enough, its content are stored to file system and the list is then emptied.
         Args:
            @param simplify: a custom function that returns only the bare minimum that is necessary from a message
            @param key: the topic name, which serves as key to obtain the last read value on the small cache
            @param store: the file handle of the hdf5 storage for this topic
            @param min_itemsize: the minimum itemsize for the appended block in HDF5. Useful for images.
                    """
        if self.get_clock().now() - self.last_time[key] < self.period:
            return

        lst = self.cache[key]
        msg = simplify(msg)

        if self.should_save[key] and len(lst) > 0:
            df = pd.DataFrame(lst).drop_duplicates('time').set_index(
                'time')  # TODO: no race condition, no artifacts?
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=PerformanceWarning)
                store.append(key, df, min_itemsize=min_itemsize)
            lst.clear()
            self.should_save[key] = False

        self.last_time[key] = time = self.get_clock().now()
        msg['time'] = time.nanoseconds
        lst.append(msg)

    def display_summary(self, simple=False):
        """ Displays a quick summaries of the files that have been created so far and their sizes """
        # print(self.topics['run_counter']['store'].info())
        if simple:
            sum_bytes = sum([os.path.getsize(v["file_path"])
                             for v in self.topics.values() if os.path.exists(v["file_path"])])
            print(f'Files total size: {sum_bytes / 1000000.:.2f} MB')
        else:
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

    recorder = Recorder(rclpy, target_node='random_controller',
                        topics={'odom': {'type': Odometry, 'qos': 20, 'simplify': odom},
                                'ground_truth/odom': {'type': Odometry, 'qos': 10, 'reliability': best_effort,
                                                      'simplify': odom},
                                'head_camera/image_raw': {'type': Image, 'qos': 30, 'reliability': best_effort,
                                                          'simplify': compress,
                                                          'min_itemsize': {'image': 20000}},
                                'virtual_sensor/signal': {'type': Bool, 'qos': 30,
                                                          'simplify': lambda msg: {'sensor': msg.data}},
                                'run_counter': {'type': Int64, 'qos': 30,
                                                'simplify': lambda msg: {'run': msg.data}}},
                        save_topic={'name': 'run_counter', 'save_value': -1})
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        print("\nRecording stopped")
    finally:
        for v in recorder.topics.values():
            v['store'].close()

        recorder.display_summary()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
