import inspect
import multiprocessing
import sys
import time
from functools import partial

from gazebo_msgs.srv._delete_entity import DeleteEntity_Response
from gazebo_msgs.srv._get_model_list import GetModelList_Response
from gazebo_msgs.srv._spawn_entity import SpawnEntity_Response
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node




class AsyncServiceCaller:
    """ Allows for multi-threaded requests on gazebo/ros listening services """

    def __init__(self, cache=None):

        if cache is None:
            cache = multiprocessing.cpu_count()
            print("Cache defaulting to", cache)
        self.cache = cache
        self.timeout = 1.  # any number > 0 should be good
        self.executor = MultiThreadedExecutor()
        self.total_tasks = 0

        self.reset_time = True

    def add_node(self, node):
        """ Add node to the execution pool """

        if self.reset_time:
            self.start = time.time()
            self.reset_time = False

        node.add_done_callback(self.executor)
        self.executor.add_node(node)
        self.total_tasks += 1
        if len(self.executor.get_nodes()) >= self.cache:
            self.spin_and_join(final=False)

    def spin_and_join(self, final=True):
        """ Empties the execution pool """

        if len(self.executor.get_nodes()) == 0:
            return

        elapsed = (time.time() - self.start)
        try:
            while len(self.executor.get_nodes()) > 0:
                self.executor.spin_once(self.timeout)
                print(f'\rLeft: {len(self.executor.get_nodes())}/{self.total_tasks}, {elapsed:.3f}s', end='')
        except KeyboardInterrupt:
            print(" Stopping.")
        finally:
            print(
                '\r' + f'Task completed so far: {self.total_tasks} - '
                       f'Total elapsed time: {elapsed:.3f}s - '
                       f'Speed: {self.total_tasks / elapsed:.2f} t/s',
                end='')

            if final:
                self.reset_time = True
                self.total_tasks = 0
                print()

    def __del__(self):
        self.executor.shutdown()


class AsyncServiceCall(Node):
    def __init__(self, srv, srv_namespace, request_dict=None, id=''):

        id = 'minimal_client_async' + id
        super().__init__(id)

        self.cli = self.create_client(srv, srv_namespace)
        while not self.cli.wait_for_service(timeout_sec=0.01):
            self.get_logger().info(f'service {srv_namespace} not available, waiting again...')
        self.req = srv.Request()

        if request_dict is None:
            request_dict = []
        for key in request_dict:
            setattr(self.req, key, request_dict[key])

        self.send_request()

    def add_done_callback(self, executor):
        self.executor = executor
        self.future.add_done_callback(self.destroy)

    def destroy(self, future):
        self.executor.remove_node(self)
        self.destroy_node()

    def send_request(self):
        self.future = self.cli.call_async(self.req)


def function_that_I_want(obj):
    mod = inspect.getmodule(obj)
    base, _sep, _stem = mod.__name__.partition('.')
    return sys.modules[base]

class SyncServiceCaller:

    def __init__(self, rclpy, node=None, destroy_node=True):
        self.rclpy = rclpy
        self.node = node
        self.destroy_node = destroy_node

    def __call__(self, srv=None, srv_namespace=None, request_dict=None, verbose=False):
        char_limit = 20
        if verbose:
            char_limit = sys.maxsize

        reqdic = ''
        if request_dict is not None:
            reqdic = ', '.join([f'{k}: \'{v if len(str(v)) < char_limit else "..."}\''
                      for k,v in request_dict.items()])
            reqdic = f'"{reqdic}"'

        t = f'{function_that_I_want(srv).__name__}/srv/{srv.__name__}'
        print(f'/{srv_namespace.ljust(20)} {t.ljust(30)} {reqdic}')
        node = AsyncServiceCall(srv=srv, srv_namespace=srv_namespace, request_dict=request_dict)

        while self.rclpy.ok():
            self.rclpy.spin_once(node)
            if node.future.done():
                try:
                    response = node.future.result()
                except Exception as e:
                    node.get_logger().info(
                        'Service call failed %r' % (e,))
                else:
                    response_interpreter(node.get_logger(), response)
                break

        node.destroy_node()

        return response


def response_interpreter(logger, res):
    if isinstance(res, SpawnEntity_Response) or isinstance(res, DeleteEntity_Response):
        logger.info(res.status_message)
    elif isinstance(res, GetModelList_Response):
        logger.info("Models: " + ", ".join(res.model_names))
    # elif isinstance(res, SetEntityState_Response):
    #    logger.info(res.success)
