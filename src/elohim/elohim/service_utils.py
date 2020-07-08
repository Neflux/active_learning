import time

from gazebo_msgs.srv._delete_entity import DeleteEntity_Response
from gazebo_msgs.srv._get_model_list import GetModelList_Response
from gazebo_msgs.srv._spawn_entity import SpawnEntity_Response
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class AsyncServiceCaller:
    def __init__(self):
        self.executor = MultiThreadedExecutor()
        self.nodes = []
        self.start = time.time()
        self.total_tasks = 0

    def add_node(self, node):
        self.executor.add_node(node)
        self.nodes.append(node)
        self.total_tasks += 1
        if len(self.nodes) == 50:
            self.spin_and_join()

    def spin_and_join(self):
        if len(self.nodes) == 0:
            return

        try:
            while len(self.nodes) > 0:
                self.executor.spin_once(0.5)
                print(f'\rLeft: {len(self.nodes)}', end='')
                delete_these = []
                for i, node in enumerate(self.nodes):
                    if node.future.done():
                        try:
                            node.future.result()
                        except Exception as e:
                            node.get_logger().info(
                                'Service call failed %r' % (e,))
                        finally:
                            node.destroy_node()
                            delete_these.append(i)

                for index in sorted(delete_these, reverse=True):
                    del self.nodes[index]
        finally:
            print('\r'+f'Task completed so far: {self.total_tasks} - Total elapsed time: {(time.time()-self.start):.3f}s', end='')

    def __del__(self):
        self.executor.shutdown()

class AsyncServiceCall(Node):
    def __init__(self, srv, srv_namespace, request_dict=None, id=''):

        id = 'minimal_client_async'+id
        super().__init__(id)

        self.cli = self.create_client(srv, srv_namespace)
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'service {srv_namespace} not available, waiting again...')
        self.req = srv.Request()

        if request_dict is None:
            request_dict = []
        for key in request_dict:
            setattr(self.req, key, request_dict[key])

    def send_request(self):
        self.future = self.cli.call_async(self.req)


class SyncServiceCaller:

    def __init__(self, rclpy, node=None, destroy_node=True):
        self.rclpy = rclpy
        self.node = node
        self.destroy_node = destroy_node

    def __call__(self, srv=None, srv_namespace=None, request_dict=None):

        node = AsyncServiceCall(srv=srv, srv_namespace=srv_namespace, request_dict=request_dict)
        node.send_request()

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
    #elif isinstance(res, SetEntityState_Response):
    #    logger.info(res.success)
