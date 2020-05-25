from gazebo_msgs.srv._delete_entity import DeleteEntity_Response
from gazebo_msgs.srv._get_model_list import GetModelList_Response
from gazebo_msgs.srv._spawn_entity import SpawnEntity_Response
from rclpy.node import Node


class AsyncServiceCall(Node):
    def __init__(self, srv, srv_namespace, request_dict=None):
        super().__init__('minimal_client_async')

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


class AsyncServiceCaller:

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