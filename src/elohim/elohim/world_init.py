import json
import os
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor

from service_utils import SyncServiceCaller, AsyncServiceCall, response_interpreter, AsyncServiceCaller
from utils import get_resource, generate_map, euler_to_quaternion
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelList
from geometry_msgs.msg import Pose, Point
from std_srvs.srv import Empty


def main(args=None):
    rclpy.init(args=args)

    ssc = SyncServiceCaller(rclpy)
    asc = AsyncServiceCaller()

    # Clean up current world

    ssc(srv=Empty, srv_namespace="pause_physics")
    ssc(srv=Empty, srv_namespace="reset_world")
    ssc(srv=Empty, srv_namespace="reset_simulation")

    response = ssc(srv=GetModelList, srv_namespace="get_model_list")

    for m in response.model_names:
        if "thymio" not in m:
            node = AsyncServiceCall(srv=DeleteEntity, srv_namespace="delete_entity", request_dict={"name": m}, id=m)
            node.send_request()
            asc.add_node(node)

    asc.spin_and_join()

    # Spawn new, textured, plane

    ssc(srv=SpawnEntity, srv_namespace="spawn_entity", request_dict={"name": "widmanstatten_plane",
                                                                     "xml": get_resource("widmanstatten_plane")})

    # Spawn pickup targets

    target = "coke_can"
    # xml_sdf = get_resource(target, root=os.path.join(Path('~').expanduser(), ".gazebo/models"))
    xml_sdf = get_resource(target, root=os.path.join(get_package_share_directory('elohim'), 'models'))

    _, targets = generate_map(0xDEADBEEF, density=1.5)

    for i, (x, y) in enumerate(targets):
        id = f"{target}{i}"
        theta = np.random.uniform(0, np.pi * 2)
        node = AsyncServiceCall(srv=SpawnEntity, srv_namespace='spawn_entity', request_dict={
            "name": id,
            "xml": xml_sdf,
            "initial_pose": Pose(position=Point(x=x, y=y), orientation=euler_to_quaternion(yaw=theta))
        }, id=id)
        node.send_request()
        asc.add_node(node)

    asc.spin_and_join()
    ssc(srv=Empty, srv_namespace="unpause_physics")



if __name__ == '__main__':
    main()
