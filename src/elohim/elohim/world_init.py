import json
import os
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from elohim.service_utils import AsyncServiceCaller
from elohim.utils import get_resource, generate_map, euler_to_quaternion
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelList
from geometry_msgs.msg import Pose, Point
from std_srvs.srv import Empty


def main(args=None):
    rclpy.init(args=args)
    asc = AsyncServiceCaller(rclpy)

    # Clean up current world

    asc(srv=Empty, srv_namespace="pause_physics")
    asc(srv=Empty, srv_namespace="reset_world")
    asc(srv=Empty, srv_namespace="reset_simulation")

    response = asc(srv=GetModelList, srv_namespace="get_model_list")
    for m in response.model_names:
        if "thymio" not in m:
            asc(srv=DeleteEntity, srv_namespace="delete_entity", request_dict={"name": m})

    # Spawn new, textured, plane

    asc(srv=SpawnEntity, srv_namespace="spawn_entity", request_dict={"name": "widmanstatten_plane",
                                                                     "xml": get_resource("widmanstatten_plane")})

    # Spawn pickup targets

    target = "coke_can"
    # xml_sdf = get_resource(target, root=os.path.join(Path('~').expanduser(), ".gazebo/models"))
    xml_sdf = get_resource(target, root=os.path.join(get_package_share_directory('elohim'), 'models'))

    def spawn_target(x, y):
        theta = np.random.uniform(0, np.pi * 2)
        asc(srv=SpawnEntity, srv_namespace="spawn_entity",
            request_dict={
                "name": f"{target}{i}",
                "xml": xml_sdf,
                "initial_pose": Pose(position=Point(x=x, y=y), orientation=euler_to_quaternion(yaw=theta))
            })

    _, targets = generate_map(0xDEADBEEF)

    for i in range(len(targets)):
        spawn_target(*targets[i])

    asc(srv=Empty, srv_namespace="unpause_physics")


if __name__ == '__main__':
    main()
