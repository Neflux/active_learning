import os
from pathlib import Path

import numpy as np
import rclpy
from elohim.utils import get_resource, handy_pose, generate_map
from elohim.utils import get_resource, handy_pose, generate_map
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelList
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
    # TODO: copy it in the local folder /models, now that everything works
    xml_sdf = get_resource(target, root=os.path.join(Path('~').expanduser(), ".gazebo/models"))
    def spawn_target(x, y):
        theta = np.random.uniform(0, np.pi * 2)
        asc(srv=SpawnEntity, srv_namespace="spawn_entity",
            request_dict={
                "name": f"{target}{i}",
                "xml": xml_sdf,
                "initial_pose": handy_pose(x, y, theta)
            })

    spawn_coords, targets = generate_map(0xDEADBEEF)

    for i in range(len(targets)):
        spawn_target(*targets[i])

    asc(srv=Empty, srv_namespace="unpause_physics")

    virtual_sensor = VirtualSensor(targets=random_positions, threshold=1)

    rclpy.spin(virtual_sensor)

    virtual_sensor.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
