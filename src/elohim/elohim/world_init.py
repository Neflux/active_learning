import json
import os
import random

import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelList
from geometry_msgs.msg import Pose, Point
from std_srvs.srv import Empty

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from service_utils import SyncServiceCaller, AsyncServiceCall, AsyncServiceCaller
    from utils_ros import get_resource, generate_map, euler_to_quaternion, generate_safe_map
    import config
except ImportError:
    from elohim.service_utils import SyncServiceCaller, AsyncServiceCall, AsyncServiceCaller
    from elohim.utils_ros import get_resource, generate_map, euler_to_quaternion, generate_safe_map
    import elohim.config as config


def main(args=None):
    rclpy.init(args=args)

    ssc = SyncServiceCaller(rclpy)  # una tantum operations, sync -> ~1 request/sec
    asc = AsyncServiceCaller(
        cache=config.async_requests_pool_size)  # parallel operations, async -> ~2-4 requests/sec depending on service

    # Clean up current world

    ssc(srv=Empty, srv_namespace="pause_physics")
    ssc(srv=Empty, srv_namespace="reset_world")
    ssc(srv=Empty, srv_namespace="reset_simulation")

    response = ssc(srv=GetModelList, srv_namespace="get_model_list")
    if len(response.model_names) > 0:
        print('Removing any existing model (except for the Thymio)..')
    for m in response.model_names:
        if "thymio" not in m:
            node = AsyncServiceCall(srv=DeleteEntity, srv_namespace="delete_entity", request_dict={"name": m}, id=m)
            asc.add_node(node)

    asc.spin_and_join()

    # Spawn new, textured, plane

    ssc(srv=SpawnEntity, srv_namespace="spawn_entity", request_dict={"name": "widmanstatten_plane",
                                                                     "xml": get_resource("widmanstatten_plane")})

    # ssc(srv=SpawnEntity, srv_namespace="spawn_entity",
    #     request_dict={"name": "baylands",
    #                   "xml": get_resource("baylands"),
    #                   "initial_pose": Pose(position=Point(x=37.185863,y=191.361298, z=float(-0.91 + 1.3 - 0.5)))})
    # Spawn pickup targets

    # Car wheel, cinder block, cinderblock 2, construction cone, cordless drill,
    # first 2015 trash can, frc 2016 ball, grey tote, lamp post

    # xml_sdf = get_resource(target, root=os.path.join(Path('~').expanduser(), ".gazebo/models"))
    xml_sdf_standard = get_resource(config.standard_obstacle,
                                    root=os.path.join(get_package_share_directory('elohim'), 'models'))
    xml_sdf_new = get_resource(config.new_obstacle, root=os.path.join(get_package_share_directory('elohim'), 'models'))

    np.random.seed(config.poisson_generation_seed)
    print('Calculating obstacles..')

    with open(os.path.join(get_package_share_directory('elohim'), 'points.json'), 'r') as f:
       targets = np.array([[t["x"], t["y"]] for t in json.load(f)["targets"]])
    # _, targets = generate_safe_map(num_obs=1250, center_spawn=True)

    print(f'Placing {len(targets)} obstacles..')
    for i, (x, y) in enumerate(targets):
        xml = xml_sdf_standard
        target = config.standard_obstacle
        if random.random() > 1.:
            xml = xml_sdf_new
            target = config.new_obstacle

        id = f"{target}{i}"
        theta = np.random.uniform(0, np.pi * 2)
        node = AsyncServiceCall(srv=SpawnEntity, srv_namespace='spawn_entity', request_dict={
            "name": id, "xml": xml,
            "initial_pose": Pose(position=Point(x=x, y=y), orientation=euler_to_quaternion(yaw=theta))
        }, id=id)
        asc.add_node(node)

    asc.spin_and_join()
    ssc(srv=Empty, srv_namespace="unpause_physics")


if __name__ == '__main__':
    main()
