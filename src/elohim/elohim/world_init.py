import os

import numpy as np
import rclpy
from ament_index_python import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelList
from geometry_msgs.msg import Pose, Point
from std_srvs.srv import Empty

try: # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from service_utils import SyncServiceCaller, AsyncServiceCall, AsyncServiceCaller
    from utils import get_resource, generate_map, euler_to_quaternion
except ImportError:
    from elohim.service_utils import SyncServiceCaller, AsyncServiceCall, AsyncServiceCaller
    from elohim.utils import get_resource, generate_map, euler_to_quaternion


def main(args=None):
    rclpy.init(args=args)

    ssc = SyncServiceCaller(rclpy)  # una tantum operations, sync -> ~1 request/sec
    asc = AsyncServiceCaller(cache=40)  # parallel operations, async -> ~2-4 requests/sec depending on service

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

    # Spawn pickup targets

    # Car wheel, cinder block, cinderblock 2, construction cone, cordless drill,
    # first 2015 trash can, frc 2016 ball, grey tote, lamp post
    target = "coke_can"
    # xml_sdf = get_resource(target, root=os.path.join(Path('~').expanduser(), ".gazebo/models"))
    xml_sdf = get_resource(target, root=os.path.join(get_package_share_directory('elohim'), 'models'))

    np.random.seed(0xDEADBEEF)
    _, targets = generate_map(0xDEADBEEF, density=1.5)
    for i, (x, y) in enumerate(targets):
        id = f"{target}{i}"
        theta = np.random.uniform(0, np.pi * 2)
        node = AsyncServiceCall(srv=SpawnEntity, srv_namespace='spawn_entity', request_dict={
            "name": id,
            "xml": xml_sdf,
            "initial_pose": Pose(position=Point(x=x, y=y), orientation=euler_to_quaternion(yaw=theta))
        }, id=id)
        asc.add_node(node)

    asc.spin_and_join()
    ssc(srv=Empty, srv_namespace="unpause_physics")


if __name__ == '__main__':
    main()
