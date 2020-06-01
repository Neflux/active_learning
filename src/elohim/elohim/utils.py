import json
import os

import numpy as np
from ament_index_python import get_package_share_directory
from elohim.poisson_disc import Grid
from geometry_msgs.msg import Pose, Point, Twist, Quaternion


def generate_map(seed):
    np.random.seed(seed)

    spawn_area = 20
    threshold = 1
    step = 5

    spawn_coords = np.stack(np.meshgrid(
        np.linspace(0, spawn_area, int(2 * spawn_area / step) + 1),
        np.linspace(0, spawn_area, int(2 * spawn_area / step) + 1)
    )).reshape([2, -1]).T

    n_targets = len(spawn_coords) // 2

    grid = Grid(threshold, spawn_area, spawn_area)
    points = grid.poisson(seed, static=spawn_coords)
    targets = np.array(points[len(spawn_coords):]) - spawn_area / 2
    np.random.shuffle(targets)
    targets = targets[:n_targets]
    spawn_coords -= spawn_area / 2

    print(f"Spawn points generated: {len(spawn_coords)}, \
            pick up targets: {n_targets}, radius: {threshold}")

    with open(os.path.join(get_package_share_directory('elohim'), 'points.json'), 'w') as f:
        json.dump({"targets": [{"x": x[0], "y": x[1]} for x in targets],
                   "spawn_coords": [{"x": x[0], "y": x[1]} for x in spawn_coords]}, f)

    return spawn_coords, targets


def get_resource(name, root=None):
    if root is None:
        root = os.path.join(get_package_share_directory('elohim'), "models")

    path = os.path.join(root, f"{name}/model.sdf")

    with open(path) as f:
        return f.read()


def euler_to_quaternion(yaw, pitch, roll):
    x = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    y = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    z = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    w = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return Quaternion(x=x, y=y, z=z, w=w)