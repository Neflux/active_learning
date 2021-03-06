import json
import math
import os

import numpy as np
from PIL import Image
from ament_index_python import get_package_share_directory
from elohim.poisson_disc import Grid
from geometry_msgs.msg import Quaternion
from scipy.spatial.distance import cdist

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    import config
except ImportError:
    import elohim.config as config

def generate_safe_map(spawn_area = 20, step = 5, limit = 2, num_obs=625):

    # Safe internal spawn coordinates
    ls = np.linspace(0, spawn_area, int(2 * spawn_area / step) + 1) - spawn_area / 2
    spawn_coords = np.stack(np.meshgrid(ls[limit:-limit], ls[limit:-limit])).reshape([2, -1]).T

    # Obstacle coordinates
    ls = np.linspace(0, spawn_area, int(np.sqrt(num_obs))+1)
    a = np.stack(np.meshgrid(ls, ls)).reshape([2, -1]).T
    a = a + np.random.normal(0, 0.2, size=a.shape)
    a -= spawn_area /2

    # Remove obstacles too close too spawns
    d = cdist(spawn_coords, a)
    mask = np.min(d, axis=0) < 0.4
    a = a[~mask]

    # Remove obstacles too close to each other
    d = cdist(a, a)+np.eye(a.shape[0])
    mask = np.min(d, axis=0) < 0.15
    targets = a[~mask]

    with open(os.path.join(get_package_share_directory('elohim'), 'points.json'), 'w') as f:
        json.dump({"targets": [{"x": x[0], "y": x[1]} for x in targets],
                   "spawn_coords": [{"x": x[0], "y": x[1]} for x in spawn_coords]}, f)

    return spawn_coords, targets

def generate_map(seed, density=1, threshold=1, spawn_area=20, step=5, limit = 2):
    np.random.seed(seed)

    ls = np.linspace(0, spawn_area, int(2 * spawn_area / step) + 1)
    spawn_coords = np.stack(np.meshgrid(ls[limit:-limit], ls[limit:-limit])).reshape([2, -1]).T

    n_targets = int(len(spawn_coords) * density)

    grid = Grid(threshold, spawn_area, spawn_area)
    points = grid.poisson(seed, static=spawn_coords)
    targets = np.array(points[len(spawn_coords):]) - spawn_area / 2
    np.random.shuffle(targets)

    targets = targets[:n_targets]
    spawn_coords -= spawn_area / 2

    print(f"Spawn points generated: {len(spawn_coords)}, "
          f"Pick up targets: {n_targets}, "
          f"Poisson disk radius: {threshold}")

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


def euler_to_quaternion(roll=0, pitch=0, yaw=0):
    x = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    y = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    z = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    w = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return Quaternion(x=x, y=y, z=z, w=w)





def quaternion_to_euler(q):
    # roll x
    t0 = +2.0 * (q.w * q.x + q.y * q.z)
    t1 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (q.w * q.y - q.z * q.x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    Z = math.atan2(t3, t4)

    # result = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
    # if dict:
    #    return result

    return X, Y, Z


def quaternion2yaw(i):
    x, y, z, w = i
    q = Quaternion(x=x, y=y, z=z, w=w)
    return quaternion_to_euler(q)[2]



def random_PIL():
    a = np.random.rand(*config.camera_shape) * 255
    return Image.fromarray(a.astype('uint8')).convert('RGB')


import matplotlib
import matplotlib.pyplot as plt


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def add_arrow(line, xdata, ydata, step=1, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    #xdata = line.get_xdata()
    #ydata = line.get_ydata()

    # if position is None:
    #     position = xdata.mean()
    # # find closest index
    # start_ind = np.argmin(np.absolute(xdata - position))

    for start_ind in range(len(xdata)-step, step, -step):

        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1
        line.axes.annotate('',
                           xytext=(xdata[start_ind], ydata[start_ind]),
                           xy=(xdata[end_ind], ydata[end_ind]),
                           arrowprops=dict(arrowstyle="->", color=color),
                           size=size
                           )
