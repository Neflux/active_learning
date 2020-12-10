import io
import json
import os
from random import choice

import cv2
import numpy as np
import torch
from PIL import Image
from ament_index_python import get_package_share_directory
from elohim.poisson_disc import Grid
from geometry_msgs.msg import Quaternion
import math

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    import config
except ImportError:
    import elohim.config as config


def generate_map(seed, density=1, threshold=1, spawn_area=20, step=5):
    np.random.seed(seed)

    spawn_coords = np.stack(np.meshgrid(
        np.linspace(0, spawn_area, int(2 * spawn_area / step) + 1),
        np.linspace(0, spawn_area, int(2 * spawn_area / step) + 1)
    )).reshape([2, -1]).T

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


def mktransf(pose):
    """Returns a trasnformation matrix given a (x, y, theta) pose."""
    assert len(pose) == 3
    cos = np.cos(pose[2])
    sin = np.sin(pose[2])
    return np.array([[cos, -sin, pose[0]],
                     [sin, cos, pose[1]],
                     [0, 0, 1]])


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


ROBOT_GEOMETRY_FULL = [
    mktransf((0.0630, 0.0493,
              quaternion2yaw([0.0, 0.0, 0.3256, 0.9455]))),  # left
    mktransf((0.0756, 0.0261,
              quaternion2yaw([0.0, 0.0, 0.1650, 0.9863]))),  # center_left
    mktransf((0.0800, 0.0000, 0.0000)),  # center
    mktransf((0.0756, -0.0261,
              quaternion2yaw([0.0, 0.0, -0.1650, 0.9863]))),  # center_right
    mktransf((0.0630, -0.0493,
              quaternion2yaw([0.0, 0.0, -0.3256, 0.9455])))  # right
]

ROBOT_GEOMETRY_SIMPLE = [
    mktransf((0.0800, 0.0000, 0.0000)),  # center
]

COORDS = np.stack(np.meshgrid(
    np.linspace(0, .8, int(.8 / .04)),
    np.linspace(-.4, .4, int(.8 / .04))
)).reshape([2, -1]).T

def binary_from_cv(cv2_img, jpeg_quality=90):
    retval, buf = cv2.imencode('.JPEG', cv2_img,
                               [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                                cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    with io.BytesIO() as memfile:
        np.save(memfile, buf)
        memfile.seek(0)
        return memfile.read().decode('latin-1')


def cv_from_binary(serialized):
    with io.BytesIO() as memfile:
        memfile.write(serialized.encode('latin-1'))
        memfile.seek(0)
        buf = np.load(memfile)
    return cv2.imdecode(buf, flags=cv2.IMREAD_UNCHANGED)


import pandas as pd


class print_full():
    def __init__(self, x=None):
        if x is None:
            x = []
        self.xlen = len(x)

    def __enter__(self):
        pd.set_option('display.max_rows', self.xlen)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        pd.set_option('display.float_format', '{:20,.2f}'.format)
        pd.set_option('display.max_colwidth', None)

    def __exit__(self, type, value, traceback):
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.float_format')
        pd.reset_option('display.max_colwidth')


maxcppfloat = 340282346638528859811704183484516925440


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

def _moveaxis(tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)


def random_session_name():
    colors = os.path.join('utils', 'colors.txt')
    animals = os.path.join('utils', 'animals.txt')
    if os.path.exists(colors) and os.path.exists(animals):
        with open(colors) as col:
            colors = [l.strip() for l in col.readlines()]
        with open(animals) as ani:
            animals = [l.strip() for l in ani.readlines()]
    else:
        print('Animals/colors .txt files not found in /utils/*, using hardcoded combinations')
        colors = ['green', 'red', 'blue']
        animals = ['panda', 'maverick', 'fox']

    result = choice(colors) + '-' + choice(animals)
    return result.lower()