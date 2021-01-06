import io
import os
from random import choice

import cv2
import numpy as np
import pandas as pd
import torch


import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    import config
except ImportError:
    import elohim.config as config


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


def _moveaxis(tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)


def random_session_name():
    colors = os.path.join('colors.txt')
    animals = os.path.join('animals.txt')
    if os.path.exists(colors) and os.path.exists(animals):
        with open(colors) as col:
            colors = [l.strip() for l in col.readlines()]
        with open(animals) as ani:
            animals = [l.strip() for l in ani.readlines()]
    else:
        print('Animals/colors .txt files not found in /utils/*, using hardcoded combinations')
        colors = ['green', 'red', 'blue', 'cyan', 'orange', 'grey', 'black', 'white', 'brown', 'yellow', 'purple']
        animals = ['panda', 'maverick', 'fox', 'hippo', 'monkey', 'donkey', 'gnu', 'snake', 'eagle', 'zebra', 'rabbit']

    result = choice(colors) + '-' + choice(animals)
    return result.lower()


def plot_transform(ax, tr, color='b', length=1, head_width=0.2):
    origin = (tr @ np.array([0, 0, 1]))[:2]
    xhat = (tr @ np.array([length, 0, 1]))[:2]
    # yhat = (tr @ np.array([0, length_y, 1]))[:2]
    ax.arrow(*origin, *(xhat - origin), head_width=head_width, color=color, zorder=3)


def mktransf(pose):
    """Returns a trasnformation matrix given a (x, y, theta) pose."""
    assert len(pose) == 3
    cos = np.cos(pose[2])
    sin = np.sin(pose[2])
    return np.array([[cos, -sin, pose[0]],
                     [sin, cos, pose[1]],
                     [0, 0, 1]])


COORDS = np.stack(np.meshgrid(
    np.linspace(0, .8, int(.8 / .04)),
    np.linspace(-.4, .4, int(.8 / .04))
)).reshape([2, -1]).T

ROBOT_GEOMETRY_FULL = [
    mktransf((0.0630, 0.0493, 0.6632885724142987)),  # left
    mktransf((0.0756, 0.0261, 0.3315180299646234)),  # center_left
    mktransf((0.0800, 0.0000, 0.0000)),  # center
    mktransf((0.0756, -0.0261, -0.3315180299646234)),  # center_right
    mktransf((0.0630, -0.0493, -0.6632885724142987))  # right
]



def scaled_full_robot_geometry(ratio):
    return [
        mktransf((0.0630 * ratio, 0.0493 * ratio, 0.6632885724142987)),  # left
        mktransf((0.0756 * ratio, 0.0261 * ratio, 0.3315180299646234)),  # center_left
        mktransf((0.0800 * ratio, 0.0000 * ratio, 0.0000)),  # center
        mktransf((0.0756 * ratio, -0.0261 * ratio, -0.3315180299646234)),  # center_right
        mktransf((0.0630 * ratio, -0.0493 * ratio, -0.6632885724142987))  # right
    ]


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5 * height, width, 0, length_includes_head=True, head_width=0.75 * height)
    return p


class HandlerRect(HandlerPatch):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):
        x = width // 2 - width // 3
        y = 0
        w = h = 10

        # create
        p = mpatches.Rectangle(xy=(x, y), width=w, height=h)

        # update with data from oryginal object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]
