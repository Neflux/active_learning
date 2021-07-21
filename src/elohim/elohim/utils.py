import io
import os
from random import choice

import cv2
import numpy as np
import pandas as pd
import torch

import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from tqdm import tqdm

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    import config
except ImportError:
    import elohim.config as config


def str_from_bytes(buf):
    with io.BytesIO() as memfile:
        np.save(memfile, buf)
        memfile.seek(0)
        return memfile.read().decode('latin-1')


def binary_from_cv(cv2_img, jpeg_quality=90):
    retval, buf = cv2.imencode('.JPEG', cv2_img,
                               [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
                                cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    return str_from_bytes(buf)


def bytes_from_str(serialized):
    with io.BytesIO() as memfile:
        memfile.write(serialized.encode('latin-1'))
        memfile.seek(0)
        return np.load(memfile)


def cv_from_binary(serialized):
    buf = bytes_from_str(serialized)
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


def fov_mask():
    mask = np.full((20, 20), fill_value=1)
    mask[np.tril_indices(20, k=-10)] = 0
    mask = np.rot90(mask, k=-1)
    mask[np.tril_indices(20, k=-10)] = 0
    return np.rot90(mask).astype(bool)


def predictions_from_dataframe(model, df, batch_size, device, bayes, transform, target_column='image'):
    model.eval()
    with torch.no_grad():
        batches = np.split(df, np.arange(batch_size, len(df), batch_size))
        omap, entr, mi = [], [], []
        appropriate_free = flexible_free_tensor(bayes)
        for x in tqdm(batches, desc=f'Computing predictions (batch size: {batch_size})'):
            image = torch.stack(x[target_column].apply(transform).values.tolist())
            preds = appropriate_free(model(image.to(device)))

            if bayes:
                preds = np.moveaxis(preds, 1, 0)

            ave_preds = preds
            if bayes:
                ave_preds = np.mean(ave_preds, axis=1)
                entropy_cells_exp = entropy(preds, axis=-1).mean(axis=1)
            else:
                entropy_cells_exp = 0
            entropy_cells = entropy(ave_preds, axis=-1)
            mutual_info = entropy_cells - entropy_cells_exp

            omap.append(ave_preds)
            entr.append(entropy_cells)
            mi.append(mutual_info)

        omap, entr, mi = np.vstack(omap)[..., -1], np.vstack(entr), np.vstack(mi)

    return [pd.Series(x.reshape(-1, 20, 20).tolist(), index=df.index) for x in [omap, entr, mi]]


def free_tensor(x: torch.Tensor):
    return x.detach().cpu().numpy()


def flexible_free_tensor(bayes):
    return (lambda x: np.array([free_tensor(s) for s in x])) if bayes else (lambda x: free_tensor(x))


def batch_entropy_mi(preds, ave_preds, bayes, minibatch_mean=True):
    # (minibatch, 400, 2) -> (minibatch, 400) -> (400)
    entropy_cells = entropy(ave_preds, axis=-1)
    if minibatch_mean:
        entropy_cells = entropy_cells.mean(axis=0)

    if bayes:

        entropy_cells_exp = entropy(preds, axis=-1).mean(axis=0)
        if minibatch_mean:
            # (samples, minibatch, 400, 2) -> (samples, minibatch, 400) -> (samples, 400) -> (400)
            entropy_cells_exp = entropy(preds, axis=-1).mean(axis=1).mean(axis=0)
    else:
        entropy_cells_exp = 0

    mutual_info = entropy_cells - entropy_cells_exp

    return entropy_cells, mutual_info


def entropy(x, axis=-1):
    return np.sum(-x * np.log2(x + 1e-10), axis=axis)
