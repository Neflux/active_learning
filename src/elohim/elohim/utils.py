import io
import os
from random import choice

import cv2
import numpy as np
import torch
import pandas as pd

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
