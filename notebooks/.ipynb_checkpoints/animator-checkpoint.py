import glob
import itertools
import json
import logging as log
import os
import random
import subprocess
import time
import warnings
from math import sin, cos, sqrt

import matplotlib.colors as mcolors
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ament_index_python import get_package_share_directory
from matplotlib import axes, figure, transforms, animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Circle
from scipy.spatial import distance
from scipy.stats import stats
from tables import PerformanceWarning

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from utils import cv_from_binary, ROBOT_GEOMETRY_SIMPLE, print_full, mktransf, COORDS
    import config
except ImportError:
    from elohim.utils import cv_from_binary, ROBOT_GEOMETRY_SIMPLE, print_full, mktransf, COORDS
    import elohim.config as config


class Animator:
    def __init__(self, df, rate=30, save_path=None, frames=None):
        self.rate = rate

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=rate, metadata=dict(artist='Me'), bitrate=1800)
        if frames is None:
            frames = len(df)

        # Cache

        self.sensor_readings = df['sensor'].to_numpy()
        self.odom_pos_data = df[['x', 'y', 'theta']].to_numpy()
        self.gt_pos_data = df[['ground_truth_odom_x', 'ground_truth_odom_y', 'ground_truth_odom_theta']].to_numpy()
        self.images = df['image'].map(cv_from_binary).map(Image.fromarray, "RGB").to_numpy()

        self.run_counter = df['run'].to_numpy()
        rd = df.groupby('run')['x'].count().astype(int)
        rd = pd.DataFrame({'len': rd, 'end': rd.cumsum().astype(int)})
        self.run_dict = rd.to_dict()

        self.omaps = [x.reshape(20, 20) if isinstance(x, np.ndarray) else x for x in df['occupancy_map']]
        self.colored_omaps = []
        for x in self.omaps:
            if type(x) is np.ndarray:
                x = x.reshape(20, 20)
                res = np.empty((20, 20, 3))
                res[x == -1] = (190, 190, 190)
                res[x == 0] = (0, 255, 0)
                res[x == 1] = (255, 0, 0)
                self.colored_omaps.append(res)
            else:
                self.colored_omaps.append(np.full((20, 20, 3), 0))

        self.fov_segments = 40

        def gradient_line(cmap):
            """ Returns a LineCollection composition that results in a single line colored with a gradient """
            gradient = np.linspace(0.0, 1.0, self.fov_segments)
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(1, 0, cmap.N)
            my_cmap = ListedColormap(my_cmap)
            return LineCollection([], array=gradient, cmap=my_cmap, linewidth=1.2)

        self.length = 10

        def gradient_area(color):
            """ Returns an AxesImage and its Polygon perimeter for the Thymio FOV """
            gradient = np.empty((100, 1, 4), dtype=float)
            rgb = mcolors.colorConverter.to_rgb(color)
            gradient[:, :, :3] = rgb
            gradient[:, :, -1] = np.linspace(0., .1, 100)[:, None]

            ext = [-self.length / sqrt(2), self.length / sqrt(2), -self.length / sqrt(2), 0]
            fov_area = ax1.imshow(gradient, aspect='auto', extent=ext, origin='lower')

            clip_path = Polygon(np.empty((3, 2)), fill=False, alpha=0)
            return fov_area, clip_path

        from matplotlib import rc
        rc('axes', unicode_minus=False)
        # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('font', **{'family': 'serif', 'serif': ['Sathu']})  # Hiragino Maru Gothic Pro
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5),
                                            dpi=120)  # type:figure.Figure, (axes.Axes, axes.Axes, axes.Axes)
        self.ax1 = ax1

        """ Subplot 1 - Top View """
        half_side = config.plane_side // 2
        bounds = [-half_side - 1, half_side + 1]
        ax1.set(title="Top view", xlim=bounds, ylim=bounds, autoscale_on=False, adjustable='box')
        major_ticks = np.arange(-half_side, half_side + 1, 5)
        minor_ticks = np.arange(-half_side, half_side + 1, 2.5)
        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.set_yticks(major_ticks)
        ax1.set_yticks(minor_ticks, minor=True)
        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.5)

        # Boundaries
        b = half_side + 0.1
        ax1.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], color='red', linewidth=1.2)

        # Thymio objects
        def thymio(color, fovline_cmap):
            """ Returns a dictionary of matplotlib objects for a Thymio instance on the top-down plot """
            thymio, = ax1.plot([], [], 'o', color=color, zorder=3, ms=1.)
            sensor_radius = ax1.add_artist(Circle((0, 0), 0, fill=False, linestyle='--', alpha=0.5, linewidth=0.7))
            intent = ax1.add_line(Line2D([], [], linestyle='--', color='black', linewidth=1.0, alpha=0.8))
            lc1, lc2 = [ax1.add_collection(gradient_line(fovline_cmap)) for _ in range(2)]
            fov_area, clip_path = gradient_area(color)
            ax1.add_patch(clip_path)
            fov_area.set_clip_path(clip_path)
            return {'thymio': thymio, 'sensor_radius': sensor_radius, 'intent': intent, 'lc1': lc1,
                    'lc2': lc2, 'fov_area': fov_area, 'clip_path': clip_path}

        self.gt_artists = thymio('blue', pl.cm.Blues_r)
        self.odom_artists = thymio('orange', pl.cm.Oranges_r)

        # Targets
        self.targets, = ax1.plot(static_targets[:, 0], static_targets[:, 1], 'ro', ms=2)

        """ Subplot 2 - Camera Feed """

        ax2.set_title("Camera view")
        ax2.axis('off')
        self.camera = ax2.imshow(self.images[0])
        self.run_indicator = ax2.text(0.5, -0.2, '', ha="center", va="center", color='black', transform=ax2.transAxes,
                                      fontdict={'family': 'serif', 'size': 10})

        """ Subplot 3 - Occupancy Map """

        ax3.set_title('Occupancy map')
        ax3.axis('off')
        self.occ_map = ax3.imshow(self.colored_omaps[0])
        self.labels = []
        for i, j in self.OCC_MAP_SQUARES:
            self.labels.append(ax3.text(j, i, '', ha="center", va="center", color="w",
                                        fontdict={'family': 'serif', 'weight': 'bold', 'size': 8}))

        self.ax1.set_aspect('equal')
        fig.canvas.set_window_title("Simulation")
        self.anim = FuncAnimation(fig, self.animate, frames=frames, interval=1000. / rate, blit=True,
                                  init_func=self.init, repeat=True)

        if save_path is None:
            plt.show()
        else:
            plt.close()
            self.anim.save(save_path, writer=writer,
                           progress_callback=lambda i, n: print(f'\rSaving: {i * 100. / n:.2f} %', end=''))
            print(f"\rSaving process complete. File location: {save_path}")

    def init(self):
        """ FuncAnimation init function, mandatory for the setup """
        for artist in [self.gt_artists, self.odom_artists]:
            artist['thymio'].set_data([], [])
            artist['intent'].set_data([], [])
            artist['lc1'].set_segments([])
            artist['lc2'].set_segments([])
            artist['fov_area'].set_transform(self.ax1.transData)
            artist['clip_path'].set_xy(np.empty((3, 2)))

        self.camera.set_data(self.images[0])
        self.occ_map.set_data(np.random.choice([0, 128, 255], size=config.occupancy_map_shape))

        self.occ_map.set_data(self.colored_omaps[0])
        # for i, j in self.OCC_MAP_SQUARES:
        #    self.labels[i * config.occupancy_map_shape[0] + j].set_text('')

        return *self.gt_artists.values(), *self.odom_artists.values(), \
               self.camera, self.run_indicator, \
               self.occ_map, *self.labels,

    def animate(self, i):
        """ FuncAnimation animate function """

        for artist, positional_data in zip([self.gt_artists, self.odom_artists],
                                           [self.gt_pos_data, self.odom_pos_data]):
            x, y, theta = positional_data[i]

            artist['thymio'].set_data([x], [y])
            artist['intent'].set_data((x, x + cos(theta)), (y, y + sin(theta)))

            t = np.c_[[x, y], [x, y]].T + self.length * np.array([np.cos(theta + (d45 * np.array([-1, 1]))),
                                                                  np.sin(theta + (d45 * np.array([-1, 1])))]).T

            def segments(x, y, t):
                """ Returns a list of coordinates for the LineCollection's updated position """
                tx, ty = t
                rx = np.linspace(x, tx, self.fov_segments)
                ry = np.linspace(y, ty, self.fov_segments)
                points = np.array([rx, ry]).T.reshape(-1, 1, 2)
                return np.concatenate([points[:-1], points[1:]], axis=1)

            artist['lc1'].set_segments(segments(x, y, t[0]))
            artist['lc2'].set_segments(segments(x, y, t[1]))

            tr = transforms.Affine2D().rotate(theta + np.pi / 2).translate(x, y)
            artist['fov_area'].set_transform(tr + self.ax1.transData)
            artist['clip_path'].set_xy(np.array([[x, y], t[0], t[1]]))
            artist['fov_area'].set_clip_path(artist['clip_path'])

        d = np.cbrt(distance.euclidean(self.gt_pos_data[i], self.odom_pos_data[i]))
        for name, obj in self.odom_artists.items():
            if name not in ['sensor_radius', 'clip_path']:
                obj.set_alpha(min(1, max(0, d)))

        img = self.images[i]
        self.camera.set_data(img)
        run = self.run_counter[i]
        end = self.run_dict['end'][run]
        length = self.run_dict['len'][run]
        self.run_indicator.set_text(f"run {str(run).rjust(2)}: {str(length + i - end).rjust(4)}/{length} "
                                    f"({str(i).rjust(4)}/{len(self.images)})")

        if type(self.omaps[i]) is np.ndarray:
            for k, j in self.OCC_MAP_SQUARES:
                cell = f'{self.omaps[i][k, j]:.0f}'
                self.labels[k * config.occupancy_map_shape[0] + j].set_text(cell)

        else:
            if i != 0 and type(self.omaps[i - 1]) is np.ndarray:
                for k, j in self.OCC_MAP_SQUARES:
                    self.labels[k * config.occupancy_map_shape[0] + j].set_text('')

        self.occ_map.set_data(self.colored_omaps[i].astype('uint8'))

        return *self.gt_artists.values(), *self.odom_artists.values(), \
               self.camera, self.run_indicator, \
               self.occ_map, *self.labels,