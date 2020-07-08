import glob
import json
import logging as log
import os
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
from tables import PerformanceWarning

from utils import cv_from_binary

d45 = np.pi / 4

class Animator:
    def __init__(self, df, static_targets, side=10, rate=30, save_path=None):
        self.rate = rate
        self.length = 10
        self.fov_segments = 40

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=rate, metadata=dict(artist='Me'), bitrate=1800)

        # Cache

        self.sensor_readings = df['sensor'].to_numpy()
        self.odom_pos_data = df[['x', 'y', 'theta']].to_numpy()
        self.gt_pos_data = df[['ground_truth_odom_x', 'ground_truth_odom_y', 'ground_truth_odom_theta']].to_numpy()
        self.images = df['image'].map(cv_from_binary).map(Image.fromarray, "RGB").to_numpy()

        def gradient_line(cmap):
            gradient = np.linspace(0.0, 1.0, self.fov_segments)
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(1, 0, cmap.N)
            my_cmap = ListedColormap(my_cmap)
            return LineCollection([], array=gradient, cmap=my_cmap, linewidth=1.2)

        def gradient_area(color):
            gradient = np.empty((100, 1, 4), dtype=float)
            rgb = mcolors.colorConverter.to_rgb(color)
            gradient[:, :, :3] = rgb
            gradient[:, :, -1] = np.linspace(0., .1, 100)[:, None]

            ext = [-self.length / sqrt(2), self.length / sqrt(2), -self.length / sqrt(2), 0]
            fov_area = ax1.imshow(gradient, aspect='auto', extent=ext, origin='lower')

            clip_path = Polygon(np.empty((3, 2)), fill=False, alpha=0)
            return fov_area, clip_path

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)  # type:figure.Figure, (axes.Axes, axes.Axes)
        self.ax1 = ax1

        """ Subplot 1 - Top View """

        bounds = [-side - 1, side + 1]
        ax1.set(title="Top view", xlim=bounds, ylim=bounds, autoscale_on=False, adjustable='box')
        major_ticks = np.arange(-10, 11, 5)
        minor_ticks = np.arange(-10, 11, 2.5)
        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.set_yticks(major_ticks)
        ax1.set_yticks(minor_ticks, minor=True)
        ax1.grid(which='both')
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.5)

        # Boundaries
        b = side + 0.1
        ax1.plot([-b, b, b, -b, -b], [-b, -b, b, b, -b], color='red', linewidth=1.2)

        # Thymio
        def thymio(color, fovline_cmap):
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

        self.ax1.set_aspect('equal')
        fig.canvas.set_window_title("Simulation")
        self.anim = FuncAnimation(fig, self.animate, frames=len(df), interval=1000. / rate, blit=True,
                                  init_func=self.init, repeat=True)

        if save_path is None:
            plt.show()
        else:
            self.anim.save(f"{save_path[0]}/{save_path[1]}.mp4", writer=writer,
                           progress_callback=lambda i, n: print(f'\rSaving: {i * 100. / n:.2f} %', end=''))
            print(f"\rSaving process complete. File location: {save_path[0]}/{save_path[1]}.mp4")


    def init(self):
        for artist in [self.gt_artists, self.odom_artists]:
            artist['thymio'].set_data([], [])
            artist['intent'].set_data([], [])
            artist['lc1'].set_segments([])
            artist['lc2'].set_segments([])
            artist['fov_area'].set_transform(self.ax1.transData)
            artist['clip_path'].set_xy(np.empty((3, 2)))

        self.camera.set_data(self.images[0])

        return *self.gt_artists.values(), *self.odom_artists.values(), self.camera,

    def animate(self, i):
        for artist, positional_data in zip([self.gt_artists, self.odom_artists],
                                           [self.gt_pos_data, self.odom_pos_data]):
            x, y, theta = positional_data[i]

            artist['thymio'].set_data([x], [y])
            artist['intent'].set_data((x, x + cos(theta)), (y, y + sin(theta)))

            t = np.c_[[x, y], [x, y]].T + self.length * np.array([np.cos(theta + (d45 * np.array([-1, 1]))),
                                                                  np.sin(theta + (d45 * np.array([-1, 1])))]).T
            def segments(x, y, t):
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
                obj.set_alpha(d)

        img = self.images[i]
        self.camera.set_data(img)

        return *self.gt_artists.values(), *self.odom_artists.values(), self.camera,


def mergedfs(dfs, tolerance='1s'):
    min_topic = None
    seen_cols = set()
    tot_with_dupes = sum([len(df) for topic, df in dfs.items()])
    max_topic_name = max([len(os.path.basename(topic)) for topic, _ in dfs.items()])
    for topic, df in dfs.items():

        # Column name formatting
        dfcols = set(df.columns)
        if len(dfcols & seen_cols) > 0:
            dfs[topic] = df = df.add_prefix(os.path.basename(topic)[:-3] + '_')
        seen_cols |= dfcols

        # Check how healthy is the bag
        duplicated = df.index.duplicated()
        if any(duplicated):
            amount = sum(duplicated.astype(int))
            log.warning(
                f'Dataframe \'{topic}\' contains {amount} ({amount * 100. / len(df):.2f}%) duplicates (TimeIndex)')
            dfs[topic] = df = df.loc[~df.index.duplicated(keep='first')]

        # Get minimum sized df
        print('\t', os.path.basename(topic).ljust(max_topic_name + 1), len(df))
        if not min_topic or len(dfs[min_topic]) > len(df):
            min_topic = topic

    # Merge on time index
    ref_df = dfs[min_topic]
    other_dfs = dfs
    other_dfs.pop(min_topic)
    result = pd.concat(
        [ref_df] +
        [df.reindex(index=ref_df.index, method='nearest', tolerance=pd.Timedelta(tolerance).value) for _, df in
         other_dfs.items()],
        axis=1)
    result.dropna(inplace=True)
    result.index = pd.to_datetime(result.index)

    num, den = tot_with_dupes - len(result) * (len(dfs) + 1), tot_with_dupes
    print(f'Resulting length: {len(result)}. Total discarded records: {num * 100 / den:.1f}% ({num}/{den}).')

    return result





def reset_odom_run(df, instructions):
    assert (isinstance(instructions, dict))
    assert ({'translation', 'rotation'} <= set(instructions))

    if isinstance(instructions['translation'], tuple):
        instructions['translation'] = [instructions['translation']]
    gts, odoms = np.array(instructions['translation']).T

    indices = df.reset_index().groupby('run').first()['time'].tolist()

    def offset_trasform(gt_col, odom_col):
        df['odom_offset'] = pd.NA
        df.loc[indices, 'odom_offset'] = df[odom_col]
        df['gt_offset'] = pd.NA
        df.loc[indices, 'gt_offset'] = df[gt_col]
        df[odom_col] = df[odom_col] - df['odom_offset'].ffill() + df['gt_offset'].ffill()
        df.drop(['gt_offset', 'odom_offset'], axis=1, inplace=True, errors='ignore')

    def rotate_points(p, origin=(0, 0), angle=0):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        o = np.atleast_2d(origin)
        p = np.atleast_2d(p)
        return np.squeeze((R @ (p.T - o.T) + o.T).T)

    def rotate_single_run(run_data):
        gt_theta, theta = instructions['rotation']
        rot_angle = run_data[gt_theta].iloc[0] - run_data[theta].iloc[0]
        rot_center = run_data[gts].iloc[0].tolist()
        run_data[odoms] = rotate_points(run_data[odoms], rot_center, rot_angle)
        return run_data

    # Offset xy path, rotate it (start of run xy is center) and then finally offset theta
    for gt_col, odom_col in instructions['translation']:
        offset_trasform(gt_col, odom_col)
    df = df.groupby('run').apply(rotate_single_run)
    offset_trasform(*instructions['rotation'])
    return df


points_file = os.path.join(get_package_share_directory('elohim'), 'points.json')
try:
    with open(points_file) as f:
        points = json.load(f)
    targets = np.array([[t["x"], t["y"]] for t in points["targets"]])

    for dir in list(os.scandir('history')):
        fp_files = glob.glob(f'{dir.path}/*.h5')
        files = [os.path.basename(x) for x in fp_files]

        if len(files) == 5 and 'summary.hdf5' not in files:

            last_snapshot = None
            while last_snapshot is None:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=PerformanceWarning)
                        last_snapshot = {f: pd.read_hdf(f) for f in fp_files}
                except Exception as ex:
                    print("Recorder is probably locking the file, trying again in 3 second")
                    time.sleep(3)

            # print(dir.path)
            df = mergedfs(last_snapshot)

            # Remove meaningless runs
            df = df[df['run'].map(df['run'].value_counts()) > 10]

            # Reset odometry at the start of each run
            instructions = {'translation': [('ground_truth_odom_x', 'x'),
                                            ('ground_truth_odom_y', 'y')],
                            'rotation': ('ground_truth_odom_theta', 'theta')}
            df = reset_odom_run(df, instructions)

            # Fix timezone
            df.index = df.index.tz_localize('UTC').tz_convert('Europe/Rome')

            # Select a random window of n sec for a preview video

            rate, seconds = 30, 15
            window = min(rate*seconds, len(df) - 1)
            start = np.random.randint(len(df) - window)
            Animator(df.reset_index().loc[start:start + window, :], targets, save_path=(dir.path, 'preview'), rate=rate)

except FileNotFoundError:
    print(f"Cannot find points.json file (at {os.path.dirname(points_file)})")
    print("Have you set up your environment at least once after your latest clean rebuild?"
          "\n\tros2 run elohim init")


