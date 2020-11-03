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

d45 = np.pi / 4


class Animator:
    OCC_MAP_SQUARES = list(itertools.product(range(config.occupancy_map_shape[0]),
                                             range(config.occupancy_map_shape[0])))

    def __init__(self, df, static_targets, rate=30, save_path=None, frames=None):
        """Given a dataframe and a list of coordinates of static targets, builds a FuncAnimation.
            The animations shows the top down view, and the live camera feed (ax1 and ax2).
            TODO: occupancy map ax3

            The top down view also features the unprecise odometry, but displays it only when the difference from
                ground truth is large enough (blue: ground truth, orange: displaced odometry)

            Args:
                    @param df:  the merged dataframe that contains all the necessary data (sensors and odometry)
                    @param static_targets:  a list of sensors' readings.
                    @param rate: rate of the animation
                    @param save_path: if specified, the animation will be saved at this path instead of being shown
            """

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

        self.omaps = [x.reshape(20,20) if isinstance(x, np.ndarray) else x for x in df['occupancy_map']]
        self.colored_omaps = []
        for x in self.omaps:
            if type(x) is np.ndarray:
                x = x.reshape(20,20)
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


def mergedfs(dfs, tolerance='1s'):
    """Merges different dataframes into a single synchronized dataframe.
        Args:
            @param dfs: a dictionary of dataframes divided by ros topic
            @param tolerance: Maximum time distance between original and new labels for inexact matches
        Returns:
            a single dataframe composed of the various dataframes synchronized.
        """
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
    """Displaces and rotates the cumulative odometry to match the starting pose of ground truth for each run
        Args:
            @param df: the already merged df that contains all the synced odometry data
            @param instructions: a dictionary that specifies the translation and rotation labels in the df
                'translation' contains a list of tuples in the format ('reference ground truth','offset odometry')
                'rotation' contains a single tuple in the same format for the yaw angle
            TODO: a boolean param to have it start at 0,0 instead of ground truth start pose? useful?
        Returns:
            a single dataframe with the odometry aligned to ground truth odometry on each run
        """
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


def get_map(rel_transform, sensor_readings, delta):
    '''Given a pose, constructs the occupancy map w.r.t. that pose.
    An occupancy map has 3 possible values:
    1 = object present;
    0 = object not present;
    -1 = unknown;
    Args:
            rel_transform:  the tranformation matrix from which to compute the occupancy map.
            sensor_readings:  a list of sensors' readings.
            ROBOT_GEOMETRY_SIMPLE: the transformations from robot frame to sensors' frames.
            COORDS: a list of relative coordinates of the form [(x1, y1), ...].
            delta: the maximum distance between a sensor reading and a coord to be matched.
    Returns:
            an occupancy map generated from the relative pose using coords and sensors' readings.
    '''
    #print([type(x) for x in [rel_transform, sensor_readings, delta]])
    # locate objects based on the distances read by the sensors
    sensor_readings_homo = np.array([[r, 0., 1.] for r in sensor_readings])
    # batch matrix multiplication
    rel_object_poses = np.einsum('ijk,ik->ij',
                                 np.matmul(rel_transform,
                                           ROBOT_GEOMETRY_SIMPLE),
                                 sensor_readings_homo)[:, :-1]
    # initialize occupancy map to -1
    occupancy_map = np.full((COORDS.shape[0],), -1, dtype=np.float)
    # compute distances between object poses and coords
    distances = np.linalg.norm(
        COORDS[:, None, :] - rel_object_poses[None, :, :],
        axis=-1)
    # find all coords with distance <= delta
    closer_than_delta = distances <= delta
    icoords, isens = np.where(closer_than_delta)
    # note: 0.11 is the maximum distance of a detected obstacle
    occupancy_map[icoords] = sensor_readings[isens] < 0.11
    return occupancy_map


def aggregate(maps):
    """Aggregates a list of occupancy maps into a single one.
    Args:
            maps: a list of occupancy maps.
    Returns:
            an aggregate occupancy map.
    """
    aggregated_map = np.full_like(maps[0], -1)
    if (maps == -1).all():
        return aggregated_map
    map_mask = (maps != -1).any(1)
    nonempty_maps = maps[map_mask]
    cell_mask = (nonempty_maps != -1).any(0)
    nonempty_cells = nonempty_maps[:, cell_mask]
    nonempty_cells[nonempty_cells == -1] = np.nan
    aggregated_map[cell_mask] = stats.mode(
        nonempty_cells, 0, nan_policy='omit')[0][0]
    return aggregated_map


def add_occupancy_maps(df: pd.DataFrame, window_size=100):
    """
    Adds occupancy maps in the input DataFrame from index half_window to len(df) - half_window;
    The remaining iterations have NaN.
    @param df: DataFrame with ground truth odometry
    @return:
    """

    gt_labels = ['ground_truth_odom_x', 'ground_truth_odom_y', 'ground_truth_odom_theta']
    df['gt_pose'] = df[gt_labels].apply(mktransf, axis=1)

    half_window = window_size // 2
    empty_block = [np.NaN] * half_window

    def rolling_occupancy_map(group):
        """ Set the occupancy maps to the right indices """
        print('Processing new run')
        gt_pose_id_col = group.columns.get_loc("gt_pose")

        maps = []
        for i in range(half_window, len(group) - half_window):
            start_pose = group.iat[i, gt_pose_id_col]
            win = group.iloc[i - half_window:i + half_window]
            rt = start_pose @ np.linalg.inv(np.stack(win['gt_pose']))
            sr = np.where(win['sensor'], 0.12, 0.)[..., np.newaxis]
            local_maps = np.array([get_map(rel_transform=rt[j], sensor_readings=sr[j], delta=config.occupancy_map_delta)
                                   for j in range(window_size)])
            #maps.append(np.rot90(omap.reshape(20, 20), 1))
            maps.append(aggregate(local_maps))

        return pd.Series(empty_block + maps + empty_block)

    df['occupancy_map'] = df.drop(['image']+gt_labels,axis=1)\
        .groupby('run').apply(rolling_occupancy_map).to_numpy().flatten()
    df.drop('gt_pose', axis=1, inplace=True)


def main(args=None):
    # TODO: load from local file
    try:
        for dir in list(os.scandir('history')):
            points_file = os.path.join(dir, 'points.json')
            with open(points_file) as f:
                points = json.load(f)
            targets = np.array([[t["x"], t["y"]] for t in points["targets"]])

            fp_files = glob.glob(f'{dir.path}/*.h5')
            files = [os.path.basename(x) for x in fp_files]

            if 'unified.h5' in files:
                print('Using cached dataset')
                df = pd.read_hdf(f'{dir.path}/unified.h5', key='df')
            elif len(files) >= 4:
                print('Preparing dataset')
                last_snapshot = {}
                while len(last_snapshot) == 0:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=PerformanceWarning)
                            last_snapshot = {}
                            for ff in fp_files:
                                last_snapshot[ff] = pd.read_hdf(ff)
                    except ValueError:
                        print(f'No dataset in HDF5 file: {ff}')
                        exit()
                    except Exception:
                        print("Recorder is probably locking the file, trying again in 3 second")
                        time.sleep(3)

                # print(dir.path)
                df = mergedfs(last_snapshot)

                # Remove meaningless runs
                too_short = df['run'].map(df['run'].value_counts()) > config.window_size // 2
                df = df[too_short]
                print(f'Unique runs: {len(df["run"].unique())} ({too_short.astype(int).sum()} iterations discarded)')

                # Reset odometry at the start of each run
                instructions = {'translation': [('ground_truth_odom_x', 'x'),
                                                ('ground_truth_odom_y', 'y')],
                                'rotation': ('ground_truth_odom_theta', 'theta')}
                df = reset_odom_run(df, instructions)

                # Fix timezone
                df.index = df.index.tz_localize('UTC').tz_convert('Europe/Rome')

                active_sensor_ratio = len(df[df['sensor']]) / len(df)
                print(f'Iterations with active virtual sensor: {active_sensor_ratio * 100.:.1f}%')

                add_occupancy_maps(df, window_size=config.window_size)

                df.to_hdf(f'{dir.path}/unified.h5', key='df', mode='w')

            # Select a random window of n sec for a preview video
            path = f"{dir.path}/preview.mp4"
            # if not os.path.exists(path):
            Animator(df.reset_index(), targets, save_path=path, rate=30, frames=None)
            subprocess.run(['/Applications/mpv.app/Contents/MacOS/mpv', '--loop-file', 'yes', path],
                           capture_output=True)

    except FileNotFoundError:
        print(f"Cannot find points.json file (at {os.path.dirname(points_file)})")
        print("Have you set up your environment at least once after your latest clean rebuild?"
              "\n\tros2 run elohim init")


if __name__ == '__main__':
    main()
