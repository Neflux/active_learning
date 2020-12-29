import glob
import json
import os
import subprocess
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams['axes.unicode_minus'] = False
from scipy.stats import stats
from tables import PerformanceWarning
from tqdm import tqdm

tqdm.pandas()

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from utils_ros import cv_from_binary, ROBOT_GEOMETRY_SIMPLE, print_full, mktransf, COORDS
    import config
except ImportError:
    from elohim.utils import cv_from_binary, ROBOT_GEOMETRY_SIMPLE, print_full, mktransf, COORDS
    import elohim.config as config

d45 = np.pi / 4


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
            print(f'Dataframe \'{topic}\' contains {amount} ({amount * 100. / len(df):.2f}%) duplicates (TimeIndex)')
            dfs[topic] = df = df.loc[~df.index.duplicated(keep='first')]

        # Get minimum sized df
        print(os.path.basename(topic).ljust(max_topic_name + 1), len(df))
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


def reset_odom_run(df):
    input_cols = ['ground_truth_odom_x', 'ground_truth_odom_y', 'ground_truth_odom_theta']
    output_cols = ['gt_rel_' + label.rsplit('_', 1)[-1] for label in input_cols]

    def rotate(p):
        angle = -p.iat[0, -1]
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        p[input_cols] = p - p.iloc[0]
        p[input_cols[:2]] = np.squeeze((R @ p.loc[:, input_cols[:2]].T).T).values
        return p

    df[output_cols] = df.groupby('run')[input_cols].apply(rotate)


def reset_odom_run_old(df, instructions):
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
    # print([type(x) for x in [rel_transform, sensor_readings, delta]])
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


def add_occupancy_maps(df: pd.DataFrame, window_size=100, empty_value=-2):
    """
    Adds occupancy maps in the input DataFrame from index half_window to len(df) - half_window;
    The remaining iterations have NaN.
    @param df: DataFrame with ground truth odometry
    @return:
    """

    gt_labels = ['ground_truth_odom_x', 'ground_truth_odom_y', 'ground_truth_odom_theta']
    print('Computing roto-translational matrices for the occupancy maps')
    df['gt_pose'] = df[gt_labels].progress_apply(mktransf, axis=1)

    half_window = window_size // 2
    empty_block = [np.ones(400) * empty_value] * half_window

    def rolling_occupancy_map(group):
        """ Set the occupancy maps to the right indices """
        # print('Processing new run')
        gt_pose_id_col = group.columns.get_loc("gt_pose")

        maps = []
        for i in range(half_window, len(group) - half_window):
            start_pose = group.iat[i, gt_pose_id_col]
            win = group.iloc[i - half_window:i + half_window]
            rt = start_pose @ np.linalg.inv(np.stack(win['gt_pose']))
            sr = np.where(win['sensor'], 0.12, 0.)[..., np.newaxis]
            local_maps = np.array([get_map(rel_transform=rt[j], sensor_readings=sr[j], delta=config.occupancy_map_delta)
                                   for j in range(window_size)])
            # maps.append(np.rot90(omap.reshape(20, 20), 1))
            maps.append(aggregate(local_maps))

        result = pd.Series(empty_block + maps + empty_block)
        if len(result) != len(group):
            print('ERROR', group.iloc[0]['run'], len(result), len(group))
            return pd.Series([np.ones(400) * empty_value] * len(group))
        return result

    print('Computing occupancy maps')
    result = df.drop(['image'] + gt_labels, axis=1) \
        .groupby('run').progress_apply(rolling_occupancy_map).to_numpy().flatten()

    df.drop('gt_pose', axis=1, inplace=True)
    return result


def main(args=None):
    # TODO: load from local file
    try:
        for dir in sorted(os.scandir(os.path.join('../../../history')), key=lambda x: x.path, reverse=True):
            points_file = os.path.join(dir, 'points.json')
            if not os.path.exists(points_file):
                continue
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
                print(f'Unique runs: {len(df["run"].unique())}')

                df = df[df['run'].map(df['run'].value_counts()) > config.window_size]
                print(f'Unique runs after removing short ones: {len(df["run"].unique())}')

                orco = 0.3
                before_len = len(df)

                def test(rundf):
                    diff = rundf['ground_truth_odom_x'].diff()
                    safe = diff.dropna().between(-orco, orco).sum() / len(diff.dropna())
                    if safe != 1:
                        safe_mask = diff.between(-orco, orco)
                        blocks = (safe_mask.shift() != safe_mask).cumsum()
                        assert len(set(blocks)) == 2, 'Teleport occurs after a valid start'
                        return rundf[safe_mask]
                    return rundf

                df = df.groupby('run').apply(test).reset_index(drop=True)  # *100 / len(df['run'].unique())
                print(f'Teleport iterations dropped: {before_len - len(df)}')

                print(
                    f'media di rapporto letture positive/totali per runt: {df.groupby("run")["sensor"].apply(lambda x: x.sum() / len(x)).mean() * 100:2.2f}%')
                # Remove meaningless runs
                # too_short = df['run'].map(df['run'].value_counts()) > config.window_size // 2
                # df = df[too_short]
                # print(f'Unique runs: {len(df["run"].unique())} ({too_short.astype(int).sum()} iterations discarded)')
                #
                # # Reset odometry at the start of each run
                # instructions = {'translation': [('ground_truth_odom_x', 'x'),
                #                                 ('ground_truth_odom_y', 'y')],
                #                 'rotation': ('ground_truth_odom_theta', 'theta')}
                # df = reset_odom_run(df, instructions)

                # Fix timezone
                # df.index = df.index.tz_localize('UTC').tz_convert('Europe/Rome')

                active_sensor_ratio = len(df[df['sensor']]) / len(df)
                print(f'% of total iterations with active virtual sensor: {active_sensor_ratio * 100.:.1f}%')

                df = df[df['run'].isin(df['run'].unique()[:3])]

                df['occupancy_map'] = add_occupancy_maps(df, window_size=config.window_size)

                df.to_hdf(f'{dir.path}/unified.h5', key='df', mode='w')

    except FileNotFoundError:
        print(f"Cannot find points.json file (at {os.path.dirname(points_file)})")
        print("Have you set up your environment at least once after your latest clean rebuild?"
              "\n\tros2 run elohim init")


if __name__ == '__main__':
    main()
