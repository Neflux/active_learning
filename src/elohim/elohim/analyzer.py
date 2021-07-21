import glob
import json
import os
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd
import tables

matplotlib.rcParams['axes.unicode_minus'] = False
from scipy.stats import stats
from tables import PerformanceWarning
from tqdm import tqdm

tqdm.pandas()

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from utils import cv_from_binary, print_full, mktransf, COORDS
    import config
except ImportError:
    from elohim.utils import cv_from_binary, print_full, mktransf, COORDS
    import elohim.config

d45 = np.pi / 4


def load_ros_data(dir_path, verbose=False):
    fp_files = glob.glob(f'{dir_path}/*.h5')
    files = [os.path.basename(x) for x in fp_files]
    assert len(files) >= 4

    print('Preparing dataset..', end=' ')
    last_snapshot = {}
    while len(last_snapshot) == 0:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=tables.PerformanceWarning)
                last_snapshot = {}
                for ff in fp_files:
                    last_snapshot[ff] = pd.read_hdf(ff)
                    if verbose:
                        print(last_snapshot[ff].columns)
        except ValueError:
            print(f'No dataset in HDF5 file: {ff}')
            exit()
        except Exception:
            print("Recorder is probably locking the file, trying again in 3 second")
            time.sleep(3)
    df = mergedfs(last_snapshot, verbose=verbose)
    print(f'Unique runs: {len(df["run"].unique())}')
    return df


def mergedfs(dfs, tolerance='1s', verbose=False):
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
            if verbose:
                print(set(df.columns), seen_cols)
            dfs[topic] = df = df.add_prefix(os.path.basename(topic)[:-3] + '_')
        seen_cols |= dfcols

        # Check how healthy is the bag
        duplicated = df.index.duplicated()
        if any(duplicated):
            amount = sum(duplicated.astype(int))
            print(f'Dataframe \'{topic}\' contains {amount} ({amount * 100. / len(df):.2f}%) duplicates (TimeIndex)')
            dfs[topic] = df = df.loc[~df.index.duplicated(keep='first')]

        # Get minimum sized df
        if verbose:
            print(os.path.basename(topic).ljust(max_topic_name + 1), len(df))
        if not min_topic or len(dfs[min_topic]) > len(df):
            min_topic = topic

    # Merge on time index
    ref_df = dfs[min_topic]
    other_dfs = dfs
    backup = other_dfs.pop(min_topic)
    result = pd.concat(
        [ref_df] +
        [df.reindex(index=ref_df.index, method='nearest', tolerance=pd.Timedelta(tolerance).value) for _, df in
         other_dfs.items()],
        axis=1)
    result.dropna(inplace=True)
    result.index = pd.to_datetime(result.index)
    other_dfs[min_topic] = backup
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


def main(args=None):
    print('Nothing to do here')
    pass


if __name__ == '__main__':
    main()
