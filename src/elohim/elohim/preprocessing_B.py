import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import config
from analyzer import mergedfs
import seaborn as sns

from preprocessing_A import coords, window, delta
from occupancy_map import compute_occupancy_map


def plot_map(spawn_coords, targets, d_path):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.plot(*targets.T, 'o', c='red', ms=1, label='Obstacle')
    ax.plot(*spawn_coords.T, 'o', ms=3, label='Spawn point')
    ax.set_xlabel('x')
    ax.set_ylabel('y', rotation=0)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title('Map top-view')
    plt.savefig(os.path.join(d_path, 'map_topview.png'))


def setup_topview(ax, fixed=True):
    half_side = config.plane_side // 2
    bounds = [-half_side - 1, half_side + 1]
    if fixed:
        ax.set(xlim=bounds, ylim=bounds, autoscale_on=False)
        major_ticks = np.arange(-half_side, half_side + 1, 5)
        minor_ticks = np.arange(-half_side, half_side + 1, 2.5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
    ax.set(title="Top view", adjustable='box')

    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)


def plot_relative_runs(df, d_path, target='run'):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
    input_cols = [f'ground_truth_odom_{axis}' for axis in ['x', 'y', 'theta']]
    output_cols = ['gt_rel_' + label.rsplit('_', 1)[-1] for label in input_cols]
    df = df.copy()

    def rotate(p):
        angle = -p.iat[0, -1]
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        p[input_cols] = p - p.iloc[0]
        p[input_cols[:2]] = np.squeeze((R @ p.loc[:, input_cols[:2]].T).T).values
        return p

    df[output_cols] = df.groupby(target)[input_cols].progress_apply(rotate)

    setup_topview(ax, fixed=False)

    def show_proof(run):
        x, y, t = run[output_cols].values.T
        ax.plot(x, y, linestyle='--', alpha=0.3, color='grey')
        sns.scatterplot(x=x[-1:], y=y[-1:], style=['x'], alpha=0.7, color='crimson', ax=ax, legend=False, zorder=3)
        sns.lineplot(x=[x[0], x[0] + np.cos(t[0]) * 2], y=[y[0], y[0] + np.sin(t[0]) * 2], color='black', ax=ax)

    df.groupby(target).progress_apply(show_proof)
    ax.set_title('Relative poses of Thymio at beginning of big blocks run')
    plt.savefig(os.path.join(d_path, 'map_topview.png'))


def extract_big_runs(df):
    df = df.copy()
    out_of_map = (df[['ground_truth_odom_x', 'ground_truth_odom_y']].abs() > 10).any(axis=1)
    df['big_block'] = (out_of_map != out_of_map.shift(1)).cumsum().astype(int)
    df = df[~out_of_map]
    df['big_block'] = (df['big_block'] != df['big_block'].shift(1)).cumsum().astype(int)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train the convolutional network')
    parser.add_argument('--data', '-d', metavar='path', dest='d_path', help='Directory with hdf5', required=True)
    args = parser.parse_args()
    d_path = args.d_path
    assert os.path.isdir(d_path), 'Not a directory'

    points_file = os.path.join(d_path, 'points.json')
    assert os.path.exists(points_file)
    with open(points_file) as f:
        points = json.load(f)
    spawn_coords = np.array([[t["x"], t["y"]] for t in points["spawn_coords"]])
    targets = np.array([[t["x"], t["y"]] for t in points["targets"]])
    plot_map(spawn_coords, targets, d_path)

    fp_files = glob.glob(f'{d_path}/*.h5')
    files = [os.path.basename(x) for x in fp_files]
    print(files)

    last_snapshot = {}
    for ff in fp_files:
        last_snapshot[ff] = pd.read_hdf(ff)
    df = mergedfs(last_snapshot)
    print(f'Unique runs: {len(df["run"].unique())}')

    df = extract_big_runs(df)
    print(f'Unique run blocks: {len(df["big_block"].unique())}')

    plot_relative_runs(df, d_path, target='big_block')

    occ_map = df.groupby('big_block').progress_apply(
        lambda x: x.apply(compute_occupancy_map, axis=1, args=(x, coords, ['sensor'], window, delta)))
    df = pd.concat([df, occ_map], axis=1)

    big_block = np.random.choice(df['big_block'].unique())
if __name__ == '__main__':
    main()
