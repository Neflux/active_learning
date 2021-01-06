import glob
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import tables
from tqdm import tqdm

import config
from analyzer import mergedfs
from occupancy_map import compute_occupancy_map

tqdm.pandas()

ds = [d for d in sorted(os.scandir('history'), key=lambda x: x.path, reverse=False) if
      re.match(r'(.*?)[0-9]{4}-[0-9]{6}', d.path) is not None]
print([d.path for d in ds])
d = ds[-1]
print(f'The most recent simulation folder has been selected "{d.path}":')
points_file = os.path.join(d.path, 'points.json')
assert os.path.exists(points_file)
with open(points_file) as f:
    points = json.load(f)
targets = np.array([[t["x"], t["y"]] for t in points["targets"]])
fp_files = glob.glob(f'{d.path}/*.h5')
files = [os.path.basename(x) for x in fp_files]

points_file = os.path.join(d.path, 'points.json')
with open(points_file) as f:
    points_d = json.load(f)
points = np.array([(c['x'], c['y']) for c in points_d['targets']])

assert len(files) >= 4
print('Loading hdf5 sensor data into dataframes..', end=' ')
last_snapshot = {}
while len(last_snapshot) == 0:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=tables.PerformanceWarning)
            last_snapshot = {}
            for ff in fp_files:
                last_snapshot[ff] = pd.read_hdf(ff)
    except ValueError:
        print(f'No dataset in HDF5 file: {ff}')
        exit()
    except Exception:
        print("Recorder is probably locking the file, trying again in 3 second")
        time.sleep(3)
print('Done')

print('Mergin dataframes into a single one..')
df = mergedfs(last_snapshot)

window = '20s'
df = df.groupby('run').filter(lambda x: (x.index[-1] - x.index[0]) > pd.Timedelta(window))
df = df[df['run'] != -1]
df = df.groupby("run").apply(lambda x: x.assign(out_of_map=True if (
        x[[f"ground_truth_odom_{a}" for a in ["x", "y"]]].iloc[
            -1].abs() > config.plane_side / 2).any() else False)).reset_index(level=0, drop=True)
df = df.groupby('run').apply(lambda x: x.iloc[10:]).reset_index(level=0, drop=True)
print(f'Unique runs: {len(df["run"].unique())}')

coords = np.stack(np.meshgrid(
    np.linspace(0, .8, int(.8 / .04)),
    np.linspace(-.4, .4, int(.8 / .04))
)).reshape([2, -1]).T

delta = 0.025
occ_map = df.groupby('run').progress_apply(
    lambda x: x.apply(compute_occupancy_map, axis=1, args=(x, coords, ['sensor'], window, delta)))
df = pd.concat([df, occ_map], axis=1)
df.to_pickle(os.path.join(d.path, 'dataset.pkl'))
