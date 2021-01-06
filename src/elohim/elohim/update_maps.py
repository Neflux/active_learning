import argparse
import glob
import os
import pandas as pd

from config import SENSORS
from occupancy_map import compute_occupancy_map
from utils import COORDS
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()

parser = argparse.ArgumentParser(description='Update')
parser.add_argument('--dataset', '-d', metavar='.pkl', dest='d_path', help='Directory with pickles', required=True)
parser.add_argument('--window', '-w', metavar='s', dest='window', help='Time window', type=int, required=True)
parser.add_argument('--delta', '-de', metavar='m', dest='delta', help='Delta', type=float, required=True)
args = parser.parse_args()

other_files = glob.glob(os.path.join(args.d_path, '*.pkl'))
sorted_sets = ['train', 'valid', 'test']

ffs = [os.path.basename(x)[:-4] for x in other_files]
assert set(sorted_sets) <= set(ffs), ffs

for path, clean in zip(other_files, ffs):
    df = pd.read_pickle(path)
    df['target_map'] = df.groupby('big_block').progress_apply(
        lambda x: x.apply(compute_occupancy_map, axis=1, args=(x, COORDS, SENSORS, f'{args.window}s', args.delta)))
    df.to_pickle(os.path.join(args.d_path, f'{clean}.pkl'))

Path(os.path.join(args.d_path, f'w{args.window}_d{args.delta}')).touch()
print('Done')
