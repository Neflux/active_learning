import glob

import pandas as pd
import numpy as np

import os


def mergedfs(dfs, tolerance='1s'):
    min_topic = None
    for topic, df in dfs.items():
        if not min_topic or len(dfs[min_topic]) > len(df):
            min_topic = topic
    ref_df = dfs[min_topic]
    other_dfs = dfs
    other_dfs.pop(min_topic)
    result = pd.concat(
        [ref_df] +
        [df.reindex(index=ref_df.index, method='nearest', tolerance=pd.Timedelta(tolerance).value) for _, df in other_dfs.items()],
        axis=1)
    result.dropna(inplace=True)
    result.index = pd.to_datetime(result.index)
    return result


for dir in os.scandir('history'):
    fp_files = glob.glob(f'{dir.path}/*.h5')
    files = [os.path.basename(x) for x in fp_files]
    print(files)
    if 'summary.hdf5' not in files:
        df = mergedfs({f: pd.read_hdf(f) for f in fp_files})
