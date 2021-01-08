import glob
import os
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import transforms
from tqdm import tqdm

import config
from utils import cv_from_binary, _moveaxis, scaled_full_robot_geometry, plot_transform, mktransf

to_pil = transforms.Compose([transforms.ToPILImage(), transforms.Resize(120)])
to_tensor = transforms.Compose([
        transforms.ToTensor(),  # outputs a (C x H x W) in the range [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
def transform_function(resize=False, height=120):
    # Bayesian is lighter with this trick
    #height = 60
    #if resize:
    #    # A quarter of the original size
    #    height = 60

    def transform(x, flip=False):
        x = cv_from_binary(x)  # outputs (240, 320, 3)
        x = to_pil(x)
        if flip:
            x = F.hflip(x)
        return to_tensor(x)

    return transform


class ConvDataset(Dataset):
    def __init__(self, images, labels=None, transform=None, random_flip=False):
        self.X = images.values
        if labels is not None:
            labels = labels.map(lambda x: torch.as_tensor(x, dtype=torch.float32)).values
        self.y = labels
        self.transform = transform

        self.horizontal_flip = lambda: False
        if random_flip:
            self.horizontal_flip = lambda: random.random() < 0.5

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]

        flip = self.horizontal_flip()
        data = self.transform(data, flip)

        desired_output = self.y[i]
        if flip:
            desired_output = torch.flip(torch.reshape(desired_output, shape=(20, 20)), dims=[1]).flatten()
        return data, desired_output


def get_interesting_block_ids(df):
    comp = pd.DataFrame(
        {'runs': df.groupby('big_block')['run'].nunique() ** 2, 'iterations': df.groupby('big_block')['run'].count()})
    # Award a high number of obstacles and penalize number of iterations
    comp['ratio'] = comp['runs'] ** 2 / comp['iterations']
    # Normalize
    comp['p'] = comp['ratio'] / comp['ratio'].sum()
    # Draft all of them (without replacement)
    pos = np.random.choice(comp.index, p=comp['p'], size=len(comp), replace=False)

    # Reindex to add the positions
    comp = comp.reindex(pos)
    comp['pos'] = np.arange(len(comp))

    return comp


def get_dataset(d_path, batch_size=8, keep=1., resize=False, test_only=False, shuffle_train=True, perc_train_set=1.):
    other_files = glob.glob(os.path.join(d_path, '*.pkl'))

    sorted_sets, random_flips, shuffle_sets = ['test'], [False], [False]
    if not test_only:
        sorted_sets = ['train', 'valid'] + sorted_sets
        random_flips = [True, False, False]
        shuffle_sets = [shuffle_train, False, False]

    ffs = [os.path.basename(x)[:-4] for x in other_files]
    assert set(sorted_sets) <= set(ffs), (sorted_sets,ffs)

    # print('Loading dataset, this could take a while.. ')
    # start = time.time()

    dfs = [pd.read_pickle(os.path.join(d_path, f'{x}.pkl'))
           for x in tqdm(sorted_sets, desc='Loading splits pickles into memory')]

    if not test_only and perc_train_set != 1.:
        comp = get_interesting_block_ids(dfs[0]).sort_values('pos')
        t = dfs[0]

        too_big_mask = comp['iterations'] < len(t) * perc_train_set
        if len(comp[too_big_mask]) == 0:
            print(f'the smallest block is bigger than the desired dataset ({perc_train_set*100}%)')
        else:
            comp = comp[too_big_mask]

        ids = [comp.index[0]]
        comp = comp.iloc[1:]
        diff = 0
        for i in range(2, len(comp) + 1):

            # If we have achieved the right size, exit
            diff = len(t) * perc_train_set - len(t[t['big_block'].isin(ids)])
            if diff < 0:
                break

            # Remove candidates that are too big
            candidates = comp['iterations'] < diff
            if len(comp[candidates]) == 0:
                # If no more candidates exit, but keep last one
                break
            comp = comp[candidates]

            ids.append(comp.index[0])
            comp = comp.iloc[1:]

        dfs[0] = t[t['big_block'].isin(ids)]
        # TODO: concatenate last candidate until :diff to achieve exactly what was requested
        print(f'Requested: {perc_train_set * 100}%, actual one: {len(dfs[0]) * 100 / len(t):2.2f}%')

    cds = [ConvDataset(df['image'].iloc[:int(len(df) * keep)], labels=df['target_map'].iloc[:int(len(df) * keep)],
                       transform=transform_function(resize), random_flip=r) for df, r in zip(dfs, random_flips)]
    loaders = [DataLoader(cd, batch_size=batch_size, shuffle=shuffle)
               for cd, shuffle in zip(cds, shuffle_sets)]

    # print(f'Datased loaded ({time.time() - start:2.2f}s)')

    print('Batches per split:', ','.join([str(len(x)) for x in loaders]))
    print('Datasets x,y shapes', ','.join([str(list(X.shape)) for X in next(iter(loaders[0]))]))

    return dfs, loaders


def compute_map_density(df, path, s='training', emulate_random_flip=True):
    p = 1.
    if emulate_random_flip:
        p = 0.5

    # Emulate random horizontal flip
    x = np.array(
        [np.flip(np.array(x).reshape(20, 20), axis=1).flatten() if random.random() > p else np.array(x) for x in
         df['target_map'].values])

    x2 = x.copy()
    # Calculate the occurrences per cell
    x2[x == -1] = 0
    x2[x == 0] = 0
    x2[x == 1] = 1
    obst = x2.sum(axis=0)  # / a.shape[0]

    x2[x == 0] = 1
    x2[x == 1] = 0
    free = x2.sum(axis=0)

    x2[x == 0] = 1
    x2[x == 1] = 1
    aggr = x2.sum(axis=0)
    # Plt em

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    def plot_em(data, ax, title):
        im = ax.imshow(data.reshape(20, 20), norm=mpl.colors.LogNorm())

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, ax=ax, cax=cax)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('x (forward axis)')
        ax.set_xlabel('y', rotation=0, labelpad=10)
        ax.set_aspect('equal')
        ax.set_title(title)

    plot_em(obst, axs[0], 'Obstacles')
    plot_em(free, axs[1], 'Free space')
    plot_em(aggr, axs[2], 'Aggregated')

    plt.subplots_adjust(0, 0, 1, 1)
    plt.tight_layout()

    Path(path).mkdir(parents=True, exist_ok=True)
    path = os.path.join(path, f'{s}_occurrences.png')
    plt.savefig(path)


def main():
    import argparse, numpy as np

    parser = argparse.ArgumentParser(description='Train the convolutional network')
    parser.add_argument('--dataset', '-d', metavar='dir', dest='d_path', help='Directory with pickles', required=True)
    parser.add_argument('--output', '-o', metavar='dir', dest='o_path', help='Directory for samples')
    args = parser.parse_args()
    print(args.d_path, args.o_path)

    o_path = args.d_path if args.o_path is None else args.o_path
    sample_dir = os.path.join(o_path, 'dataset_samples')
    Path(sample_dir).mkdir(parents=True, exist_ok=True)

    # Determinism doesnt hurt
    random.seed(0)
    np.random.seed(0xDEADBEEF)
    torch.manual_seed(0)

    (train_dataframe, _, test_dataframe), (train_loader, val_loader, test_loader) = get_dataset(args.d_path, batch_size=512,
                                                                                   shuffle_train=True, perc_train_set=.25)
    compute_map_density(train_dataframe, sample_dir, 'Training set', emulate_random_flip=False)
    compute_map_density(test_dataframe, sample_dir, 'Testing set', emulate_random_flip=False)

    for X, y in train_loader:
        i = 0
        while True:
            fig, axs = plt.subplots(2, 1, figsize=(5, 8))
            print(X.shape, X.numpy().min(), X.numpy().max())
            X2 = _moveaxis(X[i], 0, -1)

            X2 = (X2 - X2.min()) / (X2.max() - X2.min())
            axs[0].imshow(X2)
            axs[0].set_title(f'Re-norm back to [0,1]')
            axs[0].axis('off')

            x = y[i].reshape(20, 20)
            res = np.empty((20, 20, 3), dtype=np.uint8)
            res[x == -1] = (190, 190, 190)
            res[x == 0] = (0, 255, 0)
            res[x == 1] = (255, 0, 0)
            # print(x)
            axs[1].imshow(res)
            axs[1].axis('off')

            ratio = 20 / 0.8
            for r in scaled_full_robot_geometry(ratio):
                plot_transform(axs[1], mktransf((10 - 0.5, 20 - 0.5, -np.pi / 2)) @ r, color='black',
                               length=config.max_sensing_distance * ratio, head_width=0.2)

            axs[1].plot([10 / 20, 0], [0, 0.5], linewidth=.8, linestyle='--', color='grey', transform=axs[1].transAxes)
            axs[1].plot([10 / 20, 1], [0, 0.5], linewidth=.8, linestyle='--', color='grey', transform=axs[1].transAxes)

            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f'train_{i}.png'))
            # plt.suptitle('Very first image of train split ([C,H,W]->[H,W,C] axis flip for display purposes)', y=1.07)
            print(i)
            i += 1
            input()

    print(f'Total train batches: {len(train_loader)}')


if __name__ == '__main__':
    main()
