import glob
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

from utils import cv_from_binary, _moveaxis
from torchvision.transforms import functional as F

def transform_function(resize=False):
    # np.ndarray of C x H x W to PIL Image
    torchvision_transforms = [transforms.ToPILImage()]

    # Bayesian is lighter with this trick
    if resize:
        # Half of the original size
        torchvision_transforms.append(transforms.Resize(120))
    to_pil = transforms.Compose(torchvision_transforms)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),  # outputs a (C x H x W) in the range [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def transform(x, flip):
        x = cv_from_binary(x)  # outputs (240, 320, 3)
        x = to_pil(x)
        if flip:
            x = F.hflip(x)
        return to_tensor(x)

    return transform


class ConvDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.X = images.values
        if labels is not None:
            labels = labels.map(lambda x: torch.as_tensor(x, dtype=torch.float32)).values
        self.y = labels
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]

        horizontal_flip = random.random() < 0.5
        data = self.transform(data, horizontal_flip)

        desired_output = self.y[i]
        if horizontal_flip:
            desired_output = torch.flip(torch.reshape(desired_output, shape=(20, 20)), dims=[0]).flatten()
        return data, desired_output


def get_dataset(d_path, batch_size=8, keep=1., resize=False, test_only=False):
    other_files = glob.glob(os.path.join(d_path, '*.pkl'))

    sorted_sets = ['test']
    if not test_only:
        sorted_sets = ['train', 'valid'] + sorted_sets

    assert set(sorted_sets) <= set([os.path.basename(x)[:-4] for x in other_files])

    # print('Loading dataset, this could take a while.. ')
    # start = time.time()

    dfs = [pd.read_pickle(os.path.join(d_path, f'{x}.pkl'))
           for x in tqdm(sorted_sets, desc='Loading splits pickles into memory')]
    cds = [ConvDataset(df['image'].iloc[:int(len(df) * keep)], labels=df['target_map'].iloc[:int(len(df) * keep)],
                       transform=transform_function(resize)) for df in dfs]
    loaders = [DataLoader(cd, batch_size=batch_size, shuffle=shuffle) for cd, shuffle in zip(cds, [True, False, False])]

    # print(f'Datased loaded ({time.time() - start:2.2f}s)')

    print('Batches per split:', ','.join([str(len(x)) for x in loaders]))
    print('Datasets x,y shapes', ','.join([str(list(X.shape)) for X in next(iter(loaders[0]))]))

    # Return torch loaders and full testing Dataframe for videos
    if len(loaders) == 1:
        loaders = loaders[0]
    return loaders, dfs[-1]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train the convolutional network')
    parser.add_argument('--dataset', '-d', metavar='.pkl', dest='d_path', help='Directory with pickles', required=True)
    args = parser.parse_args()
    (train_loader, val_loader, test_loader), test_dataset = get_dataset(args.d_path, batch_size=16)

    fig, axs = plt.subplots(2, 1, figsize=(5, 8))
    for X, y in train_loader:
        print(X.shape, X.numpy().min(), X.numpy().max())
        X = _moveaxis(X[0], 0, -1)
        axs[0].imshow(X)
        # axs[0].set_title(
        #     f'ML ready format: [{X.min():1.1f},{X.max():1.1f}]\nnormalizing with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]' + \
        #     f'\n(here showing only "visible" [0,1] neighbourhood)')
        axs[0].axis('off')
        X = (X - X.min()) / (X.max() - X.min())
        axs[1].imshow(X)
        axs[1].set_title(f'Re-norm back to [0,1]')
        axs[1].axis('off')
        plt.tight_layout()
        # plt.suptitle('Very first image of train split ([C,H,W]->[H,W,C] axis flip for display purposes)', y=1.07)
        break
    plt.show()
    print(f'Total train batches: {len(train_loader)}')


if __name__ == '__main__':
    main()
