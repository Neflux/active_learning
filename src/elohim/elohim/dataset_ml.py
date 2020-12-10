import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from utils import cv_from_binary, _moveaxis


def transform(x):
    x = cv_from_binary(x)
    # x = np.moveaxis(x,-1,0)

    # Images of shape (3 x H x W), where H and W are expected to be at least 224
    mobilenet_v2_format = transforms.Compose([
        # transforms.ToPILImage(), # Tensor of shape C x H x W or a numpy ndarray of shape H x W x C
        # transforms.Resize(256),transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return mobilenet_v2_format(x)


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

        if self.transform is not None:
            data = self.transform(data)

        if self.y is not None:
            return data, self.y[i]
        else:
            return data


def get_dataset(dataset, tr_val_split, batch_size, dataset_max_size=None):
    cd = ConvDataset(dataset['image'], labels=dataset['target_map'], transform=transform)

    if dataset_max_size is not None and len(cd) > dataset_max_size:
        print(f'Dataset size reduced ({len(cd)} -> ({dataset_max_size}))')
        cd = torch.utils.data.dataset.Subset(cd, range(0, dataset_max_size))
    else:
        print(f'Warning: dataset is already smaller than {dataset_max_size}')

    train_val = [int(len(cd) * x) for x in tr_val_split]  # train, val
    train_val += [len(cd) - sum(train_val)]  # test
    train_loader, val_loader, test_loader = [DataLoader(x, batch_size=batch_size, shuffle=shuffle) for x, shuffle in
                                             zip(torch.utils.data.dataset.random_split(cd, train_val),
                                                 [True, False, False])]
    print('Batches per split:', ','.join([str(len(x)) for x in [train_loader, val_loader, test_loader]]))
    print('Dataset, x,y shapes: ', ','.join([str(list(X.shape)) for X in next(iter(train_loader))]))
    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    dataset = None
    for d in sorted(os.scandir('history'), key=lambda x: x.path, reverse=True):
        data_path = os.path.join(d.path, 'dataset.pkl')
        if os.path.exists(data_path):
            print(f'The most recent simulation folder has been selected "{d.path}":')
            df = pd.read_pickle(data_path)
            dataset = df.loc[~df['out_of_map'], ['image', 'target_map']]
            dataset.head()
            break
    assert dataset is not None

    batch_size = 16
    train_loader, val_loader, test_loader = get_dataset(dataset=dataset, tr_val_split=[0.8, 0.1],
                                                        batch_size=batch_size)

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
