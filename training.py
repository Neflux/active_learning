import os

import pandas as pd
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from pytorchtools import EarlyStopping
from utils import cv_from_binary

try:  # Prioritize local src in case of PyCharm execution, no need to rebuild with colcon
    from utils import _moveaxis
    import config
except ImportError:
    from elohim.utils import _moveaxis
    import elohim.config as config

import torch

import torch.nn as nn

from sklearn.metrics import roc_auc_score
import torch.optim as optim

t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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
        data = t(cv_from_binary(self.X[i]))

        if self.y is not None:
            return data, self.y[i]
        else:
            return data


def get_dataset(dataset, tr_val_split, batch_size):
    cd = ConvDataset(dataset['image'], labels=dataset['target_map'])

    train_val = [int(len(cd) * x) for x in tr_val_split]  # train, val
    train_val += [len(cd) - sum(train_val)]  # test
    train_loader, val_loader, test_loader = [DataLoader(x, batch_size=batch_size, shuffle=shuffle) for x, shuffle in
                                             zip(torch.utils.data.dataset.random_split(cd, train_val),
                                                 [True, False, False])]
    print('Dataset x,y shapes: ', [x.shape for x in next(iter(train_loader))])
    return train_loader, val_loader, test_loader


def pretrained_mobilenetv2_new_classifier():
    model = torchvision.models.MobileNetV2(num_classes=int(np.prod(config.occupancy_map_shape)),
                                           inverted_residual_setting=[
                                               # t, c, n, s
                                               [1, 16, 1, 1],
                                               [6, 24, 2, 2],
                                               [6, 32, 3, 2],
                                               [6, 64, 2, 2],
                                               [6, 96, 1, 1]
                                           ])
    print('Total trainable parameters:', sum(p.numel() for p in model.parameters()))
    return model


def train_val_test(model, dataset, n_epochs=10, patience=20, save_path=None):
    train_loader, val_loader, test_loader = dataset

    #optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    # ADAM
    #criterion = nn.MSELoss(reduction='mean') # sigmoide
    # 400 x 2 senza stack 1-x e softmax dentro, fuori crossentropy

    path = 'checkpoint.pt'
    if save_path is not None:
        path = os.path.join(save_path, path)

    def get_loss_auc(X, y):
        mask = (y > -1)
        output = model(X)
        auc = roc_auc_score(y[mask], output[mask].detach().numpy())
        loss = criterion(output[mask], y[mask])
        return loss, auc

    def show_progress(t, batch_idx, X, loader, loss, auc, epoch=None):
        epoch = '' if epoch is None else f'Epoch: {epoch + 1}'
        print(f'\r{t} {epoch} [{(batch_idx + 1) * len(X)}/{len(loader.dataset)} ' +
              f'({100. * (batch_idx + 1) / len(loader):.0f}%)]\tLoss: {loss.item():.6f}' +
              f'\t\tAUC: {auc:.3f}', end='')

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    # to plot the entire training process, loss and auc
    train_metrics = []
    # to plot the entire validation process, loss and auc
    val_metrics = []
    for epoch in range(n_epochs):

        model.train()
        for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
            loss, auc = get_loss_auc(X, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            train_metrics.append([loss.item(), auc])

            show_progress('Training', batch_idx, X, train_loader, loss, auc, epoch)

        model.eval()
        with torch.no_grad():
            for batch_idy, (X, y) in enumerate(val_loader):
                loss, auc = get_loss_auc(X, y)
                valid_losses.append(loss.item())
                val_metrics.append([loss.item(), auc])
                show_progress('Validation', batch_idy, X, val_loader, loss, auc, epoch)
        print()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(path))

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            loss, auc = get_loss_auc(X, y)
            show_progress('Test', batch_idx, X, test_loader, loss, auc)
    return train_metrics, val_metrics


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns


def plot_metrics(model_name, train_metrics, test_metrics, batch_size):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    def f(ax, metrics, title):
        df = pd.DataFrame(metrics, index=np.arange(0, len(metrics) * batch_size, batch_size), columns=['loss', 'auc'])
        ax.set_title(title)
        ax.xaxis.set_major_locator(MultipleLocator(base=batch_size))
        sns.lineplot(data=df['loss'], ax=ax, label='Loss')
        ax.legend(loc=2)
        ax2 = ax.twinx()
        sns.lineplot(data=df['auc'], ax=ax2, color='C1', label='AUC')
        ax2.legend(loc=3)

    f(axs[0], train_metrics, 'Training')
    f(axs[1], test_metrics, 'Validation')
    plt.suptitle(model_name)


if __name__ == '__main__':
    d = sorted(os.scandir(os.path.join('history')), key=lambda x: x.path, reverse=True)[0]
    print(f'The most recent simulation folder has been selected "{d.path}":')
    unified_dataset_path = os.path.join(d.path, 'dataset.pkl')
    assert os.path.exists(unified_dataset_path), 'Selected folder does not contain the dataset'
    dataset_raw = pd.read_pickle(unified_dataset_path)

    batch_size = 16
    torch_dataset = get_dataset(dataset=dataset_raw, tr_val_split=[0.795, 0.05], batch_size=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device detected:', device)

    model = pretrained_mobilenetv2_new_classifier()
    model.to(device)

    train_metrics, val_metrics = train_val_test(model, dataset=torch_dataset, patience=3, save_path=d.path)

    plot_metrics('Pretrained MobileNetv2', train_metrics, val_metrics, batch_size)
    print('\nEnd of execution')
