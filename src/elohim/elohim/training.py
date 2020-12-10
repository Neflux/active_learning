import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import bnn
from dataset_ml import get_dataset
from model import ConvNet, BayesConvNet
from pytorchtools import EarlyStopping
from utils import random_session_name

from datetime import datetime as dt
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


def train_val_test(model, loss_function, target_transform, dataset, bayes=False, n_epochs=10, patience=20,
                   save_path=None):
    train_loader, val_loader, test_loader = dataset

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    # TODO: eventually append bayes to the codename
    # TODO: save in subfolder next to dataset
    code_name = random_session_name()
    path = f'{dt.now().strftime("%m%d-%H%M")}_{code_name}.pt'
    if save_path is not None:
        path = os.path.join(save_path, path)

    print(f'New training session, code name: {code_name} (path: {path})')

    def get_loss_auc(X, y, model):
        output = model(X)
        mask = (y > -1)
        loss = loss_function(output[mask], target_transform(y)[mask])
        stacked_y = torch.stack([1 - y, y], dim=2)
        auc = roc_auc_score(stacked_y[mask], output[mask].detach().numpy())
        return loss, auc

    if bayes:
        kld_function = bnn.nn.KLDivergence(number_of_batches=len(train_loader))

        def get_loss_auc(X, y, model):
            output = model(X)
            mask = (y > -1)

            divergence = kld_function(model)
            masked_y = target_transform(y)[mask]
            likelihood = torch.stack([
                loss_function(pred[mask], masked_y) for pred in output
            ]).mean() / model.samples
            loss = likelihood + divergence

            masked_stacked_y = torch.stack([1 - y, y], dim=2)[mask]
            auc = np.array([
                roc_auc_score(masked_stacked_y, pred[mask].detach().numpy()) for pred in output
            ]).mean() / model.samples
            return loss, auc

    def show_progress(t, batch_idx, X, loader, loss, auc, epoch=None):
        epoch = '' if epoch is None else f'Epoch: {epoch + 1}'
        print(f'\r{t} {epoch} [{(batch_idx + 1) * len(X)}/{len(loader.dataset)} ' +
              f'({100. * (batch_idx + 1) / len(loader):.0f}%)]\tLoss: {loss.item():.6f}' +
              f'\t\tAUC: {auc:.3f}', end='')

    # to track the training\valid loss as the model trains
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    # to plot the entire training process, loss and auc
    train_metrics = []
    # to plot the entire validation process, loss and auc
    val_metrics = []
    for epoch in range(n_epochs):

        model.train()
        for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
            X, y = X.to(device), y.to(device)

            loss, auc = get_loss_auc(X, y, model)

            # TODO: ENTROPY

            # Calculate error contribution
            loss.backward()

            # Update the weights
            optimizer.zero_grad()
            optimizer.step()

            train_losses.append(loss.item())

            train_metrics.append([loss.item(), auc])

            show_progress('Training', batch_idx, X, train_loader, loss, auc, epoch)

        model.eval()
        with torch.no_grad():
            for batch_idy, (X, y) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                loss, auc = get_loss_auc(X, y, model)
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
            X, y = X.to(device), y.to(device)
            loss, auc = get_loss_auc(X, y, model)
            show_progress('Test', batch_idx, X, test_loader, loss, auc)
    return train_metrics, val_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the convolutional network')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', '-d', metavar='.pkl', dest='pickle_path',
                       help='Pickle file that holds the data')
    group.add_argument('--simulation-dir', '-sd', metavar='path', dest='simulations_dir_path',
                       help='Folder containing datetime-indexed subfolders of simulations. '
                            'The most recent will be automatically selected')
    parser.add_argument('--max_size', '-ms', metavar='size', dest='max_dataset_size', type=int,
                        help='How many rows of the dataset to keep (last time I checked: 261K rows). '
                             'Useful for potato computers')
    parser.add_argument('--batch-size', '-bs', metavar='batch_size', dest='batch_size', type=int, default=32,
                        help='Batch size for the convolutional network')

    parser.add_argument('--bayes', '-b', dest='bayes', action='store_true',
                        help='If present, the network will be bayesian')
    parser.set_defaults(bayes=False)

    args = parser.parse_args()

    if args.pickle_path is None:
        d_path = None
        for d in sorted(os.scandir(args.simulations_dir_path), key=lambda x: x.path, reverse=True):
            data_path = os.path.join(d.path, 'dataset.pkl')
            if os.path.exists(data_path):
                d_path = d.path
                print(f'The most recent simulation folder has been selected "{d_path}":')
                dataset_found = True
                break
        assert d_path is not None
    else:
        d_path = os.path.dirname(args.pickle_path)
        data_path = args.pickle_path

    df = pd.read_pickle(data_path)
    dataset = df.loc[~df['out_of_map'], ['image', 'target_map']]

    torch_dataset = get_dataset(dataset=dataset, tr_val_split=[0.8, 0.1],
                                batch_size=args.batch_size, dataset_max_size=args.max_dataset_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device detected:', device)

    # Initialize model
    #    t,  c, n, s
    irs = [[1, 16, 1, 1],
           [6, 24, 2, 2],
           [6, 32, 3, 2],
           [6, 96, 1, 1]]
    # Initialize model

    if args.bayes:
        # TODO: samples argparse
        model = BayesConvNet(inverted_residual_setting=irs,
                             in_planes=200, out_planes=400, samples=8,
                             num_classes=400, mode='softmax')
    else:
        model = ConvNet(inverted_residual_setting=irs,
                        in_planes=200, out_planes=400,
                        num_classes=400, mode='softmax')
    model.to(device)

    train_metrics, val_metrics = train_val_test(model, loss_function=nn.NLLLoss(),
                                                target_transform=lambda y: y.type(torch.LongTensor),
                                                bayes=args.bayes,
                                                dataset=torch_dataset, patience=3, save_path=d_path)

    print('\nEnd of execution')
