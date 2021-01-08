import json
import os
import random
from datetime import datetime as dt
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import bnn
import config
from animator import Animator
from dataset_ml import get_dataset, compute_map_density
from dataset_ml import transform_function
from models import initialize_model
from pytorchtools import EarlyStopping
from utils import random_session_name, scaled_full_robot_geometry, mktransf, plot_transform


def plot_metrics(train_metrics, val_metrics, test_metrics, params, save_path):
    fig, axs = plt.subplots(3, 3, figsize=(15, 8), sharex='col')
    axss = np.array(axs)

    def plot(title, axs, metrics, bs=None, x_label='batches'):
        if bs is None:
            bs = params['batch_size']

        df = pd.DataFrame(metrics, index=list(np.arange(0, len(metrics) * bs, bs)),
                          columns=['loss', 'entropy', 'accuracy', 'auc'])
        sns.lineplot(data=df['loss'], ax=axs[0])
        axs[0].set_title(title)
        sns.lineplot(data=df['entropy'], ax=axs[1])
        sns.lineplot(data=df['accuracy'], ax=axs[2], label='Accuracy')
        sns.lineplot(data=df['auc'], ax=axs[2], label='AUC')
        axs[2].set_xlabel(x_label)
        # [ax.xaxis.set_minor_locator(MultipleLocator(base=bs)) for ax in axs]
        for ax in axs:
            ax.grid(axis='x', which='major', linestyle='-')
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plot('Training', axss[:, 0], train_metrics.copy())
    plot('Validation', axss[:, 1], val_metrics.copy())
    plot('Testing', axss[:, 2], test_metrics.copy())
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.suptitle(f'Dataset size: {params["dataset"]}, '
                 f'{"bayesian network, samples:" + str(params["samples"]) if params["samples"] is not None else ""}',
                 y=1.02)
    plt.savefig(save_path, dpi=150)


def visual_test(model, dataset, targets, path, device, n=1, bayes=False, batch_size=8, run_ids=None,
                target_type='big_block'):
    tf = transform_function(resize=bayes)

    def store_animation(run_id):
        run_df = dataset[dataset[target_type] == run_id].copy()
        model.eval()
        with torch.no_grad():
            batches = np.split(run_df, np.arange(batch_size, len(run_df), batch_size))
            preds = []
            for x in tqdm(batches, desc=f'Computing predictions (batch size: {batch_size})'):
                p = model(torch.stack(x['image'].map(tf).values.tolist()).to(device))
                if bayes:
                    p = torch.stack(p, dim=0).mean(dim=0)
                preds.append(free_tensor(p))
            occ_map = np.vstack(preds)[..., -1]
        run_df['predicted_map'] = pd.Series(occ_map.tolist(), index=run_df.index)
        video_path = os.path.join(path, f'{run_id}.mp4')
        Animator(run_df, targets, save_path=video_path)

    interesting_run_ids = dataset[target_type].unique()

    if run_ids is None:
        n = min(n, len(interesting_run_ids))
        run_ids = np.random.choice(interesting_run_ids, size=n)
    else:
        n = len(run_ids)
    print(f'Computing animations with predictions for {n} {target_type.replace("_"," ")}{"s" if n > 1 else ""}:')
    sleep(0.2)
    for r in run_ids:
        store_animation(r)
        sleep(0.2)


def compute_masks(y, device):
    # Gets rid of 'unknown' -1 cells, this masks considers 0 and 1 values, empty and full
    mask = y > -1

    # To filter out unknown cells in the occupancy map
    masked_y = y.type(torch.LongTensor)[mask].to(device)

    # Detach it so sklearn behaves
    masked_stacked_y = free_tensor(torch.stack([1 - y, y], dim=2)[mask])

    return mask, masked_y, masked_stacked_y


def free_tensor(x: torch.Tensor):
    return x.detach().cpu().numpy()


def get_loss(loss_function, divergence=None, data_parallel=False):
    # The divergence function needs to be told exactly where the bnn is
    # Otherwise it won't find the various methods
    module_if_parallel = (lambda x: x.module) if data_parallel else (lambda x: x)

    def f(output, mask, masked_y, model):
        # Applying log before feeding it to NLLLoss, since it demands a LogSoftmax input
        loss = loss_function(torch.log(output[mask] + 1e-10), masked_y)
        return loss

    def g(output, mask, masked_y, model):
        likelihood = torch.stack([
            loss_function(torch.log(pred[mask] + 1e-10), masked_y) for pred in output
        ]).mean()  # / model.samples
        loss = likelihood + divergence(module_if_parallel(model))
        return loss

    return f if divergence is None else g


def get_entropy_and_accuracy(entropy_function):
    # Output here can be aggregated predictions from bayesian or a simple traditional output
    def test(output, mask, masked_y):
        # TODO: not correct, give a non aggregated input to bnn one
        entropy = entropy_function(output[mask])

        # Compare masked softmax logits and desired output to compute accuracy
        accuracy = torch.eq(output.argmax(dim=-1)[mask], masked_y).float().mean()
        return entropy, accuracy

    return test


def get_auc(bayes):
    """
     Need to check for corner cases in which the y_true contains only one class
     More likely to happen in the first part of a run, with contiguous batches (DataLoader's shuffle=False)

     Example of a bad y_true (only 3 free cells in the masked, stacked desired output of cells):
        [[1, 0]
         [1, 0]
         [1, 0]]

    """

    def f(output, mask, masked_stacked_y):
        if len(np.unique(masked_stacked_y, axis=0)) == 1:
            return np.nan
        return roc_auc_score(masked_stacked_y, free_tensor(output[mask]))

    def g(output, mask, masked_stacked_y):
        if len(np.unique(masked_stacked_y, axis=0)) == 1:
            return np.nan
        return np.nanmean([
            roc_auc_score(masked_stacked_y, free_tensor(pred[mask])) for pred in output
        ])

    return g if bayes else f


def training_step(model, x, y, optimizer, loss_function, auc_function, aggregate_samples, entropy_accuracy_function,
                  device):
    # Clear the gradients
    optimizer.zero_grad()

    # Run the model for minibatch prediction
    output = model(x)

    # Compute all needed masked of desired output
    mask, masked_y, masked_stacked_y = compute_masks(y, device)

    # Also feed model into the parameters, for the bayesian case where the divergence needs to be calculated
    loss = loss_function(output, mask, masked_y, model)
    # if torch.isnan(loss).any() or torch.isinf(loss).any():
    #     print('loss!')
    #     code.interact(local=locals())

    # Calculate error contribution
    loss.backward()

    # for name, param in model.named_parameters():
    #     if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
    #         print(name, torch.isinf(param.grad).any(), torch.isnan(param.grad).any())
    #         code.interact(local=locals())

    # Update the weights
    optimizer.step()

    # Sklearn function to calculate area under ROC curve
    # Also feed model into the parameters, for the bayesian case where the divergence needs to be calculated
    auc = auc_function(output, mask, masked_stacked_y)

    # Aggregate output in case we are in the bayesian case
    output = aggregate_samples(output)

    # Calculate entropy with the specified function, get accuracy with softmax logits
    entropy, accuracy = entropy_accuracy_function(output, mask, masked_y)

    return loss, auc, entropy, accuracy


def testing_step(model, x, y, loss_function, auc_function, aggregate_samples, entropy_accuracy_function, device):
    # Run the model for mini-batch prediction
    output = model(x)

    # Compute all needed masked of desired output
    mask, masked_y, masked_stacked_y = compute_masks(y, device)

    # Also feed model into the parameters, for the bayesian case where the divergence needs to be calculated
    loss = loss_function(output, mask, masked_y, model)

    # Sklearn function to calculate area under ROC curve
    # Also feed model into the parameters, for the bayesian case where the divergence needs to be calculated
    auc = auc_function(output, mask, masked_stacked_y)

    # Aggregate output in case we are in the bayesian case
    output = aggregate_samples(output)

    # Calculate entropy with the specified function, get accuracy with softmax logits
    entropy, accuracy = entropy_accuracy_function(output, mask, masked_y)

    return output, loss, auc, entropy, accuracy


# This only works with a model that normalizes its output with a softmax
def train_val_test(model, device, dataset, batch_size, val_dataset, targets, bayes=False, n_epochs=10, patience=1,
                   path=None, parallel=False):
    train_loader, val_loader, test_loader = dataset

    # Training
    # filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(path, 'checkpoint.pt'))

    # For the training step
    loss_function = get_loss(
        nn.NLLLoss(),
        bnn.nn.KLDivergence(number_of_batches=len(train_loader)) if bayes else None,
        data_parallel=parallel
    )
    entropy_accuracy_function = get_entropy_and_accuracy(bnn.nn.Entropy(dim=-1))
    auc_function = get_auc(bayes=bayes)
    aggregate_samples = (lambda x: torch.stack(x, dim=0).mean(dim=0)) if bayes else (lambda x: x)

    # To track the metrics, also necessary for early stopping
    train_losses, valid_losses, avg_train_losses, avg_valid_losses, train_metrics, val_metrics, test_metrics = \
        ([] for _ in range(7))

    for epoch in range(1, n_epochs + 1):

        # Training

        model.train()
        t = tqdm(train_loader)
        for batch_idx, (X, y) in enumerate(t):
            loss, auc, entropy, accuracy = training_step(model, X.to(device), y.to(device), optimizer, loss_function,
                                                         auc_function, aggregate_samples, entropy_accuracy_function,
                                                         device)
            t.set_description(f'[E{epoch:02}] Training, last loss: {loss.item():2.4f})')
            train_losses.append(loss.item())
            train_metrics.append([loss.item(), entropy.item(), accuracy.item(), auc])

        # Validation

        model.eval()
        with torch.no_grad():
            t = tqdm(val_loader)
            for batch_idy, (X, y) in enumerate(t):
                _, loss, auc, entropy, accuracy = testing_step(model, X.to(device), y.to(device), loss_function,
                                                               auc_function, aggregate_samples,
                                                               entropy_accuracy_function, device)
                t.set_description(f'[E{epoch:02}] Validation, last loss: {loss.item():2.4f})')
                valid_losses.append(loss.item())
                val_metrics.append([loss.item(), entropy.item(), accuracy.item(), auc])
            visual_test(model, val_dataset, targets, os.path.join(path, 'video', 'validation', f'epoch{epoch}'),
                        device, bayes=bayes, n=1, batch_size=batch_size, target_type='run')

        # Early stopping logic block

        train_loss, valid_loss = np.average(train_losses), np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(n_epochs))
        print(f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] '
              f'train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}')
        train_losses, valid_losses = [], []

        # Makes a new checkpoint if validation loss is good
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Testing')
    model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pt')))
    model.eval()

    outputs, ys = [], []
    appropriate_free = (lambda x: np.array([free_tensor(s) for s in x])) if bayes else (lambda x: free_tensor(x))
    with torch.no_grad():
        t = tqdm(test_loader)
        for batch_idx, (X, y) in enumerate(t):
            output, loss, auc, entropy, accuracy = testing_step(model, X.to(device), y.to(device), loss_function,
                                                                auc_function, aggregate_samples,
                                                                entropy_accuracy_function, device)
            t.set_description(f'Testing, last loss: {loss.item():2.4f})')
            outputs.append(appropriate_free(model(X.to(device))))
            ys.append(free_tensor(y))
            test_metrics.append([loss.item(), entropy.item(), accuracy.item(), auc])

    return (train_metrics, val_metrics, test_metrics), (outputs, ys)


def entropy(x, axis=-1):
    return np.sum(-x * np.log2(x + 1e-10), axis=axis)


def plot_entropy_auc_summary(outputs, ys, bayes, save_path):
    if bayes:
        preds = np.moveaxis(np.hstack(outputs), 1, 0)
    else:
        preds = np.vstack(outputs)

    assert (bayes and (preds.ndim == 4)) or ((not bayes) and (preds.ndim == 3))

    # Aggregate if necessary, but keep the original before ensembling
    ave_preds = preds
    if bayes:
        ave_preds = np.mean(ave_preds, axis=1)

    # (minibatch, 400, 2) -> (minibatch, 400) -> (400)
    entropy_cells = entropy(ave_preds, axis=-1).mean(axis=0)

    if bayes:
        #(samples, minibatch, 400, 2) -> (samples, minibatch, 400) -> (samples, 400) -> (400)
        entropy_cells_exp = entropy(preds, axis=-1).mean(axis=1).mean(axis=0)
    else:
        entropy_cells_exp = 0

    mutual_info = entropy_cells - entropy_cells_exp

    y = np.vstack(ys)
    y[y == -1] = np.nan
    stacked_y = np.dstack([1 - y, y])

    auc_cells = []
    for i in range(400):
        # Default value for the plot: NaN -> white transparent pixel
        val = np.nan

        # Select all the iterations of the dataset for this cell
        s, p = stacked_y[:, i], ave_preds[:, i]
        mask = ~np.isnan(s)
        mask = mask.any(axis=-1)  # mask.shape == (bs,)
        s, p = s[mask, :], p[mask, :]

        # If this cell sees more than 1 class(free/obstacle), calculate the roc_auc_score
        if s.squeeze().ndim > 1 and len(np.unique(s[:, 0])) > 1:
            val = roc_auc_score(s, p)
        auc_cells.append(val)
    auc_cells = np.array(auc_cells)

    fig, axs = plt.subplots(1, 3 if bayes else 2, figsize=(10, 5), dpi=100)

    def plot_map(data, ax, title, arrow_color='white', vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cm.get_cmap('magma'), vmax=vmax, vmin=vmin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('x (forward axis)')
        ax.set_xlabel('y', rotation=0, labelpad=10)
        ratio = 20 / 0.8
        for r in scaled_full_robot_geometry(ratio):
            plot_transform(ax, mktransf((10 - 0.5, 20 - 0.5, -np.pi / 2)) @ r, color=arrow_color,
                           length=config.max_sensing_distance * ratio)
        ax.plot([10 / 20, 0], [0, 0.5], linewidth=.8, linestyle='--', color=arrow_color, transform=ax.transAxes)
        ax.plot([10 / 20, 1], [0, 0.5], linewidth=.8, linestyle='--', color=arrow_color, transform=ax.transAxes)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(im, ax=ax, cax=cax)
        ax.set_title(title)

    plot_map(auc_cells.reshape(20, 20), axs[0], title='AUC', arrow_color='black', vmax=None, vmin=0.5)
    plot_map(entropy_cells.reshape(20, 20), axs[1], title='Entropy', vmax=0.25)
    if bayes:
        plot_map(mutual_info.reshape(20, 20), axs[2], title=f'Mutual Information')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def reset_random_generators():
    random.seed(0)
    np.random.seed(0xDEADBEEF)
    torch.set_rng_state(torch.manual_seed(0).get_state())
    torch.manual_seed(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train the convolutional network')
    parser.add_argument('--dataset', '-d', metavar='.pkl', dest='d_path', help='Directory with pickles', required=True)
    parser.add_argument('--output-dir', '-o', metavar='path', dest='output_path', required=True,
                        help='Experiments folder containing models and stats')
    parser.add_argument('--max_size', '-ms', metavar='size', dest='max_dataset_size', type=int,
                        help='How many rows of the dataset to keep (last time I checked: 261K rows). '
                             'Useful for potato computers')
    parser.add_argument('--batch-size', '-bs', metavar='bs', dest='batch_size', type=int, default=32,
                        help='Batch size for the convolutional network')
    parser.add_argument('--samples', '-s', dest='samples', type=int,
                        help='Number of samples for the bayesian network (ensemble)')
    args = parser.parse_args()

    bayes = args.samples is not None

    code_name = random_session_name()

    # Determinism doesnt hurt
    reset_random_generators()

    # Load targets
    points_file = os.path.join(args.d_path, 'points.json')
    with open(points_file) as f:
        points = json.load(f)
    targets = np.array([[t["x"], t["y"]] for t in points["targets"]])

    # Slower but catches any backward NaN
    # torch.autograd.set_detect_anomaly(mode=True)

    # Initialize model, different inverted residual structure to achieve ~1mil in both despite the bayesian layer
    model, batch_size, device, parallel = initialize_model(args.batch_size, args.samples)

    # Initialize a new session
    experiment_dir = f'{dt.now().strftime("%m%d-%H%M")}_{code_name}{f"_b{args.samples}" if bayes else ""}'
    o_path = os.path.join(args.output_path, experiment_dir)
    Path(o_path).mkdir(parents=True, exist_ok=True)
    print(f'New training session, code name: {code_name} (folder: {o_path})')

    # Backup weights
    pretrained_weights_path = os.path.join(o_path, 'pretrained.pt')
    torch.save(model.state_dict(), pretrained_weights_path)

    for perc in [0.0625, .125, .25, .50, .75, 1][-1:]: # TODO
        print(f'Now training using only {perc*100}%')

        # Hopefully the same subsets are drafted
        reset_random_generators()

        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))

        session_folder = os.path.join(o_path, f'training_{int(perc*100):03}')

        # Prepare a torch-ready tensor dataset for training
        (train_dataframe, val_dataframe, test_dataframe), torch_datasets = \
            get_dataset(args.d_path, batch_size=batch_size, resize=bayes, perc_train_set=perc)

        compute_map_density(train_dataframe, session_folder, emulate_random_flip=False)

        # Main training function
        metrics, test_stats = train_val_test(model, device, torch_datasets, batch_size, val_dataframe, targets,
                                             bayes=bayes, n_epochs=10, patience=2, path=session_folder,
                                             parallel=parallel)

        # Plot the evolution of the metrics during training, validation, testing
        plot_metrics(*metrics, params={'samples': args.samples, 'batch_size': batch_size,
                                       'dataset': len(torch_datasets[0].dataset)},
                     save_path=os.path.join(session_folder, f'metrics.png'))

        # Plot the entropy and AUC of the predicted occupancy maps, static image over the entire test set
        plot_entropy_auc_summary(*test_stats, bayes=bayes, save_path=os.path.join(session_folder, f'uncertainty.png'))

        # Predict and animate N random runs from the same test set used for the final metrics
        visual_test(model, test_dataframe, targets, os.path.join(session_folder, 'video'), device, bayes=bayes,
                    n=1, batch_size=batch_size)

    print(f'\nAll training sessions completed successfully!')


if __name__ == '__main__':
    main()
