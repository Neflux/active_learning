import code
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset_ml import get_dataset
from model import ConvNet, BayesConvNet
from training import free_tensor, plot_entropy_auc_summary


# This only works with a model that normalizes its output with a softmax
def fast_testing(model, device, dataset, bayes):
    appropriate_free = (lambda x: np.array([free_tensor(s) for s in x])) if bayes else (lambda x: free_tensor(x))
    model.eval()
    outputs, ys = [], []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(dataset, desc=f'Testing')):
            outputs.append(appropriate_free(model(X.to(device))))
            ys.append(free_tensor(y))
    return outputs, ys


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train the convolutional network')
    parser.add_argument('--dataset', '-d', metavar='.pkl', dest='d_path', help='Directory with pickles', required=True)
    parser.add_argument('--output-dir', '-o', metavar='path', dest='o_path', required=True,
                        help='Folder containing model checkpoint')
    parser.add_argument('--max_size', '-ms', metavar='size', dest='max_dataset_size', type=int,
                        help='How many rows of the dataset to keep (last time I checked: 261K rows). '
                             'Useful for potato computers')
    parser.add_argument('--batch-size', '-bs', metavar='bs', dest='batch_size', type=int, default=32,
                        help='Batch size for the convolutional network')
    parser.add_argument('--samples', '-s', dest='samples', type=int,
                        help='Number of samples for the bayesian network (ensemble)')
    args = parser.parse_args()

    weights_path = os.path.join(args.o_path, 'checkpoint.pt')
    assert os.path.exists(weights_path)

    # Determinism doesnt hurt
    np.random.seed(0xDEADBEEF)
    torch.manual_seed(0)

    # Slower but catches any backward NaN
    torch.autograd.set_detect_anomaly(mode=True)

    # Initialize model, different inverted residual structure to achieve ~1mil in both despite the bayesian layer
    common_parameters = {'num_classes': 400, 'mode': 'softmax'}
    if args.samples is not None:
        irs = [[1, 16, 1, 1],
               [6, 24, 2, 2],
               [6, 32, 3, 2],
               [6, 64, 4, 2]]
        model = BayesConvNet(inverted_residual_setting=irs, samples=args.samples, **common_parameters)
    else:
        irs = [[1, 16, 1, 1],
               [6, 24, 2, 2],
               [6, 32, 3, 2],
               [6, 64, 2, 2],
               [6, 96, 1, 1]]
        model = ConvNet(inverted_residual_setting=irs, **common_parameters)

    # Deal with multiple GPUs if present
    batch_size = args.batch_size
    parallel = torch.cuda.device_count() > 1
    weights_loaded = False
    if parallel:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        batch_size = int(batch_size * torch.cuda.device_count())
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            weights_loaded = True
        except RuntimeError:
            print('Model was probably trained on multiple GPUs or incompatible')
        model = nn.DataParallel(model)
        model.samples = args.samples
    else:
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        print('Using device:', device)
    model.to(device)

    if not weights_loaded:
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except RuntimeError:
            print('Model weights are incompatible')
    print('Weights have been loaded successfully')

    # Prepare a torch-ready tensor dataset for training
    torch_dataset, _ = get_dataset(args.d_path, batch_size=batch_size, resize=args.samples is not None,
                                   test_only=True)
    # Produce outputs fast
    test_stats = fast_testing(model, device, dataset=torch_dataset, bayes=args.samples is not None)

    # Plot the entropy and AUC of the predicted occupancy maps, static image over the entire test set
    plot_entropy_auc_summary(*test_stats, bayes=args.samples is not None,
                             save_path=os.path.join(args.o_path, f'uncertainty.png'))

    print(f'\nTesting completed successfully!')


if __name__ == '__main__':
    main()
