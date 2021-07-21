import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from dataset_ml import get_dataset
from models import initialize_model
from training import plot_entropy_auc_summary, visual_test
from utils import free_tensor


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
    bayes = args.samples is not None

    # Determinism doesnt hurt
    random.seed(0)
    np.random.seed(0xDEADBEEF)
    torch.manual_seed(0)

    # Load weights
    weights_path = os.path.join(args.o_path, 'checkpoint.pt')
    assert os.path.exists(weights_path)

    # Load targets for final animations
    points_file = os.path.join(args.d_path, 'points.json')
    with open(points_file) as f:
        points = json.load(f)
    targets = np.array([[t["x"], t["y"]] for t in points["targets"]])

    # Slower but catches any backward NaN
    torch.autograd.set_detect_anomaly(mode=True)

    # Initialize model, different inverted residual structure to achieve ~1mil in both despite the bayesian layer
    model, batch_size, device, parallel = initialize_model(args.batch_size, args.sample, weights_path)

    # Prepare a torch-ready tensor dataset for training
    test_dataframe, torch_datasets = get_dataset(args.d_path, batch_size=batch_size, resize=bayes, test_only=True)

    # Produce outputs fast
    test_stats = fast_testing(model, device, dataset=torch_datasets[0], bayes=bayes)

    # Plot the entropy and AUC of the predicted occupancy maps, static image over the entire test set
    plot_entropy_auc_summary(*test_stats, bayes=bayes, save_path=os.path.join(args.o_path, f'uncertainty.png'))

    # Predict and animate N random runs from the same test set used for the final metrics
    visual_test(model, test_dataframe[0], targets, os.path.join(args.o_path, 'video'), device, bayes=bayes, n=2,
                batch_size=batch_size)

    print(f'\nTesting completed successfully!')


if __name__ == '__main__':
    main()


