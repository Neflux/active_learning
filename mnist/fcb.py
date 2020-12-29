import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as tnn

import bnn.nn as nn

epochs = 100
learning_rate = 0.001

transform = transforms.ToTensor()
trainset = datasets.MNIST('', train=True, download=True, transform=transform)
testset = datasets.MNIST('', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)


class Net(nn.BayesianNetworkModule):
    def __init__(self, samples=12):
        super(Net, self).__init__(28 * 28, 10, samples)

        self.layers = torch.nn.Sequential(
            nn.NormalLinear(28 * 28, 64),
            tnn.ReLU(),
            nn.NormalLinear(64, 64),
            tnn.ReLU(),
            nn.NormalLinear(64, 64),
            tnn.ReLU(),
            nn.NormalLinear(64, 10),
            tnn.Softmax(dim=-1)
        )

    def _forward(self, x):
        return self.layers(x)


net = Net()
print(net)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_function = torch.nn.NLLLoss() if isinstance(net.layers[-1], tnn.LogSoftmax) else torch.nn.CrossEntropyLoss()
KLdiv = nn.KLDivergence(number_of_batches=len(train_loader))
get_entropy = nn.Entropy(dim=-1)

aggregator = (lambda preds, dim: preds.sum(dim=dim)) if isinstance(loss_function, torch.nn.NLLLoss) else (
    lambda preds, dim: preds.prod(dim=dim))

for epoch in range(epochs):
    for batch_idx, (X, y) in enumerate(train_loader):
        net.zero_grad()

        preds = net(X.view(-1, 28 * 28))

        likelihood = torch.stack([
            loss_function(pred, y) for pred in preds
        ]).mean() / net.samples

        kld = KLdiv(net)
        loss = likelihood + kld

        loss.backward()
        optimizer.step()

        aggregated_preds = aggregator(torch.stack(preds, dim=0), dim=0)
        logits = aggregated_preds.argmax(dim=-1)
        accuracy = 100.*logits.eq(y).float().mean()
        entropy = get_entropy(aggregated_preds)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch + 1} [{batch_idx * len(X)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLikelihood: {likelihood.item():.6f}'
                  f'\tKL Divergence {kld.item():.6f}\tLoss: {loss.item():.6f}'
                  f'\tAccuracy: {accuracy.item():.1f}%\tEntropy: {entropy:.6f}')

    correct = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            preds = net(X.view(-1, 28 * 28))
            logits = aggregator(torch.stack(preds, dim=-1), dim=-1).argmax(dim=-1)
            test_loss += torch.stack([loss_function(pred, y) for pred in preds]).mean() / net.samples + KLdiv(net)
            correct += logits.eq(y).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

while input() != 'q':
    X, y = next(iter(train_loader))
    plt.imshow(X[0].view(28, 28))
    first_pred = aggregator(torch.stack(net(X[0].view(-1, 28 * 28)), dim=-1), dim=-1).argmax(dim=-1)[0]
    plt.title(f"{y[0]} is classified as {first_pred}")
    plt.show()
