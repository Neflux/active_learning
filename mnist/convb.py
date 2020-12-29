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
train_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

per_epoch = 5
log_freq = int(len(train_loader) * 1. / per_epoch)


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class Net(nn.BayesianNetworkModule):
    def __init__(self, samples=12):
        super(Net, self).__init__(1, 10, samples)

        self.layers = torch.nn.Sequential(
            tnn.Conv2d(1, 32, 3, 1),
            tnn.ReLU(),
            nn.NormalConv2d(32, 64, 3, 1),
            tnn.ReLU(),
            tnn.MaxPool2d(2),
            Flatten(),
            nn.NormalLinear(9216, 128),
            tnn.ReLU(),
            nn.NormalLinear(128, 10),
            tnn.Softmax(dim=-1)
        )

    def _forward(self, x):
        return self.layers(x)


net = Net()
print(net)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss()
KLdiv = nn.KLDivergence(number_of_batches=len(train_loader))
get_entropy = nn.Entropy(dim=-1)

for epoch in range(epochs):
    for batch_idx, (X, y) in enumerate(train_loader):
        # why should I zero out the optimizer gradient instead of doing this?
        net.zero_grad()

        # why do I need to reshape while you don't?
        outputs = net(X.view(-1, 1, 28, 28))

        # why should I divide by net.sample if I'm already averaging?
        likelihood = torch.stack([loss_function(output, y) for output in outputs]).mean()
        kld = KLdiv(net)

        loss = likelihood + kld
        loss.backward()
        optimizer.step()

        aggregated_preds = torch.stack(outputs, dim=0).prod(dim=0)
        entropy = get_entropy(aggregated_preds)

        logits = aggregated_preds.argmax(dim=-1)
        accuracy = 100. * logits.eq(y).float().mean()

        if batch_idx % log_freq == 0:
            print(f'Train Epoch: {epoch + 1} [{batch_idx * len(X)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLikelihood: {likelihood.item():.6f}'
                  f'\tKL Divergence {kld.item():.6f}\tLoss: {loss.item():.6f}'
                  f'\tAccuracy: {accuracy.item():.1f}%\tEntropy: {entropy:.6f}')

    correct = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = net(X.view(-1, 1, 28, 28))
            logits = torch.stack(outputs, dim=-1).prod(dim=-1).argmax(dim=-1)
            test_loss += torch.stack([loss_function(output, y) for output in outputs]).mean()
            correct += logits.eq(y).sum().item()

    # why is this loss so low compared to the training one?
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss (without kld): {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

while input() != 'q':
    X, y = next(iter(train_loader))
    plt.imshow(X[0].view(28, 28))
    first_pred = torch.stack(net(X[0].view(-1, 1, 28, 28)), dim=-1).prod(dim=-1).argmax(dim=-1)[0]
    plt.title(f"{y[0]} is classified as {first_pred}")
    plt.show()
