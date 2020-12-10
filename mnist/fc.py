import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

epochs = 1
learning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST('', train=True, download=True, transform=transform)
testset = datasets.MNIST('', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)


net = Net()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
print(net)

for epoch in range(epochs):
    for batch_idx, (X, y) in enumerate(train_loader):
        net.zero_grad()

        preds = net(X.view(-1, 28 * 28))

        loss = F.nll_loss(preds, y)
        loss.backward()

        optimizer.step()

        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    correct = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            preds = net(X.view(-1, 28 * 28))
            logits = preds.argmax(dim=1)
            test_loss += F.nll_loss(preds, y, reduction='sum').item()
            correct += logits.eq(y.view_as(logits)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

while input() != 'q':
    X, y = next(iter(train_loader))
    plt.imshow(X[0].view(28, 28))
    first_pred = torch.argmax(net(X[0].view(-1, 28 * 28)))
    plt.title(f"{y[0]} is classified as {first_pred}")
    plt.show()
