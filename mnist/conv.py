import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

epochs = 100
learning_rate = 0.001
batch_size = 1000

transform = transforms.ToTensor()
trainset = datasets.MNIST('', train=True, download=True, transform=transform)
testset = datasets.MNIST('', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

per_epoch = 5
log_freq = int(len(train_loader) * 1./per_epoch)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(32 * 3 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)


net = Net()
print(net)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_idx, (X, y) in enumerate(train_loader):
        net.zero_grad()

        preds = net(X.view(-1, 1, 28, 28))
        loss = loss_function(preds, y)

        loss.backward()
        optimizer.step()

        logits = preds.argmax(dim=-1)
        accuracy = 100. * logits.eq(y).float().mean()

        if batch_idx % log_freq == 0:
            print(f'Train Epoch: {epoch + 1} [{batch_idx * len(X)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
                  f'\tAccuracy: {accuracy.item():.1f}%')

    correct = 0
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            preds = net(X.view(-1, 1, 28, 28))
            test_loss += loss_function(preds, y).item()

            logits = preds.argmax(dim=1)
            correct += logits.eq(y).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

while input() != 'q':
    X, y = next(iter(train_loader))
    plt.imshow(X[0].view(28, 28))
    first_pred = torch.argmax(net(X[0].view(-1, 1, 28, 28)))
    plt.title(f"{y[0]} is classified as {first_pred}")
    plt.show()
