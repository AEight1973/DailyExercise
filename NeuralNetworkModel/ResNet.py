import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import datetime

# Data path
data_path = './data/cifar10/'

# Pre-load without normalizing  # <1>
tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=True,
                                  transform=transforms.ToTensor())
tensor_cifar10_val = datasets.CIFAR10(data_path, train=False, download=True,
                                      transform=transforms.ToTensor())

# Computer the mean value and the std of each channel  # <2>
imgs = torch.stack([img_t for img_t, _ in tensor_cifar10])

mean = imgs.view(3, -1).mean(dim=1)
std = imgs.view(3, -1).std(dim=1)

# Normalize transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Reload with normalizing
transformed_cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transform
)
transformed_cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transform
)

# Take a shortcut and filter the data in cifar10  # <3>
label_map = {0: 0, 2: 1}  # airplane --> 0, bird --> 1
class_names = ['airplane', 'bird']

cifar2 = [(img, label_map[label])
          for img, label in transformed_cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label])
              for img, label in transformed_cifar10_val if label in [0, 2]]


# ResBlock  # <4>
# one group of convolutions, activation and skip connection
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)  # <5>
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  # <6>
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)  # <7>
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x


# Deep ResNet
class NetResDeep(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=5):
        super(NetResDeep, self).__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)])
        )  # <8>
        self.fc1 = nn.Linear(8 * 8 * n_chans1, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 8 * 8 * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


print(NetResDeep())

# Device
device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")


# Training loop
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = imgs.to(device=device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print("{} Epoch {}, Training loss {:.6f}".format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)
            ))


# Run  # <9>
train_loader = DataLoader(cifar2, batch_size=64, shuffle=True)
model = NetResDeep()
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=0.001)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs=100,
    optimizer=optimizer,
    model=model,
    loss_fn=loss_fn,
    train_loader=train_loader
)