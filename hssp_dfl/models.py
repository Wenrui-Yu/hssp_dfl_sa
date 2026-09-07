"""Model architectures used by the paper's experiments (Appendix G).

fc / cnn            baselines on MNIST and CIFAR-10
cnn_cifar           the CIFAR-10 classifier: three convolutional layers
                    (32, 64, 128 channels, 3x3 kernels) then two fully
                    connected layers (512 and 10 units)
purchase_fc         Purchase-100: two fully connected layers, 256 hidden
logistic_regression Sentiment140, over ada-002 embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class fc(nn.Module):
    def __init__(self, input_dim=32*32*3, output_dim=10):
        super(fc, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out=self.fc1(out)
        # out = F.softmax(out, dim=1)
        return out

class cnn(torch.nn.Module):
    def __init__(self, output_dim=10):
        super().__init__()
        # Build parameters
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)

        self.fc1 = nn.Linear(4*4*32, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.softmax(x, dim=1)
        return x

class cnn_cifar(nn.Module):
    def __init__(self, output_dim=10):
        super(cnn_cifar, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class purchase_fc(nn.Module):
    def __init__(self, input_dim=600, output_dim=100):
        super(purchase_fc, self).__init__()
        # self.fc1 = nn.Linear(input_dim, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 256)
        # self.fc4 = nn.Linear(256, output_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class logistic_regression(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(logistic_regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        return out
