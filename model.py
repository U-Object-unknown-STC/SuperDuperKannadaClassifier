import torch
import torch.nn as nn


class SuperDuperFeatureExtractor(nn.Module):
    def __init__(self):
        super(SuperDuperFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 2, 1, 1)
        self.max_pool = nn.MaxPool2d(2, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 2, 1, 1)
        self.conv3 = nn.Conv2d(64, 32, 2, 1, 1)
        self.bn = nn.BatchNorm2d(2048)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.ReLU(out)
        out = self.max_pool(out)
        out = self.conv2(out)
        out = nn.ReLU(out)
        out = self.max_pool(out)
        out = self.conv3(out)
        return out


class SuperDuperClassifier(nn.Module):
    def __init__(self):
        super(SuperDuperClassifier, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        return out


class SuperDuperReconstructor(nn.Module):
    def __init__(self):
        super(SuperDuperReconstructor, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, 2, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 2, 1, 1)
        self.fc = nn.Linear(128, 784)

    def forward(self, x):
        return x

