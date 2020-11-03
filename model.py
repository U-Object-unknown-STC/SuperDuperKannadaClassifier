import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperDuperFeatureExtractor(nn.Module):
    def __init__(self):
        super(SuperDuperFeatureExtractor, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 7 * 7, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 128 * 7 * 7)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        # x: [batch_size=128, feature_n=2048]
        return x


class SuperDuperClassifier(nn.Module):
    def __init__(self):
        super(SuperDuperClassifier, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out)
        out = self.bn(out)
        out = self.fc2(out)
        return out
