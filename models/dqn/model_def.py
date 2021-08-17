import sys

sys.path.append(r"D:\tetris-python-master\tetris-python")

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
import torch.optim as optim
from stubs import TetrisBot
import numpy as np
import random
from itertools import count
import math
from utils import ReplayMemory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 512
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 300000
EPS_DEFER = 0.75
TARGET_UPDATE = 10
NUM_OUTPUTS = None
n_actions = 6


class DQN(nn.Module):
    def __init__(self, height, width, num_outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=1)

        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x: torch.Tensor):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)

        x = self.fc1(x.view(x.size(0), -1))
        return self.fc2(x)
