import sys
sys.path.append(r'D:\tetris-python-master\tetris-python')

import torch
import torch.nn as nn
import torch.nn.functional as F
from stubs import TetrisBot
import numpy as np
import random
import math


def conv2d_output_dim(size, kernel_size=5, stride=2):
    return ((size + (kernel_size-1) - 1)//stride) + 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
NUM_OUTPUTS = None
n_actions = 6



class DQN(nn.Module):

    def __init__(self, height, width, num_outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        conv_height = conv2d_output_dim(conv2d_output_dim(conv2d_output_dim(height)))
        conv_width = conv2d_output_dim(conv2d_output_dim(conv2d_output_dim(width)))

        linear_input_size = conv_height*conv_width*32

        self.lin = nn.Linear(linear_input_size, num_outputs)


    def forward(self, x : torch.Tensor):
        x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.lin(x.view(x.size(0), -1))



class DQNBot(TetrisBot):

    def __init__(self, training_mode=False):
        height = self.get_board_height_()
        width = self.get_board_width_()
        self.policy_net = DQN(height, width, n_actions)
        self.target_net = DQN(height, width, n_actions)
        self.steps_done = 0


    def get_board(self):
        current_piece_map =  torch.Tensor(self.get_current_piece_coord_()).int()
        board_map = torch.Tensor(self.get_board_state_()).int()

        board = current_piece_map + board_map

        return board

    def step(self):
        '''
        1. Retrieves the next action
        2. Performs action
        3. Calculates reward based on action and state
        4. Checks whether there is a game over

        Returns reward and a boolean indicating whether the game is done
        '''

        action = self.select_action()


    def select_action(self):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * (self.steps_done)/EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(self.get_board()).max(1)[1].view(1, 1)
        else:
            return torch.Tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)











