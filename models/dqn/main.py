import sys, os
from pathlib import Path
path= Path(os.path.dirname(__file__))
sys.path.append(str(path.parent.parent.absolute()))
from tetris import TetrisApp
import pygame

sys.path.append(os.path.dirname(__file__))

from model_def import (
    DQN,
    n_actions,
    EPS_END,
    EPS_START,
    EPS_DECAY,
    EPS_DEFER,
    BATCH_SIZE,
    GAMMA,
    TARGET_UPDATE,
    device,
)
NUM_EPOCHS = 2000
from utils import ReplayMemory, Transition
from stubs import TetrisBot
from icecream import ic

import tetrino
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math


tetrino.set_iterations(500)
tetrino.set_seed(500)
tetrino.set_move_limit(100)
tetrino.set_silent(False)
tetrino.set_prune_size(5)

class Model(TetrisBot):

    def __init__(self, training_mode=False, policy_path = None, target_path = None):

        self.memory = ReplayMemory(10000)
        self.next_action = 0
        self.steps_done = 0
        self.internal_score = 0
        self.training_mode = training_mode
        self.prev_fitness = 0
        self.policy_path = None if policy_path is None else policy_path
        self.target_path = None if target_path is None else target_path
        self.previous_board = None
        self.current_epoch = 0

    def get_board(self):
        current_piece_map = (1/2)*torch.Tensor(self.get_current_piece_map()).int()
        board_map = torch.Tensor(self.get_board_state()).int()

        board = current_piece_map + board_map
        board = board.unsqueeze(0).unsqueeze(0).float()

        return board

    def step(self):
        """
        1. Retrieves the next action
        2. Performs action
        3. Calculates reward based on action and state
        4. Checks whether there is a game over

        Returns reward and a boolean indicating whether the game is done
        """

        # Retrieve and perform next action
        action = self.select_action()
        self.next_action = action.item()

        # Calculate reward
        current_fitness = self.fitness_function()
        #penalty = 0 if action.item() not in [3, 4] else -0.005
        penalty = 0
        reward = (current_fitness-self.prev_fitness) + penalty
        if self.training_mode:
            print(f'Took action: {action}, got reward: {reward}')
            print(f'Fitness: {current_fitness}')

        self.prev_fitness = current_fitness
        # TODO: check whether game is over
        game_over = False

        return reward, game_over
    
    def get_correct_col(self):
        for pos, col in enumerate(zip(*self.get_current_piece_map())):
            if any(col): return pos

    def defer_action(self, *args):

        tensorize = lambda x: torch.Tensor([[x]]).int()
        
        col, orientation = tetrino.get_best_move(
                self.get_board_state(), 
                [self.get_current_piece_id()],
                1000
        )
        action = 5

        current_col = self.get_correct_col()
        if current_col > col: action = 3
        if current_col < col: action = 4

        if orientation != self.get_current_piece_orientation(): 
            action = 1

        return tensorize(action)


    def select_action(self):
        sample = random.random()
        #eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
           # -1.0 * (self.steps_done) / EPS_DECAY
        #)
        eps_threshold = max(EPS_END + (EPS_START - EPS_END) * (1 - (self.current_epoch)/(0.75*NUM_EPOCHS)), EPS_END)
        eps_defer = max((EPS_DEFER) * (1 - self.current_epoch/(0.75*NUM_EPOCHS)), 0)
        self.steps_done += 1

        print(f'Epsilon: {eps_threshold}')
        print(f'Epsilon defer: {eps_defer}')

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(self.get_board()).max(1)[1].view(1, 1)

        elif 0 <= sample <= (eps_defer):
            return self.defer_action()

        else:
            rand = random.randrange(n_actions)

            return torch.Tensor([[rand]]).int()

    def optimize_model(self):
        memory = self.memory
        policy_net = self.policy_net
        target_net = self.target_net

        if len(memory) < BATCH_SIZE:
            return

        transitions = memory.sample(BATCH_SIZE)

        # converts list of transitions into a Transition tuple of arrays
        batch = Transition(*zip(*transitions))

        # Compute mask of non-terminal states and concatenate the batch elements
        non_final_mask = (
            torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state)))
            .to(torch.bool)
            .to(device)
        )

        # Concatenate the next state for each batch together into one tensor
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        ).to(device)

        # state, action, and reward fields for each batch concatenated into one tensor
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(torch.int64).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Here we compute Q(s_t, a) using the policy network. We do this by feeding the current state batch into the policy network, and selecting the columns of actions taken.
        state_action_values = policy_net(state_batch)
        state_action_values = state_action_values.index_select(1, action_batch).to(
            device
        )

        # Expected values of actions for the non-terminal next states are computed
        # by the target_net; The mask selects only non-terminal states. Terminal states are
        # left at 0.
        next_state_values = torch.zeros(BATCH_SIZE).to(device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

        # Compute expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute MSE loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Perform weight updates
        self.optimizer.zero_grad()
        loss.backward()
        if self.training_mode:
            print(f'Loss: {loss.item()}')
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train_one_step(self):

        current_screen = self.get_board()
        #ic(current_screen)

        state = current_screen
        #ic(state)
        # Calculate reward and action
        reward, game_over = self.step()
        reward = torch.Tensor([reward])
        action = torch.Tensor([self.next_action])

        # Update the state
        current_screen = self.get_board()
        if not game_over:
            next_state = current_screen

        else:
            next_state = None

        # Push the current transition into memory
        self.memory.push(state, action, next_state, reward)

        # Move to next state
        state = next_state

        # Perform one step of optimization
        self.optimize_model()

        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def next_move(self):

        if not self.steps_done == 0:
            self.train_one_step()
        else:
            height = self.get_board_height()
            width = self.get_board_width()
            self.policy_net = DQN(height, width, n_actions).cuda()
            self.target_net = DQN(height, width, n_actions).cuda()
            if self.policy_path is not None:
                self.policy_net.load_state_dict(torch.load(self.policy_path))
            if self.target_path is not None:
                self.target_net.load_state_dict(torch.load(self.target_path))
            self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
        action = self.next_action
        self.steps_done += 1
        return action

    def calculate_bumpiness(self):
        board_map = torch.Tensor(self.get_board_state()).int().to(device)
        num_rows, num_cols = board_map.size()
        heights = []

        for col in zip(*board_map):
            height = 0
            for index, val in enumerate(col):
                if val.item():
                    height = num_rows - index
                    break
            heights.append(height)

        heights = torch.Tensor(heights).to(device)
        offset_heights = torch.cat((heights[1:], torch.Tensor([0]).to(device)))
        heights = abs(heights - offset_heights)[:-1]
        bumpiness = torch.sum(heights).item()

        return bumpiness

    def calculate_aggregate_height(self):
        board_map = torch.Tensor(self.get_board_state()).int().to(device)
        num_rows, num_cols = board_map.size()
        heights = []

        for col in zip(*board_map):
            height = 0
            for index, val in enumerate(col):
                if val.item():
                    height = num_rows - index
                    break
            heights.append(height)

        heights = torch.Tensor(heights).to(device)
        aggregate_height = torch.sum(heights).item()

        return aggregate_height

    def calculate_holes(self):
        board_map = torch.Tensor(self.get_board_state()).int().to(device)
        num_rows, num_cols = board_map.size()
        heights = []

        for col in zip(*board_map):
            height = 0
            for index, val in enumerate(col):
                if val.item():
                    height = num_rows - index
                    break
            heights.append(height)

        heights = torch.Tensor(heights).to(device)
        blocks_per_col = torch.sum(board_map, dim=0)
        holes = heights - blocks_per_col
        holes = torch.sum(holes).item()

        return holes

    def fitness_function(self):
        lines = self.cleared_lines
        holes = self.calculate_holes()
        bumpiness = self.calculate_bumpiness()
        height = self.calculate_aggregate_height()

        fitness = 0.70666*lines - 0.510066*height - 0.35663*holes - 0.184483*bumpiness
        return fitness

        
if __name__ == '__main__':
    policy_path = f'./tetris_dqn_{NUM_EPOCHS}epochs_policynet.pt'
    target_path = f'./tetris_dqn_{NUM_EPOCHS}epochs_targetnet.pt'

    model = Model(training_mode=True)
    for _ in range(NUM_EPOCHS):
        print(f'EPOCH: {_}')
        seed = random.randint(0, 100)
        App = TetrisApp(model, debug=True, seed=seed)
        App.run()
        del App
        model.prev_fitness = 0
        model.current_epoch = _


    torch.save(model.policy_net.state_dict(), policy_path)
    torch.save(model.target_net.state_dict(), target_path)



