from utils import ReplayMemory, Transition
from model_def import DQN, DQNBot
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import count
from icecream import ic
import random
import math

BATCH_SIZE = 10
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
n_actions = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_model(model: DQNBot, optimizer: optim.Optimizer):
    memory = model.memory
    policy_net = model.policy_net
    target_net = model.target_net

    ic(len(memory))
    if len(memory) < BATCH_SIZE:
        ic(len(memory))
        return

    transitions = memory.sample(BATCH_SIZE)
    # converts list of transitions into a Transition tuple of arrays
    batch = Transition(*zip(transitions))

    # Compute mask of non-terminal states and concatenate the batch elements
    non_final_mask = torch.Tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )

    # Concatenate the next state for each batch together into one tensor
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # state, action, and reward fields for each batch concatenated into one tensor
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Here we compute Q(s_t, a) using the policy network. We do this by feeding the current state batch into the policy network, and selecting the columns of actions taken.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Expected values of actions for the non-terminal next states are computed
    # by the target_net; The mask selects only non-terminal states. Terminal states are
    # left at 0.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )

    # Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Perform weight updates
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train(model: DQNBot, num_episodes: int, optimizer: optim.Optimizer):

    for ep in range(num_episodes):

        last_screen = model.get_board()
        current_screen = model.get_board()

        state = current_screen - last_screen

        for t in count():

            # Calculate reward and action
            reward, game_over = model.step()
            reward = torch.Tensor([reward], device=device)
            action = torch.Tensor([model.next_action], device=device)

            # Update the state
            last_screen = current_screen
            current_screen = model.get_board()
            if not game_over:
                next_state = current_screen - last_screen

            else:
                next_state = None

            # Push the current transition into memory
            model.memory.push(state, action, next_state, reward)

            # Move to next state
            state = next_state

            # Perform one step of optimization
            optimize_model(model, optimizer)

        if ep % TARGET_UPDATE == 0:
            model.target_net.load_state_dict(model.policy_net.state_dict())
