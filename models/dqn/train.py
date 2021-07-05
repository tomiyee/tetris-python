from utils import ReplayMemory, Transition
from model_def import DQN
import torch
import torch.optim as optim
import torch.nn as nn
import random
import math

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
n_actions = 6 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def optimize_model(policy_net : DQN, target_net : DQN, memory : ReplayMemory, optimizer : optim.Optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # converts list of transitions into a Transition tuple of arrays
    batch = Transition(*zip(transitions))

    # Compute mask of non-terminal states and concatenate the batch elements
    non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)


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
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    #Compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    #Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    #Perform weight updates
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def train(num_episodes, policy_net, target_net, )












