
# Evolutionary Learning Models
# https://medium.com/ml-everything/evolutionary-learning-models-with-openai-854b5583cf97


# Policy Gradient
# Based off of the stuff here https://medium.com/ml-everything/policy-based-reinforcement-learning-with-keras-4996015a0b1

"""
When we train a policy gradient model, we need to have a trade off between
 - Exploration  : When you make a random decision so that the model
                : gets exposed to different game states and can differentiate
                : good ones from bad ones.
 - Exploitation : When you choose the model's best guess as to which decision
                : to make.
The author of the article above suggests choosing the choices with probability
proportional to the strength of the model's convictions.

Stochastic vs Non-Stochastic
--------------------------------
Stochastic is random.
Non-Stochastic is deterministic.
"""

import numpy as np
import pickle

# Hyper Parameterss
hidden = 200
batch_size = 10
learning_rate = 1e-4
gamma = 0.99 # This is the discount factor
decay_rate = 0.99
resume = False

# Init the Model
cols = 10
rows = 22
input_size = cols * rows
output_size = 6

if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    # We start initializing manually
    model = {}
    # We will use xavier initialization to pseudo-randomly initialize the weights
    model['w1'] = np.random.randn(hidden, input_size) / np.sqrt(input_size)
    model['w2'] = np.random.randn(output_size, hidden) / np.sqrt(input_size)

gradient_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = { k : np.zeros_like(v) for k, v in model.items()}

# Activation Function
# This converts numbers into probabilities
def sigmoid (n) :
    return 1.0 / (1.0 + np.exp(-n))

# Everytime something happens each game tick,
def discount_reward (r):
    discounted_r = np.zeros_like(r, dtype='f')
    running_sum = 0
    for t in reversed (range(r.size)):
        if r[t] != 0:
            running_sum = 0
        # Increment Sum
        running_sum = running_sum * gamma + r[t]
        discounted_r[t] = running_sum
    return discounted_r

def policy_forward (x):
    """x must be a numpy array"""
    hidden_layer = np.dot(model['w1'], x)
    # Apply the relu, which just says nothing less than 0
    hidden_layer[hidden_layer < 0] = 0

    log_pred = np.dot(model['w2'], hidden_layer)
    pred = sigmoid(log_pred)
    # Converts everything into a probability distribution
    pred = pred / pred.sum()

    return pred, hidden_layer

def policy_backward (eph, epdlogp):
    # We will recursively compute the chainrule
    # eph is an array of the intermediate hidden state
    dw2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['w2'])
    # Apply Relu
    dh[eph <= 0] = 0

    dw1 = np.dot(dh.T, epx)
    return {'w1': dw1, 'w2': dw2}

# Begin Training
while True:
    # Get the observation
    observation = # Stuff
    # Feed the observation to the model
    output, hidden_layer = policy_forward(observation)

    # Determine the decision to be made
    pr = np.random.uniform()
    option = None
    for i in range(output.size):
        if pr < output[i]:
            option = i
            break
        i -= option[i]
    print(option)
    #







#
