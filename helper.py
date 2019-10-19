# main function that sets up environments
# and performs training loop

from collections import deque
from agent import Agent
import numpy as np
import os
import time
import torch
from unityagents import UnityEnvironment

N_EPISODES = 2000
SOLVED_SCORE = 0.5
CONSEC_EPISODES = 100
PRINT_EVERY = 10
ADD_NOISE = True

## Helper functions

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_actions(states, add_noise):
    actions = [agent.act(states, add_noise) for agent in agents]
    # flatten action pairs into a single vector
    return np.reshape(actions, (1, num_agents*action_size))

def reset_agents():
    for agent in agents:
        agent.reset()

def learning_step(states, actions, rewards, next_states, done):
    for i, agent in enumerate(agents):
        agent.step(states, actions, rewards[i], next_states, done, i)
