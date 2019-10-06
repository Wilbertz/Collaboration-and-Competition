from typing import Tuple

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as f
import torch.optim as optim


GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # for soft update of target parameters
LR_ACTOR = 1e-3                 # learning rate of the actor
LR_CRITIC = 1e-3                # learning rate of the critic
WEIGHT_DECAY = 0.0000           # L2 weight decay
BATCH_SIZE = 1024               # mini batch size
BUFFER_SIZE = int(1e6)          # replay buffer size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""""A named tuple used to collect the different fields within the replay buffer"""
ExperienceTuple = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class Agent:
    """ The reinforcement learning agent.  """
    pass
