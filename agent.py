from typing import Tuple

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as f
import torch.optim as optim


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 5           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""""A named tuple used to collect the different fields within the replay buffer"""
ExperienceTuple = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class Agent:
    """ The reinforcement learning agent.  """
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        self.eps = EPS_START
        self.eps_decay = 1/(EPS_EP_END*LEARN_NUM)  # set decay rate based on epsilon end target
        self.timestep = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OrnsteinUhlenbeckNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)


class OrnsteinUhlenbeckNoise:
    """ Ornstein-Uhlenbeck process. The process is a stationary Gauss–Markov process,
    which means that it is a Gaussian process, a Markov process, and is temporally homogeneous."""

    def __init__(self, size: int, seed: int, mu: float = 0., theta: float = 0.15, sigma: float = 0.2) -> None:
        """
            Initialize an OrnsteinUhlenbeckNoise object.
            Args:
                size (int): The dimension of the noise vector.
                seed (int): The initialization value for the random number generator.
                mu (float): The mean value for the generated noise.
                theta (float): The drift value of the process.
                sigma (float): The diffusion value of the process.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = None
        self.reset()

    def reset(self) -> None:
        """ Reset the internal state to the mean value. """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
            Updates internal state and returns an updated state vector.
            Returns:
               The updated state vector.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples. """

    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int) -> None:
        """
            Initialize a ReplayBuffer object.
            Args:
                action_size (int): The dimension of each action
                buffer_size (int): The maximum size (number of tuples) of the buffer
                batch_size (int): The size of each training batch
                seed (int): The initialization parameter for the random number generator.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = ExperienceTuple
        self.seed = random.seed(seed)

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
            next_state: torch.Tensor, done: torch.Tensor) -> None:
        """
            Create a new experience tuple and add it to the Replay buffer..
            Args:
                state (torch.Tensor): A state vector.
                action (torch.Tensor): An action vector.
                reward  (torch.Tensor): A reward vector.
                next_state  (torch.Tensor): A vector containing the states following the given states.
                done  (torch.Tensor): A vector containing done flags.
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Retrieve a batch size random sample from the ReplayBuffer.
            Returns:
               The random sample from the ReplayBuffer.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])) \
            .float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])) \
            .float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float() \
            .to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float() \
            .to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float() \
            .to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
            Return the current number of samples within the ReplayBuffer.
            Returns:
               The current number (int) of samples within the ReplayBuffer.
        """
        return len(self.memory)
