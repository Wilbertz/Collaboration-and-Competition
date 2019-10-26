from typing import Tuple

import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as f
import torch.optim as optim

BUFFER_SIZE = int(1e5)         # replay buffer size
BATCH_SIZE = 256               # mini batch size
GAMMA = 0.9                    # discount factor
TAU = 1e-3                     # for soft update of target parameters
LR_ACTOR = 1e-3                # learning rate of the actor 
LR_CRITIC = 1e-3               # learning rate of the critic
WEIGHT_DECAY = 1e-6            # L2 weight decay

SIGMA = 0.1                    # for OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""""A named tuple used to collect the different fields within the replay buffer"""
ExperienceTuple = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


class AgentCollection:
    """ A collection of agents that share a common replay buffer."""

    def __init__(self, num_agents: int, state_size: int, action_size: int, random_seed: int):
        """Initializes an AgentCollection object.
                 Args:
                        num_agents (int): The number of agents within the collection.
                        state_size (int): The dimension of the state vector.
                        action_size (int): The dimension of the action vector.
                        random_seed (int): The initialization value for the random number generator.
                """
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.agents = [Agent(self.memory, state_size, action_size, random_seed) for _ in range(num_agents)]
        
    def reset_noise(self):
        """ Reset the Ornstein Uhlenbeck process within all agents. """
        for agent in self.agents:
            agent.reset()
        
    def step(self, states, actions, rewards, next_states, dones) -> None:
        """
           Save the experience within the ReplayBuffer.
               Args:
                   states (list of torch.Tensor): A state vector.
                   actions (List of torch.Tensor): An action vector.
                   rewards (List of torch.Tensor): A reward vector.
                   next_states (List of torch.Tensor): A vector containing the states following the given states.
                   dones (list of torch.Tensor): A vector containing done flags.
        """
        for agent, state, action, reward, next_state, done in \
                zip(self.agents, states, actions, rewards, next_states, dones):
            agent.step(state, action, reward, next_state, done)
        
    def act(self, states):
        """
            Using the actor network the method return a vector of actions given the state vector
            using the current policy.
                Args:
                    states (list of torch.Tensor): A state vector.
                Returns:
                    An list of action vectors.
        """
        actions = list()
        for agent, state in zip(self.agents, states):
            actions.append(agent.act(state))
        return actions
    
    def checkpoint(self):
        """
            Copy the actor internal state..
                Returns
                   A dictionary of actor states.
        """
        return [{
            'actor': agent.actor_local.state_dict(),
            'critic': agent.critic_local.state_dict()
        } for agent in self.agents]


class Agent:
    """Interacts with and learns from the environment."""
    
    def __init__(self, memory, state_size: int, action_size: int, seed: int):
        """Initializes an Agent object.
         Args:
                memory (ReplayBuffer): The ReplayBuffer shared between the agents.
                state_size (int): The dimension of the state vector.
                action_size (int): The dimension of the action vector.
                seed (int): The initialization value for the random number generator.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.hard_copy(self.actor_target, self.actor_local)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.hard_copy(self.critic_target, self.critic_local)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OrnsteinUhlenbeckNoise(action_size, seed)

        # Replay memory
        self.memory = memory

    @staticmethod
    def hard_copy(target, source) -> None:
        """
            Copy the actor internal state..
                Args:
                    target: The target state
                    source: The source state
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
             next_state: torch.Tensor, done: torch.Tensor):
        """
            Save the experience within the ReplayBuffer.
                Args:
                    state (torch.Tensor): A state vector.
                    action (torch.Tensor): An action vector.
                    reward (torch.Tensor): A reward vector.
                    next_state (torch.Tensor): A vector containing the states following the given states.
                    done (torch.Tensor): A vector containing done flags.
        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        """ In case there are enough experiences within ReplayBuffer, start learning. """
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state: torch.Tensor, add_noise: bool = True):
        """
            Using the actor network the method return a vector of actions given the state vector
            using the current policy.
                Args:
                    state (torch.Tensor): A state vector.
                    add_noise (bool): A flag indicating the use of noise.
                Returns:
                    An vector of actions.
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """ Reset the Ornstein Uhlenbeck process. """
        self.noise.reset()

    def learn(self, experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
              gamma: float) -> None:
        """
            Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Args:
                experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples
                gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Compute critic loss
        q_expected = self.critic_local(states, actions)
        critic_loss = f.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    @staticmethod
    def soft_update(local_model, target_model, tau: float) -> None:
        """
           Update the model parameters according to this formula:
           θ_target = τ*θ_local + (1 - τ)*θ_target

           Args:
               local_model (PyTorch model): weights will be copied from this model
               target_model (PyTorch model): weights will be copied to this model
               tau (float): interpolation parameter, tau = 1 results in complete overwrite
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OrnsteinUhlenbeckNoise:
    """ Ornstein-Uhlenbeck process. The process is a stationary Gauss–Markov process,
    which means that it is a Gaussian process, a Markov process, and is temporally homogeneous."""

    def __init__(self, size: int, seed: int, mu: float = 0., theta: float = 0.15, sigma: float = SIGMA) -> None:
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

    def reset(self):
        """ Reset the internal state to the mean value. """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
            Updates internal state and returns an updated state vector.
            Returns:
               The updated state vector.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

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
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
            Retrieve a batch size random sample from the ReplayBuffer.
            Returns:
               The random sample from the ReplayBuffer.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))\
            .float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None]))\
            .float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None]))\
            .float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None]))\
            .float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8))\
            .float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
            Return the current number of samples within the ReplayBuffer.
            Returns:
               The current number (int) of samples within the ReplayBuffer.
        """
        return len(self.memory)
