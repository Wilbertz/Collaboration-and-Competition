# Collaboration and Competition

<p align="center">
    <img src="./images/robot-tennis.jpg" width="220" title="Random Agent" alt="Robot playing tennis.">
</p>

## Table of Contents

1. [Introduction](#introduction)
2. [Directory Structure](#directoryStructure)
3. [Installation](#installation)
4. [Instructions](#instructions)
5. [Results](#results)

## Introduction <a name="introduction"></a>
<p align="center">
    <img src="./images/tennis.gif" width="800" title="Random Agent" alt="Robot playing tennis.">
</p>

The goal of this project is to train two RL agents to play tennis. 
In this environment, two agents control rackets to bounce a ball over a net. 
The environment is similar, but not identical to the Tennis environment on the Unity ML-Agents GitHub page.

- Action Space  
There are two actions for an Agent, related to the movement relative to the net, and the possibility to jump.

- Observation Space  
The observation space consists of 8 variables corresponding to position and velocity of the ball and rackets.

- Rewards  
A reward of +0.1 is given to an Agent if it hits the ball over the net. If the ball hits the ground, or is hit 
out of bounds, a negative reward of -0.01 is given. There is no discounting. This encourages the Agents to keep the 
ball in play as long as possible.

- Goal  
The training of the Agents is over, once an average score of +0.5 over 100 consecutive episodes is obtained, 
after taking the maximum score of the two Agents.

## Directory Structure <a name="directoryStructure"></a>

- Root /
    - README.md (This readme file)
    - Report.md (A report describing results)
    - Tennis.ipynb (The Jupyter notebook)
    - model.py (The neural network)
    - agent.py (The agent used for learning)
    - checkpoint.pth (The neural network weights for the actors and critics)
    - .gitignore (Git uses it to determine which files and directories to ignore)
    - images /  
        - robot-tennis.jpg  (Image of a robot playing tennis)
        - scores_plot.png (Plot of the scores during the learning process)
        - MADDPG.png (Image of MADDPG pseudocode)
        
## Installation <a name="installation"></a>

This project was written in Python 3.6, using a Jupyter Notebook on Anaconda. Currently (October 2019) you cannot use Python 3.7, since tensorflow 1.7.1 doesn't have a version corresponding to python 3.7 yet.

The relevant Python packages for this project are as follows:

- numpy  
- torch  
- torch.nn  
- torch.nn.functional  
- torch.optim  
- matplotlib.pyplot  
- unityagents 

In order to use the code you have to download the environment from one of the links below:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)  
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)  
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)  
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)  

You need only select the environment that matches your operating system:

Place the file in the GitHub repository and unzip (or decompress) the file.

## Instructions <a name="instructions"></a>

Follow the instructions in Tennis.ipynb to get started.

In order to train a network you have to create an agent collection:

agents = AgentCollection(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=42)

Execute the learning method maddpg. (This can take up to 4 hours when using GPU.)

scores = maddpg(n_episodes=10000)

## Results <a name="results"></a>

The environment was solved in 4651 episodes. An averaged score of 0.515 was reached. 
Below is a plot with the scores:

![scores](images/scores_plot.png)

A more detailed description of the results can be found in the Report.md file.
