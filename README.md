# Collaboration and Competition

## Table of Contents

1. [Introduction](#introduction)
2. [Directory Structure](#directoryStructure)
3. [Installation](#installation)
4. [Instructions](#instructions)
5. [Results](#results)

## Introduction <a name="introduction"></a>
<p align="center">
    <img src="./images/random_agent.gif" width="800" title="Random Agent" alt="Random Agent.">
</p>

## Directory Structure <a name="directoryStructure"></a>

- Root /
    - README.md (This readme file)
    - Report.md (A report describing results)
    - Collaboration_and_Competition.ipynb (The Jupyter notebook)
    - model.py (The neural network)
    - agent.py (The agent used for learning)
    - checkpoint_actor.pth (The neural network weights for the actor)
    - checkpoint_critic.pth (The neural network weights for the critic)
    - images /  
        - robot-table-tennis.gif  (Image of a robot playing tennis)
        - scores_plot.png (Plot of the scores during the learning process)
        
## Installation <a name="installation"></a>

This project was written in Python 3.6, using a Jupyter Notebook on Anaconda. Currently (Septemebr 2019) you cannot use Python 3.7, since tensorflow 1.7.1 doesn't have a version corresponding to python 3.7 yet.

## Instructions <a name="instructions"></a>

Follow the instructions in Collaboration_and_Competition.ipynb to get started.

## Results <a name="results"></a>