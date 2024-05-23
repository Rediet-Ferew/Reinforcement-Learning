# Multi-Armed Bandit and Grid World Reinforcement Learning

This repository contains implementations of various reinforcement learning algorithms applied to the Grid World problem (using the FrozenLake environment from OpenAI Gymnasium) and a single-state multi-armed bandit problem. The implemented algorithms include Value Iteration, Policy Iteration, Q-Learning, Epsilon-Greedy Policy, and the Upper Confidence Bound (UCB) algorithm.

## Table of Contents
- [Problem Definitions](#problem-definitions)
  - [Grid World](#grid-world)
  - [Multi-Armed Bandit](#multi-armed-bandit)
- [Environment Setup](#environment-setup)
- [Algorithms Implemented](#algorithms-implemented)
  - [Value Iteration](#value-iteration)
  - [Policy Iteration](#policy-iteration)
  - [Q-Learning](#q-learning)
  - [Epsilon-Greedy Policy](#epsilon-greedy-policy)
  - [UCB Algorithm](#ucb-algorithm)
- [Usage](#usage)
- [Results](#results)

## Problem Definitions

### Grid World
The Grid World environment is represented by a 2D grid with cells that can be empty, obstacles, or a goal. The agent navigates from a starting point to the goal, receiving rewards and penalties along the way.

### Multi-Armed Bandit
The single-state, multi-armed bandit problem involves selecting one of K arms to pull at each time step, with the goal of maximizing the total cumulative reward over a fixed number of steps.

## Environment Setup
The Grid World problem uses the `FrozenLake-v1` environment from OpenAI Gymnasium. The multi-armed bandit environment is custom-built to simulate K arms with random reward probabilities.

## Algorithms Implemented

### Value Iteration
Value Iteration is applied to the Grid World problem to iteratively update the value function and derive the optimal policy.

### Policy Iteration
Policy Iteration alternates between policy evaluation and policy improvement to find the optimal policy for the Grid World problem.

### Q-Learning
Q-Learning is used to learn the action-value function for the Grid World environment through exploration and exploitation.

### Epsilon-Greedy Policy
The Epsilon-Greedy Policy balances exploration and exploitation in the multi-armed bandit environment by selecting random actions with probability epsilon.

### UCB Algorithm
The Upper Confidence Bound (UCB) algorithm balances exploration and exploitation by selecting actions based on confidence bounds in the multi-armed bandit environment.

## Usage
To run the code, ensure you have Python and the required libraries installed. You can install the dependencies using:
```bash
pip install gymnasium numpy
