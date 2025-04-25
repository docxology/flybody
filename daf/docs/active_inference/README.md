# Active Inference in FlyBody

This document outlines how to implement an Active Inference-based agent for the FlyBody framework, focusing on partially observable Markov decision processes (POMDPs) for perception and policy selection.

## What is Active Inference?

Active Inference is a framework from computational neuroscience that unifies perception, learning, and decision-making under a single principle: minimizing free energy (or, equivalently, maximizing model evidence). Unlike traditional reinforcement learning where agents aim to maximize expected rewards, Active Inference agents aim to minimize the difference between their predictions and what they observe.

The key components of Active Inference include:
- **Generative models** that capture the agent's beliefs about the world
- **Variational inference** for perception (inferring hidden states)
- **Policy selection** through expected free energy minimization
- **Unified treatment** of exploration and exploitation

## Implementation in FlyBody

In the context of FlyBody, we'll implement Active Inference for the following tasks:
1. Walking with a partially observable environment
2. Vision-guided navigation
3. Obstacle avoidance with uncertainty

## Architecture Overview

Our implementation consists of:

1. **Perception Module**
   - State estimation from partial observations
   - Variational inference for hidden state representation
   - Precision-weighted prediction errors

2. **Policy Selection Module**
   - Policy evaluation using expected free energy
   - Temporal horizon planning
   - Adaptive exploration based on uncertainty

3. **Integration with FlyBody**
   - Observation processing from sensors
   - Action mapping to the fly's control system
   - Logging and visualization

## Getting Started

See the implementation details and examples in the following documents:
- [Perception Implementation](./perception.md)
- [Policy Selection](./policy_selection.md)
- [Example: Walking with Active Inference](./example_walking.md)
- [Visualization and Logging](./visualization.md) 