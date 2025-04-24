# Active Flyference Framework

This directory contains an implementation of Active Inference for the Flybody model. Active Inference is a unifying framework for understanding action, perception, and learning in biological systems based on the free energy principle.

## Overview

Active Inference frames perception and action as processes of inference and belief updating. This implementation:

1. Models the fly's perception and action as a Partially Observable Markov Decision Process (POMDP)
2. Implements active inference through variational free energy minimization
3. Demonstrates how a fly might navigate and interact with its environment based on beliefs and predictions

## Key Components

- **Generative Model**: Represents the fly's internal model of the world
- **Variational Free Energy**: The objective function to be minimized
- **Belief Updating**: Implementation of perception as Bayesian inference
- **Policy Selection**: Implementation of action selection through expected free energy minimization

## Getting Started

To run a simple active inference demo:

```bash
cd /path/to/flybody/daf
python active_flyference/deploy_active_inference.py
```

For advanced usage and training:

```bash
python active_flyference/deploy_active_inference.py --task walk_on_ball --mode train --episodes 100
```

## Directory Structure

```
active_flyference/
├── README.md                   # This file
├── active_inference_agent.py   # Core implementation of the active inference agent
├── deploy_active_inference.py  # Script to run experiments with the agent
├── models/                     # Generative models for different tasks
│   ├── generative_model.py     # Base class for generative models
│   ├── walk_model.py           # Model for walking tasks
│   └── flight_model.py         # Model for flight tasks
└── utils/                      # Utility functions
    ├── free_energy.py          # Functions for computing free energy
    ├── inference.py            # Inference algorithms
    └── visualization.py        # Visualization tools
```

## Theory: Active Inference and POMDPs

Active Inference extends traditional POMDPs by framing both perception and action as inference problems:

- **States**: The fly's true position, orientation, and other physical variables
- **Observations**: The fly's sensory inputs (vision, proprioception)
- **Actions**: Motor commands to the fly's muscles
- **Policies**: Sequences of actions that the fly might take
- **Generative Model**: The fly's model of how states generate observations and transition
- **Free Energy**: A bound on surprise that the fly minimizes through perception and action

Unlike traditional RL approaches, Active Inference doesn't require explicit reward functions. Instead, the agent has preferences encoded as prior beliefs about desired observations.

## References

1. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. Neural computation, 29(1), 1-49.
2. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. Biological cybernetics, 113(5), 495-513.
3. Sajid, N., Ball, P. J., & Friston, K. J. (2021). Active inference: demystified and compared. Neural computation, 33(3), 674-712. 