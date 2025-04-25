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

## Enhanced POMDP Implementation

The framework now features an enhanced POMDP-based implementation that more explicitly models the partially observable nature of the environment:

- **Explicit POMDP Model**: Models uncertainty in both state transitions and observations
- **Precision-Weighted Inference**: Assigns different weights to different sensory channels
- **Counterfactual Planning**: Evaluates potential action sequences by simulating future beliefs
- **Information-Seeking Behavior**: Balances exploitation (goal-seeking) and exploration (uncertainty-reduction)
- **Belief Visualization**: Visualizes belief states and their evolution over time

## Getting Started

To run a simple active inference demo:

```bash
cd /path/to/flybody/daf
python active_flyference/deploy_active_inference.py
```

To run the enhanced POMDP implementation:

```bash
python active_flyference/deploy_pomdp_agent.py --task walk_on_ball --mode demo
```

For advanced usage and training:

```bash
# Standard Active Inference
python active_flyference/deploy_active_inference.py --task walk_on_ball --mode train --episodes 100

# Enhanced POMDP Implementation
python active_flyference/deploy_pomdp_agent.py --task walk_on_ball --mode train --episodes 20 \
    --num-states 128 --planning-horizon 3 --exploration-factor 0.3 --debug
```

## Directory Structure

```
active_flyference/
├── README.md                   # This file
├── active_inference_agent.py   # Original active inference agent
├── pomdp_agent.py              # Enhanced POMDP-based agent
├── deploy_active_inference.py  # Script to run original active inference
├── deploy_pomdp_agent.py       # Script to run enhanced POMDP implementation
├── models/                     # Generative models
│   ├── generative_model.py     # Base class for generative models
│   ├── pomdp_model.py          # POMDP-specific model
│   ├── flybody_pomdp.py        # Flybody-specific POMDP implementation
│   ├── walk_model.py           # Model for walking tasks
│   └── flight_model.py         # Model for flight tasks
└── utils/                      # Utility functions
    ├── free_energy.py          # Basic free energy computations
    ├── pomdp_free_energy.py    # Enhanced POMDP-specific free energy
    ├── inference.py            # Inference algorithms
    └── visualization.py        # Visualization tools
```

## Command-Line Arguments

The POMDP agent implementation supports various command-line arguments:

```
--task              Task to run (walk_on_ball, flight_imitation, etc.)
--mode              Mode to run in (train, eval, demo)
--episodes          Number of episodes to run
--steps             Maximum steps per episode
--num-states        Number of discrete states in POMDP model
--num-actions       Number of discrete actions in POMDP model
--planning-horizon  Steps to consider in planning
--exploration-factor Weight for exploration vs. exploitation (0-1)
--learning-rate     Learning rate for model updates
--inference-iterations Number of iterations for belief updating
--precision         Precision parameter for action selection
--include-vision    Include visual observations
--debug             Enable debug output
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

### POMDP-Specific Extensions

The enhanced POMDP implementation adds several key features:

1. **Precision-Weighted Inference**: Different sensory channels (e.g., proprioception, vision) have different levels of precision, modeling the fly's varying confidence in different sensory inputs.

2. **Temporal Dynamics**: The model explicitly accounts for belief dynamics over time, enabling better prediction and planning.

3. **Information-Seeking Behavior**: The agent balances pragmatic value (achieving goals) with epistemic value (reducing uncertainty), leading to more exploratory behavior.

4. **Counterfactual Planning**: The agent can simulate multiple possible futures to evaluate which actions will lead to preferred states.

5. **Uncertainty Visualization**: The framework provides tools to visualize uncertainty in the agent's beliefs, showing how it evolves during learning.

## Output and Visualization

The framework generates several visualizations to help understand the agent's behavior:

- **Belief Evolution**: Heatmap showing how beliefs evolve over time
- **Free Energy**: Plot of free energy minimization over time
- **Uncertainty Metrics**: Plots of belief entropy and information gain
- **Action Distribution**: Histogram of selected actions
- **Belief State Visualization**: Grid-based visualization of the agent's belief state

## References

1. Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. Neural computation, 29(1), 1-49.
2. Parr, T., & Friston, K. J. (2019). Generalised free energy and active inference. Biological cybernetics, 113(5), 495-513.
3. Sajid, N., Ball, P. J., & Friston, K. J. (2021). Active inference: demystified and compared. Neural computation, 33(3), 674-712.
4. Friston, K., Da Costa, L., Hafner, D., Hesp, C., & Parr, T. (2021). Sophisticated inference. Neural Computation, 33(3), 713-763. 