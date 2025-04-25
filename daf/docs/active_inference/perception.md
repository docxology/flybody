# Perception Module for Active Inference

This document details the implementation of the perception component for Active Inference in the FlyBody framework.

## Overview

In Active Inference, perception is framed as a process of variational inference where the agent infers the hidden states of the environment that best explain its observations. This is done by minimizing variational free energy, which can be understood as minimizing the difference between the predicted and actual observations, plus a regularization term.

## Key Components

### 1. State Space Definition

For the FlyBody tasks, we define:

- **Observation space**: Raw sensory inputs from the fly's sensors
  - Proprioception (joint positions, velocities)
  - Vestibular (gyro, accelerometer)
  - Visual (eye cameras)
  
- **Hidden state space**: Latent variables that the agent infers
  - Position and orientation (global frame)
  - Environmental features (terrain properties, obstacles)
  - Task-relevant variables (target position, optimal path)

### 2. Generative Model Architecture

```
Observations <-- Generation <-- Hidden States
                                    ^
                                    |
                                Inference
                                    |
                                    v
                               Predicted States
```

The generative model consists of:

- **Transition model**: P(s_t | s_{t-1}, a_{t-1}) - How states evolve given actions
- **Observation model**: P(o_t | s_t) - How states generate observations

These can be implemented as:
- Neural networks for complex, high-dimensional spaces
- Probabilistic state-space models for structured problems
- Gaussian process models for continuous dynamics

### 3. Variational Inference Implementation

The perception module performs:

```python
def update_beliefs(self, observations, previous_state, previous_action):
    # Prediction step (prior)
    predicted_state = self.transition_model.predict(previous_state, previous_action)
    predicted_state_distribution = Normal(predicted_state.mean, predicted_state.variance)
    
    # Update step (posterior)
    likelihood = self.observation_model.likelihood(observations, predicted_state)
    posterior_state = bayes_update(predicted_state_distribution, likelihood)
    
    # Free energy calculation
    free_energy = KL_divergence(posterior_state, predicted_state_distribution) - log_likelihood
    
    return posterior_state, free_energy
```

### 4. Integration with FlyBody Sensors

The perception module interfaces with FlyBody sensor data:

1. **Preprocessing**: Normalize and transform raw sensor data
2. **Feature extraction**: For high-dimensional inputs like vision
3. **Multimodal fusion**: Combine different sensor modalities
4. **Handling occlusions**: Deal with partially observable environments

## Implementation Details

### Instantiating the Perception Module

```python
from flybody.agents.active_inference import PerceptionModule

# Create perception module with appropriate dimensions
perception = PerceptionModule(
    observation_dim=env.observation_space.shape[0],
    hidden_state_dim=64,
    action_dim=env.action_space.shape[0],
    learning_rate=1e-4
)
```

### Key Methods

```python
# Update state beliefs based on new observations
state_estimation, free_energy = perception.update(observation, action)

# Get uncertainty in current state estimate
uncertainty = perception.get_state_uncertainty()

# Reset belief state for new episode
perception.reset()
```

### Visualization and Logging

The perception module provides methods to log and visualize:

- Predicted vs. actual observations
- Belief state distribution over time
- Free energy and prediction errors
- Attention maps (for visual inputs)

## Example: Walking Task Perception

For a walking task, the perception module would:

1. Process proprioceptive feedback (joint angles, forces)
2. Incorporate vestibular information (balance, orientation)
3. Use visual input to detect terrain features ahead
4. Infer the fly's global position and orientation
5. Estimate terrain properties (smoothness, slope, obstacles)

This perception can then inform policy selection to generate appropriate walking gaits. 