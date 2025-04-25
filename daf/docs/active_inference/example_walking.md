# Example: Walking with Active Inference

This document provides a practical example of implementing an Active Inference agent for a walking task in the FlyBody framework.

## Task Description

In this example, we'll implement a fly walking task with the following characteristics:

- **Environment**: A terrain with varying elevation and obstacles
- **Goal**: Navigate to a target location while maintaining balance and energy efficiency
- **Observability**: Partially observable - the fly can only see a limited range ahead
- **Challenge**: The fly must infer terrain properties from limited observations and adapt its gait

## Implementation Components

### 1. Environment Setup

We'll use the `WalkOnBall` task with some modifications to introduce partial observability:

```python
from flybody.tasks import walk_on_ball
from flybody.tasks.arenas import heightfield

# Create an arena with varying terrain
arena = heightfield.HeightFieldArena(
    terrain_smoothness=0.8,
    terrain_type='mixed',
    visible_range=2.0  # Only 2 meters ahead is visible
)

# Create the walking task
env = walk_on_ball.WalkOnBallTask(
    walker=FruitFly,
    arena=arena,
    time_limit=30.0,
    use_legs=True,
    use_wings=False,
    physics_timestep=_WALK_PHYSICS_TIMESTEP,
    control_timestep=_WALK_CONTROL_TIMESTEP,
    joint_filter=0.01,
    adhesion_filter=0.007,
    eye_camera_size=64
)
```

### 2. Active Inference Agent

Our agent combines the perception and policy modules:

```python
from flybody.agents.active_inference import ActiveInferenceAgent

# Create the agent with appropriate dimensions
agent = ActiveInferenceAgent(
    observation_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_state_dim=128,
    planning_horizon=10,
    learning_rate=3e-4,
    precision=5.0,
    log_dir='logs/active_inference_walking'
)
```

### 3. Training and Deployment

The training loop showcases how the Active Inference process works:

```python
# Training loop
for episode in range(1000):
    observation = env.reset()
    agent.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Perception: Update beliefs about the world
        agent.update_beliefs(observation)
        
        # Policy selection: Find the best policy based on expected free energy
        action = agent.select_action()
        
        # Execute action
        next_observation, reward, done, info = env.step(action)
        
        # Learning: Update internal models
        agent.update(observation, action, next_observation, reward)
        
        # Log data for visualization
        agent.log_step(observation, action, reward, next_observation)
        
        observation = next_observation
        total_reward += reward
    
    print(f"Episode {episode}, reward: {total_reward}")
    agent.log_episode(episode, total_reward)
```

## Key Active Inference Mechanisms

### Perception Process

1. **Initial observation**: The fly receives inputs from proprioceptive sensors and its eye cameras
2. **State inference**: The agent infers hidden states including:
   - Its global position and orientation
   - Properties of terrain ahead (even parts not directly visible)
   - Dynamic variables (velocity, stability)
3. **Precision weighting**: More reliable sensory channels are given higher weight
4. **Prediction errors**: Differences between predicted and actual observations drive updates to the internal model

### Policy Selection Process

1. **Goal representation**: Target position and preferred states (stable, energy-efficient posture)
2. **Candidate policies**: Generation of possible walking gaits and movement trajectories
3. **Expected free energy calculation**:
   - Epistemic value: How much each policy reduces uncertainty about the terrain
   - Pragmatic value: How much each policy gets the fly closer to its goal
4. **Policy distribution**: Probability of selecting each policy based on its EFE
5. **Action selection**: Sampling or MAP estimation from the policy distribution

## Visualization and Analysis

The agent logs key variables to visualize:

1. **Belief states over time**:
   - The fly's belief about its position
   - The estimated terrain ahead
   - Uncertainty in different state variables

2. **Policy evaluation**:
   - Expected free energy for each candidate policy
   - Contribution of epistemic vs. pragmatic components
   - Selected policy distribution

3. **Performance metrics**:
   - Walking efficiency (reward per step)
   - Stability (pitch/roll variation)
   - Accuracy of terrain predictions
   - Free energy minimization over time

## Code Structure

The full implementation includes:

```
flybody/
  agents/
    active_inference/
      __init__.py
      agent.py          # Main agent class
      perception.py     # Perception module
      policy.py         # Policy selection module
      models.py         # Neural network models
      visualization.py  # Logging and visualization tools
      utils.py          # Helper functions
  
  tasks/
    walk_inference/
      __init__.py
      active_inference_walk.py  # Task definition for this example
```

## Execution

To run this example:

```bash
# From the repository root
cd daf
python -m flybody.tasks.walk_inference.active_inference_walk --episodes 1000 --log-dir logs/active_inference_walk
```

This will train the agent and save visualization data to the specified log directory. 