# Ant Simulation

This package adapts the `flybody` simulation framework to create a biophysically realistic ant simulation. It shares the same structure and components as the original `flybody` framework but focuses on ant-specific features.

## Structure

The package follows the structure of the original `flybody` framework:

```
daf/ant/
├── ant/              # Core ant model
│   ├── assets/       # XML and meshes for the ant model
│   ├── ant.py        # Ant walker class
│   └── observables.py # Ant observables class
├── tasks/            # Task modules
│   ├── arenas/       # Task-specific arenas
│   ├── base.py       # Base task classes
│   ├── walk_on_ball.py # Example task: walking on a ball
│   └── template_task.py # Template for creating new tasks
└── ant_envs.py       # Environment factory functions
```

## Core Components

* **Ant Model**: A detailed MuJoCo model of an ant with articulated legs, antennae, and mandibles.
* **Sensory System**: Similar to the fruit fly model, includes vestibular (accelerometer, gyro, velocimeter), proprioceptive (joint positions/velocities), visual (simulated eyes), and exteroceptive (touch, force) sensory modalities.
* **Control System**: Compatible with both force and position control actuators for all limbs, with configurable filtering.

## Tasks

Currently implemented tasks:

* **WalkOnBall**: The ant tries to balance and walk on a floating ball.
* **TemplateTask**: A minimal task skeleton for implementing new ant behaviors.

## Usage

Create a basic ant environment:

```python
import numpy as np
from daf.ant import ant_envs

# Create an ant environment
env = ant_envs.walk_on_ball(
    force_actuators=True,
    use_antennae=True,
    random_state=np.random.RandomState(42)
)

# Reset the environment and get initial observation
timestep = env.reset()

# Run a simple loop
for _ in range(100):
    action = np.random.uniform(-1, 1, size=env.action_spec().shape)
    timestep = env.step(action)
    if timestep.last():
        break
```

## Creating Custom Tasks

To create a new ant task:

1. Create a new task class inheriting from `daf.ant.tasks.base.Walking` or `daf.ant.tasks.base.Base`
2. Implement the required abstract methods:
   - `get_reward_factors`: Define the reward function
   - `check_termination`: Define termination conditions
3. Add environment creation function to `ant_envs.py`

## Customizing the Ant Model

The default ant model can be customized through parameters in the `Ant` constructor:

* `use_antennae`: Enable/disable antennae and their sensors
* `use_mouth`: Enable/disable mouth parts (mandibles)
* `force_actuators`: Use force control instead of position control
* `joint_filter`/`adhesion_filter`: Smoothing parameters for control signals

Further customization requires modifying the `ant.xml` file in `ant/assets/`. 