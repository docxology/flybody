# FlyBody Tasks (`tasks/`)

This document describes the `flybody/tasks` directory, which defines the various simulation environments and objectives for the fruit fly model within the FlyBody framework. These tasks are built upon the `dm_control` composer API.

## Overview

Each task defines the environment, the fly configuration, the goal, observables, and reward structure for a specific scenario (e.g., walking, flying, imitation).

## Directory Structure

```
flybody/tasks/
├── __init__.py
├── arenas/                     # Defines the environments/terrains for tasks
│   ├── __init__.py
│   ├── corridors.py
│   ├── flat_plane.py
│   ├── gaps.py
│   ├── heightfield.py
│   ├── hills.py
│   └── obstacles.py
├── base.py                     # Base classes for tasks (FruitFlyTask, Walking, Flying)
├── constants.py                # Physical and simulation constants
├── flight_imitation.py         # Task: Imitate a recorded flight trajectory
├── pattern_generators.py       # Wing beat pattern generators (e.g., for flight)
├── rewards.py                  # Common reward functions (e.g., tolerance)
├── synthetic_trajectories.py   # Generators for synthetic movement trajectories
├── task_utils.py               # Utility functions specific to tasks
├── template_task.py            # A simple template task example
├── trajectory_loaders.py       # Loaders for motion capture or reference trajectories
├── vision_flight.py            # Task: Vision-guided flight (potentially using WBPG)
├── walk_imitation.py           # Task: Imitate a recorded walking trajectory
└── walk_on_ball.py             # Task: Tethered fly walking on a freely rotating ball
```

## Key Components

- **`base.py`**: Contains the abstract base class `FruitFlyTask` which all specific tasks inherit from. It handles common setup like initializing the walker (`FruitFly` model), the arena, timesteps, basic observables, and termination conditions. It also defines intermediate base classes `Walking` and `Flying` which configure the fly model appropriately for these modes (e.g., disabling wings for walking, setting pitch angles for flight).
    - **Core methods**: `initialize_episode_mjcf`, `initialize_episode`, `before_step`, `get_reward`, `get_discount`, `action_spec`.

- **`arenas/`**: This subdirectory contains different arena definitions. Arenas define the static environment surrounding the fly, such as flat planes, terrains with hills or trenches (`hills.py`, `heightfield.py`), corridors, or obstacles.

- **Specific Task Files** (e.g., `walk_imitation.py`, `flight_imitation.py`, `vision_flight.py`, `walk_on_ball.py`):
    - Each file defines a concrete task class inheriting from `Walking` or `Flying` (or directly from `FruitFlyTask`).
    - They specify the particular arena, reward function (`get_reward_factors`), termination conditions (`check_termination`), observables (often adding task-specific ones), and potentially reference trajectories or targets.
    - **Imitation Tasks**: Load reference motion data (`trajectory_loaders.py`) and reward the agent for matching the reference pose and velocity.
    - **Vision Tasks**: Enable eye cameras and often use specific arenas with visual features (e.g., trenches in `vision_flight.py`).
    - **WalkOnBall**: Simulates a common experimental setup where a tethered fly walks on a ball.

- **`rewards.py`**: Provides reusable reward shaping functions, like `rewards.tolerance` from `dm_control`, used within task-specific `get_reward_factors` methods.

- **`constants.py`**: Defines shared constants like default physics/control timesteps, body pitch angles, wing parameters, etc.

- **`task_utils.py`**: Helper functions used across different task implementations.

- **`pattern_generators.py`**: Specifically for flight, this likely defines models like the Wing Beat Pattern Generator (WBPG) used in `vision_flight.py`, allowing control over wing beat frequency and amplitude.

- **`trajectory_loaders.py`**: Handles loading and processing reference trajectory data (e.g., from CGS or MoCap files) for imitation tasks.

## Core Concepts

- **`dm_control.composer`**: Tasks are built using the composer API, which facilitates combining a walker (agent embodiment), an arena (environment), and task logic (rewards, termination, observables).
- **Walker**: The `FruitFly` model instance configured for the specific task.
- **Arena**: The environment definition (terrain, obstacles).
- **Observables**: Defines the information available to the agent at each step.
- **Reward Function**: Defines the objective the agent tries to maximize.
- **Termination**: Defines conditions under which an episode ends. 