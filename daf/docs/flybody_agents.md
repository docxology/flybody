# FlyBody Agents (`agents/`)

This document describes the `flybody/agents` directory, which contains implementations of reinforcement learning agents and related components, primarily based on the Acme framework.

## Overview

The `agents` directory houses the logic for agent decision-making, learning algorithms, and network architectures used to control the simulated fly.

## Key Modules

- **`__init__.py`**: Package initializer.

- **`agent_dmpo.py`**: Implements the Distributed Maximum a Posteriori Policy Optimization (DMPO) agent. This is a key actor-critic RL algorithm used in the repository. It defines the `DMPO` agent class, the `DMPOBuilder` for constructing agent components, and the `DMPONetworks` data structure.

- **`actors.py`**: Contains implementations of Acme `Actor` classes. Actors are responsible for selecting actions based on observations from the environment using a policy network. The `DelayedFeedForwardActor` is a notable example, allowing for action delays.

- **`learning_dmpo.py`**: Implements the `DistributionalMPOLearner`, which handles the learning updates for the DMPO agent. It defines the loss functions, optimizer steps, and target network updates based on data sampled from a replay buffer.

- **`losses_mpo.py`**: Defines the MPO loss function, including its components like KL constraints (e.g., `epsilon`, `epsilon_mean`, `epsilon_stddev`) and dual variable updates (e.g., temperature, alphas). Contains the `MPO` Sonnet module.

- **`network_factory.py`**: Provides functions (`make_policy_network`, `make_critic_network`) for creating the neural network architectures (policy and critic networks) used by the agents. Typically uses MLPs (Multi-Layer Perceptrons) built with Sonnet.

- **`network_factory_vis.py`**: Similar to `network_factory.py`, but specifically designed for vision-based tasks. It defines a `VisNet` module (often using convolutions) to process visual input and a `TwoLevelController` architecture, which can integrate a pre-trained low-level controller with a trainable high-level controller that processes visual features and task inputs.

- **`ray_distributed_dmpo.py`**: Implements a distributed version of the DMPO agent using the Ray framework. This allows for scaling up training by distributing actors and learners across multiple processes or machines.

- **`remote_as_local_wrapper.py`**: A utility likely used in the distributed setup (Ray) to wrap remote objects (like actors or learners) to behave like local objects.

- **`utils_tf.py`**: Contains TensorFlow-specific utility functions used within the agents module, such as functions for creating or restoring networks from checkpoints (`restore_dmpo_networks_from_checkpoint`).

- **`utils_ray.py`**: Contains Ray-specific utility functions for the distributed agent implementations.

- **`counting.py`**: Utilities for counting steps or episodes, often used for coordinating updates or logging (based on Acme's `counting` module).

## Core Concepts

- **Acme Framework**: The agent implementations heavily leverage the DeepMind Acme framework for RL, using its interfaces for Actors, Learners, Adders (replay buffers), and Environment Loops.
- **DMPO Algorithm**: The primary learning algorithm is DMPO, an off-policy actor-critic method suitable for continuous control tasks.
- **Sonnet**: Neural networks are typically defined using DeepMind's Sonnet library.
- **Reverb**: Used for efficient replay buffering, especially in distributed settings.
- **TensorFlow/TF-Agents**: The underlying deep learning framework is TensorFlow, with potential use of TF-Agents components. 