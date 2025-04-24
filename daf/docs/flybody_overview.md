# FlyBody Package Overview

This documentation provides a technical overview of the `flybody` Python package and associated tools for simulating fruit fly locomotion and flight using reinforcement learning.

## Repository Structure

The repository is organized into two main directories:

*   `flybody/`: Contains the core Python package with simulation models, tasks, and agents.
*   `daf/`: Contains deployment and automation framework scripts, including setup, testing, agent deployment, and this documentation.

### `flybody` Package Structure

```
flybody/
├── __init__.py         # Package initializer
├── agents/             # Reinforcement learning agents (DMPO, etc.) and components
├── fruitfly/           # Fruit fly model definition and assets
│   ├── assets/         # Model XML, meshes, etc.
│   └── build_fruitfly/ # Scripts/tools for building/modifying the model
├── tasks/              # Simulation task definitions (e.g., walking, flight, vision)
│   ├── arenas/         # Arena definitions for tasks
│   └── ...             # Base task classes, rewards, utils, etc.
└── ...                 # Other potential helper modules (e.g., utils)
```

### `daf` Directory Structure

```
daf/
├── __init__.py             # Package initializer (if treated as a package)
├── ant/                    # Potentially related to Ant build tool or another project? (Structure unclear)
├── docs/                   # Documentation files (Markdown)
├── output/                 # Default location for generated outputs (logs, images, videos, checkpoints)
├── tasks/                  # DAF-specific task configurations or extensions? (Structure unclear)
├── utils/                  # Utilities specific to the DAF framework
├── venv/                   # Default virtual environment location for daf/setup_and_test.sh
├── deploy_agents.py        # Main script for training/evaluating agents
├── setup_and_test.sh       # Script for environment setup, installation, testing, and example generation
└── ...                     # Other potential helper scripts or config files
```

## Key Modules and Functionality

Detailed documentation for each major component can be found here:

- **[Root Level Files](./flybody_root_files.md):** Explanation of important files in the repository root.
- **[Fruit Fly Model (`flybody/fruitfly/`)](./flybody_fruitfly.md):** Description of the MuJoCo fruit fly model, its components, build process, and observables.
- **[Tasks (`flybody/tasks/`)](./flybody_tasks.md):** Overview of the task structure, base classes, reward functions, and specific task implementations (walking, flight, vision).
- **[Agents (`flybody/agents/`)](./flybody_agents.md):** Details on the RL agents (DMPO), network architectures, and actor implementations.
- **[Perception](./perception.md):** How the agent perceives the environment (observations).
- **[Action](./action.md):** How the agent's actions are structured and applied.

This documentation aims to provide a clear understanding of how the different parts of the `flybody` repository interact to simulate and control the virtual fruit fly, and how to use the `daf` tools to manage experiments. 