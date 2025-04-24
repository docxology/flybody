# DAF Getting Started with FlyBody

This directory contains tools for using the FlyBody model within the DeepMind Active Inference (DAF) framework.

## daf_getting_started.py

The `daf_getting_started.py` script provides a standalone demonstration of the FlyBody model and its capabilities. It is based on the original Jupyter notebook (`docs/getting-started.ipynb`) but is adapted to run as a Python script within the DAF framework.

### Features

- **Automatic Environment Setup**: Checks for FlyBody installation and offers to run the `setup_and_test.sh` script if needed.
- **Comprehensive Demonstration**: Takes you through the key features of the FlyBody model:
  1. Loading and inspecting the model
  2. Visualizing the fly from different angles
  3. Performing kinematic manipulations (rotating, folding wings, etc.)
  4. Creating animations of kinematic movements
  5. Creating and running a reinforcement learning environment (fly-on-ball)
- **Output Management**: All outputs (images, animations, logs) are saved to organized directories within `daf/output`.

### Usage

To run the demonstration:

```bash
cd /path/to/flybody
python daf/daf_getting_started.py
```

If FlyBody is not installed, the script will offer to run the setup script for you.

### Output Directories

- **Images**: `daf/output/images`
- **Animations**: `daf/output/animations`
- **Logs**: `daf/output/logs`

## Related Files

- `setup_and_test.sh`: Sets up the Python environment with all necessary dependencies
- `deploy_agents.py`: For deploying trained agents
- Other DAF-specific scripts for different simulations and experiments

## Requirements

- Python 3.10+
- MuJoCo physics engine
- Dependencies as listed in the repository's `pyproject.toml`

The `setup_and_test.sh` script will install all necessary dependencies in a virtual environment. 