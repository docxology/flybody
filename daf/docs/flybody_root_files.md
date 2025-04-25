# Repository Root Files

This document describes the key files and directories located at the root of the `flybody` repository.

- **`README.md`**: The main entry point for understanding the project. Provides a high-level overview, installation instructions, usage examples, and links to further documentation (like this one).

- **`pyproject.toml`**: Defines the project metadata, build system configuration (e.g., using `setuptools`), and crucially, the package dependencies. Specifies core dependencies and optional extras like `[tf]` (for TensorFlow/Acme) and `[ray]` (for distributed training).

- **`LICENSE`**: Contains the software license information for the repository (e.g., MIT, Apache 2.0).

- **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (e.g., virtual environments, compiled files, output directories).

- **`flybody/`**: The core Python package containing the simulation models, tasks, and agents. See `flybody_overview.md` for its internal structure.

- **`daf/`**: The Deployment and Automation Framework directory. Contains scripts for setup, testing, agent deployment, managing outputs, and documentation. See `flybody_overview.md` for its internal structure.

- **`docs/`**: Contains general documentation or high-level design documents for the project (distinct from the `daf/docs` which focuses on the `flybody` package and `daf` usage).

- **`tests/`**: Contains automated tests for the `flybody` package, likely using `pytest`.

- **`.github/workflows/`**: Contains GitHub Actions workflow definitions, e.g., for continuous integration (CI) testing.

- **`flybody.egg-info/`**: Directory generated during package installation (specifically when using `pip install -e .`), containing metadata about the installed package.

- **`fly-white.png`**: An image file, possibly a logo or asset used in the README.

# FlyBody Root Level Files

This document describes the Python modules located directly within the `flybody` package directory.

- **`__init__.py`**: Standard Python package initializer. Makes `flybody` recognizable as a package.

- **`loggers.py`**: Contains utility functions for logging simulation data and experiment results, potentially integrating with frameworks like TensorBoard. Includes functions for creating loggers and formatting log messages.

- **`quaternions.py`**: Provides a suite of functions for performing mathematical operations with quaternions, crucial for handling 3D rotations of the fly's body and limbs. Supports vectorized operations and batch dimensions.
    - Key functions: `mult_quat`, `conj_quat`, `reciprocal_quat`, `rotate_vec_with_quat`, `quat_to_angvel`, `axis_angle_to_quat`.

- **`train_dmpo_ray.py`**: An example script demonstrating how to train a Distributed MPO (DMPO) agent using the Ray framework for distributed computing. Sets up the environment, agent, and training loop.

- **`utils.py`**: Contains general utility functions used across different modules within the `flybody` package. This might include helper functions for data manipulation, array operations, or string processing.

- **`inverse_kinematics.py`**: Implements solvers for inverse kinematics (IK). Given a desired target position or orientation for an end-effector (like a leg tip), IK calculates the necessary joint angles to achieve that pose.

- **`download_data.py`**: A script likely used to download required assets or datasets, such as motion capture data or pre-trained models, needed for certain tasks or examples.

- **`ellipsoid_fluid_model.py`**: Implements a model for simulating aerodynamic forces on the fly, approximated as an ellipsoid. This is important for realistic flight simulation, calculating drag and lift based on the fly's shape and movement through the air.

- **`fly_envs.py`**: Provides wrappers and potentially base classes for creating `dm_env` compatible environments from the FlyBody tasks. It might handle environment registration, standardizing interfaces, or adding common wrappers (e.g., for observation normalization or action scaling). 