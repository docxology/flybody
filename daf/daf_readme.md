# Flybody Data Analysis Framework (DAF)

This directory contains scripts and tools for setting up, testing, and visualizing the Flybody package. It's designed to be a self-contained accessory unit to the main repository that automates installation, testing, and visualization.

## Overview

The DAF framework automates:
- Environment setup (with or without Conda)
- Package installation (in various modes: core, tf, ray)
- Comprehensive testing
- Generation of visualizations and animations for all available fly environments
- Organized output storage

## Quick Start

```bash
# From the repository root directory:
cd daf
chmod +x setup_and_test.sh
./setup_and_test.sh

# Or from anywhere in the repository:
chmod +x daf/setup_and_test.sh
./daf/setup_and_test.sh
```

## Directory Structure

After running the setup script, the following directory structure is created:

```
daf/
├── setup_and_test.sh  # Main setup and testing script
├── venv/              # Python virtual environment (if not using Conda)
└── output/            # All outputs are stored here
    ├── logs/          # Test results and setup logs
    ├── images/        # Individual frames of simulations (PNG files)
    ├── animations/    # Video animations of the fly models (MP4 files)
    └── examples/      # Example scripts for each environment
```

## Setup Script Options

The `setup_and_test.sh` script includes several configuration options at the top of the file:

```bash
# --- Configuration ---
VENV_NAME="flybody_env"    # Name of the virtual environment
PYTHON_VERSION="3.10"      # Python version to use
INSTALL_MODE="core"        # Installation mode (core, tf, or ray)
```

You can modify these variables before running the script to customize the setup:
- `core`: Minimal installation for experimenting with the fly model
- `tf`: Includes TensorFlow and Acme dependencies for ML capabilities
- `ray`: Includes Ray for distributed training support

## Environment Setup

The script is designed to be flexible regarding the Python environment:

1. If Conda is available, it will create or use a Conda environment
2. If Conda is not available, it falls back to a standard Python virtual environment

To activate the environment after running the script:

```bash
# If using Conda:
conda activate flybody_daf

# If using virtual environment:
source daf/venv/bin/activate
```

## Available Examples

The framework includes examples for all available fly environments:

1. **walk_imitation** - A fly tracking a reference walking trajectory
2. **flight_imitation** - A fly tracking a reference flying trajectory
3. **walk_on_ball** - A tethered fly walking on a floating ball
4. **vision_guided_flight** - Vision-guided flight with obstacle avoidance (bumps/trenches)
5. **template_task** - A simple test environment

Each example is available both as a standalone script and as part of a combined visualization.

## Visualizations

The script generates several types of visualizations:

1. **Individual frames** - PNG images of the simulation at each timestep
2. **Single-view animations** - MP4 videos showing the simulation from a single viewpoint
3. **Multi-view animations** - MP4 videos showing multiple camera angles in a grid format
4. **Combined visualization** - A grid showing all environments side-by-side

## Running Individual Examples

You can run any of the examples individually:

```bash
# Activate the environment first
source daf/venv/bin/activate

# Run an example
cd daf/output/examples
python walk_imitation.py
```

## Current Limitations and Considerations

1. **Action Dimensions**: The examples are configured to use the correct action dimensions for each environment, automatically detecting the action space:
   - Walking environments: 59 dimensions
   - Flight environments: 12 dimensions

2. **Rendering**: The script attempts to detect the available rendering system:
   - With X display server: Uses standard rendering
   - With NVIDIA GPU but no display: Sets up headless rendering with EGL
   - Without either: Skips visualizations (user needs to set MUJOCO_GL manually)

3. **Memory Usage**: Creating visualizations for all environments simultaneously (in the combined visualization) may be memory-intensive, especially for longer simulations.

## Extending the Framework

### Adding New Examples

To add a new example:
1. Edit the `setup_and_test.sh` script
2. Add a new call to the `create_example_script` function
3. Provide the appropriate environment name, function, and action dimensions

### Modifying Existing Examples

Each example is a standalone Python script in the `output/examples` directory. You can modify these directly to change:
- Number of simulation frames
- Rendering options
- Action generation behavior
- Output formats

### Adding Custom Metrics or Analyses

To add custom metrics or analysis to the existing examples:
1. Modify the example scripts to collect the desired data
2. Add plotting or reporting code to visualize results
3. Update the output structure as needed

## Troubleshooting

### Rendering Issues
- If you encounter rendering issues, try setting the MUJOCO_GL environment variable:
  ```bash
  # For EGL (NVIDIA GPU headless):
  export MUJOCO_GL=egl
  export MUJOCO_EGL_DEVICE_ID=0
  
  # For software rendering:
  export MUJOCO_GL=osmesa
  ```

### Action Space Errors
- If you see "could not broadcast input array" errors, there's likely a mismatch between the provided action dimensions and what the environment expects. The examples are now configured to automatically determine the correct action dimensions.

### Environment Setup Issues
- If the script fails to create or update the Python environment, try manually creating it following the instructions in the main repository README. 