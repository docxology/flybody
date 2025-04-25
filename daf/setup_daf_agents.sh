#!/bin/bash

# Setup script for DAF deployment framework
# This script sets up the deployment infrastructure for the DAF framework

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Flybody DAF Agent Deployment Framework...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Make sure output directories exist
mkdir -p output/logs
mkdir -p output/images
mkdir -p output/animations
mkdir -p output/stats
mkdir -p output/checkpoints

# Make all Python scripts executable
chmod +x deploy_agents.py
chmod +x deploy_flight_imitation.py
chmod +x deploy_vision_flight.py
chmod +x deploy_walk_imitation.py

# Check if the Python environment is set up properly
if command -v python3 >/dev/null 2>&1; then
    echo -e "${GREEN}Python 3 found.${NC}"
    
    # Check if required libraries are installed
    echo "Checking for required Python packages..."
    REQUIRED_PACKAGES=("tensorflow" "numpy" "matplotlib" "dm_env" "acme" "sonnet")
    MISSING_PACKAGES=()

    for package in "${REQUIRED_PACKAGES[@]}"; do
        # Use a more robust check that handles submodules like acme.tf
        if ! python3 -c "import $package" >/dev/null 2>&1; then
             if [ "$package" == "acme" ]; then
                 # Try importing a common submodule as well
                 if ! python3 -c "import acme.agents" >/dev/null 2>&1; then
                    MISSING_PACKAGES+=("$package")
                 fi
             elif [ "$package" == "sonnet" ]; then
                  if ! python3 -c "import sonnet.v2" >/dev/null 2>&1; then
                    MISSING_PACKAGES+=("$package (dm-sonnet)")
                 fi
             else
                 MISSING_PACKAGES+=("$package")
             fi
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
        echo -e "${GREEN}All required Python packages seem to be installed.${NC}"
    else
        echo -e "${YELLOW}Some potentially required packages might be missing:${NC}"
        for package in "${MISSING_PACKAGES[@]}"; do
            echo "  - $package"
        done
        echo -e "${YELLOW}If scripts fail, please ensure you have run the main setup script first:${NC}"
        echo "  ./daf/setup_and_test.sh"
    fi
else
    echo -e "${RED}Python 3 not found. Please make sure Python 3 is installed.${NC}"
    exit 1
fi

# Create a README file with usage instructions (ensure it mentions prerequisite setup)
cat > README_AGENTS.md << 'EOF'
# Flybody Agent Deployment Framework

This framework provides tools for deploying, training, and evaluating agents on various Flybody tasks.

## Prerequisites

**IMPORTANT:** Before using these scripts, ensure you have successfully run the main DAF setup script to create the necessary environment and install all dependencies:

```bash
# From the repository root
cd daf
chmod +x setup_and_test.sh
./setup_and_test.sh
# Activate the created environment (e.g., conda activate flybody_env or source venv/bin/activate)
```

## Quick Start

Once the environment is set up and activated:

```bash
# (Assuming you are in the daf/ directory)

# Make scripts executable if needed
chmod +x *.py

# Deploy an agent on the template task
./deploy_agents.py --task template --mode deploy

# Train an agent on the flight imitation task
./deploy_flight_imitation.py --mode train --episodes 50

# Evaluate a trained agent
./deploy_vision_flight.py --mode eval --checkpoint output/vision_flight_trench_train/checkpoints/best_agent
```

## Available Scripts

### Main Deployment Script

The main deployment script can be used for any task:

```bash
./deploy_agents.py --task <task_name> --mode <mode>
```

### Task-Specific Scripts

For convenience, specialized scripts are provided for common tasks:

- `deploy_flight_imitation.py`: For flight imitation tasks
- `deploy_vision_flight.py`: For vision-guided flight tasks
- `deploy_walk_imitation.py`: For walk imitation tasks

Each script has task-specific configurations optimal for that environment.

## Common Options

All scripts support the following options:

- `--mode`: Operation mode
  - `train`: Train an agent from scratch
  - `eval`: Evaluate a trained agent
  - `deploy`: Run a single episode with visualization
- `--episodes`: Number of episodes to run
- `--max-steps`: Maximum steps per episode
- `--checkpoint`: Path to a checkpoint for evaluation
- `--output-name`: Custom name for the output directory
- `--save-frames`: Save individual frames as images
- `--seed`: Random seed for reproducibility

## Output Structure

Each run creates a timestamped directory under `output/` with the following structure:

```
output/<run_name>_<timestamp>/
├── animations/     # MP4 animations of episodes
├── checkpoints/    # Agent checkpoints
├── images/         # Individual frames as images
├── logs/           # TensorBoard logs and metrics (.jsonl)
├── stats/          # Summary statistics (.json)
├── args.json         # Command line arguments for the run
└── metadata.json     # Run metadata
```

## Examples

### Training a Walking Agent

```bash
./deploy_walk_imitation.py --mode train --episodes 100 --max-steps 500
```

### Evaluating a Trained Flight Agent

```bash
./deploy_flight_imitation.py --mode eval --checkpoint output/flight_imitation_train_YYYYMMDD_HHMMSS/checkpoints/best_agent --episodes 5
```

### Deploying a Vision-Guided Flight Agent

```bash
./deploy_vision_flight.py --mode deploy --scene-type trench --checkpoint output/vision_flight_trench_train_YYYYMMDD_HHMMSS/checkpoints/best_agent
```

## Advanced Usage

For more complex scenarios, you can write custom deployment scripts that import and use the components from this framework:

```python
# Make sure your environment is activated first

from deploy_agents import create_environment, create_agent, run_episode
from daf_agent_utils import create_run_dirs, MetricLogger

# Create environment and agent
env = create_environment('your_custom_task')
agent = create_agent(env_spec)

# Create output directories and logger
output_dirs = create_run_dirs("my_custom_run")
logger = MetricLogger(output_dirs['logs'])

# Run episodes and collect data
run_episode(env, agent, output_dirs=output_dirs, logger=logger)
```
EOF

echo -e "${GREEN}DAF Agent Deployment Framework setup script finished.${NC}"
echo -e "${YELLOW}IMPORTANT: Ensure you have run './daf/setup_and_test.sh' first to install dependencies.${NC}"
echo -e "${YELLOW}See README_AGENTS.md for usage instructions.${NC}"
echo ""
echo "Quick start example (after running setup_and_test.sh and activating env):"
echo "  cd daf"
echo "  ./deploy_agents.py --task template --mode deploy"
echo ""
echo "Happy deploying!" 