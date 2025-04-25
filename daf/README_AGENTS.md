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
