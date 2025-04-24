#!/bin/bash
# This script automates the setup and testing of the flybody repository.
# It creates a virtual environment, installs the package, and runs tests.
# All outputs (logs, images, animations) are saved to daf/output directory.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
VENV_NAME="flybody_env"
PYTHON_VERSION="3.10"
# Choose installation mode: 'core', 'tf', or 'ray'
# 'core': Minimal installation for model interaction.
# 'tf': Includes TensorFlow and Acme for running ML policies.
# 'ray': Includes Ray for distributed training.
INSTALL_MODE="core"

# --- Path Setup ---
# Detect if we're running from daf directory or from repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
if [[ "$(basename "$SCRIPT_DIR")" == "daf" ]]; then
    # We're running from daf directory
    REPO_DIR="$(dirname "$SCRIPT_DIR")"
    echo "Running from daf directory. Repository root: $REPO_DIR"
else
    # We're running from somewhere else (likely repo root)
    echo "Running from directory: $SCRIPT_DIR"
    # Check if daf is a subdirectory
    if [[ -d "$SCRIPT_DIR/daf" ]]; then
        REPO_DIR="$SCRIPT_DIR"
        SCRIPT_DIR="$SCRIPT_DIR/daf"
        echo "Repository root: $REPO_DIR"
    else
        echo "ERROR: Cannot determine repository structure. Please run from daf directory or repository root."
        exit 1
    fi
fi

# Create output directory structure
OUTPUT_DIR="$SCRIPT_DIR/output"
LOGS_DIR="$OUTPUT_DIR/logs"
IMAGES_DIR="$OUTPUT_DIR/images"
ANIMATIONS_DIR="$OUTPUT_DIR/animations"
EXAMPLES_DIR="$OUTPUT_DIR/examples"

mkdir -p "$LOGS_DIR" "$IMAGES_DIR" "$ANIMATIONS_DIR" "$EXAMPLES_DIR"
LOG_FILE="$LOGS_DIR/setup_log.txt"

# --- Helper Functions ---
log_step() {
    echo "" # Add spacing
    echo "========================================" | tee -a "${LOG_FILE}"
    echo "STEP: $1" | tee -a "${LOG_FILE}"
    echo "========================================" | tee -a "${LOG_FILE}"
}

create_example_script() {
    # Create a common example script template
    local ENV_NAME="$1"
    local ENV_FUNCTION="$2"
    local ACTION_DIMS="$3"
    local FRAMES="${4:-30}"
    local ADDITIONAL_IMPORTS="${5:-}"
    local ADDITIONAL_CODE="${6:-}"
    local EXAMPLE_FILE="$EXAMPLES_DIR/${ENV_NAME}.py"
    
    local CAMERA_ID="0"
    # Use camera_id=1 for walk_imitation as it has a better view
    if [[ "$ENV_FUNCTION" == "walk_imitation" ]]; then
        CAMERA_ID="1"
    fi
    
    echo "import os
import numpy as np
import mediapy
from pathlib import Path
$ADDITIONAL_IMPORTS

# Example name
EXAMPLE_NAME = '${ENV_NAME}'

# Ensure output paths exist
output_dir = Path('$OUTPUT_DIR')
images_dir = Path('$IMAGES_DIR')
animations_dir = Path('$ANIMATIONS_DIR')

try:
    # Import the flybody module
    from flybody.fly_envs import $ENV_FUNCTION
    
    print(f'Creating {EXAMPLE_NAME} environment...')
    env = $ENV_FUNCTION()
    
    $ADDITIONAL_CODE
    
    # Run simulation for a few steps with random actions
    print('Running simulation with random actions...')
    frames = []
    for i in range($FRAMES):
        action = np.random.normal(size=$ACTION_DIMS)
        timestep = env.step(action)
        
        # Render and save current frame
        pixels = env.physics.render(camera_id=$CAMERA_ID)
        frames.append(pixels)
        
        # Save intermediate frame as image
        frame_path = images_dir / f'{EXAMPLE_NAME}_frame_{i:03d}.png'
        mediapy.write_image(frame_path, pixels)
        print(f'Saved frame {i} to {frame_path}')
    
    # Save animation
    print('Saving animation...')
    animation_path = animations_dir / f'{EXAMPLE_NAME}.mp4'
    mediapy.write_video(animation_path, frames, fps=10)
    print(f'Saved animation to {animation_path}')
    
    # Create multi-view animation if available
    try:
        print('Creating multi-view animation...')
        multi_frames = []
        for i in range(min(10, len(frames))):  # Use first 10 frames for multi-view
            # Get images from different camera angles
            views = []
            for cam_id in range(4):  # Try cameras 0-3
                try:
                    view = env.physics.render(camera_id=cam_id)
                    views.append(view)
                except:
                    pass  # Skip if camera doesn't exist
            
            if len(views) > 1:
                # Arrange views in a grid
                h, w, c = views[0].shape
                grid_size = int(np.ceil(np.sqrt(len(views))))
                grid = np.zeros((grid_size * h, grid_size * w, c), dtype=np.uint8)
                
                for j, view in enumerate(views):
                    row = j // grid_size
                    col = j % grid_size
                    grid[row*h:(row+1)*h, col*w:(col+1)*w] = view
                
                multi_frames.append(grid)
                
                # Save grid as image
                grid_path = images_dir / f'{EXAMPLE_NAME}_multiview_{i:03d}.png'
                mediapy.write_image(grid_path, grid)
        
        if multi_frames:
            # Save multi-view animation
            multiview_path = animations_dir / f'{EXAMPLE_NAME}_multiview.mp4'
            mediapy.write_video(multiview_path, multi_frames, fps=5)
            print(f'Saved multi-view animation to {multiview_path}')
    except Exception as e:
        print(f'Error creating multi-view animation: {e}')
    
    print('Example completed successfully!')
except Exception as e:
    print(f'Error running example: {e}')
    import traceback
    traceback.print_exc()
" > "$EXAMPLE_FILE"
    
    echo "Created example script: $EXAMPLE_FILE"
}

run_example() {
    # Run an example script
    local EXAMPLE_NAME="$1"
    local EXAMPLE_FILE="$EXAMPLES_DIR/${EXAMPLE_NAME}.py"
    local EXAMPLE_OUTPUT="$EXAMPLES_DIR/${EXAMPLE_NAME}_output.txt"
    
    if [ ! -f "$EXAMPLE_FILE" ]; then
        echo "Error: Example script $EXAMPLE_FILE does not exist!"
        return 1
    fi
    
    log_step "Running example: $EXAMPLE_NAME"
    echo "Running $EXAMPLE_NAME example..."
    cd "$REPO_DIR"
    python "$EXAMPLE_FILE" | tee "$EXAMPLE_OUTPUT"
}

# --- Initial Setup ---
# Clear previous log file if it exists
> "${LOG_FILE}"
# Redirect stdout and stderr to log file and console
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Starting flybody setup and test script..."
echo "Full log will be available at: ${LOG_FILE}"
echo "All outputs will be saved to: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo "Current working directory: $(pwd)"
echo "Script directory: $SCRIPT_DIR"
echo "Repository directory: $REPO_DIR"
echo "----------------------------------------"

log_step "Verifying script location and environment"
# Ensure we can find the repo files
cd "$REPO_DIR"
if [ ! -f "pyproject.toml" ] || [ ! -d "flybody" ]; then
    echo "ERROR: Required repository files not found at $REPO_DIR."
    echo "       Current directory: $(pwd)"
    exit 1
fi
echo "Verified repository structure at: $(pwd)"

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not found in PATH."
    exit 1
fi
echo "Verified Python 3 installation: $(python3 --version)"

# --- Environment Setup ---
log_step "Setting up Python environment"

# Check if venv module is available
if ! python3 -m venv --help &> /dev/null; then
    echo "Python venv module not available. Attempting to install it..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3-venv
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3-venv
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-venv
    elif command -v pacman &> /dev/null; then
        sudo pacman -S python-virtualenv --noconfirm
    else
        echo "WARNING: Could not automatically install python3-venv."
        echo "You may need to install it manually."
    fi
fi

# Try to use Conda if available, otherwise fall back to venv
USE_CONDA=false
VENV_DIR="$SCRIPT_DIR/venv"

if command -v conda &> /dev/null; then
    echo "Conda found! Using Conda for environment management."
    USE_CONDA=true
    VENV_NAME="flybody_daf"
    
    if conda env list | grep -q "^${VENV_NAME}\s"; then
        echo "Conda environment '${VENV_NAME}' already exists. Updating..."
        conda install --name "${VENV_NAME}" -c conda-forge python pip ipython --yes
    else
        echo "Creating new Conda environment '${VENV_NAME}'..."
        conda create --name "${VENV_NAME}" -c conda-forge python pip ipython --yes
    fi
    
    # Activate conda environment
    echo "Activating Conda environment: ${VENV_NAME}"
    eval "$(conda shell.bash hook)"
    conda activate "${VENV_NAME}"
    
    # Verify activation
    if [[ "$CONDA_PREFIX" != *"${VENV_NAME}"* ]]; then
        echo "WARNING: Failed to activate conda environment '${VENV_NAME}' automatically within the script."
        echo "         Consider sourcing the script instead: 'source $SCRIPT_DIR/setup_and_test.sh'"
    fi
    
    PYTHON_CMD="python"
else
    echo "Conda not found. Using Python virtual environment instead."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating new virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Using existing virtual environment in $VENV_DIR..."
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Update pip to latest version
    pip install --upgrade pip
    
    PYTHON_CMD="python"
fi

# Display Python version and environment info
echo "Using Python: $($PYTHON_CMD --version)"
echo "Environment: $(which $PYTHON_CMD)"
echo "Pip: $(pip --version)"

# --- Installation ---
log_step "Installing flybody package (Mode: ${INSTALL_MODE})"
echo "Using pip from: $(which pip)"
case ${INSTALL_MODE} in
    core)
        # Install core requirements in editable mode
        pip install -e .
        ;;
    tf)
        # Install core + tensorflow/acme requirements in editable mode
        pip install -e .[tf]
        ;;
    ray)
        # Install core + tf + ray requirements in editable mode
        pip install -e .[ray]
        ;;
    *)
        echo "ERROR: Invalid INSTALL_MODE: '${INSTALL_MODE}'. Choose 'core', 'tf', or 'ray'."
        exit 1
        ;;
esac
echo "flybody installation (Mode: ${INSTALL_MODE}) complete."

# Install additional packages needed for examples and visualizations
echo "Installing additional packages for examples and visualizations..."
pip install mediapy pytest matplotlib

echo "Installed packages:"
pip list | grep -E 'flybody|mediapy|pytest|matplotlib'

# --- Environment Variable Configuration ---
log_step "Environment Variable Configuration Notes (User Action May Be Required)"
echo "The following environment variables might be needed for specific functionalities:"
echo ""
echo "1. MuJoCo Rendering:"
echo "   For rendering using specific hardware backends (like EGL on headless servers)."
echo "   Uncomment and set these if needed, especially if you encounter rendering issues:"
echo "   # export MUJOCO_GL=egl"
echo "   # export MUJOCO_EGL_DEVICE_ID=0  # Adjust '0' based on your GPU ID"
echo ""

if [ "${INSTALL_MODE}" = "tf" ] || [ "${INSTALL_MODE}" = "ray" ]; then
    echo "2. Shared Libraries (for TF/ML/Ray extensions):"
    echo "   If using the '[tf]' or '[ray]' extras, ensure CUDA/CuDNN libraries are findable."
    echo "   You might need to set LD_LIBRARY_PATH to include your CUDA and CuDNN libraries."
fi

# --- Testing ---
log_step "Running tests with pytest"
echo "Running pytest..."
# Run tests from the repository root. Add '-v' for more verbose output.
# Save output to a file
TEST_OUTPUT_FILE="$LOGS_DIR/pytest_output.txt"
$PYTHON_CMD -m pytest -v | tee "$TEST_OUTPUT_FILE"
TEST_EXIT_CODE=${PIPESTATUS[0]}  # Get the exit code of pytest, not tee
echo "Test results saved to $TEST_OUTPUT_FILE"

# --- Create Example Scripts ---
log_step "Creating example scripts"

# Create walk_imitation example (59 action dims)
echo "Creating walk_imitation example..."
create_example_script "walk_imitation" "walk_imitation" "59"

# Create flight_imitation example (36 action dims)
echo "Creating flight_imitation example..."
create_example_script "flight_imitation" "flight_imitation" "36" "20"

# Create walk_on_ball example (59 action dims)
echo "Creating walk_on_ball example..."
create_example_script "walk_on_ball" "walk_on_ball" "59" "40"

# Create vision_guided_flight example (36 action dims)
echo "Creating vision_guided_flight example..."
create_example_script "vision_guided_flight" "vision_guided_flight" "36" "20" "" "    # Configure for vision guided flight with bumps
    env = vision_guided_flight(bumps_or_trench='bumps')"

# Create template_task example (59 action dims)
echo "Creating template_task example..."
create_example_script "template_task" "template_task" "59" "20"

# Create combined visualization example
echo "Creating combined example..."
cat > "$EXAMPLES_DIR/combined_visualization.py" << 'EOL'
import os
import numpy as np
import mediapy
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Ensure output paths exist
output_dir = Path('REPLACE_OUTPUT_DIR')
images_dir = Path('REPLACE_IMAGES_DIR')
animations_dir = Path('REPLACE_ANIMATIONS_DIR')

try:
    # Import all flybody environments
    from flybody.fly_envs import (
        walk_imitation, 
        flight_imitation, 
        walk_on_ball,
        vision_guided_flight,
        template_task
    )
    
    # Create all environments
    print("Creating all environments...")
    envs = {
        "walk_imitation": walk_imitation(),
        "flight_imitation": flight_imitation(),
        "walk_on_ball": walk_on_ball(),
        "vision_guided_flight": vision_guided_flight(bumps_or_trench='bumps'),
        "template_task": template_task()
    }
    
    # Take a few random actions in each environment
    print("Running simulations with random actions...")
    # Action dimensions for each environment
    action_dims = {
        "walk_imitation": 59,
        "flight_imitation": 36,
        "walk_on_ball": 59,
        "vision_guided_flight": 36,
        "template_task": 59
    }
    
    # Camera IDs that work well for each environment
    camera_ids = {
        "walk_imitation": 1,
        "flight_imitation": 0,
        "walk_on_ball": 0,
        "vision_guided_flight": 0,
        "template_task": 0
    }
    
    # Run each environment for a few steps and capture frames
    all_env_frames = {}
    for env_name, env in envs.items():
        print(f"Running {env_name}...")
        frames = []
        for i in range(10):  # Just 10 frames for the combined visualization
            action = np.random.normal(size=action_dims[env_name])
            timestep = env.step(action)
            pixels = env.physics.render(camera_id=camera_ids[env_name])
            frames.append(pixels)
        all_env_frames[env_name] = frames
    
    # Create a grid of animations from all environments
    print("Creating combined visualization...")
    num_frames = 10
    num_envs = len(envs)
    rows = int(np.ceil(np.sqrt(num_envs)))
    cols = int(np.ceil(num_envs / rows))
    
    # Get the frame sizes (assume all frames for a given env have the same size)
    frame_heights = {env_name: frames[0].shape[0] for env_name, frames in all_env_frames.items()}
    frame_widths = {env_name: frames[0].shape[1] for env_name, frames in all_env_frames.items()}
    
    # Calculate the grid size based on the largest frame
    max_height = max(frame_heights.values())
    max_width = max(frame_widths.values())
    
    # Create the combined frames
    combined_frames = []
    for frame_idx in range(num_frames):
        # Create a figure with custom size
        fig = plt.figure(figsize=(cols*5, rows*4))
        gs = GridSpec(rows, cols, figure=fig)
        
        # Add each environment's frame to the grid
        for i, (env_name, frames) in enumerate(all_env_frames.items()):
            row = i // cols
            col = i % cols
            
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(frames[frame_idx])
            ax.set_title(env_name)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Convert figure to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        combined_frames.append(img)
        
        # Save frame as image
        frame_path = images_dir / f"combined_frame_{frame_idx:03d}.png"
        plt.savefig(frame_path, dpi=120)
        print(f"Saved combined frame {frame_idx} to {frame_path}")
        
        plt.close(fig)
    
    # Save combined animation
    print("Saving combined animation...")
    combined_path = animations_dir / "combined_environments.mp4"
    mediapy.write_video(combined_path, combined_frames, fps=5)
    print(f"Saved combined animation to {combined_path}")
    
    print("Combined visualization completed successfully!")
except Exception as e:
    print(f"Error creating combined visualization: {e}")
    import traceback
    traceback.print_exc()
EOL

# Replace placeholders with actual paths
sed -i "s|REPLACE_OUTPUT_DIR|$OUTPUT_DIR|g" "$EXAMPLES_DIR/combined_visualization.py"
sed -i "s|REPLACE_IMAGES_DIR|$IMAGES_DIR|g" "$EXAMPLES_DIR/combined_visualization.py"
sed -i "s|REPLACE_ANIMATIONS_DIR|$ANIMATIONS_DIR|g" "$EXAMPLES_DIR/combined_visualization.py"

echo "Created example scripts in $EXAMPLES_DIR"

# --- Run Examples and Visualizations ---
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    # Try to detect if we're in a graphical environment
    if [[ -z "${MUJOCO_GL}" ]]; then
        # If MUJOCO_GL is not set, we'll try to auto-detect
        if [[ -n "$DISPLAY" ]] && command -v xdpyinfo >/dev/null 2>&1; then
            echo "Detected X display server. Will run examples with visualizations."
            RUN_EXAMPLES=true
        elif command -v nvidia-smi >/dev/null 2>&1; then
            echo "No X display detected, but NVIDIA GPU found. Setting up headless rendering."
            export MUJOCO_GL=egl
            export MUJOCO_EGL_DEVICE_ID=0
            RUN_EXAMPLES=true
        else
            echo "Cannot detect suitable rendering backend. Skipping examples with visualizations."
            echo "To run examples, please set MUJOCO_GL environment variable manually."
            RUN_EXAMPLES=false
        fi
    else
        echo "MUJOCO_GL is set to '$MUJOCO_GL'. Will run examples with visualizations."
        RUN_EXAMPLES=true
    fi

    if [ "$RUN_EXAMPLES" = true ]; then
        # Run individual examples
        run_example "walk_imitation"
        run_example "flight_imitation"
        run_example "walk_on_ball"
        run_example "vision_guided_flight"
        run_example "template_task"
        
        # Run combined visualization at the end
        log_step "Running combined visualization"
        echo "Running combined visualization..."
        cd "$REPO_DIR"
        python "$EXAMPLES_DIR/combined_visualization.py" | tee "$EXAMPLES_DIR/combined_visualization_output.txt"
    fi
fi

# --- Completion ---
log_step "Script Finished"
if [ ${TEST_EXIT_CODE} -eq 0 ]; then
    echo "----------------------------------------"
    echo " Setup and tests completed successfully."
    echo " All outputs saved to: ${OUTPUT_DIR}"
    echo "----------------------------------------"
else
    echo "----------------------------------------"
    echo " WARNING: Setup completed, but pytest encountered errors (Exit code: ${TEST_EXIT_CODE})."
    echo " Please check the logs in ${LOGS_DIR} for details."
    echo "----------------------------------------"
fi

if [ "$USE_CONDA" = true ]; then
    echo "To use the installed environment, activate it in your shell:"
    echo "conda activate ${VENV_NAME}"
else
    echo "To use the installed environment, activate it in your shell:"
    echo "source $VENV_DIR/bin/activate"
fi

echo ""
echo "To view generated outputs:"
echo "- Logs: $LOGS_DIR"
echo "- Images: $IMAGES_DIR"
echo "- Animations: $ANIMATIONS_DIR"
echo "- Examples: $EXAMPLES_DIR"

exit ${TEST_EXIT_CODE} 