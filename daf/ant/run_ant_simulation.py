#!/usr/bin/env python
# coding: utf-8

"""
# Getting started with Ant Simulation from DAF

This script runs a basic ant simulation using the adapted framework
in the `daf/ant` directory, based on the `flybody` structure.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import importlib.util

import numpy as np
# import matplotlib.pyplot as plt # Removed as not used in the simplified version
import mediapy

# Setup paths
ANT_DIR = Path(__file__).parent.absolute()
DAF_DIR = ANT_DIR.parent # Should still be daf/
REPO_DIR = DAF_DIR.parent
# <<< Change output directory to be inside daf/ant/output >>>
OUTPUT_DIR = ANT_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
ANIMATIONS_DIR = OUTPUT_DIR / "animations"
LOGS_DIR = OUTPUT_DIR / "logs" # Keep logs dir for potential future use

# Create output directories if they don't exist
for directory in [OUTPUT_DIR, IMAGES_DIR, ANIMATIONS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# <<< Keep environment setup, but maybe simplify the message >>>
def setup_environment():
    """Check for necessary packages (dm_control, etc.) and suggest setup if needed."""
    # Simplified check - assume basic dependencies like dm_control are needed
    # A more robust check would verify specific versions or key imports.
    if importlib.util.find_spec("dm_control") is None or importlib.util.find_spec("mujoco") is None:
        print("\n========== DEPENDENCIES MISSING ==========")
        print("Required packages (dm_control, mujoco) not found.")
        print(f"Please ensure your environment is set up correctly.")
        print(f"Consider activating the venv: source {DAF_DIR}/venv/bin/activate")
        # Removed automatic setup script execution for simplicity
        sys.exit(1)
    
    print("Required packages found.")
    return True

def save_frames_as_video(frames, filename, fps=30):
    """Save frames as a video file."""
    output_path = ANIMATIONS_DIR / filename
    mediapy.write_video(str(output_path), frames, fps=fps)
    print(f"Saved animation to {output_path}")
    return output_path

def save_image(pixels, filename):
    """Save a single image to the images directory."""
    # <<< Ensure PIL is imported if needed here >>>
    try:
        from PIL import Image
        output_path = IMAGES_DIR / filename
        Image.fromarray(pixels).save(output_path)
        print(f"Saved image to {output_path}")
        return output_path
    except ImportError:
        print("PIL (Pillow) not found. Cannot save images.")
        return None

# Check environment and continue only if setup is complete
if not setup_environment():
    sys.exit(1)

# Import required modules - these imports are placed here to ensure
# they only run after the environment check
try:
    # <<< Remove PIL imports here if only used in save_image >>>
    # import PIL.Image
    # import PIL.ImageDraw
    
    from dm_control import mujoco
    # from dm_control import mjcf # Not directly used in the simplified run
    # from dm_control.mujoco.wrapper.mjbindings import enums # Not used
    # from dm_control.mujoco.wrapper.mjbindings import mjlib # Not used
    
    # <<< Change import from flybody to daf.ant >>>
    from daf.ant import ant_envs
    # from flybody.utils import any_substr_in_str # Not used
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure the environment is correctly set up.")
    sys.exit(1)

# Frame size and camera name-to-idx mapping for rendering.
frame_size = {'width': 640, 'height': 480}
# <<< Assuming cameras might be different or simpler for ant. Start with default/track1 >>>
# Check the ant XML or environment definition if specific cameras are needed.
# For now, using a generic name that might work.
cameras = {'track1': 0} # Simplified, might need adjustment based on ant model

def main():
    """Main demonstration script for the ant simulation."""
    print("\n" + "="*50)
    print("ANT SIMULATION DEMONSTRATION")
    print("="*50)
    
    # <<< Remove Fly-specific Parts 1-4 >>>
    # Part 1: Stand-alone fly model ... removed ...
    # Part 2: Visualizing the fly ... removed ...
    # Part 3: Kinematic manipulations ... removed ...
    # Part 4: Kinematic replay animation ... removed ...
    
    # Part 5: Adapted for ant-on-ball RL environment
    print("\n[...] Creating ant-on-ball RL environment")
    # <<< Use the ant_envs function >>>
    # Consider adding arguments like force_actuators=False if needed
    env = ant_envs.walk_on_ball() 
    
    # Visualization of the environment
    print("\n[...] Resetting environment and rendering initial state")
    timestep = env.reset()
    # <<< Use a potentially available camera, default to 0 if 'track1' fails >>>
    camera_key = 'track1' 
    camera_id_to_use = cameras.get(camera_key, 0) 
    try:
        pixels = env.physics.render(camera_id=camera_id_to_use, **frame_size)
        save_image(pixels, "ant_on_ball_initial.png")
    except Exception as e:
        print(f"Warning: Could not render initial state with camera '{camera_key}' (ID: {camera_id_to_use}). Error: {e}")
        print("Rendering might require specific camera names defined in the Ant XML.")

    # Run a short simulation with random actions
    print("\n[...] Running short simulation with random actions")
    n_actions = env.action_spec().shape[0]
    frames = []
    rewards = []
    n_steps = 50 # Reduced number of steps for quick test
    
    # Environment loop
    timestep = env.reset() # Reset again just before the loop
    for i in range(n_steps):
        # Random action policy
        # <<< Ensure action range is appropriate, check env.action_spec() if needed >>>
        # Assuming [-1, 1] is a reasonable default range for many dm_control envs
        random_action = np.random.uniform(-1.0, 1.0, n_actions) 
        try:
            timestep = env.step(random_action)
            rewards.append(timestep.reward)
            
            # Render frame
            pixels = env.physics.render(camera_id=camera_id_to_use, **frame_size)
            frames.append(pixels)
            
            # Save a few key frames
            if i % 10 == 0 or i == n_steps-1:
                save_image(pixels, f"ant_on_ball_frame_{i:03d}.png")

            # Print progress lightly
            if (i+1) % 10 == 0:
                print(f"Step {i+1}/{n_steps} completed.")

        except Exception as e:
            print(f"Error during simulation step {i}: {e}")
            break # Stop simulation if an error occurs

    if frames:
        print("\n[...] Saving simulation frames as video")
        save_frames_as_video(frames, "ant_on_ball_random.mp4", fps=15)
    else:
        print("\n[...] No frames recorded, skipping video saving.")
        
    if rewards:
        avg_reward = np.mean(rewards)
        print(f"\nAverage reward over {len(rewards)} steps: {avg_reward:.4f}")
    
    print("\n" + "="*50)
    print("ANT SIMULATION DEMONSTRATION COMPLETE")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main() 