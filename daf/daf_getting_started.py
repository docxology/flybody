#!/usr/bin/env python
# coding: utf-8

"""
# Getting started with `flybody` from DAF

This is a standalone Python script version of the flybody getting-started notebook,
adapted to run within the DeepMind Active Inference (DAF) framework.

`flybody` is an anatomically-detailed body model of the fruit fly _Drosophila melanogaster_
for MuJoCo physics simulator and reinforcement learning (RL) applications.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
import mediapy

# Setup paths
DAF_DIR = Path(__file__).parent.absolute()
REPO_DIR = DAF_DIR.parent
OUTPUT_DIR = DAF_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
ANIMATIONS_DIR = OUTPUT_DIR / "animations"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create output directories if they don't exist
for directory in [OUTPUT_DIR, IMAGES_DIR, ANIMATIONS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

def setup_environment():
    """Check for the flybody installation and suggest running setup_and_test.sh if needed."""
    # Check if flybody is installed
    if importlib.util.find_spec("flybody") is None:
        print("\n========== FLYBODY NOT INSTALLED ==========")
        print("The flybody package is not installed in your current Python environment.")
        print(f"Would you like to run the setup script at {DAF_DIR}/setup_and_test.sh?")
        response = input("Run setup script? (y/n): ").strip().lower()
        
        if response == 'y':
            # Run the setup script
            setup_script = DAF_DIR / "setup_and_test.sh"
            if setup_script.exists():
                print("\nRunning setup script. This may take a while...")
                try:
                    # Execute setup script
                    result = subprocess.run(
                        ["bash", str(setup_script)],
                        check=True,
                        stderr=subprocess.STDOUT
                    )
                    print("\nSetup completed successfully.")
                    print("You may need to restart this script to use the installed environment.")
                    sys.exit(0)
                except subprocess.CalledProcessError as e:
                    print(f"\nSetup failed with error code {e.returncode}.")
                    print("Please check the logs for details.")
                    sys.exit(1)
            else:
                print(f"\nSetup script not found at {setup_script}")
                sys.exit(1)
        else:
            print("\nSkipping setup. Please install flybody manually before running this script.")
            print("You can install it by running:")
            print(f"  bash {DAF_DIR}/setup_and_test.sh")
            sys.exit(1)
    
    print("flybody package is installed.")
    return True

def save_frames_as_video(frames, filename, fps=30):
    """Save frames as a video file."""
    output_path = ANIMATIONS_DIR / filename
    mediapy.write_video(str(output_path), frames, fps=fps)
    print(f"Saved animation to {output_path}")
    return output_path

def save_image(pixels, filename):
    """Save a single image to the images directory."""
    from PIL import Image
    output_path = IMAGES_DIR / filename
    Image.fromarray(pixels).save(output_path)
    print(f"Saved image to {output_path}")
    return output_path

# Check environment and continue only if setup is complete
if not setup_environment():
    sys.exit(1)

# Import flybody modules - these imports are placed here to ensure
# they only run after the environment check
try:
    import PIL.Image
    import PIL.ImageDraw
    
    from dm_control import mujoco
    from dm_control import mjcf
    from dm_control.mujoco.wrapper.mjbindings import enums
    from dm_control.mujoco.wrapper.mjbindings import mjlib
    
    import flybody
    from flybody.fly_envs import walk_on_ball
    from flybody.utils import any_substr_in_str
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please make sure the flybody environment is correctly set up.")
    sys.exit(1)

# Frame size and camera name-to-idx mapping for rendering.
frame_size = {'width': 640, 'height': 480}
cameras = {'track1': 0, 'track2': 1, 'track3': 2,
           'back': 3, 'side': 4, 'bottom': 5, 'hero': 6,
           'eye_right': 7, 'eye_left': 8}

def main():
    """Main demonstration script for flybody."""
    print("\n" + "="*50)
    print("FLYBODY DEMONSTRATION")
    print("="*50)
    
    # Part 1: Stand-alone fly model
    print("\n[1/5] Loading the MuJoCo fly model")
    flybody_path = os.path.dirname(flybody.__file__)
    xml_path = os.path.join(flybody_path, 'fruitfly/assets/fruitfly.xml')
    
    physics = mjcf.Physics.from_xml_path(xml_path)  # Load and compile.
    
    print('# of bodies:', physics.model.nbody)
    print('# of degrees of freedom:', physics.model.nv)
    print('# of joints:', physics.model.njnt)
    print('# of actuators:', physics.model.nu)
    print("fly's mass (gr):", physics.model.body_subtreemass[1])
    
    # Save qpos info to a text file
    with open(LOGS_DIR / "qpos_info.txt", "w") as f:
        f.write(str(physics.data.qpos))
    
    # Render and save images from different camera views
    print("\n[2/5] Visualizing the fly from different angles")
    pixels = physics.render(camera_id=cameras['hero'], **frame_size)
    save_image(pixels, "fly_hero_view.png")
    
    pixels = physics.render(camera_id=cameras['bottom'], **frame_size)
    save_image(pixels, "fly_bottom_view.png")
    
    # Show collision geoms
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.geomgroup[1] = 0  # Hide external meshes.
    scene_option.geomgroup[3] = 1  # Make fluid-interactions wing ellipsoids visible.
    scene_option.geomgroup[4] = 1  # Make collision geoms visible.
    pixels = physics.render(camera_id=cameras['side'], **frame_size, scene_option=scene_option)
    save_image(pixels, "fly_collision_geoms.png")
    
    # Load with floor and visualize
    xml_path_floor = os.path.join(flybody_path, 'fruitfly/assets/floor.xml')
    physics = mjcf.Physics.from_xml_path(xml_path_floor)
    pixels = physics.render(camera_id=cameras['side'], **frame_size)
    save_image(pixels, "fly_with_floor.png")
    
    # Kinematic manipulations
    print("\n[3/5] Performing kinematic manipulations")
    
    # Rotate fly around z-axis
    angle = np.pi / 2
    quat = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    with physics.reset_context():
        physics.named.data.qpos[3:7] = quat
    pixels = physics.render(camera_id=cameras['side'], **frame_size)
    save_image(pixels, "fly_rotated.png")
    
    # Fold wings
    physics.reset()  # Reset to default state
    wing_joints = []
    folded_wing_angles = []
    for i in range(physics.model.njnt):
        joint_name = physics.model.id2name(i, 'joint')
        if 'wing' in joint_name:
            wing_joints.append(joint_name)
            folded_wing_angles.append(
                physics.named.model.qpos_spring[joint_name].item())
    
    with physics.reset_context():
        physics.named.data.qpos[wing_joints] = folded_wing_angles
    pixels = physics.render(camera_id=cameras['side'], **frame_size)
    save_image(pixels, "fly_folded_wings.png")
    
    # Retract legs for flight
    with physics.reset_context():
        for i in range(physics.model.njnt):
            name = physics.model.id2name(i, 'joint')
            if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus'], name):
                physics.named.data.qpos[name] = physics.named.model.qpos_spring[name]
        physics.data.qpos[2] = 1.  # Lift fly by 1cm
    pixels = physics.render(camera_id=cameras['side'], **frame_size)
    save_image(pixels, "fly_retracted_legs.png")
    
    # Wing folding kinematic replay
    print("\n[4/5] Creating kinematic replay animation")
    n_steps = 50  # Reduced from 150 for faster execution
    frames = []
    for i in range(n_steps):
        with physics.reset_context():
            wing_angles = np.array(folded_wing_angles) * np.sin(np.pi/2 * i/n_steps)
            physics.named.data.qpos[wing_joints] = wing_angles
        pixels = physics.render(camera_id=cameras['back'], **frame_size)
        frames.append(pixels)
        
        # Save a few key frames
        if i % 10 == 0 or i == n_steps-1:
            save_image(pixels, f"wing_fold_frame_{i:03d}.png")
    
    save_frames_as_video(frames, "wing_folding.mp4", fps=15)
    
    # Example: fly-on-ball RL environment
    print("\n[5/5] Creating fly-on-ball RL environment")
    env = walk_on_ball()
    
    # Visualization of the environment
    timestep = env.reset()
    pixels = env.physics.render(camera_id=cameras['track1'], **frame_size)
    save_image(pixels, "fly_on_ball_initial.png")
    
    # Run a short simulation with random actions
    n_actions = env.action_spec().shape[0]
    frames = []
    rewards = []
    
    # Environment loop
    timestep = env.reset()
    for i in range(50):  # Reduced from 200 for faster execution
        # Random action policy
        random_action = np.random.uniform(-.5, .5, n_actions)
        timestep = env.step(random_action)
        rewards.append(timestep.reward)
        
        pixels = env.physics.render(camera_id=cameras['track1'], **frame_size)
        frames.append(pixels)
        
        # Save a few key frames
        if i % 10 == 0 or i == 49:
            save_image(pixels, f"fly_on_ball_frame_{i:03d}.png")
    
    save_frames_as_video(frames, "fly_on_ball_random.mp4", fps=15)
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('timestep')
    plt.ylabel('reward')
    plt.title('Rewards from Random Actions')
    plt.savefig(str(IMAGES_DIR / "rewards_plot.png"))
    plt.close()
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE")
    print("="*50)
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Animations saved to: {ANIMATIONS_DIR}")
    print(f"Logs saved to: {LOGS_DIR}")

if __name__ == "__main__":
    main() 