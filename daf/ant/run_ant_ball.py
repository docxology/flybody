#!/usr/bin/env python3
"""Run the ant walk on ball task."""

import os
import datetime
import numpy as np
from dm_control import viewer
from dm_control.suite import common

from daf.ant.task_envs import ant_walk_on_ball


def main():
    """Run the ant walk on ball task."""
    # Create random state for reproducibility
    random_state = np.random.RandomState(42)
    
    # Create the environment
    env = ant_walk_on_ball(
        force_actuators=True,  # Use force actuators for more natural movement
        random_state=random_state,
    )
    
    # Define output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', f'ant_walk_on_ball_{timestamp}')
    os.makedirs(os.path.join(output_dir, 'animations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Set up viewer options
    width = 640
    height = 480
    
    # Create a callback for saving frames
    frames = []
    
    def policy_callback(time_step):
        """Policy function that captures frames and returns random actions."""
        # Capture frame
        if len(frames) < 300:  # Limit to 300 frames (10 seconds at 30 fps)
            frame = env.physics.render(height=height, width=width, camera_id=0)
            frames.append(frame)
        
        # Generate random action
        action_spec = env.action_spec()
        return np.random.uniform(
            action_spec.minimum, 
            action_spec.maximum, 
            size=action_spec.shape
        )
    
    # Launch the viewer
    viewer.launch(env, policy=policy_callback)
    
    # Save video after viewer is closed
    if frames:
        print(f"Saving animation with {len(frames)} frames")
        video_path = os.path.join(output_dir, 'animations', 'ant_walk_on_ball.mp4')
        
        try:
            # Use dm_control's video saving utility
            fps = 30
            common.save_video(frames, video_path, fps=fps)
            print(f"Animation saved to {video_path}")
            
            # Save the last frame as an image
            image_path = os.path.join(output_dir, 'images', 'ant_walk_on_ball_last_frame.png')
            common.save_image(frames[-1], image_path)
            print(f"Last frame saved to {image_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")


if __name__ == "__main__":
    main() 