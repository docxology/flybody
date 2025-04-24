import os
import numpy as np
import mediapy
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Ensure output paths exist
output_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output')
images_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output/images')
animations_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output/animations')

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
