import os
import numpy as np
import mediapy
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
        
        # Get the correct action spec for this environment
        action_spec = env.action_spec()
        print(f"Action space for {env_name}: {action_spec.shape[0]} dimensions")
        
        for i in range(10):  # Just 10 frames for the combined visualization
            # Generate random action with proper dimensions for this environment
            action = np.random.uniform(
                low=action_spec.minimum, 
                high=action_spec.maximum, 
                size=action_spec.shape
            )
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
        
        # Save frame as image directly in RGB format
        frame_path = images_dir / f"combined_frame_{frame_idx:03d}.png"
        plt.savefig(frame_path, dpi=120, format='png', transparent=False)
        print(f"Saved combined frame {frame_idx} to {frame_path}")
        
        # Read the saved image and convert to RGB if needed
        saved_img = mediapy.read_image(frame_path)
        # Convert from RGBA to RGB if needed
        if saved_img.shape[2] == 4:
            saved_img = saved_img[:, :, :3]
        
        combined_frames.append(saved_img)
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
