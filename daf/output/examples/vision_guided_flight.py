import os
import numpy as np
import mediapy
from pathlib import Path


# Example name
EXAMPLE_NAME = 'vision_guided_flight'

# Ensure output paths exist
output_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output')
images_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output/images')
animations_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output/animations')

try:
    # Import the flybody module
    from flybody.fly_envs import vision_guided_flight
    
    print(f'Creating {EXAMPLE_NAME} environment...')
    # Configure for vision guided flight with bumps
    env = vision_guided_flight(bumps_or_trench='bumps')
    
    # Get the correct action size
    action_spec = env.action_spec()
    action_size = action_spec.shape[0]
    print(f"Action space size: {action_size}")
    
    # Run simulation for a few steps with random actions
    print('Running simulation with random actions...')
    frames = []
    for i in range(20):
        # Generate random action with proper dimensions
        action = np.random.uniform(
            low=action_spec.minimum, 
            high=action_spec.maximum, 
            size=action_spec.shape
        )
        
        timestep = env.step(action)
        
        # Render and save current frame
        pixels = env.physics.render(camera_id=0)
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

