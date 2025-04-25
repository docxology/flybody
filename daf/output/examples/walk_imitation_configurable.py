import os
import numpy as np
import mediapy
from pathlib import Path

# Configuration parameters
NUM_SIMULATION_STEPS = 500  # Total number of simulation steps
SAVE_FRAME_INTERVAL = 5    # Save every Nth frame as image
FPS = 5                    # Frames per second for main animation
MULTIVIEW_FPS = 5           # Frames per second for multiview animation
MULTIVIEW_INTERVAL = 5      # Use every Nth frame for multiview animation
NUM_MULTIVIEW_FRAMES = 100   # Number of frames to use for multiview animation

# Example name
EXAMPLE_NAME = 'walk_imitation_config'

# Ensure output paths exist
output_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output')
images_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output/images')
animations_dir = Path('/home/trim/Documents/GitHub/flybody/daf/output/animations')

try:
    # Import the flybody module
    from flybody.fly_envs import walk_imitation
    
    print(f'Creating {EXAMPLE_NAME} environment...')
    env = walk_imitation()
    
    # Run simulation with random actions
    print(f'Running simulation with {NUM_SIMULATION_STEPS} steps...')
    frames = []
    for i in range(NUM_SIMULATION_STEPS):
        action = np.random.normal(size=59)
        timestep = env.step(action)
        
        # Render and save current frame
        pixels = env.physics.render(camera_id=1)
        frames.append(pixels)
        
        # Save intermediate frame as image
        if i % SAVE_FRAME_INTERVAL == 0:
            frame_path = images_dir / f'{EXAMPLE_NAME}_frame_{i:03d}.png'
            mediapy.write_image(frame_path, pixels)
            print(f'Saved frame {i} to {frame_path}')
    
    # Save animation
    print('Saving animation...')
    animation_path = animations_dir / f'{EXAMPLE_NAME}.mp4'
    mediapy.write_video(animation_path, frames, fps=FPS)
    print(f'Saved animation to {animation_path}')
    
    # Create multi-view animation
    try:
        print('Creating multi-view animation...')
        multi_frames = []
        
        # Calculate indices for multiview frames, limited by total number of frames
        max_frames = min(NUM_MULTIVIEW_FRAMES, len(frames))
        multiview_indices = list(range(0, NUM_SIMULATION_STEPS, MULTIVIEW_INTERVAL))[:max_frames]
        
        # Reset environment to initial state
        env.reset()
        
        # Re-run the simulation to capture multiview frames
        print(f'Re-running simulation for {len(multiview_indices)} multiview frames...')
        for index, step in enumerate(range(NUM_SIMULATION_STEPS)):
            # Use same random action sequence as before to ensure consistency
            np.random.seed(step)  # Set seed based on step number for reproducibility
            action = np.random.normal(size=59)
            timestep = env.step(action)
            
            # Only process frames at specified intervals
            if step in multiview_indices:
                print(f'Capturing multiview for step {step}...')
                
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
                    grid_path = images_dir / f'{EXAMPLE_NAME}_multiview_{step:03d}.png'
                    mediapy.write_image(grid_path, grid)
                    print(f'Saved multiview frame {step} to {grid_path}')
        
        if multi_frames:
            # Save multi-view animation - ensure we have enough frames
            if len(multi_frames) > 1:  # Need at least 2 frames for a video
                multiview_path = animations_dir / f'{EXAMPLE_NAME}_multiview.mp4'
                mediapy.write_video(multiview_path, multi_frames, fps=MULTIVIEW_FPS)
                print(f'Saved multi-view animation with {len(multi_frames)} frames to {multiview_path}')
            else:
                print(f'Not enough multiview frames ({len(multi_frames)}) to create video')
        else:
            print('No multiview frames were generated')
            
    except Exception as e:
        print(f'Error creating multi-view animation: {e}')
        import traceback
        traceback.print_exc()
    
    print('Example completed successfully!')
except Exception as e:
    print(f'Error running example: {e}')
    import traceback
    traceback.print_exc() 