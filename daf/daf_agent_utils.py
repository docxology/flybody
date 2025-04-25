"""Utility functions for deploying agents and managing outputs in the DAF framework."""

import os
import time
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import dm_env
from typing import Dict, List, Optional, Tuple, Any, Callable

# Create run-specific output directories
def create_run_dirs(base_name: str = "run") -> Dict[str, str]:
    """
    Create run-specific output directories for logs, images, and animations.
    
    Args:
        base_name: Base name for the run directory
        
    Returns:
        Dictionary with paths to created directories
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{base_name}_{timestamp}"
    
    # Create main run directory under daf/output
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    run_dir = os.path.join(base_dir, run_id)
    
    # Create subdirectories
    dirs = {
        "root": run_dir,
        "logs": os.path.join(run_dir, "logs"),
        "images": os.path.join(run_dir, "images"),
        "animations": os.path.join(run_dir, "animations"),
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "stats": os.path.join(run_dir, "stats")
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    # Create a metadata file
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        metadata = {
            "run_id": run_id,
            "timestamp": timestamp,
            "base_name": base_name,
            "created_at": datetime.datetime.now().isoformat()
        }
        json.dump(metadata, f, indent=2)
        
    return dirs

# Logger class for tracking metrics during training and evaluation
class MetricLogger:
    def __init__(self, log_dir: str, log_freq: int = 100):
        """
        Initialize a metric logger.
        
        Args:
            log_dir: Directory to save log files
            log_freq: Frequency of logging to file
        """
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.metrics = {}
        self.step_counter = 0
        
        # Create TensorBoard writer
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        
        # Create a log file
        self.log_file = os.path.join(log_dir, "metrics.jsonl")
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (uses internal counter if None)
        """
        if step is None:
            step = self.step_counter
            
        if name not in self.metrics:
            self.metrics[name] = []
            
        self.metrics[name].append((step, value))
        
        # Write to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)
        
        # Write to log file every log_freq steps
        if len(self.metrics[name]) % self.log_freq == 0:
            self._write_to_file(name, step, value)
    
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """Log a histogram of values to TensorBoard."""
        if step is None:
            step = self.step_counter
            
        with self.summary_writer.as_default():
            tf.summary.histogram(name, values, step=step)
    
    def increment_step(self):
        """Increment the internal step counter."""
        self.step_counter += 1
        return self.step_counter
    
    def _write_to_file(self, name: str, step: int, value: float):
        """Write a metric entry to the log file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                "metric": name,
                "step": step,
                "value": float(value),
                "timestamp": datetime.datetime.now().isoformat()
            }) + "\n")
    
    def save_metrics_summary(self, path: Optional[str] = None):
        """
        Save a summary of all metrics to a JSON file.
        
        Args:
            path: Path to save the summary (defaults to stats directory)
        """
        if path is None:
            path = os.path.join(os.path.dirname(self.log_dir), "stats", "metrics_summary.json")
        
        # Calculate statistics for each metric
        summary = {}
        for name, values in self.metrics.items():
            if values:
                steps, metric_values = zip(*values)
                metric_values = np.array(metric_values)
                summary[name] = {
                    "min": float(np.min(metric_values)),
                    "max": float(np.max(metric_values)),
                    "mean": float(np.mean(metric_values)),
                    "std": float(np.std(metric_values)),
                    "last": float(metric_values[-1]),
                    "steps": len(values)
                }
        
        # Save to file
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
            
        return summary

# Visualization utilities
def save_frame(env_state: np.ndarray, frame_idx: int, output_dir: str, prefix: str = "frame"):
    """
    Save a single frame from the environment.
    
    Args:
        env_state: RGB array from environment render
        frame_idx: Frame index
        output_dir: Directory to save the frame
        prefix: Filename prefix
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(env_state)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"{prefix}_{frame_idx:04d}.png"), 
                bbox_inches='tight', pad_inches=0)
    plt.close()

def create_animation(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """
    Create and save an animation from a list of frames.
    
    Args:
        frames: List of RGB arrays
        output_path: Path to save the animation
        fps: Frames per second
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    
    def update(i):
        ax.clear()
        ax.imshow(frames[i])
        ax.axis('off')
        return [ax]
    
    anim = FuncAnimation(fig, update, frames=len(frames), blit=True)
    anim.save(output_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    plt.close()

# Environment execution utilities
def run_episode(
    env: dm_env.Environment,
    actor: Any,
    max_steps: int = 1000,
    render: bool = True,
    logger: Optional[MetricLogger] = None,
    output_dirs: Optional[Dict[str, str]] = None,
    save_frames: bool = False,
    create_animation_file: bool = True,
    prefix: str = "episode"
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Run a single episode with the given environment and actor.
    
    Args:
        env: DM Environment
        actor: Actor that implements select_action method
        max_steps: Maximum number of steps to run
        render: Whether to render and collect frames
        logger: Optional metric logger
        output_dirs: Output directories dict
        save_frames: Whether to save individual frames
        create_animation_file: Whether to create animation file
        prefix: Prefix for saved files
        
    Returns:
        Tuple of (rewards, frames)
    """
    timestep = env.reset()
    actor.observe_first(timestep)
    
    rewards = []
    frames = []
    
    for step_idx in range(max_steps):
        # Select action
        action = actor.select_action(timestep.observation)
        
        # Take action in environment
        timestep = env.step(action)
        actor.observe(action, timestep)
        
        # Record reward
        rewards.append(float(timestep.reward) if timestep.reward is not None else 0.0)
        
        # Log step metrics
        if logger:
            logger.log_metric(f"{prefix}_reward", rewards[-1], step=step_idx)
            logger.log_metric(f"{prefix}_episode_return", sum(rewards), step=step_idx)
            if hasattr(timestep.observation, "keys") and "vision" in timestep.observation:
                logger.log_histogram(f"{prefix}_vision", timestep.observation["vision"], step=step_idx)
        
        # Render if needed
        if render:
            frame = env.physics.render(camera_id=0, height=480, width=640)
            frames.append(frame)
            
            if save_frames and output_dirs:
                save_frame(frame, step_idx, output_dirs["images"], prefix=prefix)
        
        # Check if episode has ended
        if timestep.last():
            break
    
    # Create animation if needed
    if create_animation_file and frames and output_dirs:
        animation_path = os.path.join(output_dirs["animations"], f"{prefix}_animation.mp4")
        create_animation(frames, animation_path)
    
    # Log episode summary
    if logger:
        episode_return = sum(rewards)
        logger.log_metric(f"{prefix}_total_return", episode_return)
        logger.log_metric(f"{prefix}_episode_length", len(rewards))
    
    return rewards, frames

def save_network_summary(network: tf.Module, output_dir: str, name: str = "network"):
    """
    Save a summary of a TensorFlow network.
    
    Args:
        network: TensorFlow network/module
        output_dir: Directory to save the summary
        name: Name for the summary file
    """
    summary = {
        "name": name,
        "variables": [],
        "total_parameters": 0
    }
    
    total_params = 0
    for var in network.variables:
        var_shape = var.shape.as_list()
        var_params = np.prod(var_shape)
        total_params += var_params
        
        summary["variables"].append({
            "name": var.name,
            "shape": var_shape,
            "parameters": int(var_params)
        })
    
    summary["total_parameters"] = int(total_params)
    
    with open(os.path.join(output_dir, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2) 