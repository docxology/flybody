#!/usr/bin/env python3
"""
Deploy POMDP-based Active Inference agents on flybody tasks.

This script runs the POMDP implementation of active inference,
demonstrating the unification of perception and action in a biological framework.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List, Tuple

# Add parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import flybody components
from flybody.tasks import (
    template_task, walk_imitation, flight_imitation, walk_on_ball, vision_flight
)
from acme import specs

# Import DAF utilities
try:
    from daf_agent_utils import create_run_dirs, run_episode, MetricLogger
except ImportError:
    # Define simplified versions if the full utilities aren't available
    def create_run_dirs(base_dir: str) -> Dict[str, str]:
        """Create directories for storing run data."""
        directories = {
            "root": base_dir,
            "logs": os.path.join(base_dir, "logs"),
            "checkpoints": os.path.join(base_dir, "checkpoints"),
            "images": os.path.join(base_dir, "images"),
            "animations": os.path.join(base_dir, "animations"),
            "stats": os.path.join(base_dir, "stats")
        }
        for directory in directories.values():
            os.makedirs(directory, exist_ok=True)
        return directories
    
    class MetricLogger:
        """Simple metric logger."""
        def __init__(self, log_dir: str):
            self.log_dir = log_dir
            self.metrics = {}
            
        def log_metric(self, name: str, value: float, step: Optional[int] = None):
            """Log a metric value."""
            if name not in self.metrics:
                self.metrics[name] = []
            if step is not None:
                self.metrics[name].append((step, value))
            else:
                self.metrics[name].append(value)
        
        def save_metrics(self):
            """Save logged metrics to file."""
            for name, values in self.metrics.items():
                filename = os.path.join(self.log_dir, f"{name}.json")
                with open(filename, 'w') as f:
                    json.dump(values, f)

# Import Active Inference POMDP components
from daf.active_flyference.pomdp_agent import POMDPAgent, create_pomdp_agent_for_task
from daf.active_flyference.models.flybody_pomdp import FlybodyPOMDP

def create_environment(task_name: str, **kwargs) -> Any:
    """
    Create an environment for the specified task.
    
    Args:
        task_name: Name of the task to create
        **kwargs: Additional arguments for task creation
        
    Returns:
        Task environment
    """
    # Map task names to their environments
    task_creators = {
        'template': template_task,
        'walk_imitation': walk_imitation,
        'flight_imitation': flight_imitation,
        'walk_on_ball': walk_on_ball,
        'vision_flight': vision_flight
    }
    
    if task_name not in task_creators:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Create the task environment
    try:
        env = task_creators[task_name](**kwargs)
    except Exception as e:
        print(f"Error creating task {task_name}: {e}")
        # Fallback
        env = task_creators[task_name]()
    
    return env

def run_pomdp_episode(
    env: Any,
    agent: POMDPAgent,
    output_dirs: Dict[str, str],
    logger: MetricLogger,
    episode: int = 0,
    max_steps: int = 1000,
    render: bool = True,
    save_frames: bool = True,
    create_animation: bool = True,
    prefix: str = "pomdp_agent"
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Run a single episode with the POMDP agent.
    
    Args:
        env: Environment
        agent: POMDP agent
        output_dirs: Output directories for saving data
        logger: Metric logger
        episode: Episode number
        max_steps: Maximum steps per episode
        render: Whether to render frames
        save_frames: Whether to save individual frames
        create_animation: Whether to create animation from frames
        prefix: Prefix for saved files
        
    Returns:
        Tuple of (rewards, frames)
    """
    # Reset environment and agent
    timestep = env.reset()
    agent.observe_first(timestep)
    
    rewards = []
    frames = []
    
    # Run episode
    for step_idx in range(max_steps):
        # Select action using active inference
        action = agent.select_action()
        
        # Take action in environment
        timestep = env.step(action)
        agent.observe(action, timestep)
        
        # Record reward
        rewards.append(float(timestep.reward) if timestep.reward is not None else 0.0)
        
        # Log step metrics
        if logger:
            logger.log_metric(f"{prefix}_reward", rewards[-1], step=step_idx)
            logger.log_metric(f"{prefix}_episode_return", sum(rewards), step=step_idx)
            logger.log_metric(f"{prefix}_free_energy", agent.episode_free_energy, step=step_idx)
            
            # Log uncertainty if available
            if hasattr(agent, 'state_entropy_history') and len(agent.state_entropy_history) > 0:
                logger.log_metric(f"{prefix}_belief_entropy", agent.state_entropy_history[-1], step=step_idx)
        
        # Render if needed
        if render:
            frame = env.physics.render(camera_id=0, height=480, width=640)
            frames.append(frame)
            
            # Save frame if requested
            if save_frames and output_dirs:
                frame_path = os.path.join(output_dirs["images"], f"{prefix}_ep{episode}_step{step_idx:04d}.png")
                # Simple frame saving
                try:
                    from PIL import Image
                    Image.fromarray(frame).save(frame_path)
                except ImportError:
                    # Fallback to matplotlib
                    plt.imsave(frame_path, frame)
        
        # Check if episode has ended
        if timestep.last():
            break
    
    # Save agent history data
    if output_dirs:
        agent.save_history(episode=episode, prefix=prefix)
        
        # Create visualizations
        agent.create_visualizations(episode=episode, prefix=prefix)
    
    # Create animation if needed
    if create_animation and frames and output_dirs:
        animation_path = os.path.join(output_dirs["animations"], f"{prefix}_ep{episode}_animation.mp4")
        try:
            import mediapy as media
            media.write_video(animation_path, frames, fps=30)
        except ImportError:
            # Fallback to matplotlib animation
            try:
                from matplotlib import animation
                fig = plt.figure(figsize=(8, 6))
                plt.axis('off')
                
                def animate(i):
                    plt.imshow(frames[i])
                    return []
                
                anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
                anim.save(animation_path, writer=animation.FFMpegWriter(fps=30))
                plt.close()
            except Exception as e:
                print(f"Failed to create animation: {e}")
    
    # Log episode summary
    if logger:
        episode_return = sum(rewards)
        logger.log_metric(f"{prefix}_total_return", episode_return)
        logger.log_metric(f"{prefix}_episode_length", len(rewards))
        logger.log_metric(f"{prefix}_total_free_energy", agent.episode_free_energy)
        
        # Save metrics
        logger.save_metrics()
    
    return rewards, frames

def run_pomdp_training(
    env: Any,
    agent: POMDPAgent,
    output_dirs: Dict[str, str],
    logger: MetricLogger,
    num_episodes: int = 20,
    max_steps_per_episode: int = 1000,
    save_frames: bool = True,
    create_animation: bool = True,
    update_model: bool = True,
    eval_frequency: int = 5
) -> POMDPAgent:
    """
    Train a POMDP agent through experience.
    
    Args:
        env: Environment
        agent: POMDP agent
        output_dirs: Output directories for saving data
        logger: Metric logger
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        save_frames: Whether to save individual frames
        create_animation: Whether to create animation from frames
        update_model: Whether to update the model from experience
        eval_frequency: Frequency of evaluation episodes
        
    Returns:
        Trained agent
    """
    # Track best performance
    best_return = float('-inf')
    
    for episode in range(1, num_episodes + 1):
        print(f"Episode {episode}/{num_episodes}")
        
        # Run episode
        rewards, frames = run_pomdp_episode(
            env=env,
            agent=agent,
            output_dirs=output_dirs,
            logger=logger,
            episode=episode,
            max_steps=max_steps_per_episode,
            render=(episode % eval_frequency == 0),  # Only render eval episodes
            save_frames=(episode % eval_frequency == 0) and save_frames,
            create_animation=(episode % eval_frequency == 0) and create_animation,
            prefix=f"train_ep{episode}"
        )
        
        episode_return = sum(rewards)
        print(f"  Episode return: {episode_return:.2f}")
        print(f"  Episode free energy: {agent.episode_free_energy:.2f}")
        print(f"  Final belief entropy: {agent.state_entropy_history[-1]:.4f}" 
              if hasattr(agent, 'state_entropy_history') and len(agent.state_entropy_history) > 0 else "")
        
        # Update model from experience
        if update_model:
            agent.update_model_from_experience()
        
        # Run evaluation episode
        if episode % eval_frequency == 0:
            print(f"Evaluation after episode {episode}")
            eval_rewards, _ = run_pomdp_episode(
                env=env,
                agent=agent,
                output_dirs=output_dirs,
                logger=logger,
                episode=episode // eval_frequency,
                max_steps=max_steps_per_episode,
                render=True,
                save_frames=save_frames,
                create_animation=create_animation,
                prefix=f"eval"
            )
            
            eval_return = sum(eval_rewards)
            print(f"  Evaluation return: {eval_return:.2f}")
            
            # Save best model
            if eval_return > best_return:
                best_return = eval_return
                print(f"  New best performance: {best_return:.2f}")
    
    return agent

def main():
    """Main function to run POMDP-based Active Inference deployment."""
    parser = argparse.ArgumentParser(description="Deploy POMDP Active Inference on flybody tasks")
    parser.add_argument("--task", type=str, default="walk_on_ball",
                       choices=["template", "walk_imitation", "flight_imitation", "walk_on_ball", "vision_flight"],
                       help="Task to run")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "eval", "demo"],
                       help="Mode to run in")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000,
                       help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                       help="Render environment")
    parser.add_argument("--no-save-frames", action="store_true",
                       help="Disable saving individual frames")
    parser.add_argument("--no-animation", action="store_true",
                       help="Disable animation creation")
    parser.add_argument("--include-vision", action="store_true",
                       help="Include vision in observations")
    parser.add_argument("--output-dir", type=str, default="output/pomdp_agent",
                       help="Output directory")
    parser.add_argument("--num-states", type=int, default=128,
                       help="Number of discrete states in POMDP model")
    parser.add_argument("--num-actions", type=int, default=16,
                       help="Number of discrete actions in POMDP model")
    parser.add_argument("--planning-horizon", type=int, default=3,
                       help="Planning horizon for expected free energy")
    parser.add_argument("--exploration-factor", type=float, default=0.3,
                       help="Exploration factor (0-1, higher = more exploration)")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="Learning rate for model updates")
    parser.add_argument("--inference-iterations", type=int, default=10,
                       help="Number of iterations for belief updating")
    parser.add_argument("--no-update-model", action="store_true",
                       help="Disable model updating during training")
    parser.add_argument("--precision", type=float, default=2.0,
                       help="Precision parameter for action selection")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    # Create output directories
    output_dirs = create_run_dirs(args.output_dir)
    
    # Create logger
    logger = MetricLogger(output_dirs["logs"])
    
    # Save configuration
    config_path = os.path.join(output_dirs["root"], "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Create environment
    env = create_environment(args.task)
    
    # Create agent
    agent = create_pomdp_agent_for_task(
        task_name=args.task,
        include_vision=args.include_vision,
        num_states=args.num_states,
        num_actions=args.num_actions,
        precision=args.precision,
        learning_rate=args.learning_rate,
        inference_iterations=args.inference_iterations,
        planning_horizon=args.planning_horizon,
        exploration_factor=args.exploration_factor,
        log_dir=output_dirs["logs"],
        debug=args.debug
    )
    
    # Run in appropriate mode
    if args.mode == "train":
        agent = run_pomdp_training(
            env=env,
            agent=agent,
            output_dirs=output_dirs,
            logger=logger,
            num_episodes=args.episodes,
            max_steps_per_episode=args.steps,
            save_frames=not args.no_save_frames,
            create_animation=not args.no_animation,
            update_model=not args.no_update_model,
            eval_frequency=5
        )
    elif args.mode == "eval":
        # Run evaluation episodes
        total_returns = []
        for episode in range(args.episodes):
            print(f"Evaluation episode {episode+1}/{args.episodes}")
            rewards, _ = run_pomdp_episode(
                env=env,
                agent=agent,
                output_dirs=output_dirs,
                logger=logger,
                episode=episode,
                max_steps=args.steps,
                render=args.render,
                save_frames=not args.no_save_frames,
                create_animation=not args.no_animation,
                prefix=f"eval_ep{episode}"
            )
            
            episode_return = sum(rewards)
            total_returns.append(episode_return)
            print(f"  Episode return: {episode_return:.2f}")
        
        # Print summary statistics
        print("\nEvaluation summary:")
        print(f"  Mean return: {np.mean(total_returns):.2f} +/- {np.std(total_returns):.2f}")
        print(f"  Min return: {np.min(total_returns):.2f}")
        print(f"  Max return: {np.max(total_returns):.2f}")
    
    elif args.mode == "demo":
        # Run a single demonstration episode
        print("Running demonstration episode")
        rewards, _ = run_pomdp_episode(
            env=env,
            agent=agent,
            output_dirs=output_dirs,
            logger=logger,
            episode=0,
            max_steps=args.steps,
            render=True,
            save_frames=not args.no_save_frames,
            create_animation=not args.no_animation,
            prefix="demo"
        )
        
        episode_return = sum(rewards)
        print(f"  Episode return: {episode_return:.2f}")
    
    # Create final visualizations
    agent.create_visualizations(episode=0, prefix="final")
    
    print("\nPOMDP Agent deployment complete.")

if __name__ == "__main__":
    main() 