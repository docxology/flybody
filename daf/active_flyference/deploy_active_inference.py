#!/usr/bin/env python3
"""
Deploy Active Inference agents on flybody tasks.

This script allows running the Active Inference framework
on various flybody tasks, with detailed visualization and analysis.
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
from flybody.tasks import template_task, walk_imitation, flight_imitation, walk_on_ball, vision_flight
from acme import specs

# Import DAF utilities
from daf_agent_utils import create_run_dirs, run_episode, MetricLogger

# Import Active Inference components with absolute imports
from daf.active_flyference.active_inference_agent import ActiveInferenceAgent, create_agent_for_task
from daf.active_flyference.models.generative_model import GenerativeModel
from daf.active_flyference.models.walk_model import WalkOnBallModel
from daf.active_flyference.models.flight_model import FlightModel
from daf.active_flyference.utils.visualization import plot_active_inference_summary

def create_environment(task_name: str, **kwargs) -> Any:
    """
    Create an environment for the specified task.
    
    Args:
        task_name: Name of the task to create
        **kwargs: Additional arguments for task creation
        
    Returns:
        Task environment
    """
    # Import fly environment factory functions
    from flybody import fly_envs
    
    # Map task names to their factory functions
    task_creators = {
        'template': fly_envs.template_task,
        'walk_imitation': fly_envs.walk_imitation,
        'flight_imitation': fly_envs.flight_imitation,
        'walk_on_ball': fly_envs.walk_on_ball,
        'vision_flight': fly_envs.vision_guided_flight
    }
    
    if task_name not in task_creators:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Create the task environment using the factory function
    try:
        env = task_creators[task_name](**kwargs)
    except Exception as e:
        print(f"Error creating task {task_name}: {e}")
        # Fallback to simple creation without args
        env = task_creators[task_name]()
    
    return env

def run_active_inference_episode(
    env: Any,
    agent: ActiveInferenceAgent,
    output_dirs: Dict[str, str],
    logger: MetricLogger,
    episode: int = 0,
    max_steps: int = 1000,
    render: bool = True,
    save_frames: bool = True,
    create_animation: bool = True,
    prefix: str = "active_inference"
) -> Tuple[List[float], List[np.ndarray]]:
    """
    Run a single episode with the active inference agent.
    
    Args:
        env: Environment
        agent: Active Inference agent
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
    # Reset environment
    timestep = env.reset()
    agent.observe_first(timestep)
    
    rewards = []
    frames = []
    
    for step_idx in range(max_steps):
        # Select action using active inference
        action = agent.select_action(timestep.observation)
        
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
        
        # Render if needed
        if render:
            frame = env.physics.render(camera_id=0, height=480, width=640)
            frames.append(frame)
            
            if save_frames and output_dirs:
                from daf_agent_utils import save_frame
                save_frame(frame, step_idx, output_dirs["images"], prefix=f"{prefix}_ep{episode}")
        
        # Check if episode has ended
        if timestep.last():
            break
    
    # Save agent history data
    if output_dirs:
        agent.save_history(episode=episode, prefix=prefix)
        
        # Create agent visualizations
        agent.create_visualizations(episode=episode, prefix=prefix)
    
    # Create animation if needed
    if create_animation and frames and output_dirs:
        from daf_agent_utils import create_animation
        animation_path = os.path.join(output_dirs["animations"], f"{prefix}_ep{episode}_animation.mp4")
        create_animation(frames, animation_path)
    
    # Log episode summary
    if logger:
        episode_return = sum(rewards)
        logger.log_metric(f"{prefix}_total_return", episode_return)
        logger.log_metric(f"{prefix}_episode_length", len(rewards))
        logger.log_metric(f"{prefix}_total_free_energy", agent.episode_free_energy)
    
    return rewards, frames

def run_active_inference_training(
    env: Any,
    agent: ActiveInferenceAgent,
    output_dirs: Dict[str, str],
    logger: MetricLogger,
    num_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    save_frames: bool = True,
    create_animation: bool = True,
    update_model: bool = True,
    eval_frequency: int = 5
) -> ActiveInferenceAgent:
    """
    Train an active inference agent through experience.
    
    Args:
        env: Environment
        agent: Active Inference agent
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
        rewards, frames = run_active_inference_episode(
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
        
        # Update model from experience
        if update_model:
            agent.update_model_from_experience()
        
        # Run evaluation episode
        if episode % eval_frequency == 0:
            print(f"Evaluation after episode {episode}")
            eval_rewards, _ = run_active_inference_episode(
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
                
                # In a real implementation, would save model parameters here
                print(f"  New best performance: {best_return:.2f}")
    
    return agent

def main():
    """Main function to run active inference deployment."""
    parser = argparse.ArgumentParser(description="Deploy Active Inference on flybody tasks")
    parser.add_argument("--task", type=str, default="walk_on_ball",
                      choices=["template", "walk_imitation", "flight_imitation", "walk_on_ball", "vision_flight"],
                      help="Task to run")
    parser.add_argument("--mode", type=str, default="deploy",
                      choices=["deploy", "train", "evaluate"],
                      help="Mode to run in")
    parser.add_argument("--episodes", type=int, default=10,
                      help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000,
                      help="Maximum steps per episode")
    parser.add_argument("--output-name", type=str, default=None,
                      help="Custom name for output directory")
    parser.add_argument("--precision", type=float, default=2.0,
                      help="Precision parameter for action selection")
    parser.add_argument("--inference-iterations", type=int, default=10,
                      help="Number of inference iterations for belief updating")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                      help="Learning rate for model updates")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--no-visualization", action="store_true",
                      help="Disable visualization")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directories
    run_name = args.output_name or f"active_inference_{args.task}_{args.mode}"
    output_dirs = create_run_dirs(run_name)
    
    # Save command line arguments
    with open(os.path.join(output_dirs['root'], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create logger
    logger = MetricLogger(output_dirs['logs'])
    
    # Create environment
    env = create_environment(args.task)
    
    # Create active inference agent
    agent = create_agent_for_task(
        task_name=args.task,
        include_vision=(args.task == "vision_flight"),
        learning_rate=args.learning_rate,
        precision=args.precision,
        inference_iterations=args.inference_iterations,
        log_dir=output_dirs['root']
    )
    
    # Run based on mode
    if args.mode == "deploy":
        print(f"Deploying active inference agent on {args.task}")
        rewards, frames = run_active_inference_episode(
            env=env,
            agent=agent,
            output_dirs=output_dirs,
            logger=logger,
            max_steps=args.max_steps,
            render=not args.no_visualization,
            save_frames=not args.no_visualization,
            create_animation=not args.no_visualization,
            prefix="deploy"
        )
        
        print(f"Deployment complete. Return: {sum(rewards):.2f}")
        
    elif args.mode == "train":
        print(f"Training active inference agent on {args.task} for {args.episodes} episodes")
        agent = run_active_inference_training(
            env=env,
            agent=agent,
            output_dirs=output_dirs,
            logger=logger,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            save_frames=not args.no_visualization,
            create_animation=not args.no_visualization,
            update_model=True
        )
        
        print("Training complete")
        
    elif args.mode == "evaluate":
        print(f"Evaluating active inference agent on {args.task} for {args.episodes} episodes")
        
        all_returns = []
        for episode in range(1, args.episodes + 1):
            print(f"Evaluation episode {episode}/{args.episodes}")
            rewards, _ = run_active_inference_episode(
                env=env,
                agent=agent,
                output_dirs=output_dirs,
                logger=logger,
                episode=episode,
                max_steps=args.max_steps,
                render=not args.no_visualization,
                save_frames=not args.no_visualization,
                create_animation=not args.no_visualization,
                prefix=f"eval"
            )
            
            episode_return = sum(rewards)
            all_returns.append(episode_return)
            print(f"  Episode return: {episode_return:.2f}")
        
        # Print summary statistics
        mean_return = np.mean(all_returns)
        std_return = np.std(all_returns)
        print(f"Evaluation complete. Mean return: {mean_return:.2f} ± {std_return:.2f}")
        
        # Save summary
        with open(os.path.join(output_dirs['stats'], "evaluation_summary.json"), "w") as f:
            json.dump({
                "returns": all_returns,
                "mean_return": float(mean_return),
                "std_return": float(std_return),
                "min_return": float(np.min(all_returns)),
                "max_return": float(np.max(all_returns))
            }, f, indent=2)
    
    # Save metrics summary
    logger.save_metrics_summary()
    
    print(f"All outputs saved to: {output_dirs['root']}")

if __name__ == "__main__":
    main() 