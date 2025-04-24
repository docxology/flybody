#!/usr/bin/env python3
"""
Deploy an agent on the flight imitation task.
This script provides a simplified interface for training and evaluating agents
specifically on the flight imitation task.
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main deployment script
from deploy_agents import (
    create_environment, 
    create_agent, 
    train_agent, 
    run_evaluation,
    run_episode
)

# Import utilities
from daf_agent_utils import create_run_dirs, MetricLogger, save_network_summary

# Import specific task
from flybody.tasks.flight_imitation import FlightImitationTask
from acme import specs

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate agents on flight imitation task")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "deploy"],
                      help="Operation mode")
    parser.add_argument("--episodes", type=int, default=100,
                      help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=1000,
                      help="Maximum steps per episode")
    parser.add_argument("--checkpoint", type=str, default=None,
                      help="Path to checkpoint for evaluation")
    parser.add_argument("--output-name", type=str, default=None,
                      help="Custom name for the output directory")
    parser.add_argument("--save-frames", action="store_true",
                      help="Save individual frames")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--trajectory", type=str, default=None,
                      help="Optional path to specific flight trajectory file")
    parser.add_argument("--render-size", type=int, default=480,
                      help="Size of the rendered frames (height and width)")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create a unique output directory for this run
    run_name = args.output_name or f"flight_imitation_{args.mode}"
    output_dirs = create_run_dirs(run_name)
    
    # Save command line arguments
    with open(os.path.join(output_dirs['root'], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create logger
    logger = MetricLogger(output_dirs['logs'])
    
    # Create environment with task-specific parameters
    env_kwargs = {}
    if args.trajectory:
        env_kwargs['reference_trajectory'] = args.trajectory
    
    env = create_environment('flight_imitation', **env_kwargs)
    env_spec = specs.make_environment_spec(env)
    
    print(f"Created flight imitation environment with action space: {env_spec.actions.shape}")
    
    # Flight-specific agent configuration
    agent_config = {
        'hidden_layer_sizes': (256, 256, 128),  # Flight may benefit from deeper networks
        'learning_rate': 3e-4,  # Slightly higher learning rate for flight
        'discount': 0.99,
        'batch_size': 256,
        'target_update_period': 100,
    }
    
    # Create or load agent
    if args.mode == "eval" and args.checkpoint:
        agent = create_agent(env_spec, agent_config)
        print(f"Loading agent from checkpoint: {args.checkpoint}")
        agent.restore(args.checkpoint)
    else:
        agent = create_agent(env_spec, agent_config)
    
    # Save network architecture summary
    if hasattr(agent, '_policy_network'):
        save_network_summary(agent._policy_network, output_dirs['stats'], "policy_network")
    if hasattr(agent, '_critic_network'):
        save_network_summary(agent._critic_network, output_dirs['stats'], "critic_network")
    
    # Flight-specific task info
    task_info = {
        "task": "flight_imitation",
        "action_dimension": env_spec.actions.shape[0],
        "observation_dimension": env_spec.observations.shape[0] if hasattr(env_spec.observations, 'shape') else "dict",
        "has_vision": False,
    }
    with open(os.path.join(output_dirs['stats'], "task_info.json"), "w") as f:
        json.dump(task_info, f, indent=2)
    
    # Run based on the specified mode
    if args.mode == "train":
        # Flight-specific training configuration
        train_config = {
            'num_episodes': args.episodes,
            'eval_every': 5,  # More frequent evaluation for flight
            'max_steps_per_episode': args.max_steps,
            'save_frames': args.save_frames,
            'save_checkpoint_every': 10,
        }
        
        print(f"Training agent on flight imitation task for {args.episodes} episodes...")
        agent = train_agent(env, agent, output_dirs, logger, train_config)
        
        # Run final evaluation
        eval_config = {
            'num_episodes': 5,
            'max_steps_per_episode': args.max_steps,
            'save_frames': True,  # Always save frames for final evaluation
        }
        run_evaluation(env, agent, output_dirs, logger, eval_config)
    
    elif args.mode == "eval":
        eval_config = {
            'num_episodes': args.episodes,
            'max_steps_per_episode': args.max_steps,
            'save_frames': args.save_frames,
        }
        run_evaluation(env, agent, output_dirs, logger, eval_config)
    
    elif args.mode == "deploy":
        # Just run a single episode with rendering and save frames
        print(f"Deploying agent on flight imitation task...")
        
        # Always save frames and create animation for deployment
        rewards, frames = run_episode(
            env=env,
            actor=agent,
            max_steps=args.max_steps,
            render=True,
            logger=logger,
            output_dirs=output_dirs,
            save_frames=True,
            create_animation_file=True,
            prefix="deploy_flight"
        )
        
        print(f"Deployment complete. Return: {sum(rewards):.2f}")
    
    # Save metrics summary
    logger.save_metrics_summary()
    
    print(f"All outputs saved to: {output_dirs['root']}")

if __name__ == "__main__":
    main() 