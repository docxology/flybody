#!/usr/bin/env python3
"""
Main script for deploying agents on various tasks with the Flybody framework.
Handles environment setup, agent configuration, training, evaluation, and output management.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import tensorflow as tf
import sonnet as snt
from acme.tf import networks as acme_networks
from acme import specs
import dm_env

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import flybody modules
from flybody.tasks import template_task, walk_imitation, flight_imitation, walk_on_ball #, vision_flight <-- Still commented out
from flybody.tasks import template_task, walk_imitation, flight_imitation, walk_on_ball, vision_flight
from flybody.agents import agent_dmpo, actors, network_factory

# Import local utilities
from daf_agent_utils import (
    create_run_dirs, 
    MetricLogger, 
    run_episode, 
    save_network_summary
)

# Dictionary mapping task names to their environment creation functions
TASKS = {
    'template': template_task.TemplateTask,
    'walk_imitation': walk_imitation.WalkImitation,
    'flight_imitation': flight_imitation.FlightImitationWBPG,
    'walk_on_ball': walk_on_ball.WalkOnBall,
    # 'vision_flight': vision_flight.Vision, # Temporarily commented out
}

def create_environment(task_name, **kwargs):
    """Create an environment for the specified task."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASKS.keys())}")
    
    # Default components for task initialization (can be overridden by kwargs)
    default_walker = kwargs.pop('walker', fruitfly.FruitFly())
    default_arena = kwargs.pop('arena', Floor())
    default_time_limit = kwargs.pop('time_limit', 10.0) # Default 10 seconds
    default_joint_filter = kwargs.pop('joint_filter', ButterworthFilter())
    
    task_class = TASKS[task_name]
    
    # Check if it's a vision task to avoid passing walker/arena if not needed?
    # (Need to inspect base classes or task signatures, difficult without file reading)
    # For now, assume most tasks take these standard args. Handle exceptions if they occur.
    try:
        environment = task_class(
            walker=default_walker, 
            arena=default_arena, 
            time_limit=default_time_limit,
            joint_filter=default_joint_filter,
            **kwargs
        )
    except TypeError as e:
        print(f"Warning: TypeError initializing {task_name}: {e}")
        print(f"Attempting initialization without standard args...")
        # Fallback for tasks that might not take these standard args
        try:
            environment = task_class(**kwargs)
        except Exception as final_e:
            print(f"Error: Failed to initialize task {task_name} even with fallback.")
            raise final_e
            
    return environment

def create_agent(env_spec, config=None):
    """Create a DMPO agent for the given environment spec."""
    # Default configuration if none is provided
    if config is None:
        config = {
            'hidden_layer_sizes': (256, 256),
            'learning_rate': 1e-4,
            'discount': 0.99,
            'batch_size': 256,
            'target_update_period': 100,
        }
    
    # Create policy and critic networks
    policy_network = network_factory.make_policy_network(
        env_spec.actions.shape[0], 
        config['hidden_layer_sizes'],
        min_scale=1e-3
    )
    
    # For the critic, we use an MLP that takes state and action as input
    critic_network = network_factory.make_critic_network(
        config['hidden_layer_sizes']
    )
    
    # Identity observation network (pass-through)
    observation_network = tf.identity
    
    # Create the agent
    agent = agent_dmpo.DMPO(
        environment_spec=env_spec,
        policy_network=policy_network,
        critic_network=critic_network,
        observation_network=observation_network,
        discount=config['discount'],
        batch_size=config['batch_size'],
        target_policy_update_period=config['target_update_period'],
        target_critic_update_period=config['target_update_period'],
        policy_optimizer=snt.optimizers.Adam(config['learning_rate']),
        critic_optimizer=snt.optimizers.Adam(config['learning_rate']),
    )
    
    return agent

def train_agent(env, agent, output_dirs, logger, config=None):
    """Train an agent on the given environment."""
    if config is None:
        config = {
            'num_episodes': 100,
            'eval_every': 10,
            'max_steps_per_episode': 1000,
            'save_frames': True,
            'save_checkpoint_every': 10,
        }
    
    # Track best performance for model saving
    best_return = float('-inf')
    
    for episode in range(1, config['num_episodes'] + 1):
        # Training episode
        print(f"Training episode {episode}/{config['num_episodes']}")
        train_rewards, _ = run_episode(
            env=env,
            actor=agent,
            max_steps=config['max_steps_per_episode'],
            render=False,  # Don't render during training
            logger=logger,
            output_dirs=output_dirs,
            save_frames=False,
            create_animation_file=False,
            prefix=f"train_ep{episode}"
        )
        
        episode_return = sum(train_rewards)
        print(f"  Episode return: {episode_return:.2f}")
        
        # Save checkpoint
        if episode % config['save_checkpoint_every'] == 0:
            checkpoint_path = os.path.join(output_dirs['checkpoints'], f"checkpoint_ep{episode}")
            agent.save(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")
        
        # Evaluation
        if episode % config['eval_every'] == 0:
            print(f"Evaluating agent after episode {episode}...")
            eval_rewards, frames = run_episode(
                env=env,
                actor=agent,
                max_steps=config['max_steps_per_episode'],
                render=True,
                logger=logger,
                output_dirs=output_dirs,
                save_frames=config['save_frames'],
                create_animation_file=True,
                prefix=f"eval_ep{episode}"
            )
            
            eval_return = sum(eval_rewards)
            print(f"  Evaluation return: {eval_return:.2f}")
            
            # Save best model
            if eval_return > best_return:
                best_return = eval_return
                best_path = os.path.join(output_dirs['checkpoints'], "best_agent")
                agent.save(best_path)
                print(f"  New best model with return {best_return:.2f}")
    
    # Save final model
    final_path = os.path.join(output_dirs['checkpoints'], "final_agent")
    agent.save(final_path)
    print(f"Training completed. Final model saved to {final_path}")
    
    # Save training summary
    summary = {
        'num_episodes': config['num_episodes'],
        'max_steps_per_episode': config['max_steps_per_episode'],
        'best_return': float(best_return),
        'final_return': float(eval_return),
    }
    with open(os.path.join(output_dirs['stats'], "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return agent

def run_evaluation(env, agent, output_dirs, logger, config=None):
    """Run evaluation of a trained agent."""
    if config is None:
        config = {
            'num_episodes': 10,
            'max_steps_per_episode': 1000,
            'save_frames': True,
        }
    
    all_returns = []
    all_steps = []
    
    for episode in range(1, config['num_episodes'] + 1):
        print(f"Evaluation episode {episode}/{config['num_episodes']}")
        rewards, frames = run_episode(
            env=env,
            actor=agent,
            max_steps=config['max_steps_per_episode'],
            render=True,
            logger=logger,
            output_dirs=output_dirs,
            save_frames=config['save_frames'],
            create_animation_file=True,
            prefix=f"eval_ep{episode}"
        )
        
        episode_return = sum(rewards)
        episode_steps = len(rewards)
        all_returns.append(float(episode_return))
        all_steps.append(episode_steps)
        
        print(f"  Episode return: {episode_return:.2f}, steps: {episode_steps}")
    
    # Save evaluation summary
    summary = {
        'num_episodes': config['num_episodes'],
        'returns': all_returns,
        'steps': all_steps,
        'mean_return': float(np.mean(all_returns)),
        'std_return': float(np.std(all_returns)),
        'max_return': float(np.max(all_returns)),
        'min_return': float(np.min(all_returns)),
        'mean_steps': float(np.mean(all_steps)),
    }
    
    with open(os.path.join(output_dirs['stats'], "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Evaluation completed. Mean return: {summary['mean_return']:.2f} ± {summary['std_return']:.2f}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Deploy and train agents on Flybody tasks")
    parser.add_argument("--task", type=str, default="template", choices=TASKS.keys(),
                        help="Task to run")
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
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create a unique output directory for this run
    run_name = args.output_name or f"{args.task}_{args.mode}"
    output_dirs = create_run_dirs(run_name)
    
    # Save command line arguments
    with open(os.path.join(output_dirs['root'], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create logger
    logger = MetricLogger(output_dirs['logs'])
    
    # Create environment
    env = create_environment(args.task)
    env_spec = specs.make_environment_spec(env)
    
    # Log environment information
    with open(os.path.join(output_dirs['stats'], "environment_spec.json"), "w") as f:
        spec_info = {
            "task": args.task,
            "observation_shape": str(env_spec.observations.shape),
            "observation_dtype": str(env_spec.observations.dtype),
            "action_shape": str(env_spec.actions.shape),
            "action_dtype": str(env_spec.actions.dtype),
            "reward_shape": str(env_spec.rewards.shape),
            "reward_dtype": str(env_spec.rewards.dtype),
        }
        json.dump(spec_info, f, indent=2)
    
    # Create or load agent
    if args.mode == "eval" and args.checkpoint:
        # Load agent from checkpoint for evaluation
        agent_config = {
            'hidden_layer_sizes': (256, 256),
            'learning_rate': 1e-4,
            'discount': 0.99,
            'batch_size': 256,
            'target_update_period': 100,
        }
        agent = create_agent(env_spec, agent_config)
        print(f"Loading agent from checkpoint: {args.checkpoint}")
        agent.restore(args.checkpoint)
    else:
        # Create a new agent
        agent_config = {
            'hidden_layer_sizes': (256, 256),
            'learning_rate': 1e-4,
            'discount': 0.99,
            'batch_size': 256,
            'target_update_period': 100,
        }
        agent = create_agent(env_spec, agent_config)
    
    # Save network architecture summary
    if hasattr(agent, '_policy_network'):
        save_network_summary(agent._policy_network, output_dirs['stats'], "policy_network")
    if hasattr(agent, '_critic_network'):
        save_network_summary(agent._critic_network, output_dirs['stats'], "critic_network")
    
    # Run based on the specified mode
    if args.mode == "train":
        train_config = {
            'num_episodes': args.episodes,
            'eval_every': 10,
            'max_steps_per_episode': args.max_steps,
            'save_frames': args.save_frames,
            'save_checkpoint_every': 10,
        }
        agent = train_agent(env, agent, output_dirs, logger, train_config)
        
        # Run final evaluation after training
        eval_config = {
            'num_episodes': 5,
            'max_steps_per_episode': args.max_steps,
            'save_frames': args.save_frames,
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
        # Just run a single episode with rendering
        print(f"Deploying agent on {args.task} task...")
        rewards, frames = run_episode(
            env=env,
            actor=agent,
            max_steps=args.max_steps,
            render=True,
            logger=logger,
            output_dirs=output_dirs,
            save_frames=args.save_frames,
            create_animation_file=True,
            prefix="deploy"
        )
        print(f"Deployment episode complete. Return: {sum(rewards):.2f}")
    
    # Save metrics summary
    logger.save_metrics_summary()
    
    print(f"All outputs saved to: {output_dirs['root']}")

if __name__ == "__main__":
    main() 