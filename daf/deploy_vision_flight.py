#!/usr/bin/env python3
"""
Deploy an agent on the vision-guided flight task.
This script provides a simplified interface for training and evaluating agents
specifically on the vision flight task, which includes visual observations.
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import sonnet as snt

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main deployment script
from deploy_agents import create_environment, run_episode
from daf_agent_utils import create_run_dirs, MetricLogger, save_network_summary

# Import specific task and required modules
from flybody.tasks.vision_flight import VisualFlightTask
from flybody.agents import agent_dmpo, network_factory, network_factory_vis
from acme import specs
from acme.tf import utils as tf_utils

def create_vision_agent(env_spec, config=None):
    """Create a DMPO agent with vision network for the given environment spec."""
    # Default configuration if none is provided
    if config is None:
        config = {
            'vision_encoding_size': 128,
            'hidden_layer_sizes': (256, 256),
            'learning_rate': 1e-4,
            'discount': 0.99,
            'batch_size': 256,
            'target_update_period': 100,
        }
    
    # Create vision observation network
    vision_encoder = network_factory_vis.make_vision_network(
        config['vision_encoding_size']
    )
    
    # Create observation network that can handle both vision and proprioceptive inputs
    observation_network = network_factory_vis.make_observation_network(
        vision_encoder=vision_encoder,
        proprioception_hidden_layer_sizes=(128,)
    )
    
    # Create policy network with the output of the observation network as input
    policy_layer_sizes = config['hidden_layer_sizes']
    policy_network = network_factory.make_policy_network(
        env_spec.actions.shape[0], 
        policy_layer_sizes
    )
    
    # Create critic network
    critic_network = network_factory.make_critic_network(
        config['hidden_layer_sizes']
    )
    
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

def train_vision_agent(env, agent, output_dirs, logger, config=None):
    """Train an agent on the given vision flight environment."""
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
    last_eval_return = None
    
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
            last_eval_return = eval_return
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
        'final_return': float(last_eval_return) if last_eval_return is not None else None,
    }
    with open(os.path.join(output_dirs['stats'], "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return agent

def run_vision_evaluation(env, agent, output_dirs, logger, config=None):
    """Run evaluation of a trained agent on vision flight task."""
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
    parser = argparse.ArgumentParser(description="Train and evaluate agents on vision flight task")
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
    parser.add_argument("--scene-type", type=str, default="trench",
                      help="Type of scene to use (trench, bumpy, random)")
    parser.add_argument("--vision-size", type=int, default=64,
                      help="Size of the vision input")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create a unique output directory for this run
    run_name = args.output_name or f"vision_flight_{args.scene_type}_{args.mode}"
    output_dirs = create_run_dirs(run_name)
    
    # Save command line arguments
    with open(os.path.join(output_dirs['root'], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create logger
    logger = MetricLogger(output_dirs['logs'])
    
    # Create environment with task-specific parameters
    env_kwargs = {
        'scene_type': args.scene_type,
        'vision_size': args.vision_size
    }
    
    env = create_environment('vision_flight', **env_kwargs)
    env_spec = specs.make_environment_spec(env)
    
    print(f"Created vision flight environment with action space: {env_spec.actions.shape}")
    
    # Vision-specific agent configuration
    agent_config = {
        'vision_encoding_size': 128,
        'hidden_layer_sizes': (256, 256, 128),
        'learning_rate': 1e-4,
        'discount': 0.99,
        'batch_size': 256,
        'target_update_period': 100,
    }
    
    # Create or load agent
    if args.mode == "eval" and args.checkpoint:
        agent = create_vision_agent(env_spec, agent_config)
        print(f"Loading agent from checkpoint: {args.checkpoint}")
        agent.restore(args.checkpoint)
    else:
        agent = create_vision_agent(env_spec, agent_config)
    
    # Save task info
    task_info = {
        "task": "vision_flight",
        "scene_type": args.scene_type,
        "vision_size": args.vision_size,
        "action_dimension": env_spec.actions.shape[0],
        "has_vision": True,
    }
    with open(os.path.join(output_dirs['stats'], "task_info.json"), "w") as f:
        json.dump(task_info, f, indent=2)
    
    # Run based on the specified mode
    if args.mode == "train":
        # Vision-specific training configuration
        train_config = {
            'num_episodes': args.episodes,
            'eval_every': 5,
            'max_steps_per_episode': args.max_steps,
            'save_frames': args.save_frames,
            'save_checkpoint_every': 10,
        }
        
        print(f"Training agent on vision flight task for {args.episodes} episodes...")
        agent = train_vision_agent(env, agent, output_dirs, logger, train_config)
        
        # Run final evaluation
        eval_config = {
            'num_episodes': 5,
            'max_steps_per_episode': args.max_steps,
            'save_frames': True,
        }
        run_vision_evaluation(env, agent, output_dirs, logger, eval_config)
    
    elif args.mode == "eval":
        eval_config = {
            'num_episodes': args.episodes,
            'max_steps_per_episode': args.max_steps,
            'save_frames': args.save_frames,
        }
        run_vision_evaluation(env, agent, output_dirs, logger, eval_config)
    
    elif args.mode == "deploy":
        # Just run a single episode with rendering and save frames
        print(f"Deploying agent on vision flight task ({args.scene_type} scene)...")
        
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
            prefix="deploy_vision"
        )
        
        print(f"Deployment complete. Return: {sum(rewards):.2f}")
    
    # Save metrics summary
    logger.save_metrics_summary()
    
    print(f"All outputs saved to: {output_dirs['root']}")

if __name__ == "__main__":
    main() 