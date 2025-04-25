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
from dm_control import mjcf
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import imageio

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import flybody modules
from flybody.tasks import template_task, walk_imitation, flight_imitation, walk_on_ball, vision_flight, make_task
from flybody.agents import agent_dmpo, actors, network_factory
from flybody.fruitfly.fruitfly import FruitFly
from dm_control.locomotion.arenas.floors import Floor
import flybody.utils as utils

# Import local utilities
from daf.daf_agent_utils import (
    create_run_dirs, 
    MetricLogger, 
    run_episode, 
    save_network_summary,
    save_frame,
    create_animation
)

# Set default frame size for rendering
frame_size = {'width': 640, 'height': 480}

# Dictionary mapping task names to their environment creation functions
TASKS = {
    'template': template_task.TemplateTask,
    'walk_imitation': walk_imitation.WalkImitation,
    'flight_imitation': flight_imitation.FlightImitationWBPG,
    'walk_on_ball': walk_on_ball.WalkOnBall,
    'vision_flight': vision_flight.VisionFlightImitationWBPG,
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility functions for rendering and saving frames
def save_frame(frame, frame_path):
    """Save a single frame to the specified path."""
    if frame is not None:
        imageio.imwrite(frame_path, frame)

def create_animation(frames, animation_path, fps=30):
    """Create an animation from a list of frames."""
    if frames:
        logger.info(f"Creating animation at {animation_path}")
        imageio.mimsave(animation_path, frames, fps=fps)
        logger.info(f"Animation saved to {animation_path}")

# Function to create output directories
def create_output_dirs(base_dir):
    """Create output directories for the deployment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    deploy_dir = os.path.join(base_dir, f"template_deploy_{timestamp}")
    
    output_dirs = {
        'root': deploy_dir,
        'animations': os.path.join(deploy_dir, 'animations'),
        'images': os.path.join(deploy_dir, 'images'),
        'logs': os.path.join(deploy_dir, 'logs'),
        'checkpoints': os.path.join(deploy_dir, 'checkpoints'),
        'stats': os.path.join(deploy_dir, 'stats'),
    }
    
    # Create directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return output_dirs

class TaskEnvironmentWrapper(dm_env.Environment):
    """Wrapper that adapts a task to the dm_env.Environment interface."""
    
    def __init__(self, task):
        self._task = task
        self._physics = mjcf.Physics.from_mjcf_model(task.root_entity.mjcf_model)
        self._reset_next_step = True
        self._random_state = np.random.RandomState(42)  # Default seed
        
        # Initialize the task
        physics = self._physics.copy()
        task.initialize_episode(physics, self._random_state)
        
        # Get the action spec
        action_spec = task.action_spec(physics)
        self._action_spec = specs.BoundedArray(
            shape=action_spec.shape,
            dtype=action_spec.dtype,
            minimum=action_spec.minimum,
            maximum=action_spec.maximum,
            name='action'
        )
        
        # Use a simple, large observation spec to allow for different observations
        # In a real implementation, you would determine this more precisely
        self._observation_spec = specs.BoundedArray(
            shape=(128,),  # Large enough to accommodate various observations
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name='observation'
        )
    
    def reset(self):
        """Reset the environment."""
        physics = self._physics.copy()
        self._task.initialize_episode(physics, self._random_state)
        self._physics = physics
        self._reset_next_step = False
        return self._get_time_step(physics)
    
    def step(self, action):
        """Take a step in the environment."""
        if self._reset_next_step:
            return self.reset()
        
        physics = self._physics.copy()
        
        # Apply action
        self._task.before_step(physics, action, self._random_state)
        physics.step()
        
        # Check if the episode should terminate
        if self._task.should_terminate_episode(physics):
            self._reset_next_step = True
        
        self._physics = physics
        return self._get_time_step(physics)
    
    def _get_time_step(self, physics):
        """Convert the current physics state to a TimeStep."""
        discount = self._task.get_discount(physics)
        reward = self._task.get_reward(physics)
        
        # Get observation - handle FruitFlyObservables which doesn't have items()
        # Start with basic proprioceptive observations
        obs_values = []
        
        # Add vestibular and proprioception observations
        for sensor in self._task.walker.observables.vestibular:
            if sensor.enabled:
                obs_values.append(sensor(physics))
                
        for sensor in self._task.walker.observables.proprioception:
            if sensor.enabled:
                obs_values.append(sensor(physics))
        
        # Add any other enabled observables
        # Note: This is not a complete solution as it doesn't handle all possible observables
        # In a real implementation, you would need to enumerate all observable attributes
        special_observables = ['right_eye', 'left_eye', 'thorax_height', 'abdomen_height', 
                               'world_zaxis_hover', 'world_zaxis', 'world_zaxis_abdomen', 
                               'world_zaxis_head', 'force', 'touch', 'accelerometer', 
                               'gyro', 'velocimeter', 'actuator_activation', 'appendages_pos']
        
        for name in special_observables:
            if hasattr(self._task.walker.observables, name):
                sensor = getattr(self._task.walker.observables, name)
                if sensor.enabled:
                    obs_values.append(sensor(physics))
        
        # Flatten and concatenate all observations
        if not obs_values:
            # If no observations (unlikely), return zeros
            obs_flat = np.zeros(self._observation_spec.shape, dtype=self._observation_spec.dtype)
        else:
            try:
                obs_flat = np.concatenate([v.flatten() for v in obs_values if v is not None])
                
                # Handle observation size mismatch
                spec_size = self._observation_spec.shape[0]
                if len(obs_flat) > spec_size:
                    # Truncate if too large
                    obs_flat = obs_flat[:spec_size]
                elif len(obs_flat) < spec_size:
                    # Pad with zeros if too small
                    padding = np.zeros(spec_size - len(obs_flat), dtype=obs_flat.dtype)
                    obs_flat = np.concatenate([obs_flat, padding])
            except Exception as e:
                print(f"Error processing observations: {e}")
                obs_flat = np.zeros(self._observation_spec.shape, dtype=self._observation_spec.dtype)
        
        if self._reset_next_step:
            return dm_env.termination(reward, obs_flat)
        else:
            return dm_env.transition(reward, discount, obs_flat)
    
    def observation_spec(self):
        """Return the observation spec."""
        return self._observation_spec
    
    def action_spec(self):
        """Return the action spec."""
        return self._action_spec
    
    @property
    def physics(self):
        """Return the underlying physics object."""
        return self._physics

def create_environment(task_name, **kwargs):
    """Create an environment for the specified task."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASKS.keys())}")
    
    # Default components for task initialization (can be overridden by kwargs)
    default_walker = kwargs.pop('walker', FruitFly)
    default_arena = kwargs.pop('arena', Floor())
    default_time_limit = kwargs.pop('time_limit', 10.0) # Default 10 seconds
    default_joint_filter = kwargs.pop('joint_filter', 0.01) # Default filter value
    
    task_class = TASKS[task_name]
    
    # Check if it's a vision task to avoid passing walker/arena if not needed?
    # (Need to inspect base classes or task signatures, difficult without file reading)
    # For now, assume most tasks take these standard args. Handle exceptions if they occur.
    try:
        task = task_class(
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
            task = task_class(**kwargs)
        except Exception as final_e:
            print(f"Error: Failed to initialize task {task_name} even with fallback.")
            raise final_e
    
    # Wrap the task in an environment adapter that provides the expected interface
    environment = TaskEnvironmentWrapper(task)
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
    
    # Create policy and critic networks using the network factory
    factory_fn = network_factory.make_network_factory_dmpo(
        policy_layer_sizes=config['hidden_layer_sizes'],
        critic_layer_sizes=config['hidden_layer_sizes'],
        min_scale=1e-3
    )
    
    networks = factory_fn(env_spec.actions)
    
    # Create the agent
    agent = agent_dmpo.DMPO(
        environment_spec=env_spec,
        policy_network=networks['policy'],
        critic_network=networks['critic'],
        observation_network=networks['observation'],
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

def run_evaluation(env, actor, output_dirs, logger, config=None):
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
        rewards, frames = run_simple_episode(
            env=env,
            actor=actor,
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
    """Example of how to use the deployment code."""
    import argparse
    from flybody.fruitfly.build_fruitfly import make_fruitfly_env
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Deploy agents in FlyBody environments')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true', help='Whether to render the environment')
    parser.add_argument('--save_frames', action='store_true', help='Whether to save frames')
    parser.add_argument('--create_animation', action='store_true', help='Whether to create an animation')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('deploy_agents')
    
    # Create output directories
    output_dirs = create_output_dirs()
    
    # Create environment
    env = make_fruitfly_env()
    
    # Create agent
    action_space = env.action_spec()
    agent = RandomAgent(action_space)
    
    # Run episodes
    for episode in range(args.episodes):
        logger.info(f"Starting episode {episode+1}/{args.episodes}")
        rewards, frames = run_simple_episode(
            env=env,
            agent=agent,
            max_steps=args.max_steps,
            render=args.render,
            logger=logger,
            output_dirs=output_dirs,
            save_frames=args.save_frames,
            create_animation_file=args.create_animation,
            prefix=f"episode_{episode+1}"
        )
        
        logger.info(f"Episode {episode+1} completed with reward: {sum(rewards):.4f}")
    
    logger.info(f"All episodes completed. Output saved to: {output_dirs['base']}")
    env.close()


class RandomAgent:
    """A simple random agent that selects random actions."""
    
    def __init__(self, action_spec):
        self.action_spec = action_spec
        logger.info(f"Initialized RandomAgent with action spec: {action_spec}")
        
    def select_action(self, observation):
        """Select a random action within the action space."""
        if hasattr(self.action_spec, 'minimum') and hasattr(self.action_spec, 'maximum'):
            # Continuous action space
            shape = self.action_spec.shape
            minimum = self.action_spec.minimum
            maximum = self.action_spec.maximum
            
            # Generate random actions within bounds
            return np.random.uniform(minimum, maximum, shape)
        else:
            # Discrete action space
            return np.random.randint(0, self.action_spec.num_values)
    
    def observe_first(self, timestep):
        """Dummy method to observe first timestep."""
        pass
    
    def observe(self, action, timestep):
        """Dummy method to observe timestep."""
        pass


def run_simple_episode(env, agent, max_steps=1000, render=True, logger=None, 
                      output_dirs=None, save_frames=False, create_animation_file=False, prefix="episode"):
    """
    Run a simple episode with the given environment and agent.
    
    Args:
        env: The environment to run the episode in.
        agent: The agent to use for selecting actions (should implement select_action, observe_first, observe).
        max_steps: Maximum number of steps to run the episode for.
        render: Whether to render the environment.
        logger: Logger for metrics. If None, no logging will be performed.
        output_dirs: Dictionary of output directories for saving frames and animations.
                   Should contain 'images' and 'animations' keys.
        save_frames: Whether to save frames to the output directory.
        create_animation_file: Whether to create an animation from the frames.
        prefix: Prefix for saved files.
        
    Returns:
        Tuple of (rewards, frames) where rewards is a list of rewards for each step
        and frames is a list of rendered frames.
    """
    # Initialize frames and rewards lists
    frames = []
    rewards = []
    total_reward = 0.0
    
    # Reset the environment and get the first timestep
    timestep = env.reset()
    
    # Let the agent observe the first timestep
    agent.observe_first(timestep)
    
    # Step through the episode
    for step in range(max_steps):
        # Select an action
        action = agent.select_action(timestep.observation)
        
        # Take a step in the environment
        timestep = env.step(action)
        
        # Let the agent observe the step
        agent.observe(action, timestep)
        
        # Record the reward
        reward = timestep.reward
        rewards.append(reward if reward is not None else 0.0)
        if reward is not None:
            total_reward += reward
        
        # Render if required
        if render:
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            # Save the frame if required
            if save_frames and output_dirs and 'images' in output_dirs:
                frame_path = os.path.join(output_dirs['images'], f"{prefix}_step_{step:04d}.png")
                save_frame(frame, frame_path)
        
        # Log metrics
        if logger is not None:
            logger.log_scalar("step_reward", reward if reward is not None else 0.0, step)
            logger.log_scalar("cumulative_reward", total_reward, step)
        
        # Break if the episode is done
        if timestep.last():
            break
    
    # Create an animation if required
    if create_animation_file and frames and output_dirs and 'animations' in output_dirs:
        animation_path = os.path.join(output_dirs['animations'], f"{prefix}_animation.mp4")
        create_animation(frames, animation_path)
    
    # Log final metrics
    if logger is not None:
        logger.log_scalar("episode_length", step + 1, 0)
        logger.log_scalar("episode_return", total_reward, 0)
        logger.info(f"Episode completed with {step+1} steps and total reward: {total_reward:.4f}")
    
    return rewards, frames

if __name__ == "__main__":
    main() 