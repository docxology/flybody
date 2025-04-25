#!/usr/bin/env python3
"""
This is a simplified version of deploy_agents.py for the DAF context.
It includes only the minimum required functionality to run deploy_flight_imitation.py.
"""

import os
import sys
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from acme import specs
import dm_env

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import flybody modules
try:
    from flybody.tasks import flight_imitation
    from flybody.fruitfly.fruitfly import FruitFly
    from dm_control.locomotion.arenas.floors import Floor
    from flybody.utils.filters import ButterworthFilter
except ImportError:
    # Fallback if the filter module is not available
    print("Warning: Could not import ButterworthFilter - creating a dummy implementation")
    class ButterworthFilter:
        def __init__(self, **kwargs):
            pass
        def filter(self, x):
            return x
        def __call__(self, x):
            return x

# Import local utilities
from daf_agent_utils import (
    create_run_dirs, 
    MetricLogger, 
    run_episode, 
    save_network_summary
)

# Dictionary mapping task names to their environment creation functions
TASKS = {
    'flight_imitation': flight_imitation.FlightImitationWBPG,
}

def create_environment(task_name, **kwargs):
    """Create an environment for the specified task."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASKS.keys())}")
    
    # Default components for task initialization (can be overridden by kwargs)
    default_walker = kwargs.pop('walker', FruitFly())
    default_arena = kwargs.pop('arena', Floor())
    default_time_limit = kwargs.pop('time_limit', 10.0) # Default 10 seconds
    default_joint_filter = kwargs.pop('joint_filter', ButterworthFilter())
    
    task_class = TASKS[task_name]
    
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
    """Create a simplified agent stub for the given environment spec."""
    # This is a stub implementation that returns a simple random agent
    class RandomAgent:
        def __init__(self, action_spec):
            self.action_spec = action_spec
            
        def select_action(self, observation):
            return np.random.uniform(
                self.action_spec.minimum, 
                self.action_spec.maximum, 
                size=self.action_spec.shape
            )
            
        def observe_first(self, timestep):
            pass
            
        def observe(self, action, next_timestep):
            pass
            
        def update(self):
            pass
            
        def save(self, path):
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "agent_info.json"), "w") as f:
                json.dump({"type": "random_agent"}, f)
                
        def restore(self, path):
            pass
    
    return RandomAgent(env_spec.actions)

def train_agent(env, agent, output_dirs, logger, config=None):
    """Stub implementation for training an agent."""
    if config is None:
        config = {
            'num_episodes': 10,
            'eval_every': 5,
            'max_steps_per_episode': 100,
            'save_frames': True,
            'save_checkpoint_every': 5,
        }
    
    print("Note: This is a stub implementation that doesn't actually train the agent")
    print(f"Training would run for {config['num_episodes']} episodes")
    
    # Save a checkpoint
    checkpoint_path = os.path.join(output_dirs['checkpoints'], "stub_checkpoint")
    agent.save(checkpoint_path)
    
    return agent

def run_evaluation(env, agent, output_dirs, logger, config=None):
    """Run evaluation of an agent."""
    if config is None:
        config = {
            'num_episodes': 3,
            'max_steps_per_episode': 100,
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
    
    return summary 