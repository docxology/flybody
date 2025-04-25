#!/usr/bin/env python3
"""
Simplified script to run the POMDP-based Active Inference agent.

This script provides a streamlined way to test the POMDP implementation
without requiring command-line arguments.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import POMDP agent
from daf.active_flyference.pomdp_agent import create_pomdp_agent_for_task

# Import environment creator
from daf.active_flyference.deploy_pomdp_agent import create_environment, run_pomdp_episode

def main():
    """Run a simple POMDP agent demo."""
    print("Running POMDP-based Active Inference agent demo...")
    
    # Configuration
    task_name = "walk_on_ball"
    num_states = 128
    num_actions = 16
    observation_noise = 0.05
    transition_noise = 0.1
    precision = 2.0
    learning_rate = 0.01
    inference_iterations = 10
    planning_horizon = 3
    action_temperature = 1.0
    exploration_factor = 0.3
    include_vision = False
    debug = True
    
    # Create output directory
    output_dir = os.path.join("output", "pomdp_demo")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    output_dirs = {
        "root": output_dir,
        "logs": os.path.join(output_dir, "logs"),
        "images": os.path.join(output_dir, "images"),
        "animations": os.path.join(output_dir, "animations")
    }
    
    for directory in output_dirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Create a simple metric logger
    class SimpleLogger:
        def __init__(self, log_dir):
            self.log_dir = log_dir
            self.metrics = {}
            
        def log_metric(self, name, value, step=None):
            if name not in self.metrics:
                self.metrics[name] = []
            if step is not None:
                self.metrics[name].append((step, value))
            else:
                self.metrics[name].append(value)
    
    logger = SimpleLogger(output_dirs["logs"])
    
    # Create environment
    print(f"Creating environment: {task_name}")
    env = create_environment(task_name)
    
    # Create agent
    print("Creating POMDP agent...")
    agent = create_pomdp_agent_for_task(
        task_name=task_name,
        include_vision=include_vision,
        num_states=num_states,
        num_actions=num_actions,
        observation_noise=observation_noise,
        transition_noise=transition_noise,
        precision=precision,
        learning_rate=learning_rate,
        inference_iterations=inference_iterations,
        planning_horizon=planning_horizon,
        action_temperature=action_temperature,
        exploration_factor=exploration_factor,
        log_dir=output_dirs["logs"],
        debug=debug
    )
    
    # Run a simple episode
    print("\nRunning demonstration episode...")
    rewards, frames = run_pomdp_episode(
        env=env,
        agent=agent,
        output_dirs=output_dirs,
        logger=logger,
        episode=0,
        max_steps=200,  # Shorter episode for testing
        render=True,
        save_frames=True,
        create_animation=True,
        prefix="demo"
    )
    
    # Print results
    print("\nEpisode complete!")
    print(f"Total reward: {sum(rewards):.2f}")
    print(f"Episode length: {len(rewards)}")
    print(f"Free energy: {agent.episode_free_energy:.2f}")
    
    # Create final visualizations
    print("\nCreating visualizations...")
    agent.create_visualizations(episode=0, prefix="final")
    
    print(f"\nDone! Output saved to {output_dir}")
    
    # Plot reward
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Rewards during Episode')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dirs["images"], "rewards.png"))
    plt.close()
    
    print("\nTo train the agent more extensively, use:")
    print("python deploy_pomdp_agent.py --task walk_on_ball --mode train --episodes 20 --debug")

if __name__ == "__main__":
    main() 