"""POMDP-based active inference agent for flybody environments.

This module implements an active inference agent that explicitly treats
perception and action in flybody as a Partially Observable Markov Decision Process.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
import os
import time
import dm_env
from acme import types
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from daf.active_flyference.models.pomdp_model import POMDPModel
from daf.active_flyference.models.flybody_pomdp import FlybodyPOMDP
from daf.active_flyference.utils.pomdp_free_energy import (
    compute_pomdp_variational_free_energy,
    compute_pomdp_expected_free_energy,
    compute_action_probability,
    compute_belief_entropy_reduction,
    compute_information_gain_actions
)

class POMDPAgent:
    """
    POMDP-based active inference agent for flybody environments.
    
    This agent implements active inference principles within an explicit POMDP framework,
    focusing on belief updating, uncertainty-driven exploration, and preference-based action.
    """
    
    def __init__(
        self,
        model: POMDPModel,
        task_name: str,
        inference_iterations: int = 10,
        planning_horizon: int = 3,
        action_temperature: float = 1.0,
        exploration_factor: float = 0.5,
        logging: bool = True,
        log_dir: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the POMDP-based active inference agent.
        
        Args:
            model: POMDP generative model
            task_name: Name of the task
            inference_iterations: Number of iterations for belief updating
            planning_horizon: How many steps to consider in planning
            action_temperature: Temperature parameter for action selection
            exploration_factor: Weight for exploration vs. exploitation
            logging: Whether to log agent data
            log_dir: Directory for logging
            debug: Whether to print debug information
        """
        self.model = model
        self.task_name = task_name
        self.inference_iterations = inference_iterations
        self.planning_horizon = planning_horizon
        self.action_temperature = action_temperature
        self.exploration_factor = exploration_factor
        self.debug = debug
        
        # Initialize POMDP state tracking
        self.current_observation = None
        self.last_action = None
        self.time_step = 0
        self.episode_step = 0
        
        # Initialize free energy tracking
        self.total_free_energy = 0.0
        self.episode_free_energy = 0.0
        self.expected_free_energies = None
        
        # Track uncertainty and information gain
        self.state_entropy_history = []
        self.information_gain_history = []
        
        # Set up logging
        self.logging = logging
        self.log_dir = log_dir or os.path.expanduser("~/pomdp_agent_logs")
        
        if self.logging:
            os.makedirs(self.log_dir, exist_ok=True)
            self.belief_history = []
            self.action_history = []
            self.observation_history = []
            self.free_energy_history = []
            self.uncertainty_history = []
    
    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        """
        Process the first observation in an episode.
        
        Args:
            timestep: First timestep from the environment
        """
        # Reset internal state
        self.model.reset()
        self.episode_step = 0
        self.episode_free_energy = 0.0
        
        # Clear history
        if self.logging:
            self.belief_history = []
            self.action_history = []
            self.observation_history = []
            self.free_energy_history = []
            self.uncertainty_history = []
            self.state_entropy_history = []
            self.information_gain_history = []
        
        # Process the first observation
        self._process_observation(timestep.observation)
        
        if self.debug:
            print(f"Initial belief entropy: {self._compute_belief_entropy():.4f}")
    
    def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        """
        Process a new observation after taking an action.
        
        Args:
            action: Action that was taken
            next_timestep: Resulting timestep from the environment
        """
        # Increment counters
        self.time_step += 1
        self.episode_step += 1
        
        # Convert continuous action back to discrete if needed
        if hasattr(self, 'last_action_idx'):
            discrete_action = self.last_action_idx
        else:
            # Try to infer which action was closest
            discrete_action = 0  # Default
        
        # Process the new observation with temporal dynamics
        self._process_observation_with_dynamics(next_timestep.observation, discrete_action)
        
        # Compute and accumulate free energy
        fe = self._compute_free_energy()
        self.episode_free_energy += fe
        self.total_free_energy += fe
        
        # Log free energy
        if self.logging:
            self.free_energy_history.append(fe)
        
        # Debug output
        if self.debug:
            print(f"Step {self.episode_step}: Free Energy = {fe:.4f}, "
                  f"Belief Entropy = {self._compute_belief_entropy():.4f}")
    
    def _process_observation(self, observation: Any) -> jnp.ndarray:
        """
        Process an observation and update beliefs.
        
        Args:
            observation: Observation from the environment
            
        Returns:
            Processed observation vector
        """
        # Convert observation to model's format
        obs_vector = self.model.observation_mapping(observation)
        
        # Update belief state
        self.model.update_belief(obs_vector, iterations=self.inference_iterations)
        
        # Store for later use
        self.current_observation = obs_vector
        
        # Log if enabled
        if self.logging:
            self.belief_history.append(self.model.belief_states.copy())
            self.observation_history.append(obs_vector)
            self.uncertainty_history.append(self.model.estimate_uncertainty())
            self.state_entropy_history.append(self._compute_belief_entropy())
        
        return obs_vector
    
    def _process_observation_with_dynamics(
        self, 
        observation: Any, 
        last_action: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Process an observation considering temporal dynamics.
        
        Args:
            observation: Observation from the environment
            last_action: Last action taken (if available)
            
        Returns:
            Processed observation vector
        """
        # Convert observation to model's format
        obs_vector = self.model.observation_mapping(observation)
        
        # Update belief using temporal dynamics if available
        if hasattr(self.model, 'update_belief_with_temporal_dynamics') and last_action is not None:
            # POMDP-specific belief update with temporal dynamics
            self.model.update_belief_with_temporal_dynamics(
                obs_vector, 
                action=last_action,
                iterations=self.inference_iterations
            )
        else:
            # Fallback to standard belief update
            self.model.update_belief(obs_vector, iterations=self.inference_iterations)
        
        # Store for later use
        self.current_observation = obs_vector
        
        # Compute information gain if we have history
        if len(self.belief_history) > 0 and self.logging:
            prev_belief = self.belief_history[-1]
            info_gain = compute_belief_entropy_reduction(
                prev_belief, self.model.belief_states
            )
            self.information_gain_history.append(info_gain)
        
        # Log if enabled
        if self.logging:
            self.belief_history.append(self.model.belief_states.copy())
            self.observation_history.append(obs_vector)
            self.uncertainty_history.append(self.model.estimate_uncertainty())
            self.state_entropy_history.append(self._compute_belief_entropy())
        
        return obs_vector
    
    def select_action(self, observation: Any = None) -> np.ndarray:
        """
        Select an action using POMDP-based active inference principles.
        
        Args:
            observation: Current observation (ignored if None, using stored belief)
            
        Returns:
            Action vector for environment
        """
        # Use current model state to select action, considering
        # both pragmatic value (preferences) and epistemic value (information gain)
        
        # Get number of actions
        num_actions = self.model.num_actions
        
        # Initialize array for expected free energies
        expected_free_energies = jnp.zeros(num_actions)
        
        # For each action, compute expected free energy
        for a in range(num_actions):
            # If our model has the enhanced POMDP function, use it
            if hasattr(self.model, 'compute_counterfactual_beliefs'):
                # Sophisticated counterfactual trajectory analysis
                trajectories = self.model.compute_counterfactual_beliefs(
                    [a], planning_horizon=self.planning_horizon
                )
                
                # Extract expected free energy from trajectory
                if a in trajectories:
                    # Get predicted observations from final belief
                    final_belief = trajectories[a]['final_belief']
                    predicted_obs = self.model.predict_observation(final_belief)
                    
                    # Compute divergence from preferences
                    from daf.active_flyference.utils.free_energy import kl_divergence
                    pragmatic_value = kl_divergence(
                        predicted_obs, self.model.preferred_observations
                    )
                    
                    # Compute information gain (entropy reduction)
                    epistemic_value = compute_belief_entropy_reduction(
                        self.model.belief_states, final_belief
                    )
                    
                    # Combine with exploration factor
                    efe = pragmatic_value - self.exploration_factor * epistemic_value
                    expected_free_energies = expected_free_energies.at[a].set(efe)
            else:
                # Use standard expected free energy computation
                efe = compute_pomdp_expected_free_energy(
                    self.model.belief_states,
                    self.model.predict_next_state(a),
                    self.model.likelihood_model,
                    self.model.preferred_observations,
                    self.model.transition_model,
                    a,
                    planning_horizon=self.planning_horizon
                )
                expected_free_energies = expected_free_energies.at[a].set(efe)
        
        # Store for visualization
        self.expected_free_energies = expected_free_energies
        
        # Convert to action probabilities
        action_probs = compute_action_probability(
            expected_free_energies, alpha=self.action_temperature
        )
        
        # Select action with lowest expected free energy
        action_idx = jnp.argmin(expected_free_energies)
        
        # For exploration, occasionally sample from the distribution
        if np.random.random() < self.exploration_factor:
            action_idx = np.random.choice(num_actions, p=np.array(action_probs))
        
        # Store the selected action index for later reference
        self.last_action_idx = int(action_idx)
        
        # Convert to environment action
        env_action = self.model.action_mapping(action_idx)
        
        # Log if enabled
        if self.logging:
            self.action_history.append(action_idx)
        
        if self.debug:
            action_label = self.model.action_labels[action_idx] if hasattr(self.model, 'action_labels') else f"action_{action_idx}"
            print(f"Selected {action_label} (EFE: {expected_free_energies[action_idx]:.4f})")
        
        return env_action
    
    def _compute_free_energy(self) -> float:
        """
        Compute variational free energy for current belief and observation.
        
        Returns:
            Free energy value
        """
        # Check if we have the current observation
        if self.current_observation is None:
            return 0.0
        
        # Get predicted observation from current belief
        predicted_obs = self.model.predict_observation()
        
        # Get observation precision if available
        obs_precision = None
        if hasattr(self.model, 'predict_observation_with_precision'):
            _, obs_precision = self.model.predict_observation_with_precision()
        
        # Compute variational free energy
        free_energy = compute_pomdp_variational_free_energy(
            self.model.belief_states,
            predicted_obs,
            self.current_observation,
            self.model.likelihood_model,
            self.model.prior_states,
            obs_precision
        )
        
        return float(free_energy)
    
    def _compute_belief_entropy(self) -> float:
        """
        Compute entropy of current belief state.
        
        Returns:
            Entropy value
        """
        from daf.active_flyference.utils.free_energy import entropy
        return float(entropy(self.model.belief_states))
    
    def update_model_from_experience(self, learning_rate: Optional[float] = None) -> None:
        """
        Update the model based on recent experience.
        
        Args:
            learning_rate: Optional learning rate override
        """
        if not hasattr(self.model, 'update_from_experience') or len(self.action_history) == 0:
            return  # Model doesn't support learning or no history
        
        # Override learning rate if provided
        original_lr = None
        if learning_rate is not None and hasattr(self.model, 'learning_rate'):
            original_lr = self.model.learning_rate
            self.model.learning_rate = learning_rate
        
        # Determine state indices based on highest belief probability
        states = [jnp.argmax(belief) for belief in self.belief_history]
        
        # Update model for each transition
        for t in range(len(self.action_history)):
            if t + 1 >= len(states):
                break  # Not enough history
                
            state = states[t]
            action = self.action_history[t]
            next_state = states[t + 1]
            observation = self.observation_history[t + 1]  # Observation after action
            
            # Update model parameters
            self.model.update_from_experience(state, action, next_state, observation)
        
        # Restore original learning rate if changed
        if original_lr is not None:
            self.model.learning_rate = original_lr
        
        if self.debug:
            print(f"Updated model from {len(self.action_history)} experiences")
    
    def save_history(self, episode: int = 0, prefix: str = "") -> Dict[str, str]:
        """
        Save agent history data.
        
        Args:
            episode: Episode number for filename
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary of saved file paths
        """
        if not self.logging or not self.belief_history:
            return {}
        
        # Create episode directory
        episode_dir = os.path.join(self.log_dir, f"{prefix}episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save histories as NumPy arrays
        files = {}
        
        # Belief history
        belief_path = os.path.join(episode_dir, "belief_history.npy")
        np.save(belief_path, np.array(self.belief_history))
        files['beliefs'] = belief_path
        
        # Action history
        action_path = os.path.join(episode_dir, "action_history.npy")
        np.save(action_path, np.array(self.action_history))
        files['actions'] = action_path
        
        # Observation history
        obs_path = os.path.join(episode_dir, "observation_history.npy")
        np.save(obs_path, np.array(self.observation_history))
        files['observations'] = obs_path
        
        # Free energy history
        fe_path = os.path.join(episode_dir, "free_energy_history.npy")
        np.save(fe_path, np.array(self.free_energy_history))
        files['free_energy'] = fe_path
        
        # Uncertainty and information gain
        if self.uncertainty_history:
            uncert_path = os.path.join(episode_dir, "uncertainty_history.npy")
            # Convert list of dicts to structured array
            uncert_data = {}
            for key in self.uncertainty_history[0].keys():
                uncert_data[key] = [u[key] for u in self.uncertainty_history]
            np.savez(uncert_path, **uncert_data)
            files['uncertainty'] = uncert_path
        
        # State entropy history
        if self.state_entropy_history:
            entropy_path = os.path.join(episode_dir, "entropy_history.npy")
            np.save(entropy_path, np.array(self.state_entropy_history))
            files['entropy'] = entropy_path
        
        # Information gain history
        if self.information_gain_history:
            info_path = os.path.join(episode_dir, "information_gain_history.npy")
            np.save(info_path, np.array(self.information_gain_history))
            files['info_gain'] = info_path
        
        return files
    
    def create_visualizations(self, episode: int = 0, prefix: str = "") -> None:
        """
        Create visualizations of agent performance and beliefs.
        
        Args:
            episode: Episode number for filenames
            prefix: Optional prefix for filenames
        """
        if not self.logging or not self.belief_history:
            return
        
        # Create episode directory
        episode_dir = os.path.join(self.log_dir, f"{prefix}episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Plot belief evolution
        self._plot_belief_evolution(os.path.join(episode_dir, "belief_evolution.png"))
        
        # Plot free energy
        self._plot_free_energy(os.path.join(episode_dir, "free_energy.png"))
        
        # Plot entropy and information gain
        self._plot_uncertainty_metrics(os.path.join(episode_dir, "uncertainty.png"))
        
        # Plot action distribution
        self._plot_action_distribution(os.path.join(episode_dir, "action_distribution.png"))
        
        # Create belief state visualization images
        self._create_belief_state_images(episode_dir)
    
    def _plot_belief_evolution(self, save_path: str) -> None:
        """
        Plot the evolution of belief states over time.
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.belief_history) == 0:
            return
        
        # Convert belief history to array
        belief_array = np.array(self.belief_history)
        num_timesteps, num_states = belief_array.shape
        
        # Plot heatmap of belief evolution
        plt.figure(figsize=(12, 8))
        plt.imshow(belief_array.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Belief Probability')
        plt.xlabel('Time Step')
        plt.ylabel('State Index')
        plt.title('Belief State Evolution')
        
        # Add state labels if available
        if hasattr(self.model, 'state_labels') and len(self.model.state_labels) > 0:
            # Sample state labels if too many
            max_labels = 20
            if num_states > max_labels:
                step = num_states // max_labels
                label_indices = np.arange(0, num_states, step)
                plt.yticks(label_indices, [self.model.state_labels[i] for i in label_indices], fontsize=8)
            else:
                plt.yticks(np.arange(num_states), self.model.state_labels, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_free_energy(self, save_path: str) -> None:
        """
        Plot free energy over time.
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.free_energy_history) == 0:
            return
        
        # Plot free energy
        plt.figure(figsize=(10, 6))
        plt.plot(self.free_energy_history, 'b-', linewidth=2)
        plt.xlabel('Time Step')
        plt.ylabel('Variational Free Energy')
        plt.title('Free Energy over Time')
        plt.grid(True, alpha=0.3)
        
        # Add cumulative free energy
        cumulative_fe = np.cumsum(self.free_energy_history)
        plt.plot(cumulative_fe / (np.arange(len(cumulative_fe)) + 1), 'r--', 
                 label='Cumulative Average')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_uncertainty_metrics(self, save_path: str) -> None:
        """
        Plot uncertainty metrics over time.
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.state_entropy_history) == 0:
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot belief entropy
        ax1.plot(self.state_entropy_history, 'b-', linewidth=2)
        ax1.set_ylabel('Belief Entropy')
        ax1.set_title('Uncertainty Metrics')
        ax1.grid(True, alpha=0.3)
        
        # Plot information gain if available
        if len(self.information_gain_history) > 0:
            # Pad with zero for first step
            info_gain = [0] + self.information_gain_history
            ax2.plot(info_gain, 'g-', linewidth=2)
            ax2.set_ylabel('Information Gain')
            ax2.set_xlabel('Time Step')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _plot_action_distribution(self, save_path: str) -> None:
        """
        Plot distribution of selected actions.
        
        Args:
            save_path: Path to save the plot
        """
        if len(self.action_history) == 0:
            return
        
        # Count action occurrences
        action_counts = {}
        for action in self.action_history:
            if action in action_counts:
                action_counts[action] += 1
            else:
                action_counts[action] = 1
        
        # Sort by action index
        sorted_actions = sorted(action_counts.items())
        actions, counts = zip(*sorted_actions)
        
        # Get action labels if available
        labels = actions
        if hasattr(self.model, 'action_labels'):
            labels = [self.model.action_labels[a] if a < len(self.model.action_labels) else f"action_{a}" 
                     for a in actions]
        
        # Plot action distribution
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(counts)), counts, tick_label=labels)
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.title('Action Distribution')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def _create_belief_state_images(self, output_dir: str) -> None:
        """
        Create visualization images for belief states.
        
        Args:
            output_dir: Directory to save images
        """
        if len(self.belief_history) == 0 or not hasattr(self.model, 'state_dims'):
            return
        
        # Only create for specific episodes to avoid too many files
        max_images = 10
        step = max(1, len(self.belief_history) // max_images)
        
        # Create images directory
        images_dir = os.path.join(output_dir, "belief_states")
        os.makedirs(images_dir, exist_ok=True)
        
        # For each selected time step
        for t in range(0, len(self.belief_history), step):
            belief = self.belief_history[t]
            
            # Create an image based on the structure of state space
            if self.task_name == 'walk_on_ball' and 'position_x' in self.model.state_dims:
                # 2D grid visualization for walk-on-ball
                self._create_grid_visualization(
                    belief, 
                    os.path.join(images_dir, f"belief_t{t:03d}.png"),
                    time_step=t
                )
            elif self.task_name == 'flight' and 'position_x' in self.model.state_dims:
                # 3D grid visualization for flight
                self._create_grid_visualization(
                    belief, 
                    os.path.join(images_dir, f"belief_t{t:03d}.png"),
                    time_step=t,
                    include_z=True
                )
    
    def _create_grid_visualization(
        self, 
        belief: jnp.ndarray, 
        save_path: str,
        time_step: int = 0,
        include_z: bool = False
    ) -> None:
        """
        Create a grid visualization of belief state.
        
        Args:
            belief: Belief state vector
            save_path: Path to save the image
            time_step: Current time step
            include_z: Whether to include z dimension
        """
        # Extract dimensions from model
        dims = self.model.state_dims
        x_dim = dims.get('position_x', 1)
        y_dim = dims.get('position_y', 1)
        z_dim = dims.get('position_z', 1)
        
        # Determine grid size
        grid_height = y_dim
        grid_width = x_dim
        
        # Create a base image (larger for better visualization)
        cell_size = 50
        img_width = grid_width * cell_size + 100  # Extra space for labels
        img_height = grid_height * cell_size + 100
        
        # Create image and drawing context
        image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("Arial", 12)
            title_font = ImageFont.truetype("Arial", 16)
        except IOError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Draw title
        title = f"Belief State - Step {time_step}"
        draw.text((img_width // 2 - 100, 10), title, fill='black', font=title_font)
        
        # Project 3D belief state to 2D grid if needed
        grid = np.zeros((grid_height, grid_width))
        
        # Marginalize over orientation and other dimensions
        for s, prob in enumerate(belief):
            if s < len(self.model.state_mapping):
                state_dict = self.model.state_mapping[s]
                x = state_dict.get('position_x', 0)
                y = state_dict.get('position_y', 0)
                
                if x < grid_width and y < grid_height:
                    grid[y, x] += prob
        
        # Normalize grid for visualization
        grid = grid / (np.max(grid) + 1e-8)
        
        # Draw grid
        for y in range(grid_height):
            for x in range(grid_width):
                # Calculate grid cell position
                cell_x = 50 + x * cell_size
                cell_y = 50 + y * cell_size
                
                # Determine cell color based on belief probability
                # Convert from white (low) to blue (high)
                intensity = int(255 * (1 - grid[y, x]))
                color = (intensity, intensity, 255)
                
                # Draw cell
                draw.rectangle(
                    [cell_x, cell_y, cell_x + cell_size, cell_y + cell_size],
                    fill=color,
                    outline='black'
                )
                
                # Add probability text
                text = f"{grid[y, x]:.2f}"
                draw.text(
                    (cell_x + cell_size // 2 - 10, cell_y + cell_size // 2 - 6),
                    text,
                    fill='black',
                    font=font
                )
        
        # Add coordinate labels
        for x in range(grid_width):
            draw.text(
                (50 + x * cell_size + cell_size // 2 - 5, 35),
                str(x),
                fill='black',
                font=font
            )
        
        for y in range(grid_height):
            draw.text(
                (35, 50 + y * cell_size + cell_size // 2 - 5),
                str(y),
                fill='black',
                font=font
            )
        
        # Add axis labels
        draw.text((img_width // 2, img_height - 20), "X Position", fill='black', font=font)
        draw.text((15, img_height // 2), "Y Position", fill='black', font=font)
        
        # Save the image
        image.save(save_path)


def create_pomdp_agent_for_task(
    task_name: str,
    include_vision: bool = False,
    num_states: int = 128,
    num_actions: int = 16,
    observation_noise: float = 0.05,
    transition_noise: float = 0.1,
    precision: float = 2.0,
    learning_rate: float = 0.01,
    inference_iterations: int = 10,
    planning_horizon: int = 3,
    action_temperature: float = 1.0,
    exploration_factor: float = 0.3,
    log_dir: Optional[str] = None,
    debug: bool = False
) -> POMDPAgent:
    """
    Create a POMDP-based active inference agent for a specific task.
    
    Args:
        task_name: Name of the task
        include_vision: Whether to include visual observations
        num_states: Number of discrete states
        num_actions: Number of discrete actions
        observation_noise: Noise level in observations
        transition_noise: Noise level in transitions
        precision: Precision for action selection
        learning_rate: Learning rate for model updates
        inference_iterations: Number of iterations for belief updating
        planning_horizon: Steps to consider in planning
        action_temperature: Temperature for action selection
        exploration_factor: Weight for exploration vs. exploitation
        log_dir: Directory for logs
        debug: Whether to print debug information
        
    Returns:
        Configured POMDP agent
    """
    # Create POMDP model for the task
    model = FlybodyPOMDP(
        task_name=task_name,
        include_vision=include_vision,
        num_states=num_states,
        num_actions=num_actions,
        observation_noise=observation_noise,
        transition_noise=transition_noise,
        precision=precision,
        learning_rate=learning_rate
    )
    
    # Create agent with the model
    agent = POMDPAgent(
        model=model,
        task_name=task_name,
        inference_iterations=inference_iterations,
        planning_horizon=planning_horizon,
        action_temperature=action_temperature,
        exploration_factor=exploration_factor,
        log_dir=log_dir,
        debug=debug
    )
    
    return agent 