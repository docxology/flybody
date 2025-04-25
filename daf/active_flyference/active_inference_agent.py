"""Active Inference agent for flybody environments."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
import os
import time
import dm_env
from acme import types

from daf.active_flyference.models.generative_model import GenerativeModel
from daf.active_flyference.models.walk_model import WalkOnBallModel
from daf.active_flyference.models.flight_model import FlightModel
from daf.active_flyference.utils.free_energy import compute_variational_free_energy
from daf.active_flyference.utils.inference import belief_updating_full_cycle

class ActiveInferenceAgent:
    """
    Active Inference agent for flybody environments.
    
    This agent implements the Active Inference framework for perception and action
    in flybody tasks, using a POMDP formulation and variational inference.
    """
    
    def __init__(
        self,
        model: GenerativeModel,
        task_name: str,
        inference_iterations: int = 10,
        action_alpha: float = 1.0,
        logging: bool = True,
        log_dir: Optional[str] = None
    ):
        """
        Initialize the Active Inference agent.
        
        Args:
            model: Generative model for the task
            task_name: Name of the task
            inference_iterations: Number of iterations for belief updating
            action_alpha: Temperature parameter for action selection
            logging: Whether to log agent data
            log_dir: Directory for logging
        """
        self.model = model
        self.task_name = task_name
        self.inference_iterations = inference_iterations
        self.action_alpha = action_alpha
        
        # Initialize agent state
        self.current_observation = None
        self.last_action = None
        self.time_step = 0
        
        # Track free energy for monitoring
        self.total_free_energy = 0.0
        self.episode_free_energy = 0.0
        
        # Set up logging
        self.logging = logging
        self.log_dir = log_dir or os.path.expanduser("~/active_inference_logs")
        
        if self.logging:
            os.makedirs(self.log_dir, exist_ok=True)
            self.belief_history = []
            self.action_history = []
            self.free_energy_history = []
            self.observation_history = []
    
    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        """
        Process the first observation in an episode.
        
        Args:
            timestep: First timestep from the environment
        """
        # Reset the model's belief state to prior
        self.model.reset()
        
        # Reset tracking variables
        self.time_step = 0
        self.episode_free_energy = 0.0
        
        # Clear history
        if self.logging:
            self.belief_history = []
            self.action_history = []
            self.free_energy_history = []
            self.observation_history = []
        
        # Process the first observation
        self._process_observation(timestep.observation)
    
    def _process_observation(self, observation: Any) -> jnp.ndarray:
        """
        Process an observation and update beliefs.
        
        Args:
            observation: Observation from the environment
            
        Returns:
            Processed observation vector
        """
        # Convert observation to the format expected by the model
        obs_vector = self.model.observation_mapping(observation)
        
        # Update the model's belief state
        self.model.update_belief(obs_vector, iterations=self.inference_iterations)
        
        # Store for later use
        self.current_observation = obs_vector
        
        # Log if enabled
        if self.logging:
            self.belief_history.append(self.model.belief_states.copy())
            self.observation_history.append(obs_vector)
        
        return obs_vector
    
    def observe(self, action: np.ndarray, next_timestep: dm_env.TimeStep) -> None:
        """
        Process a new observation after taking an action.
        
        Args:
            action: Action that was taken
            next_timestep: Resulting timestep from the environment
        """
        # Increment time step
        self.time_step += 1
        
        # Process the new observation
        obs_vector = self._process_observation(next_timestep.observation)
        
        # Compute and accumulate free energy
        fe = self._compute_free_energy(obs_vector)
        self.episode_free_energy += fe
        self.total_free_energy += fe
        
        # Log free energy if enabled
        if self.logging:
            self.free_energy_history.append(fe)
    
    def select_action(self, observation: Any) -> np.ndarray:
        """
        Select an action using active inference principles.
        
        Args:
            observation: Current observation (ignored, using stored beliefs)
            
        Returns:
            Action vector for environment
        """
        # Use the current belief state to select an action
        action_idx, action_probs = self.model.select_action(alpha=self.action_alpha)
        
        # Convert discrete action to continuous vector for environment
        env_action = self.model.action_mapping(action_idx)
        
        # Store last action for reference
        self.last_action = action_idx
        
        # Log if enabled
        if self.logging and self.last_action is not None:
            self.action_history.append(self.last_action)
        
        return env_action
    
    def _compute_free_energy(self, observation: jnp.ndarray) -> float:
        """
        Compute variational free energy for current belief and observation.
        
        Args:
            observation: Current observation
            
        Returns:
            Free energy value
        """
        # Compute variational free energy
        free_energy = compute_variational_free_energy(
            belief_states=self.model.belief_states,
            belief_observations=self.model.predict_observation(),
            prior_states=self.model.prior_states,
            likelihood=self.model.likelihood_model,
            observation=observation
        )
        
        return float(free_energy)
    
    def update_model_from_experience(self, learning_rate: Optional[float] = None) -> None:
        """
        Update the model based on recent experience.
        
        Args:
            learning_rate: Optional learning rate override
        """
        if len(self.action_history) == 0 or len(self.belief_history) < 2:
            return  # Not enough history to learn from
        
        # Override learning rate if provided
        if learning_rate is not None:
            original_lr = self.model.learning_rate
            self.model.learning_rate = learning_rate
        
        # Determine state indices based on highest belief probability
        states = [jnp.argmax(belief) for belief in self.belief_history]
        
        # Update model for each transition
        for t in range(len(self.action_history)):
            state = states[t]
            action = self.action_history[t]
            next_state = states[t + 1] if t + 1 < len(states) else states[t]
            observation = self.observation_history[t]
            
            # Update model parameters
            self.model.update_from_experience(state, action, next_state, observation)
        
        # Restore original learning rate if changed
        if learning_rate is not None:
            self.model.learning_rate = original_lr
    
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
        
        belief_path = os.path.join(episode_dir, "belief_history.npy")
        np.save(belief_path, np.array(self.belief_history))
        files['beliefs'] = belief_path
        
        action_path = os.path.join(episode_dir, "action_history.npy")
        np.save(action_path, np.array(self.action_history))
        files['actions'] = action_path
        
        fe_path = os.path.join(episode_dir, "free_energy_history.npy")
        np.save(fe_path, np.array(self.free_energy_history))
        files['free_energy'] = fe_path
        
        obs_path = os.path.join(episode_dir, "observation_history.npy")
        np.save(obs_path, np.array(self.observation_history))
        files['observations'] = obs_path
        
        # Save summary information
        summary = {
            'episode': episode,
            'task': self.task_name,
            'time_steps': self.time_step,
            'total_free_energy': float(self.episode_free_energy),
            'average_free_energy': float(self.episode_free_energy / max(1, self.time_step)),
            'timestamp': time.time()
        }
        
        import json
        summary_path = os.path.join(episode_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        files['summary'] = summary_path
        
        return files
    
    def create_visualizations(self, episode: int = 0, prefix: str = "") -> None:
        """
        Create visualizations of agent performance.
        
        Args:
            episode: Episode number
            prefix: Optional prefix for filenames
        """
        if not self.logging or not self.belief_history:
            return
        
        # Create episode directory
        episode_dir = os.path.join(self.log_dir, f"{prefix}episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        try:
            from daf.active_flyference.utils.visualization import create_active_inference_animation
            
            # Prepare data for visualization
            beliefs = self.belief_history
            
            # Compute expected free energies for each time step
            free_energies = []
            for t, belief in enumerate(beliefs):
                if t < len(self.action_history):
                    efe_values = []
                    for a in range(self.model.num_actions):
                        efe = self.model.compute_expected_free_energy(a)
                        efe_values.append(efe)
                    free_energies.append(jnp.array(efe_values))
            
            # Prepare observations and predictions
            observations = self.observation_history
            predictions = [self.model.predict_observation(belief) for belief in beliefs]
            
            # Get state and action labels if available
            state_labels = getattr(self.model, 'state_labels', None)
            action_labels = getattr(self.model, 'action_labels', None)
            
            # Create animation
            create_active_inference_animation(
                beliefs=beliefs,
                free_energies=free_energies,
                observations=observations,
                predictions=predictions,
                state_labels=state_labels,
                action_labels=action_labels,
                output_dir=episode_dir
            )
            
            print(f"Created visualizations in {episode_dir}")
            
        except ImportError as e:
            print(f"Could not create visualizations: {e}")

def create_agent_for_task(
    task_name: str, 
    include_vision: bool = False,
    learning_rate: float = 0.01,
    precision: float = 2.0,
    inference_iterations: int = 10,
    log_dir: Optional[str] = None
) -> ActiveInferenceAgent:
    """
    Create an Active Inference agent for a specific task.
    
    Args:
        task_name: Name of the task (walk_on_ball, flight_imitation, vision_flight)
        include_vision: Whether to include vision in observations
        learning_rate: Learning rate for model updates
        precision: Precision parameter for action selection
        inference_iterations: Number of iterations for belief updating
        log_dir: Directory for logging
        
    Returns:
        Configured ActiveInferenceAgent
    """
    # Create the appropriate model for the task
    if task_name in ['walk_on_ball', 'walk_imitation']:
        model = WalkOnBallModel(
            precision=precision,
            learning_rate=learning_rate
        )
    elif task_name in ['flight_imitation', 'vision_flight']:
        model = FlightModel(
            precision=precision,
            include_vision=(include_vision or 'vision' in task_name),
            learning_rate=learning_rate
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Create and return the agent
    return ActiveInferenceAgent(
        model=model,
        task_name=task_name,
        inference_iterations=inference_iterations,
        action_alpha=precision,
        logging=True,
        log_dir=log_dir
    ) 