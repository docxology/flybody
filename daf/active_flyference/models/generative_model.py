"""Base generative model for Active Inference in flybody."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
import abc

class GenerativeModel(abc.ABC):
    """
    Base class for generative models in Active Inference.
    
    A generative model represents the agent's internal model of the world,
    including state transitions, observation likelihoods, and preferences.
    """
    
    def __init__(
        self,
        num_states: int,
        num_observations: int,
        num_actions: int,
        precision: float = 1.0
    ):
        """
        Initialize the generative model.
        
        Args:
            num_states: Number of states in the model
            num_observations: Number of observation dimensions
            num_actions: Number of possible actions
            precision: Precision parameter for action selection
        """
        self.num_states = num_states
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.precision = precision
        
        # Initialize model components with uniform distributions
        self._init_model_components()
    
    def _init_model_components(self) -> None:
        """Initialize the core components of the generative model."""
        # Prior over initial states: p(s_0)
        self.prior_states = jnp.ones(self.num_states) / self.num_states
        
        # Transition model: p(s'|s,a)
        self.transition_model = jnp.ones((self.num_states, self.num_states, self.num_actions)) / self.num_states
        
        # Observation model (likelihood): p(o|s)
        self.likelihood_model = jnp.ones((self.num_observations, self.num_states)) / self.num_observations
        
        # Prior preferences: p(o) - desired observations
        self.preferred_observations = jnp.ones(self.num_observations) / self.num_observations
        
        # Current belief state
        self.belief_states = self.prior_states.copy()
    
    @abc.abstractmethod
    def observation_mapping(self, environment_observation: Any) -> jnp.ndarray:
        """
        Map environment observations to the format expected by the generative model.
        
        Args:
            environment_observation: Observation from the environment
            
        Returns:
            Observation vector in the format expected by the model
        """
        pass
    
    @abc.abstractmethod
    def action_mapping(self, model_action: int) -> Any:
        """
        Map model actions to the format expected by the environment.
        
        Args:
            model_action: Action selected by the model
            
        Returns:
            Action in the format expected by the environment
        """
        pass
    
    def update_belief(self, observation: jnp.ndarray, iterations: int = 10) -> jnp.ndarray:
        """
        Update belief states based on new observation.
        
        Args:
            observation: Current observation
            iterations: Number of iterations for belief updating
            
        Returns:
            Updated belief state
        """
        # Import here to avoid circular imports
        from ..utils.inference import update_beliefs_with_observation
        
        self.belief_states = update_beliefs_with_observation(
            self.belief_states,
            self.likelihood_model,
            observation,
            iterations=iterations
        )
        
        return self.belief_states
    
    def select_action(self, alpha: Optional[float] = None) -> Tuple[int, jnp.ndarray]:
        """
        Select an action using active inference principles.
        
        Args:
            alpha: Temperature parameter for softmax (uses self.precision if None)
            
        Returns:
            Tuple of (selected_action, action_probabilities)
        """
        # Import here to avoid circular imports
        from ..utils.inference import select_action_active_inference
        
        if alpha is None:
            alpha = self.precision
        
        action, action_probs = select_action_active_inference(
            self.belief_states,
            self.transition_model,
            self.likelihood_model,
            self.preferred_observations,
            self.num_actions,
            alpha=alpha
        )
        
        return action, action_probs
    
    def predict_next_state(self, action: int) -> jnp.ndarray:
        """
        Predict the next state distribution given an action.
        
        Args:
            action: Selected action
            
        Returns:
            Predicted next state distribution
        """
        # Import here to avoid circular imports
        from ..utils.inference import predict_next_state
        
        next_belief = predict_next_state(
            self.belief_states,
            self.transition_model,
            action
        )
        
        return next_belief
    
    def predict_observation(self, state_belief: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Predict expected observations given current belief state.
        
        Args:
            state_belief: State belief to use (uses self.belief_states if None)
            
        Returns:
            Predicted observation
        """
        if state_belief is None:
            state_belief = self.belief_states
        
        predicted_obs = self.likelihood_model @ state_belief
        return predicted_obs
    
    def compute_expected_free_energy(self, action: int, alpha: Optional[float] = None) -> float:
        """
        Compute expected free energy for a given action.
        
        Args:
            action: Action to evaluate
            alpha: Temperature parameter (uses self.precision if None)
            
        Returns:
            Expected free energy value
        """
        # Import here to avoid circular imports
        from ..utils.free_energy import compute_expected_free_energy
        
        if alpha is None:
            alpha = self.precision
        
        # Predict next state given action
        next_state_belief = self.predict_next_state(action)
        
        # Compute expected free energy
        efe = compute_expected_free_energy(
            self.belief_states,
            next_state_belief,
            self.likelihood_model,
            self.preferred_observations,
            self.transition_model,
            action,
            alpha=alpha
        )
        
        return efe
    
    def update_model_parameters(self, 
                               transitions: Optional[jnp.ndarray] = None, 
                               likelihood: Optional[jnp.ndarray] = None,
                               preferences: Optional[jnp.ndarray] = None) -> None:
        """
        Update model parameters with new values.
        
        Args:
            transitions: New transition model
            likelihood: New likelihood model
            preferences: New preference distribution
        """
        if transitions is not None:
            self.transition_model = transitions
            
        if likelihood is not None:
            self.likelihood_model = likelihood
            
        if preferences is not None:
            self.preferred_observations = preferences
    
    def reset(self) -> None:
        """Reset belief to prior."""
        self.belief_states = self.prior_states.copy() 