"""POMDP-based generative model for Active Inference in flybody.

This model explicitly represents the Partially Observable Markov Decision Process
(POMDP) structure for active inference in the flybody environment.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
import os
import abc

from .generative_model import GenerativeModel

class POMDPModel(GenerativeModel):
    """
    POMDP-based generative model for Active Inference.
    
    This model explicitly represents the POMDP structure for active inference,
    including hidden states, observations, and actions for the flybody agent.
    """
    
    def __init__(
        self,
        num_states: int,
        num_observations: int,
        num_actions: int,
        observation_noise: float = 0.1,
        transition_noise: float = 0.1,
        precision: float = 2.0,
        learning_rate: float = 0.01
    ):
        """
        Initialize the POMDP model.
        
        Args:
            num_states: Number of states in the model
            num_observations: Number of observation dimensions
            num_actions: Number of possible actions
            observation_noise: Noise level in the observation model
            transition_noise: Noise level in the transition model
            precision: Precision parameter for action selection
            learning_rate: Learning rate for parameter updates
        """
        super().__init__(num_states, num_observations, num_actions, precision)
        
        self.observation_noise = observation_noise
        self.transition_noise = transition_noise
        self.learning_rate = learning_rate
        
        # Additional POMDP-specific components
        self._init_pomdp_components()
    
    def _init_pomdp_components(self) -> None:
        """Initialize POMDP-specific model components."""
        # Observation noise precision (inverse variance)
        self.observation_precision = 1.0 / (self.observation_noise ** 2)
        
        # Transition noise precision
        self.transition_precision = 1.0 / (self.transition_noise ** 2)
        
        # Belief history for temporal dynamics
        self.belief_history = []
        
        # Innovations (prediction errors)
        self.state_prediction_errors = jnp.zeros(self.num_states)
        self.obs_prediction_errors = jnp.zeros(self.num_observations)
    
    def update_belief_with_temporal_dynamics(
        self, 
        observation: jnp.ndarray, 
        action: Optional[int] = None,
        iterations: int = 10
    ) -> jnp.ndarray:
        """
        Update beliefs considering both observations and temporal dynamics.
        
        This method performs belief updating in the POMDP setting, accounting for:
        1. Prior predictions based on previous actions
        2. Likelihood of current observations
        3. Temporal dynamics of the system
        
        Args:
            observation: Current observation
            action: Previous action taken (if available)
            iterations: Number of iterations for belief updating
            
        Returns:
            Updated belief state
        """
        # If we have previous action information, use it for prediction
        if action is not None and len(self.belief_history) > 0:
            # Get previous belief
            prev_belief = self.belief_history[-1]
            
            # Predict current belief based on previous belief and action
            predicted_belief = self.predict_next_state(action)
            
            # Use this as our prior instead of the static prior
            prior = predicted_belief
        else:
            # If no history, use static prior
            prior = self.prior_states
        
        # Import here to avoid circular imports
        from ..utils.inference import update_beliefs_with_observation
        
        # Update belief with new observation
        posterior = update_beliefs_with_observation(
            prior,
            self.likelihood_model,
            observation,
            iterations=iterations
        )
        
        # Store prediction errors
        if len(self.belief_history) > 0:
            self.state_prediction_errors = posterior - prior
        
        # Predicted observation
        predicted_obs = self.predict_observation(posterior)
        self.obs_prediction_errors = observation - predicted_obs
        
        # Store updated belief
        self.belief_states = posterior
        self.belief_history.append(posterior)
        
        # Maintain reasonable history length
        if len(self.belief_history) > 100:
            self.belief_history = self.belief_history[-100:]
        
        return posterior
    
    def compute_counterfactual_beliefs(
        self, 
        actions: List[int], 
        planning_horizon: int = 3
    ) -> Dict[str, Any]:
        """
        Compute counterfactual belief trajectories for planning.
        
        This method simulates belief trajectories for different action sequences,
        allowing the agent to plan by evaluating expected free energy.
        
        Args:
            actions: List of actions to consider
            planning_horizon: How many steps to look ahead
            
        Returns:
            Dictionary of counterfactual beliefs and their metrics
        """
        counterfactuals = {}
        
        # Current belief is starting point
        current_belief = self.belief_states
        
        # For each action, compute belief trajectory
        for action in actions:
            # Initialize trajectory with current belief
            belief_trajectory = [current_belief]
            
            # Generate trajectory through planning horizon
            for t in range(planning_horizon):
                # Predict next belief
                next_belief = self.predict_next_state(action)
                belief_trajectory.append(next_belief)
                
                # Use this belief for next prediction
                current_belief = next_belief
            
            # Reset current belief
            current_belief = self.belief_states
            
            # Store trajectory
            counterfactuals[action] = {
                'belief_trajectory': belief_trajectory,
                'final_belief': belief_trajectory[-1],
                'predicted_observations': [self.predict_observation(b) for b in belief_trajectory]
            }
        
        return counterfactuals
    
    def update_from_experience(
        self, 
        state: int, 
        action: int, 
        next_state: int, 
        observation: jnp.ndarray
    ) -> None:
        """
        Update model parameters based on experience.
        
        Args:
            state: Current state index
            action: Action taken
            next_state: Next state index
            observation: Observation received
        """
        # Update transition model
        # Increase probability of observed transition
        current_prob = self.transition_model[next_state, state, action]
        update = self.learning_rate * (1.0 - current_prob)
        self.transition_model = self.transition_model.at[next_state, state, action].set(current_prob + update)
        
        # Normalize to maintain valid probabilities
        column_sum = jnp.sum(self.transition_model[:, state, action])
        self.transition_model = self.transition_model.at[:, state, action].set(
            self.transition_model[:, state, action] / column_sum
        )
        
        # Update likelihood model
        # Increase probability of observed observation given state
        for o in range(self.num_observations):
            obs_value = observation[o]
            current_prob = self.likelihood_model[o, state]
            
            # Simple update rule: move probability toward observation
            update = self.learning_rate * (obs_value - current_prob)
            self.likelihood_model = self.likelihood_model.at[o, state].set(current_prob + update)
        
        # Normalize likelihood to maintain valid probabilities
        for o in range(self.num_observations):
            column_sum = jnp.sum(self.likelihood_model[o, :])
            self.likelihood_model = self.likelihood_model.at[o, :].set(
                self.likelihood_model[o, :] / column_sum
            )
    
    def estimate_uncertainty(self) -> Dict[str, jnp.ndarray]:
        """
        Estimate uncertainty in the current belief state.
        
        Returns:
            Dictionary with different uncertainty metrics
        """
        # State entropy (higher = more uncertain)
        state_entropy = -jnp.sum(self.belief_states * jnp.log(self.belief_states + 1e-8))
        
        # Observation entropy based on current belief
        predicted_obs = self.predict_observation()
        obs_entropy = -jnp.sum(predicted_obs * jnp.log(predicted_obs + 1e-8))
        
        # Information gain (reduction in expected uncertainty)
        if len(self.belief_history) > 1:
            prev_entropy = -jnp.sum(self.belief_history[-2] * jnp.log(self.belief_history[-2] + 1e-8))
            info_gain = prev_entropy - state_entropy
        else:
            info_gain = 0.0
        
        return {
            'state_entropy': state_entropy,
            'observation_entropy': obs_entropy,
            'information_gain': info_gain,
            'state_pred_error': jnp.sum(jnp.abs(self.state_prediction_errors)),
            'obs_pred_error': jnp.sum(jnp.abs(self.obs_prediction_errors))
        } 