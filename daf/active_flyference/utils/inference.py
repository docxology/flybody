"""Inference utilities for Active Inference."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Callable
from .free_energy import softmax, kl_divergence

def update_beliefs_with_observation(
    prior_belief: jnp.ndarray,
    likelihood: jnp.ndarray,
    observation: jnp.ndarray,
    iterations: int = 10,
    inference_lr: float = 0.1
) -> jnp.ndarray:
    """
    Update beliefs using a variational inference approach.
    
    Args:
        prior_belief: Prior belief over states (shape: [num_states])
        likelihood: Likelihood mapping from states to observations (shape: [num_observations, num_states])
        observation: Current observation (shape: [num_observations])
        iterations: Number of inference iterations
        inference_lr: Learning rate for gradient updates
        
    Returns:
        Updated posterior belief over states
    """
    # Initialize posterior with prior
    posterior = prior_belief.copy()
    
    # Ensure observation has the right shape for matrix operations
    if len(observation.shape) == 1:
        # Convert to column vector if needed
        observation = observation.reshape(-1, 1)  # [num_observations, 1]
    
    # Get dimensions
    num_obs, num_states = likelihood.shape
    
    # Check if observation shape matches likelihood shape
    if observation.shape[0] != num_obs:
        print(f"Warning: Observation shape {observation.shape} doesn't match likelihood shape {likelihood.shape}")
        # Resize observation to match likelihood by padding or truncating
        if observation.shape[0] < num_obs:
            # Pad with zeros
            padded_obs = jnp.zeros((num_obs, 1))
            padded_obs = padded_obs.at[:observation.shape[0], :].set(observation)
            observation = padded_obs
        else:
            # Truncate
            observation = observation[:num_obs]
    
    # Variational inference loop
    for i in range(iterations):
        # Compute predicted observations based on current posterior
        predicted_obs = likelihood @ posterior  # [num_observations, 1]
        
        # Prediction error between observation and predicted observations
        pred_error = observation - predicted_obs
        
        # Update posterior using gradient (prediction error weighted by likelihood)
        gradient = jnp.matmul(jnp.transpose(likelihood), pred_error).flatten()
        posterior = posterior + inference_lr * gradient
        
        # Ensure posterior is a valid probability distribution
        posterior = jnp.clip(posterior, 1e-8, 1.0)
        posterior = posterior / jnp.sum(posterior)
    
    return posterior

def predict_next_state(
    belief_states: jnp.ndarray,
    transition_model: jnp.ndarray,
    action: int
) -> jnp.ndarray:
    """
    Predict the next state distribution given current belief and action.
    
    Args:
        belief_states: Current belief over states
        transition_model: Transition probabilities P(s'|s,a)
        action: Selected action
        
    Returns:
        Predicted next state distribution
    """
    # Extract transition matrix for the given action
    transition_matrix = transition_model[:, :, action]
    
    # Compute predicted next state
    next_belief = transition_matrix @ belief_states
    
    # Normalize
    next_belief = next_belief / jnp.sum(next_belief)
    
    return next_belief

def select_action_active_inference(
    belief_states: jnp.ndarray,
    transition_model: jnp.ndarray,
    likelihood_model: jnp.ndarray,
    preferred_observations: jnp.ndarray,
    num_actions: int,
    alpha: float = 1.0
) -> Tuple[int, jnp.ndarray]:
    """
    Select an action using active inference principles (minimizing expected free energy).
    
    Args:
        belief_states: Current belief over states
        transition_model: Transition probabilities P(s'|s,a)
        likelihood_model: Likelihood mapping from states to observations
        preferred_observations: Prior preferences over observations
        num_actions: Number of possible actions
        alpha: Temperature parameter for softmax
        
    Returns:
        Tuple of (selected_action, action_distribution)
    """
    # Initialize array to store expected free energy for each action
    expected_free_energies = jnp.zeros(num_actions)
    
    # Compute expected free energy for each action
    for a in range(num_actions):
        # Predict next state distribution given action
        predicted_next_belief = predict_next_state(belief_states, transition_model, a)
        
        # Predict observations from next state
        predicted_obs = likelihood_model @ predicted_next_belief
        
        # Compute expected free energy components
        
        # 1. Ambiguity (risk): Entropy of predicted observations
        ambiguity = -jnp.sum(predicted_obs * jnp.log(predicted_obs + 1e-8))
        
        # 2. Risk: KL divergence from preferred observations
        risk = kl_divergence(predicted_obs, preferred_observations)
        
        # Combine components into expected free energy
        expected_free_energies = expected_free_energies.at[a].set(ambiguity + risk)
    
    # Convert to action probabilities using softmax with negative EFE
    # (Minimize free energy = maximize negative free energy)
    action_probs = softmax(-expected_free_energies, temp=alpha)
    
    # Sample action from distribution
    action = jnp.argmin(expected_free_energies)
    
    return action, action_probs

def belief_updating_full_cycle(
    prior_belief: jnp.ndarray,
    transition_model: jnp.ndarray,
    likelihood_model: jnp.ndarray,
    observation: jnp.ndarray,
    action: int,
    inference_iterations: int = 10
) -> jnp.ndarray:
    """
    Perform a full cycle of belief updating: perception, action, prediction.
    
    Args:
        prior_belief: Prior belief over states
        transition_model: Transition probabilities P(s'|s,a)
        likelihood_model: Likelihood mapping from states to observations
        observation: Current observation
        action: Action taken
        inference_iterations: Number of iterations for perception
        
    Returns:
        Updated belief after perception and prediction
    """
    # 1. Perception: Update beliefs based on observation
    posterior = update_beliefs_with_observation(
        prior_belief,
        likelihood_model,
        observation,
        iterations=inference_iterations
    )
    
    # 2. Prediction: Predict next state after action
    next_belief = predict_next_state(
        posterior,
        transition_model,
        action
    )
    
    return next_belief 