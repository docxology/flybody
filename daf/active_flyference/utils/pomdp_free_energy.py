"""POMDP-specific free energy calculations for active inference.

This module provides enhanced free energy calculations for POMDP scenarios,
considering both observation and transition uncertainties in active inference.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Callable, Any

from .free_energy import softmax, kl_divergence, entropy

def compute_pomdp_variational_free_energy(
    belief_states: jnp.ndarray,
    predicted_obs: jnp.ndarray,
    observation: jnp.ndarray,
    likelihood_model: jnp.ndarray,
    prior_states: jnp.ndarray,
    obs_precision: Optional[jnp.ndarray] = None
) -> float:
    """
    Compute variational free energy for POMDP with weighted observation precision.
    
    Args:
        belief_states: Current belief over states
        predicted_obs: Predicted observations from current belief
        observation: Actual observation
        likelihood_model: Observation likelihood model
        prior_states: Prior belief over states
        obs_precision: Optional precision for different observation dimensions
        
    Returns:
        Variational free energy value
    """
    # Complexity term: KL divergence between posterior and prior
    complexity = kl_divergence(belief_states, prior_states)
    
    # Accuracy term: Weighted prediction error
    if obs_precision is None:
        # Without precision weighting
        prediction_error = jnp.sum((predicted_obs - observation)**2)
    else:
        # With precision weighting for different sensory channels
        prediction_error = jnp.sum(obs_precision * (predicted_obs - observation)**2)
    
    # Free energy = complexity + accuracy
    # Note: Lower prediction error means higher accuracy, so we use negative
    return complexity + 0.5 * prediction_error

def compute_pomdp_expected_free_energy(
    belief_states: jnp.ndarray,
    predicted_next_states: jnp.ndarray,
    likelihood_model: jnp.ndarray,
    preferred_observations: jnp.ndarray,
    transition_model: jnp.ndarray,
    action: int,
    planning_horizon: int = 1,
    obs_precision: Optional[jnp.ndarray] = None,
    discount_factor: float = 0.9
) -> float:
    """
    Compute expected free energy for action selection in POMDP.
    
    This function computes the expected free energy for a given action,
    considering multiple steps into the future (planning horizon).
    
    Args:
        belief_states: Current belief over states
        predicted_next_states: Predicted next states given action
        likelihood_model: Observation likelihood model
        preferred_observations: Prior preferences over observations
        transition_model: State transition model
        action: The action to evaluate
        planning_horizon: How many steps to look ahead
        obs_precision: Optional precision for different observation dimensions
        discount_factor: Discount factor for future time steps
        
    Returns:
        Expected free energy value
    """
    # Initialize total expected free energy
    total_efe = 0.0
    
    # Start with current belief
    current_belief = belief_states
    
    # Compute EFE for each step in planning horizon
    for t in range(planning_horizon):
        # Apply discount factor based on time step
        discount = discount_factor ** t
        
        # Predict next state distribution given current belief and action
        next_belief = transition_model[:, :, action] @ current_belief
        next_belief = next_belief / jnp.sum(next_belief)  # Normalize
        
        # Predict expected observations from next belief
        predicted_obs = likelihood_model @ next_belief
        
        # 1. Epistemic value: How much would this action reduce uncertainty?
        # Expected information gain
        predicted_states = transition_model[:, :, action] @ current_belief
        state_entropy_before = entropy(current_belief)
        state_entropy_after = entropy(predicted_states)
        information_gain = state_entropy_before - state_entropy_after
        
        # 2. Pragmatic value: How close are predicted observations to preferences?
        # KL divergence from preferred observations
        divergence = kl_divergence(predicted_obs, preferred_observations)
        
        # Combine epistemic and pragmatic components with discount
        step_efe = discount * (divergence - information_gain)
        
        # Add to total
        total_efe += step_efe
        
        # Update current belief for next iteration
        current_belief = next_belief
    
    return total_efe

def compute_precision_weighted_prediction_error(
    observation: jnp.ndarray,
    predicted_obs: jnp.ndarray,
    precision: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute precision-weighted prediction error for observations.
    
    Args:
        observation: Actual observation
        predicted_obs: Predicted observation
        precision: Precision (inverse variance) for each observation dimension
        
    Returns:
        Precision-weighted prediction error
    """
    # Compute the difference between actual and predicted
    error = observation - predicted_obs
    
    # Weight by precision
    weighted_error = precision * error
    
    # Return the squared weighted error
    return jnp.sum(weighted_error ** 2)

def compute_action_probability(
    expected_free_energies: jnp.ndarray,
    alpha: float = 1.0,
    action_prior: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Compute action probability distribution based on expected free energies.
    
    Args:
        expected_free_energies: Expected free energy for each action
        alpha: Inverse temperature parameter
        action_prior: Optional prior over actions
        
    Returns:
        Action probability distribution
    """
    # Convert EFE to action probabilities (lower EFE = higher probability)
    action_probs = softmax(-expected_free_energies, temp=alpha)
    
    # If we have a prior over actions, incorporate it
    if action_prior is not None:
        # Multiply by prior and renormalize
        action_probs = action_probs * action_prior
        action_probs = action_probs / jnp.sum(action_probs)
    
    return action_probs

def compute_belief_entropy_reduction(
    current_belief: jnp.ndarray,
    next_belief: jnp.ndarray
) -> float:
    """
    Compute the reduction in belief entropy after an observation.
    
    Args:
        current_belief: Current belief distribution
        next_belief: Updated belief after observation
        
    Returns:
        Reduction in entropy (positive value indicates information gain)
    """
    # Compute entropy before and after
    entropy_before = entropy(current_belief)
    entropy_after = entropy(next_belief)
    
    # Reduction in entropy = information gain
    return entropy_before - entropy_after

def compute_hierarchical_free_energy(
    belief_states: Dict[str, jnp.ndarray],
    predicted_obs: Dict[str, jnp.ndarray],
    observation: Dict[str, jnp.ndarray],
    prior_states: Dict[str, jnp.ndarray],
    precision_factors: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute free energy across hierarchical levels of inference.
    
    This function handles hierarchical models where beliefs at different
    levels interact (e.g., high-level states inform low-level priors).
    
    Args:
        belief_states: Beliefs at each level
        predicted_obs: Predicted observations at each level
        observation: Actual observations at each level
        prior_states: Prior beliefs at each level
        precision_factors: Precision weighting for each level
    
    Returns:
        Dictionary of free energy at each level and total
    """
    # Initialize results
    results = {}
    total_fe = 0.0
    
    # Compute free energy at each level
    for level, level_belief in belief_states.items():
        # Skip if we don't have all components for this level
        if (level not in predicted_obs or level not in observation or
            level not in prior_states or level not in precision_factors):
            continue
        
        # Get components for this level
        level_pred = predicted_obs[level]
        level_obs = observation[level]
        level_prior = prior_states[level]
        level_precision = precision_factors[level]
        
        # Compute complexity term
        complexity = kl_divergence(level_belief, level_prior)
        
        # Compute accuracy term with precision weighting
        prediction_error = jnp.sum((level_pred - level_obs) ** 2)
        accuracy = 0.5 * level_precision * prediction_error
        
        # Compute free energy for this level
        level_fe = complexity + accuracy
        
        # Store result and add to total
        results[f"{level}_fe"] = float(level_fe)
        total_fe += level_fe
    
    # Store total
    results["total_fe"] = float(total_fe)
    
    return results

def compute_information_gain_actions(
    belief_states: jnp.ndarray,
    transition_model: jnp.ndarray,
    likelihood_model: jnp.ndarray,
    num_actions: int
) -> jnp.ndarray:
    """
    Compute expected information gain for each possible action.
    
    This function quantifies how much each action would reduce uncertainty
    about hidden states, supporting active exploration strategies.
    
    Args:
        belief_states: Current belief over states
        transition_model: State transition model
        likelihood_model: Observation likelihood model
        num_actions: Number of actions to evaluate
        
    Returns:
        Expected information gain for each action
    """
    # Current entropy of belief
    current_entropy = entropy(belief_states)
    
    # Initialize array for information gains
    info_gains = jnp.zeros(num_actions)
    
    # For each action, compute expected information gain
    for a in range(num_actions):
        # Predict next state distribution
        next_belief = transition_model[:, :, a] @ belief_states
        next_belief = next_belief / jnp.sum(next_belief)  # Normalize
        
        # Compute entropy of predicted next state
        next_entropy = entropy(next_belief)
        
        # Information gain is reduction in entropy
        info_gain = current_entropy - next_entropy
        
        # Store result
        info_gains = info_gains.at[a].set(info_gain)
    
    return info_gains 