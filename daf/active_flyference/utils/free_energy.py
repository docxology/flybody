"""Utilities for computing free energy in Active Inference."""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Callable
import jax
import jax.numpy as jnp

def compute_variational_free_energy(
    belief_states: jnp.ndarray, 
    belief_observations: jnp.ndarray,
    prior_states: jnp.ndarray,
    likelihood: jnp.ndarray,
    observation: jnp.ndarray
) -> float:
    """
    Compute the variational free energy for a given belief and observation.
    
    Args:
        belief_states: Posterior belief over states (q(s))
        belief_observations: Predicted observations based on belief (p(o|s)q(s))
        prior_states: Prior belief over states (p(s))
        likelihood: Likelihood mapping from states to observations (p(o|s))
        observation: Actual observation (o)
        
    Returns:
        Variational free energy value
    """
    # Complexity: KL divergence between posterior and prior
    complexity = kl_divergence(belief_states, prior_states)
    
    # Accuracy: Expected log likelihood of observation given belief
    accuracy = -jnp.sum(belief_states * jnp.log(likelihood @ observation.reshape(-1, 1)))
    
    # Free energy is complexity - accuracy
    return complexity + accuracy

def compute_expected_free_energy(
    belief_states: jnp.ndarray,
    belief_next_states: jnp.ndarray,
    likelihood: jnp.ndarray,
    preferred_observations: jnp.ndarray,
    state_transition: jnp.ndarray,
    action: int,
    alpha: float = 1.0
) -> float:
    """
    Compute the expected free energy for a policy (action sequence).
    
    Args:
        belief_states: Current posterior belief over states
        belief_next_states: Predicted next states given action
        likelihood: Likelihood mapping from states to observations
        preferred_observations: Prior preferences over observations
        state_transition: Transition model for next states given action
        action: The action to evaluate
        alpha: Temperature parameter for action selection
        
    Returns:
        Expected free energy value
    """
    # Compute predicted observations (G matrix in Active Inference)
    predicted_obs = likelihood @ belief_next_states
    
    # Risk: Ambiguity (entropy) of predicted observations
    ambiguity = -jnp.sum(predicted_obs * jnp.log(predicted_obs + 1e-8))
    
    # Risk: Divergence (KL) of predicted observations from preferences
    divergence = kl_divergence(predicted_obs, preferred_observations)
    
    # Return expected free energy
    return alpha * (ambiguity + divergence)

def kl_divergence(p: jnp.ndarray, q: jnp.ndarray) -> float:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        KL divergence
    """
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    q_safe = q + epsilon
    p_safe = p + epsilon
    
    # Normalize
    p_safe = p_safe / jnp.sum(p_safe)
    q_safe = q_safe / jnp.sum(q_safe)
    
    return jnp.sum(p_safe * jnp.log(p_safe / q_safe))

def softmax(x: jnp.ndarray, temp: float = 1.0) -> jnp.ndarray:
    """
    Compute the softmax function.
    
    Args:
        x: Input array
        temp: Temperature parameter
        
    Returns:
        Softmax-transformed array
    """
    x_temp = x / temp
    exp_x = jnp.exp(x_temp - jnp.max(x_temp))
    return exp_x / jnp.sum(exp_x)

def entropy(p: jnp.ndarray) -> float:
    """
    Compute the entropy of a probability distribution.
    
    Args:
        p: Probability distribution
        
    Returns:
        Entropy value
    """
    # Add small epsilon to prevent log(0)
    p_safe = p + 1e-8
    p_safe = p_safe / jnp.sum(p_safe)
    
    return -jnp.sum(p_safe * jnp.log(p_safe)) 