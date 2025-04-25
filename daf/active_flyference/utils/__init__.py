"""
Utility functions for Active Inference in flybody.

This module contains utilities for free energy computation, inference, and visualization.
"""

from .free_energy import compute_variational_free_energy, compute_expected_free_energy, kl_divergence, softmax, entropy
from .inference import update_beliefs_with_observation, predict_next_state, select_action_active_inference, belief_updating_full_cycle
from .visualization import plot_belief_distribution, plot_free_energy_landscape, plot_observation_prediction, plot_active_inference_summary, create_active_inference_animation

__all__ = [
    # Free energy utilities
    'compute_variational_free_energy',
    'compute_expected_free_energy',
    'kl_divergence',
    'softmax',
    'entropy',
    
    # Inference utilities
    'update_beliefs_with_observation', 
    'predict_next_state', 
    'select_action_active_inference',
    'belief_updating_full_cycle',
    
    # Visualization utilities
    'plot_belief_distribution',
    'plot_free_energy_landscape',
    'plot_observation_prediction',
    'plot_active_inference_summary',
    'create_active_inference_animation'
] 