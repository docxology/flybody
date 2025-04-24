"""Visualization utilities for Active Inference."""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Any
import os

def plot_belief_distribution(
    belief: jnp.ndarray,
    state_labels: Optional[List[str]] = None,
    title: str = "Belief Distribution",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> plt.Axes:
    """
    Plot a belief distribution over states.
    
    Args:
        belief: Probability distribution over states
        state_labels: Optional labels for states
        title: Plot title
        ax: Optional matplotlib axis to plot on
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # If no labels are provided, use indices
    if state_labels is None:
        state_labels = [f"S{i}" for i in range(len(belief))]
    
    # Plot belief as a bar chart
    ax.bar(state_labels, belief, alpha=0.7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(belief):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    return ax

def plot_free_energy_landscape(
    free_energies: jnp.ndarray,
    action_labels: Optional[List[str]] = None,
    title: str = "Expected Free Energy Landscape",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> plt.Axes:
    """
    Plot the free energy landscape across actions.
    
    Args:
        free_energies: Array of free energy values for each action
        action_labels: Optional labels for actions
        title: Plot title
        ax: Optional matplotlib axis to plot on
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # If no labels are provided, use indices
    if action_labels is None:
        action_labels = [f"A{i}" for i in range(len(free_energies))]
    
    # Plot free energies as a bar chart
    ax.bar(action_labels, free_energies, alpha=0.7, color='orange')
    ax.set_ylabel("Expected Free Energy")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(free_energies):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # Highlight the minimum free energy action
    min_idx = jnp.argmin(free_energies)
    ax.bar(action_labels[min_idx], free_energies[min_idx], alpha=0.7, color='green', 
           label=f"Selected: {action_labels[min_idx]}")
    ax.legend()
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    return ax

def plot_observation_prediction(
    actual_obs: jnp.ndarray,
    predicted_obs: jnp.ndarray,
    obs_labels: Optional[List[str]] = None,
    title: str = "Observation vs Prediction",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> plt.Axes:
    """
    Plot actual observation alongside predicted observation.
    
    Args:
        actual_obs: Actual observation vector
        predicted_obs: Predicted observation vector
        obs_labels: Optional labels for observation dimensions
        title: Plot title
        ax: Optional matplotlib axis to plot on
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # If no labels are provided, use indices
    if obs_labels is None:
        obs_labels = [f"O{i}" for i in range(len(actual_obs))]
    
    # Set up the x positions for the bars
    x = np.arange(len(obs_labels))
    width = 0.35
    
    # Plot actual and predicted observations as grouped bar chart
    ax.bar(x - width/2, actual_obs, width, alpha=0.7, label='Actual')
    ax.bar(x + width/2, predicted_obs, width, alpha=0.7, label='Predicted')
    
    ax.set_xticks(x)
    ax.set_xticklabels(obs_labels)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    
    return ax

def plot_active_inference_summary(
    belief: jnp.ndarray,
    free_energies: jnp.ndarray,
    actual_obs: jnp.ndarray,
    predicted_obs: jnp.ndarray,
    state_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    obs_labels: Optional[List[str]] = None,
    step: int = 0,
    output_dir: str = ".",
) -> None:
    """
    Create a combined plot summarizing the active inference process.
    
    Args:
        belief: Current belief over states
        free_energies: Expected free energies for actions
        actual_obs: Actual observation
        predicted_obs: Predicted observation
        state_labels: Optional labels for states
        action_labels: Optional labels for actions
        obs_labels: Optional labels for observations
        step: Current time step
        output_dir: Directory to save the figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot belief distribution
    plot_belief_distribution(
        belief,
        state_labels=state_labels,
        title=f"Belief Distribution (Step {step})",
        ax=axes[0]
    )
    
    # Plot free energy landscape
    plot_free_energy_landscape(
        free_energies,
        action_labels=action_labels,
        title=f"Expected Free Energy Landscape (Step {step})",
        ax=axes[1]
    )
    
    # Plot observation prediction
    plot_observation_prediction(
        actual_obs,
        predicted_obs,
        obs_labels=obs_labels,
        title=f"Observation vs Prediction (Step {step})",
        ax=axes[2]
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"active_inference_summary_step_{step}.png"))
    plt.close(fig)

def create_active_inference_animation(
    beliefs: List[jnp.ndarray],
    free_energies: List[jnp.ndarray],
    observations: List[jnp.ndarray],
    predictions: List[jnp.ndarray],
    state_labels: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    obs_labels: Optional[List[str]] = None,
    output_dir: str = "."
) -> None:
    """
    Create a series of plots for each time step that can be converted to an animation.
    
    Args:
        beliefs: List of belief distributions for each time step
        free_energies: List of expected free energies for each time step
        observations: List of actual observations for each time step
        predictions: List of predicted observations for each time step
        state_labels: Optional labels for states
        action_labels: Optional labels for actions
        obs_labels: Optional labels for observations
        output_dir: Directory to save the figures
    """
    # Create frames directory if it doesn't exist
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate a plot for each time step
    for t in range(len(beliefs)):
        plot_active_inference_summary(
            beliefs[t],
            free_energies[t],
            observations[t],
            predictions[t],
            state_labels=state_labels,
            action_labels=action_labels,
            obs_labels=obs_labels,
            step=t,
            output_dir=frames_dir
        )
    
    print(f"Generated {len(beliefs)} frames in {frames_dir}")
    print("To create an animation from these frames, you can use:")
    print(f"ffmpeg -framerate 2 -i {frames_dir}/active_inference_summary_step_%d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/active_inference_animation.mp4") 