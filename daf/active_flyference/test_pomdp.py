#!/usr/bin/env python3
"""
Simple tests for the POMDP-based Active Inference implementation.

This module provides basic unit tests to validate the core functionality
of the POMDP implementation without requiring a full environment.
"""

import numpy as np
import jax
import jax.numpy as jnp
import unittest
import os
import sys

# Add parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import POMDP components
from daf.active_flyference.models.pomdp_model import POMDPModel
from daf.active_flyference.models.flybody_pomdp import FlybodyPOMDP
from daf.active_flyference.utils.pomdp_free_energy import (
    compute_pomdp_variational_free_energy,
    compute_pomdp_expected_free_energy,
    compute_action_probability,
    compute_belief_entropy_reduction
)

class TestPOMDPModel(unittest.TestCase):
    """Test cases for the POMDP model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a simple POMDP model for testing
        self.num_states = 8
        self.num_obs = 4
        self.num_actions = 3
        
        # Create model
        self.model = POMDPModel(
            num_states=self.num_states,
            num_observations=self.num_obs,
            num_actions=self.num_actions,
            observation_noise=0.1,
            transition_noise=0.1,
            precision=2.0,
            learning_rate=0.01
        )
    
    def test_initialization(self):
        """Test model initialization."""
        # Check model dimensions
        self.assertEqual(self.model.num_states, self.num_states)
        self.assertEqual(self.model.num_observations, self.num_obs)
        self.assertEqual(self.model.num_actions, self.num_actions)
        
        # Check prior beliefs
        self.assertEqual(self.model.belief_states.shape, (self.num_states,))
        self.assertAlmostEqual(jnp.sum(self.model.belief_states), 1.0)
        
        # Check transition model dimensions
        self.assertEqual(self.model.transition_model.shape, 
                         (self.num_states, self.num_states, self.num_actions))
    
    def test_belief_update(self):
        """Test belief updating."""
        # Create a simple observation
        observation = jnp.ones(self.num_obs) / self.num_obs
        
        # Initial belief should be uniform
        self.assertTrue(jnp.allclose(self.model.belief_states, 
                                    jnp.ones(self.num_states) / self.num_states))
        
        # Update belief
        updated_belief = self.model.update_belief(observation, iterations=5)
        
        # Check properties of updated belief
        self.assertEqual(updated_belief.shape, (self.num_states,))
        self.assertAlmostEqual(jnp.sum(updated_belief), 1.0)
        
        # Belief should have changed
        self.assertFalse(jnp.allclose(updated_belief, 
                                     jnp.ones(self.num_states) / self.num_states))
    
    def test_action_selection(self):
        """Test action selection."""
        # Select an action
        action, action_probs = self.model.select_action()
        
        # Action should be a valid index
        self.assertTrue(0 <= action < self.num_actions)
        
        # Action probabilities should sum to 1
        self.assertEqual(action_probs.shape, (self.num_actions,))
        self.assertAlmostEqual(jnp.sum(action_probs), 1.0)
    
    def test_temporal_dynamics(self):
        """Test belief updating with temporal dynamics."""
        # Skip if the method doesn't exist
        if not hasattr(self.model, 'update_belief_with_temporal_dynamics'):
            return
        
        # Create a simple observation
        observation = jnp.ones(self.num_obs) / self.num_obs
        
        # Action
        action = 0
        
        # Update with dynamics
        updated_belief = self.model.update_belief_with_temporal_dynamics(
            observation, action, iterations=5
        )
        
        # Check properties
        self.assertEqual(updated_belief.shape, (self.num_states,))
        self.assertAlmostEqual(jnp.sum(updated_belief), 1.0)
    
    def test_counterfactual_beliefs(self):
        """Test counterfactual belief computation."""
        # Skip if the method doesn't exist
        if not hasattr(self.model, 'compute_counterfactual_beliefs'):
            return
        
        # Compute counterfactuals
        counterfactuals = self.model.compute_counterfactual_beliefs(
            actions=[0, 1], planning_horizon=2
        )
        
        # Check structure
        self.assertIn(0, counterfactuals)
        self.assertIn(1, counterfactuals)
        
        # Check trajectory for action 0
        self.assertIn('belief_trajectory', counterfactuals[0])
        self.assertIn('final_belief', counterfactuals[0])
        
        # Should have planning_horizon + 1 beliefs in trajectory
        self.assertEqual(len(counterfactuals[0]['belief_trajectory']), 3)

class TestFlybodyPOMDP(unittest.TestCase):
    """Test cases for the FlybodyPOMDP implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a FlybodyPOMDP model
        self.task_name = "walk_on_ball"
        self.model = FlybodyPOMDP(
            task_name=self.task_name,
            include_vision=False,
            num_states=64,
            num_actions=8,
            observation_noise=0.05,
            transition_noise=0.1
        )
    
    def test_initialization(self):
        """Test model initialization."""
        # Check task-specific properties
        self.assertEqual(self.model.task_name, self.task_name)
        
        # Check state dimension setup
        self.assertIn('position_x', self.model.state_dims)
        self.assertIn('position_y', self.model.state_dims)
        
        # Check observation dimension setup
        self.assertIn('proprioception', self.model.obs_dims)
        
        # Check action labels
        self.assertTrue(hasattr(self.model, 'action_labels'))
        self.assertEqual(len(self.model.action_labels), self.model.num_actions)
    
    def test_observation_mapping(self):
        """Test observation mapping."""
        # Test with dictionary input
        obs_dict = {
            'joint_positions': np.random.randn(20),
            'ball_velocity': np.random.randn(2),
            'orientation': np.random.randn(3)
        }
        
        obs_vector = self.model.observation_mapping(obs_dict)
        
        # Should convert to expected size
        expected_size = sum(self.model.obs_dims.values())
        self.assertEqual(obs_vector.shape, (expected_size,))
        
        # Test with array input
        obs_array = np.random.randn(expected_size + 5)  # Intentionally larger
        obs_vector = self.model.observation_mapping(obs_array)
        
        # Should truncate to expected size
        self.assertEqual(obs_vector.shape, (expected_size,))
    
    def test_action_mapping(self):
        """Test action mapping."""
        # Get mapped action
        action_idx = 0
        action_vector = self.model.action_mapping(action_idx)
        
        # Should match action dimension
        self.assertEqual(action_vector.shape, (self.model.action_dim,))

class TestPOMDPFreeEnergy(unittest.TestCase):
    """Test cases for POMDP free energy computations."""
    
    def setUp(self):
        """Set up test environment."""
        # Create simple arrays for testing
        self.num_states = 4
        self.num_obs = 3
        self.num_actions = 2
        
        # Prior and posterior beliefs
        self.prior = jnp.ones(self.num_states) / self.num_states
        posterior = jnp.array([0.4, 0.3, 0.2, 0.1])
        self.posterior = posterior / jnp.sum(posterior)
        
        # Likelihood model
        self.likelihood = jnp.ones((self.num_obs, self.num_states)) / self.num_obs
        
        # Observations
        self.observation = jnp.ones(self.num_obs) / self.num_obs
        
        # Preferred observations
        self.preferences = jnp.array([0.6, 0.3, 0.1])
        
        # Predicted observations
        self.predicted_obs = self.likelihood @ self.posterior
        
        # Transition model
        self.transitions = jnp.ones((self.num_states, self.num_states, self.num_actions)) / self.num_states
    
    def test_variational_free_energy(self):
        """Test variational free energy computation."""
        # Compute VFE
        vfe = compute_pomdp_variational_free_energy(
            self.posterior,
            self.predicted_obs,
            self.observation,
            self.likelihood,
            self.prior
        )
        
        # Should be a scalar
        self.assertTrue(jnp.isscalar(vfe))
        
        # With uniform distributions, should be close to KL divergence
        kl = jnp.sum(self.posterior * jnp.log(self.posterior / self.prior))
        self.assertAlmostEqual(float(vfe), float(kl), places=4)
    
    def test_precision_weighted_vfe(self):
        """Test precision-weighted variational free energy."""
        # Create precision vector
        precision = jnp.array([2.0, 1.0, 0.5])
        
        # Compute precision-weighted VFE
        vfe = compute_pomdp_variational_free_energy(
            self.posterior,
            self.predicted_obs,
            self.observation,
            self.likelihood,
            self.prior,
            precision
        )
        
        # Should be a scalar
        self.assertTrue(jnp.isscalar(vfe))
    
    def test_expected_free_energy(self):
        """Test expected free energy computation."""
        # Compute EFE
        efe = compute_pomdp_expected_free_energy(
            self.posterior,
            self.posterior,  # Use same belief for next state
            self.likelihood,
            self.preferences,
            self.transitions,
            action=0
        )
        
        # Should be a scalar
        self.assertTrue(jnp.isscalar(efe))
    
    def test_planning_horizon(self):
        """Test multi-step planning horizon."""
        # Compute EFE with different planning horizons
        efe1 = compute_pomdp_expected_free_energy(
            self.posterior,
            self.posterior,
            self.likelihood,
            self.preferences,
            self.transitions,
            action=0,
            planning_horizon=1
        )
        
        efe3 = compute_pomdp_expected_free_energy(
            self.posterior,
            self.posterior,
            self.likelihood,
            self.preferences,
            self.transitions,
            action=0,
            planning_horizon=3
        )
        
        # Longer horizon should influence the value
        self.assertNotEqual(float(efe1), float(efe3))

if __name__ == "__main__":
    unittest.main() 