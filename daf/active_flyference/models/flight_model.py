"""Active Inference generative model for flight tasks."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
import os

from .generative_model import GenerativeModel

class FlightModel(GenerativeModel):
    """
    Active Inference generative model for flight tasks.
    
    This model represents the fly's internal model of flight tasks,
    including vision-guided flight and flight imitation.
    """
    
    def __init__(
        self,
        num_states: int = 64,  # Discretized state space
        num_actions: int = 8,   # Discretized action primitives
        precision: float = 2.0,
        include_vision: bool = False,
        vision_size: Tuple[int, int] = (16, 16),
        learning_rate: float = 0.01
    ):
        """
        Initialize the flight model.
        
        Args:
            num_states: Number of discrete states in the model
            num_actions: Number of discrete action primitives
            precision: Precision parameter for action selection
            include_vision: Whether to include vision in observations
            vision_size: Size of vision input (if included)
            learning_rate: Learning rate for updating model parameters
        """
        # The observation space depends on whether vision is included
        self.include_vision = include_vision
        self.vision_size = vision_size
        
        # Basic proprioceptive observation space:
        # - Wing angles (2 wings * 3 angles)
        # - Body orientation (3 axes)
        # - Velocity (3 dimensions)
        num_proprioceptive = 2 * 3 + 3 + 3
        
        # If vision is included, add vision dimensions
        if include_vision:
            # We'll use a lower-dimensional representation of vision
            # rather than raw pixels for computational feasibility
            num_vision_features = 16  # Simplified vision encoding
            num_observations = num_proprioceptive + num_vision_features
            self.vision_encoder = self._create_vision_encoder(vision_size, num_vision_features)
        else:
            num_observations = num_proprioceptive
            self.vision_encoder = None
        
        # Call parent constructor
        super().__init__(num_states, num_observations, num_actions, precision)
        
        self.learning_rate = learning_rate
        
        # Create structured state and action spaces
        self._setup_state_representation()
        self._setup_flight_actions()
        
        # Create models based on flight dynamics
        self._setup_structured_transition_model()
        self._setup_preferences()
    
    def _create_vision_encoder(self, vision_size: Tuple[int, int], num_features: int) -> Callable:
        """
        Create a simple encoder function for vision.
        
        Args:
            vision_size: Size of vision input (height, width)
            num_features: Number of features to extract
            
        Returns:
            Encoder function that converts vision to features
        """
        # A more complex implementation would use CNNs or similar
        # For simplicity, we'll use a function that extracts basic statistics
        
        def encode_vision(vision_input: jnp.ndarray) -> jnp.ndarray:
            # Reshape if needed
            if len(vision_input.shape) > 2:
                # Grayscale conversion if needed (assuming RGB input)
                if vision_input.shape[-1] == 3:
                    vision_input = jnp.mean(vision_input, axis=-1)
            
            # Resize to expected dimensions if needed
            if vision_input.shape != vision_size:
                # In a real implementation, would use proper resizing
                # Here we'll just use a placeholder
                vision_input = vision_input[:vision_size[0], :vision_size[1]]
            
            # Compute features:
            features = []
            
            # 1. Mean brightness in each quadrant (4 features)
            h, w = vision_input.shape
            h_mid, w_mid = h // 2, w // 2
            
            top_left = jnp.mean(vision_input[:h_mid, :w_mid])
            top_right = jnp.mean(vision_input[:h_mid, w_mid:])
            bottom_left = jnp.mean(vision_input[h_mid:, :w_mid])
            bottom_right = jnp.mean(vision_input[h_mid:, w_mid:])
            
            features.extend([top_left, top_right, bottom_left, bottom_right])
            
            # 2. Mean brightness in horizontal thirds (3 features)
            h_third = h // 3
            upper = jnp.mean(vision_input[:h_third, :])
            middle = jnp.mean(vision_input[h_third:2*h_third, :])
            lower = jnp.mean(vision_input[2*h_third:, :])
            
            features.extend([upper, middle, lower])
            
            # 3. Horizontal and vertical brightness gradients (2 features)
            horiz_grad = jnp.mean(vision_input[:, :w_mid]) - jnp.mean(vision_input[:, w_mid:])
            vert_grad = jnp.mean(vision_input[:h_mid, :]) - jnp.mean(vision_input[h_mid:, :])
            
            features.extend([horiz_grad, vert_grad])
            
            # 4. Standard deviation as texture measure (1 feature)
            std_dev = jnp.std(vision_input)
            features.append(std_dev)
            
            # 5. Add more features to reach num_features
            # For simplicity, we'll add some basic statistics
            min_val = jnp.min(vision_input)
            max_val = jnp.max(vision_input)
            median_val = jnp.median(vision_input)
            
            features.extend([min_val, max_val, median_val])
            
            # Ensure we have the exact number of features requested
            while len(features) < num_features:
                features.append(0.0)
            
            features = features[:num_features]
            
            return jnp.array(features)
        
        return encode_vision
    
    def _setup_state_representation(self) -> None:
        """Set up a structured representation of flight states."""
        # States represent combinations of:
        # - Horizontal position (4 regions)
        # - Altitude (4 levels)
        # - Body orientation (4 orientations)
        
        # Create meaningful labels for debugging/visualization
        self.state_labels = []
        
        for pos in ['left', 'center_left', 'center_right', 'right']:
            for alt in ['low', 'medium_low', 'medium_high', 'high']:
                for orient in ['level', 'banking_left', 'banking_right', 'pitching']:
                    self.state_labels.append(f"{pos}_{alt}_{orient}")
    
    def _setup_flight_actions(self) -> None:
        """Set up action primitives for flight control."""
        # Define basic flight maneuver primitives
        self.action_labels = [
            'forward_thrust', 'climb', 'descend', 'bank_left',
            'bank_right', 'hover', 'slow_down', 'stabilize'
        ]
        
        # In a real implementation, these would map to detailed wing control patterns
        # For now, we'll create simplified mappings to wing control parameters
        
        # Flight control requires control over:
        # - Left wing: amplitude, frequency, offset
        # - Right wing: amplitude, frequency, offset
        total_action_dim = 6  # 2 wings * 3 parameters
        
        # Create mapping from discrete actions to continuous control vector
        self.action_mapping_matrix = jnp.zeros((self.num_actions, total_action_dim))
        
        # Forward thrust: symmetric strong wing beats
        self.action_mapping_matrix = self.action_mapping_matrix.at[0].set(
            jnp.array([0.8, 1.0, 0.0, 0.8, 1.0, 0.0])  # high amplitude, high frequency, neutral offset
        )
        
        # Climb: symmetric wing beats with upward bias
        self.action_mapping_matrix = self.action_mapping_matrix.at[1].set(
            jnp.array([0.9, 0.9, 0.2, 0.9, 0.9, 0.2])  # high amplitude, medium frequency, positive offset
        )
        
        # Descend: reduced power with downward bias
        self.action_mapping_matrix = self.action_mapping_matrix.at[2].set(
            jnp.array([0.5, 0.7, -0.1, 0.5, 0.7, -0.1])  # medium amplitude, medium frequency, negative offset
        )
        
        # Bank left: asymmetric wing beats
        self.action_mapping_matrix = self.action_mapping_matrix.at[3].set(
            jnp.array([0.6, 0.8, 0.0, 0.9, 0.8, 0.1])  # stronger right wing
        )
        
        # Bank right: asymmetric wing beats (opposite of bank left)
        self.action_mapping_matrix = self.action_mapping_matrix.at[4].set(
            jnp.array([0.9, 0.8, 0.1, 0.6, 0.8, 0.0])  # stronger left wing
        )
        
        # Hover: balanced medium power
        self.action_mapping_matrix = self.action_mapping_matrix.at[5].set(
            jnp.array([0.7, 0.7, 0.1, 0.7, 0.7, 0.1])  # medium amplitude, medium frequency, slight positive offset
        )
        
        # Slow down: reduced power
        self.action_mapping_matrix = self.action_mapping_matrix.at[6].set(
            jnp.array([0.4, 0.5, 0.0, 0.4, 0.5, 0.0])  # low amplitude, low frequency, neutral offset
        )
        
        # Stabilize: corrective action to level out
        self.action_mapping_matrix = self.action_mapping_matrix.at[7].set(
            jnp.array([0.6, 0.6, 0.0, 0.6, 0.6, 0.0])  # medium amplitude, medium frequency, neutral offset
        )
    
    def _setup_structured_transition_model(self) -> None:
        """Set up a structured transition model based on flight dynamics."""
        # Initialize with uniform transitions
        self.transition_model = jnp.ones((self.num_states, self.num_states, self.num_actions)) / self.num_states
        
        # In a full implementation, we would construct a more accurate transition model
        # based on aerodynamics for each action. Here's a simplified example:
        
        # Example: Forward thrust tends to maintain altitude but move position forward
        forward_thrust_idx = 0
        for i in range(self.num_states):
            # Parse the current state
            state_label = self.state_labels[i]
            
            # For forward thrust, find states that are one position ahead
            next_positions = {
                'left': 'center_left',
                'center_left': 'center_right',
                'center_right': 'right'
            }
            
            # Keep the same altitude and orientation
            for j in range(self.num_states):
                target_label = self.state_labels[j]
                
                # Get components of current state
                parts = state_label.split('_')
                pos = parts[0]
                alt = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                orient = parts[-1]
                
                # Check if target state matches expected transition
                if pos in next_positions:
                    target_parts = target_label.split('_')
                    target_pos = target_parts[0]
                    target_alt = '_'.join(target_parts[1:-1]) if len(target_parts) > 2 else target_parts[1]
                    target_orient = target_parts[-1]
                    
                    # If position advances, altitude stays same, orientation stays same
                    if (target_pos == next_positions[pos] and
                        target_alt == alt and
                        target_orient == orient):
                        
                        # Increase transition probability to this state
                        self.transition_model = self.transition_model.at[j, i, forward_thrust_idx].set(0.3)
            
            # Normalize to keep probabilities
            self.transition_model = self.transition_model.at[:, i, forward_thrust_idx].set(
                self.transition_model[:, i, forward_thrust_idx] / jnp.sum(self.transition_model[:, i, forward_thrust_idx])
            )
        
        # Similar logic would be applied for other actions
        # This is simplified for illustration purposes
    
    def _setup_preferences(self) -> None:
        """Set up preferences for flight observations."""
        # Initialize with uniform preferences
        self.preferred_observations = jnp.ones(self.num_observations) / self.num_observations
        
        # For flight tasks, we typically prefer:
        # - Stable orientation (balanced wing angles)
        # - Specific altitude and position (task dependent)
        
        # Indices for wing angles in observation vector
        wing_indices = list(range(6))  # First 6 elements
        
        # Prefer balanced, active wing movements
        for idx in wing_indices:
            # Higher values for amplitude and frequency
            if idx % 3 < 2:  
                self.preferred_observations = self.preferred_observations.at[idx].set(0.2)
        
        # Indices for orientation in observation vector
        orient_indices = list(range(6, 9))
        
        # Prefer level flight (orientation close to neutral)
        for idx in orient_indices:
            self.preferred_observations = self.preferred_observations.at[idx].set(0.15)
        
        # If vision is included, prefer certain vision features
        if self.include_vision:
            # Prefer high contrast in forward field (useful for navigation)
            # Assuming contrast features are at specific indices
            vision_indices = list(range(12, self.num_observations))
            for idx in vision_indices[:4]:  # First few vision features
                self.preferred_observations = self.preferred_observations.at[idx].set(0.1)
        
        # Normalize
        self.preferred_observations = self.preferred_observations / jnp.sum(self.preferred_observations)
    
    def observation_mapping(self, environment_observation: Any) -> jnp.ndarray:
        """
        Map environment observations to the format expected by the generative model.
        
        Args:
            environment_observation: Observation from the flybody environment
            
        Returns:
            Observation vector in the format expected by the model
        """
        # Extract relevant components from the environment observation
        obs_vector = []
        
        # Process wing angles
        if isinstance(environment_observation, dict) and 'wing_angles' in environment_observation:
            wing_angles = environment_observation['wing_angles']
            # Assuming 6 values (3 per wing)
            obs_vector.extend(wing_angles[:6])
        else:
            # Default values
            obs_vector.extend([0.0] * 6)
        
        # Process body orientation
        if isinstance(environment_observation, dict) and 'orientation' in environment_observation:
            orientation = environment_observation['orientation']
            obs_vector.extend(orientation[:3])
        else:
            # Default values
            obs_vector.extend([0.0] * 3)
        
        # Process velocity
        if isinstance(environment_observation, dict) and 'velocity' in environment_observation:
            velocity = environment_observation['velocity']
            obs_vector.extend(velocity[:3])
        else:
            # Default values
            obs_vector.extend([0.0] * 3)
        
        # Process vision if included
        if self.include_vision and self.vision_encoder is not None:
            if isinstance(environment_observation, dict) and 'vision' in environment_observation:
                vision = environment_observation['vision']
                vision_features = self.vision_encoder(vision)
                obs_vector.extend(vision_features)
            else:
                # Default values (zeros for vision features)
                obs_vector.extend([0.0] * 16)  # 16 vision features
        
        return jnp.array(obs_vector)
    
    def action_mapping(self, model_action: int) -> np.ndarray:
        """
        Map model actions to the format expected by the environment.
        
        Args:
            model_action: Action selected by the model
            
        Returns:
            Action in the format expected by the environment
        """
        # Convert the discrete action to a continuous vector using the mapping matrix
        continuous_action = self.action_mapping_matrix[model_action]
        
        # Note: In a real implementation, this would need to be scaled and possibly
        # restructured to match the exact action space of the flybody environment
        
        return np.array(continuous_action)
    
    def update_from_experience(self, 
                              state: int, 
                              action: int, 
                              next_state: int, 
                              observation: jnp.ndarray) -> None:
        """
        Update the model based on experience.
        
        Args:
            state: Current state index
            action: Action taken
            next_state: Resulting state index
            observation: Observation received
        """
        # Update transition model
        self.transition_model = self.transition_model.at[next_state, state, action].set(
            self.transition_model[next_state, state, action] + self.learning_rate
        )
        
        # Normalize
        self.transition_model = self.transition_model.at[:, state, action].set(
            self.transition_model[:, state, action] / jnp.sum(self.transition_model[:, state, action])
        )
        
        # Update likelihood model
        for o in range(self.num_observations):
            self.likelihood_model = self.likelihood_model.at[o, next_state].set(
                self.likelihood_model[o, next_state] + self.learning_rate * observation[o]
            )
        
        # Normalize columns
        for s in range(self.num_states):
            self.likelihood_model = self.likelihood_model.at[:, s].set(
                self.likelihood_model[:, s] / jnp.sum(self.likelihood_model[:, s])
            )
    
    def select_target_state(self, target_description: str) -> int:
        """
        Find the state index that best matches a target description.
        
        Args:
            target_description: String description of desired state
            
        Returns:
            Index of the matching state
        """
        for i, label in enumerate(self.state_labels):
            if all(term in label for term in target_description.split('_')):
                return i
        
        # If no exact match, return the first state (default)
        return 0
    
    def set_target_preferences(self, target_state: int, strength: float = 0.5) -> None:
        """
        Update preference distribution to favor observations associated with a target state.
        
        Args:
            target_state: Index of target state
            strength: How strongly to prefer this state (0-1)
        """
        # Get expected observations for target state
        target_obs = self.likelihood_model[:, target_state]
        
        # Blend with current preferences
        blend_ratio = jnp.clip(strength, 0.0, 1.0)
        new_prefs = (1 - blend_ratio) * self.preferred_observations + blend_ratio * target_obs
        
        # Normalize and update
        new_prefs = new_prefs / jnp.sum(new_prefs)
        self.preferred_observations = new_prefs 