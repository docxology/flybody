"""Active Inference generative model for the walk-on-ball task."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Any
import os

from .generative_model import GenerativeModel

class WalkOnBallModel(GenerativeModel):
    """
    Active Inference generative model for the walk-on-ball task.
    
    This model represents the fly's internal model of the walk-on-ball task,
    where the goal is to maintain balance while walking on a spherical treadmill.
    """
    
    def __init__(
        self,
        num_states: int = 64,
        num_actions: int = 8,
<<<<<<< HEAD
        num_observations: int = 23,  # Update to match our actual observation vector
        action_dim: int = 59,  # Full action space dimension for the fly
=======
        leg_action_dim: int = 6,  # Simplified action space per leg
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        precision: float = 2.0,
        learning_rate: float = 0.01,
        action_mapping_file: Optional[str] = None
    ):
        """
        Initialize the walk-on-ball model.
        
        Args:
            num_states: Number of discrete states in the model (representing position/orientation)
            num_actions: Number of discrete actions (simplified control primitives)
<<<<<<< HEAD
            num_observations: Number of observation dimensions
            action_dim: Dimensionality of the full action space (59 for walk_on_ball task)
=======
            leg_action_dim: Dimensionality of individual leg controls
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
            precision: Precision parameter for action selection
            learning_rate: Learning rate for updating the model parameters
            action_mapping_file: Optional file containing the action mapping matrix
        """
<<<<<<< HEAD
        # Actual observation dimensions from our environment:
        # - 16 proprioceptive joints (simplified from leg joints)
        # - 2 ball velocity components (x, y)
        # - 3 orientation components (roll, pitch, yaw)
        # Total: 21 dimensions
        
        # Update to the correct observation size
        num_observations = 21
=======
        # The observation space includes:
        # - Proprioceptive feedback from each leg (6 legs * 3 joints)
        # - Ball rotation sensors (2)
        # - Body orientation sensors (3)
        num_observations = 6 * 3 + 2 + 3
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        
        # Call parent constructor
        super().__init__(num_states, num_observations, num_actions, precision)
        
<<<<<<< HEAD
        self.action_dim = action_dim
=======
        self.leg_action_dim = leg_action_dim
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        self.learning_rate = learning_rate
        
        # Create more informative state representation
        self._setup_state_representation()
        
        # Create action primitives for the walk-on-ball task
        self._setup_action_primitives(action_mapping_file)
        
        # Create observation bins for discretizing continuous observations
        self._setup_observation_discretization()
        
        # Create a more structured transition model based on physics
        self._setup_structured_transition_model()
        
        # Set up specific preferences for the walk-on-ball task
        self._setup_preferences()
    
    def _setup_state_representation(self) -> None:
        """Set up a more structured state representation for the model."""
        # States represent combinations of:
        # - Position on ball (4 quadrants)
        # - Rotation of ball (4 directions)
        # - Body orientation (4 states: level, tilted left, right, forward)
        
        # Create meaningful labels for debugging/visualization
        self.state_labels = []
        
        for pos in ['quadrant1', 'quadrant2', 'quadrant3', 'quadrant4']:
            for rot in ['no_rotation', 'rolling_fwd', 'rolling_back', 'rolling_side']:
                for orient in ['level', 'tilted_left', 'tilted_right', 'tilted_fwd']:
                    self.state_labels.append(f"{pos}_{rot}_{orient}")
    
    def _setup_action_primitives(self, action_mapping_file: Optional[str] = None) -> None:
        """
        Set up action primitives for the fly.
        
        Args:
            action_mapping_file: Optional file containing the action mapping matrix
        """
<<<<<<< HEAD
        # The full action space has 59 dimensions:
        # - 6 adhere_claw actions (one per leg)
        # - 3 head actions (head_abduct, head_twist, head)
        # - 2 abdomen actions (abdomen_abduct, abdomen)
        # - 48 leg joint actions (8 per leg * 6 legs)
        
        # Create labels for our simplified action space
=======
        # Full action space would be 6 legs with multiple joints
        # We simplify to 8 basic movement patterns: 
        # forward, backward, rotate-left, rotate-right, 
        # sidestep-left, sidestep-right, stop, stabilize
        
        # Create labels for actions
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        self.action_labels = [
            'forward', 'backward', 'rotate_left', 'rotate_right',
            'sidestep_left', 'sidestep_right', 'stop', 'stabilize'
        ]
        
        # Create mapping from discrete actions to continuous control vector
<<<<<<< HEAD
        self.action_mapping_matrix = jnp.zeros((self.num_actions, self.action_dim))
=======
        # Each action maps to a vector of shape (6 legs * leg_action_dim)
        total_action_dim = 6 * self.leg_action_dim
        self.action_mapping_matrix = jnp.zeros((self.num_actions, total_action_dim))
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        
        # If a mapping file is provided, load it
        if action_mapping_file and os.path.exists(action_mapping_file):
            self.action_mapping_matrix = jnp.array(np.load(action_mapping_file))
        else:
            # Otherwise, create a simplified mapping
<<<<<<< HEAD
            # These are placeholder patterns that would be refined in a real implementation
            
            # Define some default values for non-leg actions
            default_head = jnp.array([0.0, 0.0, 0.0])  # head_abduct, head_twist, head
            default_abdomen = jnp.array([0.0, 0.0])    # abdomen_abduct, abdomen
            
            # Create pattern templates for leg joints (per leg)
            # Each leg has 9 joints: adhere_claw, coxa_abduct, coxa_twist, coxa, femur_twist, femur, tibia, tarsus, tarsus2
            
            # Forward walking pattern (tripod gait)
            forward_left_stance = jnp.array([0.5, 0.3, 0.0, 0.5, 0.0, 0.5, 0.3, 0.2, 0.0])  # Stance phase (left legs)
            forward_right_swing = jnp.array([0.0, -0.3, 0.0, -0.3, 0.0, -0.3, -0.1, -0.1, 0.0])  # Swing phase (right legs)
            
            # Backward walking pattern
            backward_left_swing = jnp.array([0.0, -0.3, 0.0, -0.3, 0.0, -0.3, -0.1, -0.1, 0.0])  # Swing phase (left legs)
            backward_right_stance = jnp.array([0.5, 0.3, 0.0, 0.5, 0.0, 0.5, 0.3, 0.2, 0.0])  # Stance phase (right legs)
            
            # Rotate patterns
            rotate_left_stance = jnp.array([0.5, 0.3, 0.2, 0.4, 0.1, 0.3, 0.2, 0.1, 0.0])
            rotate_right_stance = jnp.array([0.5, -0.3, -0.2, 0.4, -0.1, 0.3, 0.2, 0.1, 0.0])
            
            # Sidestep patterns
            sidestep_left = jnp.array([0.3, 0.4, 0.0, 0.3, 0.0, 0.2, 0.1, 0.0, 0.0])
            sidestep_right = jnp.array([0.3, -0.4, 0.0, 0.3, 0.0, 0.2, 0.1, 0.0, 0.0])
            
            # Stop pattern
            stop_neutral = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Stabilize pattern
            stabilize = jnp.array([0.5, 0.0, 0.0, 0.3, 0.0, 0.4, 0.3, 0.1, 0.0])
            
            # Helper function to create full action vectors
            def create_action_vector(pattern_front_left, pattern_front_right, 
                                     pattern_middle_left, pattern_middle_right,
                                     pattern_back_left, pattern_back_right):
                # Adhere claws (first 6 dimensions)
                adhere = jnp.array([
                    pattern_front_left[0], pattern_front_right[0],
                    pattern_middle_left[0], pattern_middle_right[0],
                    pattern_back_left[0], pattern_back_right[0]
                ])
                
                # Head and abdomen (next 5 dimensions)
                head_abdomen = jnp.concatenate([default_head, default_abdomen])
                
                # Leg joints (remaining 48 dimensions)
                # Each leg has 8 joints (excluding adhere_claw, which we already handled)
                front_left_joints = pattern_front_left[1:]
                front_right_joints = pattern_front_right[1:]
                middle_left_joints = pattern_middle_left[1:]
                middle_right_joints = pattern_middle_right[1:]
                back_left_joints = pattern_back_left[1:]
                back_right_joints = pattern_back_right[1:]
                
                leg_joints = jnp.concatenate([
                    front_left_joints, front_right_joints,
                    middle_left_joints, middle_right_joints,
                    back_left_joints, back_right_joints
                ])
                
                # Combine all parts
                return jnp.concatenate([adhere, head_abdomen, leg_joints])
            
            # Forward action (tripod gait: front-left, middle-right, back-left stance)
            self.action_mapping_matrix = self.action_mapping_matrix.at[0].set(
                create_action_vector(
                    forward_left_stance, forward_right_swing,
                    forward_right_swing, forward_left_stance,
                    forward_left_stance, forward_right_swing
                )
            )
            
            # Backward action (reverse tripod gait)
            self.action_mapping_matrix = self.action_mapping_matrix.at[1].set(
                create_action_vector(
                    backward_left_swing, backward_right_stance,
                    backward_right_stance, backward_left_swing,
                    backward_left_swing, backward_right_stance
                )
            )
            
            # Rotate left
            self.action_mapping_matrix = self.action_mapping_matrix.at[2].set(
                create_action_vector(
                    rotate_left_stance, rotate_left_stance,
                    rotate_left_stance, rotate_left_stance,
                    rotate_left_stance, rotate_left_stance
                )
            )
            
            # Rotate right
            self.action_mapping_matrix = self.action_mapping_matrix.at[3].set(
                create_action_vector(
                    rotate_right_stance, rotate_right_stance,
                    rotate_right_stance, rotate_right_stance,
                    rotate_right_stance, rotate_right_stance
                )
            )
            
            # Sidestep left
            self.action_mapping_matrix = self.action_mapping_matrix.at[4].set(
                create_action_vector(
                    sidestep_left, sidestep_left,
                    sidestep_left, sidestep_left,
                    sidestep_left, sidestep_left
                )
            )
            
            # Sidestep right
            self.action_mapping_matrix = self.action_mapping_matrix.at[5].set(
                create_action_vector(
                    sidestep_right, sidestep_right,
                    sidestep_right, sidestep_right,
                    sidestep_right, sidestep_right
                )
            )
            
            # Stop (neutral position)
            self.action_mapping_matrix = self.action_mapping_matrix.at[6].set(
                create_action_vector(
                    stop_neutral, stop_neutral,
                    stop_neutral, stop_neutral,
                    stop_neutral, stop_neutral
                )
            )
            
            # Stabilize
            self.action_mapping_matrix = self.action_mapping_matrix.at[7].set(
                create_action_vector(
                    stabilize, stabilize,
                    stabilize, stabilize,
                    stabilize, stabilize
                )
            )
            
            # Verify the shape matches what we expect
            print(f"Action mapping matrix shape: {self.action_mapping_matrix.shape}")
            assert self.action_mapping_matrix.shape == (self.num_actions, self.action_dim), \
                f"Expected shape ({self.num_actions}, {self.action_dim}) but got {self.action_mapping_matrix.shape}"
=======
            # These are simplified patterns that would be refined in real implementation
            
            # Forward walking pattern (tripod gait)
            self.action_mapping_matrix = self.action_mapping_matrix.at[0].set(
                jnp.array([0.5, 0.3, 0.2] * 2 + [-0.3, -0.1, -0.1] * 2 + [0.5, 0.3, 0.2] * 2)
            )
            
            # Backward walking pattern (reverse tripod)
            self.action_mapping_matrix = self.action_mapping_matrix.at[1].set(
                jnp.array([-0.5, -0.3, -0.2] * 2 + [0.3, 0.1, 0.1] * 2 + [-0.5, -0.3, -0.2] * 2)
            )
            
            # Rotate left pattern
            self.action_mapping_matrix = self.action_mapping_matrix.at[2].set(
                jnp.array([0.3, 0.2, 0.1] * 3 + [-0.3, -0.2, -0.1] * 3)
            )
            
            # Rotate right pattern
            self.action_mapping_matrix = self.action_mapping_matrix.at[3].set(
                jnp.array([-0.3, -0.2, -0.1] * 3 + [0.3, 0.2, 0.1] * 3)
            )
            
            # Sidestep left pattern
            self.action_mapping_matrix = self.action_mapping_matrix.at[4].set(
                jnp.array([0.0, 0.3, 0.2] * 3 + [0.0, -0.3, -0.2] * 3)
            )
            
            # Sidestep right pattern
            self.action_mapping_matrix = self.action_mapping_matrix.at[5].set(
                jnp.array([0.0, -0.3, -0.2] * 3 + [0.0, 0.3, 0.2] * 3)
            )
            
            # Stop pattern (return to neutral)
            self.action_mapping_matrix = self.action_mapping_matrix.at[6].set(
                jnp.zeros(total_action_dim)
            )
            
            # Stabilize pattern (slightly bend legs for stability)
            self.action_mapping_matrix = self.action_mapping_matrix.at[7].set(
                jnp.array([0.0, 0.1, 0.2] * 6)
            )
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
    
    def _setup_observation_discretization(self) -> None:
        """Set up discretization bins for continuous observations."""
        # Define bins for proprioceptive feedback
        self.prop_bins = np.linspace(-1.0, 1.0, 5)
        
        # Define bins for ball rotation
        self.ball_rot_bins = np.linspace(-0.5, 0.5, 5)
        
        # Define bins for body orientation
        self.orient_bins = np.linspace(-0.5, 0.5, 5)
    
    def _setup_structured_transition_model(self) -> None:
        """Set up a transition model based on physics principles."""
        # Initialize with uniform transitions
        self.transition_model = jnp.ones((self.num_states, self.num_states, self.num_actions)) / self.num_states
        
        # In a full implementation, we would construct a more accurate transition model
        # based on physics principles for each action
        
        # For illustration, we'll demonstrate how actions should affect states differently
        # (In reality, this would be learned from experience or designed more carefully)
        
        # Example: Forward action is more likely to move to forward states
        forward_idx = 0
        for i in range(self.num_states):
            # Parse the current state
            state_label = self.state_labels[i]
            
            # Find states that would result from moving forward
            forward_states = []
            for j in range(self.num_states):
                target_label = self.state_labels[j]
                # If the target state shows forward ball rotation
                if 'rolling_fwd' in target_label:
                    forward_states.append(j)
            
            # Increase transition probability to forward states
            if forward_states:
                for j in forward_states:
                    self.transition_model = self.transition_model.at[j, i, forward_idx].set(0.2)
                
                # Normalize
                self.transition_model = self.transition_model.at[:, i, forward_idx].set(
                    self.transition_model[:, i, forward_idx] / jnp.sum(self.transition_model[:, i, forward_idx])
                )
        
        # Similar logic would be applied for other actions
        # This is simplified for illustration purposes
    
    def _setup_preferences(self) -> None:
        """Set up preferences for specific observations in the walk-on-ball task."""
        # Initialize with uniform preferences
        self.preferred_observations = jnp.ones(self.num_observations) / self.num_observations
        
        # In walk-on-ball, we prefer:
        # - Balanced body orientation (middle values for orientation sensors)
        # - Controlled ball rotation (small values)
        
        # Prefer balanced orientation (observations indices for orientation)
        orient_indices = list(range(self.num_observations - 3, self.num_observations))
        for idx in orient_indices:
            self.preferred_observations = self.preferred_observations.at[idx].set(0.2)
        
        # Prefer controlled ball rotation (observation indices for ball rotation)
        ball_rot_indices = list(range(self.num_observations - 5, self.num_observations - 3))
        for idx in ball_rot_indices:
            self.preferred_observations = self.preferred_observations.at[idx].set(0.1)
        
        # Normalize
        self.preferred_observations = self.preferred_observations / jnp.sum(self.preferred_observations)
    
    def observation_mapping(self, environment_observation: Any) -> jnp.ndarray:
        """
<<<<<<< HEAD
        Map from environment observations to the model's observation space.
        
        Args:
            environment_observation: Observation from the environment (OrderedDict)
            
        Returns:
            Observation vector for the model's internal representation (21 dimensions)
        """
        # Print observation structure for debugging
        print(f"Environment observation type: {type(environment_observation)}")
        print(f"Environment observation keys: {environment_observation.keys()}")
        
        # Initialize components with zeros
        proprioceptive = np.zeros(16, dtype=np.float32)  # 16 joints
        ball_rotation = np.zeros(2, dtype=np.float32)    # 2 ball velocity components
        orientation = np.zeros(3, dtype=np.float32)      # 3 orientation components
        
        # Extract joint positions if available
        if 'walker/joints_pos' in environment_observation:
            joint_pos = environment_observation['walker/joints_pos']
            print(f"Joint positions shape: {joint_pos.shape}")
            # Select key leg joints for simplified representation
            if len(joint_pos) >= 59:
                # Take every 5th joint starting from leg joints (index 11)
                indices = [11 + i*5 for i in range(16) if 11 + i*5 < len(joint_pos)]
                if len(indices) > 16:
                    indices = indices[:16]  # Ensure we don't exceed 16
                proprioceptive[:len(indices)] = joint_pos[indices]
        
        # Extract ball velocity if available
        if 'walker/ball_qvel' in environment_observation:
            ball_qvel = environment_observation['walker/ball_qvel']
            print(f"Ball velocity: {ball_qvel}")
            if len(ball_qvel) >= 2:
                ball_rotation = ball_qvel[:2]  # Use the first two components
        
        # Extract orientation if available
        if 'walker/world_zaxis' in environment_observation:
            # Use world_zaxis as orientation (gives us the gravity direction)
            zaxis = environment_observation['walker/world_zaxis']
            if len(zaxis) >= 3:
                orientation = zaxis[:3]
        elif 'walker/orientation' in environment_observation:
            ori = environment_observation['walker/orientation']
            if len(ori) >= 3:
                orientation = ori[:3]
        
        # Check gyro data for rotation information
        if 'walker/gyro' in environment_observation and np.all(orientation == 0):
            gyro = environment_observation['walker/gyro']
            if len(gyro) >= 3:
                # Use gyro for orientation if we don't have better orientation data
                orientation = gyro[:3]
        
        # Combine into a single observation vector of exactly 21 dimensions
        obs_vector = np.concatenate([proprioceptive, ball_rotation, orientation])
        
        # Double-check dimension
        assert obs_vector.shape == (21,), f"Expected observation vector of shape (21,), got {obs_vector.shape}"
        
        print(f"Final observation vector shape: {obs_vector.shape}")
        
        # Convert to jax numpy array
=======
        Map environment observations to the format expected by the generative model.
        
        Args:
            environment_observation: Observation from the flybody environment
            
        Returns:
            Observation vector in the format expected by the model
        """
        # Extract relevant components from the environment observation
        # Depending on the exact format from the environment, this may need adjustment
        
        obs_vector = []
        
        # Process proprioceptive feedback (leg joint angles)
        if isinstance(environment_observation, dict) and 'joints' in environment_observation:
            joint_angles = environment_observation['joints']
            # Take the first 18 values (6 legs * 3 joints)
            proprioception = joint_angles[:18]
            obs_vector.extend(proprioception)
        else:
            # If not available, use zeros
            obs_vector.extend([0.0] * 18)
        
        # Process ball rotation
        if isinstance(environment_observation, dict) and 'ball_rotation' in environment_observation:
            ball_rotation = environment_observation['ball_rotation']
            obs_vector.extend(ball_rotation)
        else:
            # If not available, use zeros
            obs_vector.extend([0.0, 0.0])
        
        # Process body orientation
        if isinstance(environment_observation, dict) and 'orientation' in environment_observation:
            orientation = environment_observation['orientation']
            obs_vector.extend(orientation)
        else:
            # If not available, use zeros
            obs_vector.extend([0.0, 0.0, 0.0])
        
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        return jnp.array(obs_vector)
    
    def action_mapping(self, model_action: int) -> np.ndarray:
        """
<<<<<<< HEAD
        Map from model's discrete action to environment action.
        
        Args:
            model_action: Index of action in the model's action space
            
        Returns:
            Action vector for the environment action space
        """
        # Ensure the action index is valid
        if model_action < 0 or model_action >= self.num_actions:
            model_action = 6  # Default to 'stop' action if out of range
        
        # Map to the continuous action vector
        action_vector = np.array(self.action_mapping_matrix[model_action])
        
        # Clip to ensure values are within the allowed range
        action_vector = np.clip(action_vector, -1.0, 1.0)
        
        print(f"Selected action: {self.action_labels[model_action]}, shape: {action_vector.shape}")
        
        return action_vector
=======
        Map model actions to the format expected by the environment.
        
        Args:
            model_action: Action selected by the model (index)
            
        Returns:
            Action in the format expected by the environment
        """
        # Convert the discrete action to the continuous action vector using the mapping matrix
        continuous_action = self.action_mapping_matrix[model_action]
        
        # Note: In a real implementation with flybody, this might need scaling and reshaping
        # depending on the exact action space expected by the environment
        
        # Convert to numpy array for environment
        return np.array(continuous_action)
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
    
    def update_from_experience(self, 
                              state: int, 
                              action: int, 
                              next_state: int, 
                              observation: jnp.ndarray) -> None:
        """
        Update the model based on experience (simple learning).
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            observation: Observation received
        """
        # Update transition model based on state transition
        # Simple update: increase probability of observed transition
        self.transition_model = self.transition_model.at[next_state, state, action].set(
            self.transition_model[next_state, state, action] + self.learning_rate
        )
        
        # Normalize to maintain probability distribution
        self.transition_model = self.transition_model.at[:, state, action].set(
            self.transition_model[:, state, action] / jnp.sum(self.transition_model[:, state, action])
        )
        
        # Update likelihood model based on observation
        # This is a simplified update; in practice would be more sophisticated
        # Convert observation to state probability
        obs_probs = jnp.zeros(self.num_states)
        obs_probs = obs_probs.at[next_state].set(1.0)  # Assume perfect observation for simplicity
        
        # Update likelihood
        for o in range(self.num_observations):
            # Increase likelihood of this observation for this state
            self.likelihood_model = self.likelihood_model.at[o, next_state].set(
                self.likelihood_model[o, next_state] + self.learning_rate * observation[o]
            )
        
        # Normalize columns to maintain probability interpretation
        for s in range(self.num_states):
            self.likelihood_model = self.likelihood_model.at[:, s].set(
                self.likelihood_model[:, s] / jnp.sum(self.likelihood_model[:, s])
            )
    
    def state_to_index(self, position: int, rotation: int, orientation: int) -> int:
        """
        Convert state components to state index.
        
        Args:
            position: Position component (0-3)
            rotation: Ball rotation component (0-3)
            orientation: Body orientation component (0-3)
            
        Returns:
            State index
        """
        return position * 16 + rotation * 4 + orientation 