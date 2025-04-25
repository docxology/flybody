"""POMDP model for the flybody agent.

This model implements a POMDP for the flybody, covering perception,
action selection, and learning based on Active Inference principles.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
import os

from .pomdp_model import POMDPModel

class FlybodyPOMDP(POMDPModel):
    """
    POMDP-based generative model for the flybody agent.
    
    This model represents the fly's internal model of its environment,
    including proprioceptive and exteroceptive observations, motor actions,
    and belief updating for active inference.
    """
    
    def __init__(
        self,
        task_name: str,
        include_vision: bool = False,
        num_states: int = 128,
        num_actions: int = 16,
        state_dimensionality: int = 3,
        observation_noise: float = 0.05,
        transition_noise: float = 0.1,
        precision: float = 2.0,
        learning_rate: float = 0.01,
        action_mapping_file: Optional[str] = None
    ):
        """
        Initialize the flybody POMDP model.
        
        Args:
            task_name: Name of the task (e.g., 'walk_on_ball', 'flight')
            include_vision: Whether to include visual observations
            num_states: Number of discrete states in the model
            num_actions: Number of discrete actions
            state_dimensionality: Dimensionality of the state representation
            observation_noise: Noise in the observation model
            transition_noise: Noise in the transition model
            precision: Precision parameter for action selection
            learning_rate: Learning rate for updating model parameters
            action_mapping_file: Optional file containing action mapping matrix
        """
        # Determine observation space based on task and sensors
        if task_name == 'walk_on_ball':
            # Proprioceptive joints (18) + Ball velocity (2) + Orientation (3)
            num_observations = 23
            self.action_dim = 60  # Full FlyBody action space for walk_on_ball
        elif task_name == 'flight':
            # Proprioceptive joints (18) + Air velocity (3) + Orientation (3)
            num_observations = 24
            self.action_dim = 65  # Full FlyBody action space for flight
        else:
            # Default observation space
            num_observations = 20
            self.action_dim = 60
        
        # Add visual observations if included
        if include_vision:
            # Simple visual field with 16 values
            num_observations += 16
        
        # Initialize parent class
        super().__init__(
            num_states=num_states,
            num_observations=num_observations,
            num_actions=num_actions,
            observation_noise=observation_noise,
            transition_noise=transition_noise,
            precision=precision,
            learning_rate=learning_rate
        )
        
        self.task_name = task_name
        self.include_vision = include_vision
        self.state_dimensionality = state_dimensionality
        
        # Setup specific to flybody
        self._setup_state_representation()
        self._setup_action_mapping(action_mapping_file)
        self._setup_observation_mapping()
        self._setup_preferences()
        
        # Initialize sensory mapping components
        self._setup_sensory_precision()
    
    def _setup_state_representation(self) -> None:
        """Set up a structured state representation for the flybody."""
        # Define state dimensions based on task
        if self.task_name == 'walk_on_ball':
            # State dimensions: position (x,y), orientation (phi), stability
            self.state_dims = {
                'position_x': 4,  # 4 discrete positions along x
                'position_y': 4,  # 4 discrete positions along y
                'orientation': 4,  # 4 discrete orientations
                'stability': 2    # 2 states: stable/unstable
            }
        elif self.task_name == 'flight':
            # State dimensions: position (x,y,z), orientation (roll,pitch,yaw)
            self.state_dims = {
                'position_x': 4,  # 4 discrete positions along x
                'position_y': 4,  # 4 discrete positions along y 
                'position_z': 4,  # 4 discrete positions along z
                'orientation': 8  # 8 discrete orientations
            }
        else:
            # Default state dimensions
            self.state_dims = {
                'position_x': 4,
                'position_y': 4,
                'orientation': 8
            }
        
        # Create a mapping from flat state index to structured state
        self.state_mapping = {}
        
        # Create meaningful labels for debugging
        self.state_labels = []
        
        # Create an expanding index to map between flat and structured states
        state_idx = 0
        
        # Handle walk_on_ball task
        if self.task_name == 'walk_on_ball':
            for x in range(self.state_dims['position_x']):
                for y in range(self.state_dims['position_y']):
                    for o in range(self.state_dims['orientation']):
                        for s in range(self.state_dims['stability']):
                            # Map flat index to structured state
                            self.state_mapping[state_idx] = {
                                'position_x': x,
                                'position_y': y,
                                'orientation': o,
                                'stability': s
                            }
                            
                            # Create human-readable label
                            orient_labels = ['north', 'east', 'south', 'west']
                            stability_labels = ['stable', 'unstable']
                            label = f"pos({x},{y})_orient({orient_labels[o]})_{stability_labels[s]}"
                            self.state_labels.append(label)
                            
                            state_idx += 1
        
        # Handle flight task
        elif self.task_name == 'flight':
            for x in range(self.state_dims['position_x']):
                for y in range(self.state_dims['position_y']):
                    for z in range(self.state_dims['position_z']):
                        for o in range(self.state_dims['orientation']):
                            # Map flat index to structured state
                            self.state_mapping[state_idx] = {
                                'position_x': x,
                                'position_y': y,
                                'position_z': z,
                                'orientation': o
                            }
                            
                            # Create human-readable label
                            orient_labels = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
                            label = f"pos({x},{y},{z})_orient({orient_labels[o]})"
                            self.state_labels.append(label)
                            
                            state_idx += 1
        
        # Default labeling for other tasks
        else:
            for x in range(self.state_dims['position_x']):
                for y in range(self.state_dims['position_y']):
                    for o in range(self.state_dims['orientation']):
                        # Map flat index to structured state
                        self.state_mapping[state_idx] = {
                            'position_x': x,
                            'position_y': y,
                            'orientation': o
                        }
                        
                        # Create human-readable label
                        orient_labels = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']
                        label = f"pos({x},{y})_orient({orient_labels[o if o < len(orient_labels) else 0]})"
                        self.state_labels.append(label)
                        
                        state_idx += 1
        
        # Create a structured transition model based on the state representation
        self._setup_structured_transition_model()
    
    def _setup_action_mapping(self, action_mapping_file: Optional[str] = None) -> None:
        """
        Set up action primitives for the flybody.
        
        Args:
            action_mapping_file: Optional file containing action mapping matrix
        """
        # Create labels for simplified action space
        if self.task_name == 'walk_on_ball':
            self.action_labels = [
                'forward', 'backward', 'rotate_left', 'rotate_right',
                'sidestep_left', 'sidestep_right', 'stop', 'stabilize',
                'tripod_gait_1', 'tripod_gait_2', 'wave_gait_1', 'wave_gait_2',
                'climb_up', 'climb_down', 'search', 'reset'
            ]
        elif self.task_name == 'flight':
            self.action_labels = [
                'hover', 'forward', 'backward', 'strafe_left', 'strafe_right',
                'ascend', 'descend', 'roll_left', 'roll_right',
                'pitch_up', 'pitch_down', 'yaw_left', 'yaw_right',
                'bank_left', 'bank_right', 'reset'
            ]
        else:
            # Default action labels
            self.action_labels = [f"action_{i}" for i in range(self.num_actions)]
        
        # Create mapping from discrete actions to continuous control vector
        self.action_mapping_matrix = jnp.zeros((self.num_actions, self.action_dim))
        
        # If mapping file provided, load it
        if action_mapping_file and os.path.exists(action_mapping_file):
            self.action_mapping_matrix = jnp.array(np.load(action_mapping_file))
        else:
            # Otherwise, initialize with random patterns
            # This is a simplified approach - in a real implementation, these would be
            # carefully designed action primitives
            
            # Create random action patterns with some structure
            key = jax.random.PRNGKey(42)
            
            for i in range(self.num_actions):
                # Create a random pattern with some sparsity
                key, subkey = jax.random.split(key)
                action_pattern = jax.random.normal(subkey, (self.action_dim,))
                
                # Make it sparse (70% of values close to zero)
                key, subkey = jax.random.split(key)
                mask = jax.random.uniform(subkey, (self.action_dim,)) < 0.7
                action_pattern = action_pattern * (~mask)
                
                # Normalize
                action_pattern = jnp.tanh(action_pattern)
                
                # Store in mapping matrix
                self.action_mapping_matrix = self.action_mapping_matrix.at[i].set(action_pattern)
    
    def _setup_observation_mapping(self) -> None:
        """Set up observation mapping for the flybody."""
        # Define observation component dimensions based on task
        if self.task_name == 'walk_on_ball':
            self.obs_dims = {
                'proprioception': 18,  # Joint positions
                'exteroception': 2,    # Ball velocity
                'orientation': 3       # Body orientation
            }
        elif self.task_name == 'flight':
            self.obs_dims = {
                'proprioception': 18,  # Joint positions
                'exteroception': 3,    # Air velocity
                'orientation': 3       # Body orientation
            }
        else:
            # Default observation dimensions
            self.obs_dims = {
                'proprioception': 18,
                'exteroception': 0,
                'orientation': 3
            }
        
        # Add visual observations if included
        if self.include_vision:
            self.obs_dims['vision'] = 16
        
        # Create observation bins for discretization
        self.obs_bins = {}
        
        # For each observation dimension, create bins
        for obs_type, dim in self.obs_dims.items():
            if dim > 0:
                self.obs_bins[obs_type] = {}
                for i in range(dim):
                    # 10 bins from -1 to 1
                    self.obs_bins[obs_type][i] = np.linspace(-1, 1, 10)
    
    def _setup_structured_transition_model(self) -> None:
        """Set up a structured transition model based on the state representation."""
        # Create a more physically plausible transition model
        
        # First, create directional movement patterns based on actions
        # This maps actions to changes in state dimensions
        if self.task_name == 'walk_on_ball':
            action_effects = {
                'forward':        {'position_x': 1, 'position_y': 0, 'orientation': 0},
                'backward':       {'position_x': -1, 'position_y': 0, 'orientation': 0},
                'rotate_left':    {'position_x': 0, 'position_y': 0, 'orientation': -1},
                'rotate_right':   {'position_x': 0, 'position_y': 0, 'orientation': 1},
                'sidestep_left':  {'position_x': 0, 'position_y': -1, 'orientation': 0},
                'sidestep_right': {'position_x': 0, 'position_y': 1, 'orientation': 0},
                'stop':           {'position_x': 0, 'position_y': 0, 'orientation': 0},
                'stabilize':      {'position_x': 0, 'position_y': 0, 'orientation': 0, 'stability': 1},
                'tripod_gait_1':  {'position_x': 1, 'position_y': 0, 'orientation': 0},
                'tripod_gait_2':  {'position_x': 1, 'position_y': 0, 'orientation': 0},
                'wave_gait_1':    {'position_x': 0.5, 'position_y': 0, 'orientation': 0},
                'wave_gait_2':    {'position_x': 0.5, 'position_y': 0, 'orientation': 0},
                'climb_up':       {'position_z': 1} if 'position_z' in self.state_dims else {},
                'climb_down':     {'position_z': -1} if 'position_z' in self.state_dims else {},
                'search':         {'position_x': 0, 'position_y': 0, 'orientation': 0},
                'reset':          {'position_x': 0, 'position_y': 0, 'orientation': 0}
            }
        elif self.task_name == 'flight':
            action_effects = {
                'hover':        {'position_x': 0, 'position_y': 0, 'position_z': 0, 'orientation': 0},
                'forward':      {'position_x': 1, 'position_y': 0, 'position_z': 0, 'orientation': 0},
                'backward':     {'position_x': -1, 'position_y': 0, 'position_z': 0, 'orientation': 0},
                'strafe_left':  {'position_x': 0, 'position_y': -1, 'position_z': 0, 'orientation': 0},
                'strafe_right': {'position_x': 0, 'position_y': 1, 'position_z': 0, 'orientation': 0},
                'ascend':       {'position_x': 0, 'position_y': 0, 'position_z': 1, 'orientation': 0},
                'descend':      {'position_x': 0, 'position_y': 0, 'position_z': -1, 'orientation': 0},
                'roll_left':    {'position_x': 0, 'position_y': 0, 'position_z': 0, 'orientation': -1},
                'roll_right':   {'position_x': 0, 'position_y': 0, 'position_z': 0, 'orientation': 1},
                'pitch_up':     {'position_x': 0, 'position_y': 0, 'position_z': 0, 'orientation': 2},
                'pitch_down':   {'position_x': 0, 'position_y': 0, 'position_z': 0, 'orientation': -2},
                'yaw_left':     {'position_x': 0, 'position_y': 0, 'position_z': 0, 'orientation': -4},
                'yaw_right':    {'position_x': 0, 'position_y': 0, 'position_z': 0, 'orientation': 4},
                'bank_left':    {'position_x': 0, 'position_y': -1, 'position_z': 0, 'orientation': -1},
                'bank_right':   {'position_x': 0, 'position_y': 1, 'position_z': 0, 'orientation': 1},
                'reset':        {'position_x': 0, 'position_y': 0, 'position_z': 0, 'orientation': 0}
            }
        else:
            # Default action effects
            action_effects = {label: {} for label in self.action_labels}
        
        # Initialize with small random transitions
        key = jax.random.PRNGKey(0)
        for s1 in range(self.num_states):
            for s2 in range(self.num_states):
                for a in range(self.num_actions):
                    key, subkey = jax.random.split(key)
                    small_prob = jax.random.uniform(subkey, ()) * 0.1
                    self.transition_model = self.transition_model.at[s2, s1, a].set(small_prob)
        
        # Apply structured transitions based on action effects
        for s1 in range(self.num_states):
            for a, label in enumerate(self.action_labels):
                if label in action_effects:
                    # Get the structured effects for this action
                    effects = action_effects[label]
                    
                    # Compute the target state based on effects
                    target_state = dict(self.state_mapping[s1])  # Copy current state
                    
                    # Apply effects
                    for dim, change in effects.items():
                        if dim in target_state:
                            # Apply change with wrapping
                            dim_size = self.state_dims[dim]
                            target_state[dim] = (target_state[dim] + change) % dim_size
                    
                    # Find the flat index for the target state
                    # This is a simplification - in a real implementation, we'd need a more robust mapping
                    target_s2 = None
                    for s2, state_dict in self.state_mapping.items():
                        if all(state_dict[k] == target_state[k] for k in target_state if k in state_dict):
                            target_s2 = s2
                            break
                    
                    # If we found a matching target state, set high transition probability
                    if target_s2 is not None:
                        # Set high probability for the target transition
                        self.transition_model = self.transition_model.at[target_s2, s1, a].set(0.7)
                        
                        # Add some probability to neighboring states for stochasticity
                        for s2 in range(self.num_states):
                            if s2 != target_s2:
                                # Compute "distance" between states in state space
                                dist = 0
                                s2_state = self.state_mapping[s2]
                                for dim in target_state:
                                    if dim in s2_state:
                                        # Manhattan distance in each dimension
                                        dim_size = self.state_dims[dim]
                                        dim_dist = min(
                                            (target_state[dim] - s2_state[dim]) % dim_size,
                                            (s2_state[dim] - target_state[dim]) % dim_size
                                        )
                                        dist += dim_dist
                                
                                # Add probability inversely proportional to distance
                                if dist == 1:  # Adjacent states get higher probability
                                    self.transition_model = self.transition_model.at[s2, s1, a].set(0.1)
        
        # Normalize transitions to ensure valid probability distributions
        for s1 in range(self.num_states):
            for a in range(self.num_actions):
                column_sum = jnp.sum(self.transition_model[:, s1, a])
                if column_sum > 0:
                    self.transition_model = self.transition_model.at[:, s1, a].set(
                        self.transition_model[:, s1, a] / column_sum
                    )
                else:
                    # If all zero, set uniform
                    self.transition_model = self.transition_model.at[:, s1, a].set(
                        jnp.ones(self.num_states) / self.num_states
                    )
    
    def _setup_preferences(self) -> None:
        """Set up preferences based on task goals."""
        # Default: uniform preferences
        self.preferred_observations = jnp.ones(self.num_observations) / self.num_observations
        
        # Task-specific preferences
        if self.task_name == 'walk_on_ball':
            # Set preferences for stable upright posture
            # Assuming observation vector is structured as:
            # [joint_positions(18), ball_velocity(2), orientation(3)]
            
            # Prefer zero ball velocity (stable position)
            if 'exteroception' in self.obs_dims:
                for i in range(self.obs_dims['proprioception'], 
                              self.obs_dims['proprioception'] + self.obs_dims['exteroception']):
                    # Prefer values close to zero (centered distribution)
                    prefs = jnp.exp(-((jnp.linspace(-1, 1, 10))**2) / 0.1)
                    prefs = prefs / jnp.sum(prefs)
                    
                    # Map to our observation vector
                    obs_idx = i
                    self.preferred_observations = self.preferred_observations.at[obs_idx].set(
                        0.5  # Highest preference for zero velocity
                    )
            
            # Prefer upright orientation
            if 'orientation' in self.obs_dims:
                orient_start = self.obs_dims['proprioception'] + self.obs_dims.get('exteroception', 0)
                roll_idx = orient_start  # Roll
                pitch_idx = orient_start + 1  # Pitch
                
                # Set preferences for upright orientation (roll & pitch close to 0)
                self.preferred_observations = self.preferred_observations.at[roll_idx].set(0.8)
                self.preferred_observations = self.preferred_observations.at[pitch_idx].set(0.8)
        
        elif self.task_name == 'flight':
            # Set preferences for stable flight at a target height
            # Assuming observation vector is structured as:
            # [joint_positions(18), air_velocity(3), orientation(3)]
            
            # Prefer moderate forward velocity
            if 'exteroception' in self.obs_dims:
                ext_start = self.obs_dims['proprioception']
                fwd_vel_idx = ext_start  # Forward velocity
                
                # Prefer positive forward velocity
                self.preferred_observations = self.preferred_observations.at[fwd_vel_idx].set(0.7)
            
            # Prefer stable orientation
            if 'orientation' in self.obs_dims:
                orient_start = self.obs_dims['proprioception'] + self.obs_dims.get('exteroception', 0)
                roll_idx = orient_start  # Roll
                pitch_idx = orient_start + 1  # Pitch
                
                # Set preferences for level flight (roll & pitch close to 0)
                self.preferred_observations = self.preferred_observations.at[roll_idx].set(0.8)
                self.preferred_observations = self.preferred_observations.at[pitch_idx].set(0.6)  # Slight pitch up
        
        # Normalize to ensure valid probability distribution
        self.preferred_observations = self.preferred_observations / jnp.sum(self.preferred_observations)
    
    def _setup_sensory_precision(self) -> None:
        """Set up precision (inverse variance) for different sensory channels."""
        # Initialize precision for each sensory channel
        self.sensory_precision = {}
        
        # Set different precision values for different observation types
        if 'proprioception' in self.obs_dims:
            # High precision for proprioception (fly has accurate body sense)
            self.sensory_precision['proprioception'] = 10.0
        
        if 'exteroception' in self.obs_dims:
            # Medium precision for exteroception (ball/air velocity)
            self.sensory_precision['exteroception'] = 5.0
        
        if 'orientation' in self.obs_dims:
            # Medium-high precision for orientation
            self.sensory_precision['orientation'] = 8.0
        
        if self.include_vision and 'vision' in self.obs_dims:
            # Lower precision for vision (more uncertainty/noise)
            self.sensory_precision['vision'] = 3.0
    
    def observation_mapping(self, environment_observation: Any) -> jnp.ndarray:
        """
        Map environment observations to the format expected by the generative model.
        
        Args:
            environment_observation: Observation from the environment
            
        Returns:
            Observation vector in the format expected by the model
        """
        # Convert raw environment observation to appropriate vector
        # This depends on the specific environment implementation
        
        # For simplicity, assume environment_observation is already a dictionary or array
        # In a real implementation, we'd need to extract and process the relevant components
        
        if isinstance(environment_observation, dict):
            # Extract components based on known structure
            obs_vector = []
            
            # Extract proprioception if available
            if 'joint_positions' in environment_observation and 'proprioception' in self.obs_dims:
                joints = environment_observation['joint_positions']
                # Take the first N joint values based on our model's expectation
                joints = joints[:self.obs_dims['proprioception']]
                obs_vector.extend(joints)
            
            # Extract exteroception if available
            if 'ball_velocity' in environment_observation and 'exteroception' in self.obs_dims:
                velocity = environment_observation['ball_velocity']
                # Take the appropriate number of velocity components
                velocity = velocity[:self.obs_dims['exteroception']]
                obs_vector.extend(velocity)
            elif 'air_velocity' in environment_observation and 'exteroception' in self.obs_dims:
                velocity = environment_observation['air_velocity']
                velocity = velocity[:self.obs_dims['exteroception']]
                obs_vector.extend(velocity)
            
            # Extract orientation if available
            if 'orientation' in environment_observation and 'orientation' in self.obs_dims:
                orientation = environment_observation['orientation']
                orientation = orientation[:self.obs_dims['orientation']]
                obs_vector.extend(orientation)
            
            # Extract vision if available and we're using it
            if self.include_vision and 'vision' in environment_observation and 'vision' in self.obs_dims:
                vision = environment_observation['vision']
                vision = vision[:self.obs_dims['vision']]
                obs_vector.extend(vision)
            
            return jnp.array(obs_vector)
        
        elif isinstance(environment_observation, (np.ndarray, jnp.ndarray)):
            # If already an array, ensure it's the right size
            obs_array = jnp.array(environment_observation).flatten()
            
            # Truncate or pad to match expected size
            expected_size = sum(self.obs_dims.values())
            
            if len(obs_array) > expected_size:
                # Truncate
                obs_array = obs_array[:expected_size]
            elif len(obs_array) < expected_size:
                # Pad with zeros
                padding = jnp.zeros(expected_size - len(obs_array))
                obs_array = jnp.concatenate([obs_array, padding])
            
            return obs_array
        
        else:
            # Fallback: return a vector of zeros
            expected_size = sum(self.obs_dims.values())
            return jnp.zeros(expected_size)
    
    def action_mapping(self, model_action: int) -> Any:
        """
        Map model actions to the format expected by the environment.
        
        Args:
            model_action: Action selected by the model
            
        Returns:
            Action in the format expected by the environment
        """
        # Get the continuous action vector from our mapping matrix
        if 0 <= model_action < self.num_actions:
            return self.action_mapping_matrix[model_action]
        else:
            # Return zeros if invalid action
            return jnp.zeros(self.action_dim)
    
    def predict_observation_with_precision(self, state_belief: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict expected observations with associated precision.
        
        Args:
            state_belief: State belief to use (uses self.belief_states if None)
            
        Returns:
            Tuple of (predicted_observation, observation_precision)
        """
        # Get predicted observation
        predicted_obs = self.predict_observation(state_belief)
        
        # Create precision vector matching observation dimensions
        precision_vector = jnp.ones_like(predicted_obs)
        
        # Assign precision based on sensory channel
        obs_idx = 0
        
        # Set precision for proprioception
        if 'proprioception' in self.obs_dims and 'proprioception' in self.sensory_precision:
            prop_dim = self.obs_dims['proprioception']
            precision_vector = precision_vector.at[obs_idx:obs_idx+prop_dim].set(
                self.sensory_precision['proprioception']
            )
            obs_idx += prop_dim
        
        # Set precision for exteroception
        if 'exteroception' in self.obs_dims and 'exteroception' in self.sensory_precision:
            ext_dim = self.obs_dims['exteroception']
            precision_vector = precision_vector.at[obs_idx:obs_idx+ext_dim].set(
                self.sensory_precision['exteroception']
            )
            obs_idx += ext_dim
        
        # Set precision for orientation
        if 'orientation' in self.obs_dims and 'orientation' in self.sensory_precision:
            orient_dim = self.obs_dims['orientation']
            precision_vector = precision_vector.at[obs_idx:obs_idx+orient_dim].set(
                self.sensory_precision['orientation']
            )
            obs_idx += orient_dim
        
        # Set precision for vision
        if 'vision' in self.obs_dims and 'vision' in self.sensory_precision:
            vision_dim = self.obs_dims['vision']
            precision_vector = precision_vector.at[obs_idx:obs_idx+vision_dim].set(
                self.sensory_precision['vision']
            )
        
        return predicted_obs, precision_vector 