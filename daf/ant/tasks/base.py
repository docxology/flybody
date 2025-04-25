"""Base task classes for ant tasks."""

import abc
from typing import Dict, Optional, Sequence, Union, Type, Any

import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import Arena
from dm_control.composer.observation import observable
from dm_control.composer.observation.observable import MJCFFeature
from dm_control.mjcf import Physics
from dm_control.utils import rewards

_THETA_BIAS = .5


class Base(composer.Task, metaclass=abc.ABCMeta):
    """A base class for all ant tasks."""

    @abc.abstractproperty
    def root_entity(self):
        """The root MJCF entity of the primary walker."""
        pass

    def __init__(self,
                walker,
                arena: Arena,
                time_limit: float = 2.0,
                add_ghost: bool = False,
                ghost_offset: Sequence[float] = (0., 0., 0.),
                use_antennae: bool = True,
                force_actuators: bool = False,
                joint_filter: float = 0.01,
                adhesion_filter: float = 0.007,
                num_user_actions: int = 0,
                observables_options: Optional[Dict] = None,
                **kwargs):
        """Constructor for base task.
        
        Args:
            walker: Constructor for the walker.
            arena: Arena in which the task is performed.
            time_limit: Maximum time before truncating episode.
            add_ghost: Whether to add a ghost walker for references.
            ghost_offset: (x, y, z) offset of the ghost agent.
            use_antennae: Whether to use antennae.
            force_actuators: Whether to use force or position actuators.
            joint_filter: Timescale of filter for joint actuators. 0: disabled.
            adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
            num_user_actions: Number of user actions passed to the walker.
            observables_options: A dict mapping observable names to options. Or, 
                a dict mapping options to the same value for all observables.
            **kwargs: Additional arguments passed to Task constructor.
        """
        self._walker = walker(name='walker',
                             use_antennae=use_antennae,
                             force_actuators=force_actuators,
                             joint_filter=joint_filter,
                             adhesion_filter=adhesion_filter,
                             num_user_actions=num_user_actions)
        self._ghost = None
        self._ghost_offset = np.array(ghost_offset)
        if add_ghost:
            ghost_options = dict(
                use_antennae=True,
                force_actuators=False,
                joint_filter=0.,
                num_user_actions=0,
                name='ghost')
            self._ghost = walker(**ghost_options)

        # Assign the actual root entity after adding the walker to the arena.
        # The abstract property ensures subclasses know this is expected.
        self._root_entity_instance = arena.add_free_entity(self._walker)
        arena.mjcf_model.visual.map.znear = 0.00025
        self._arena = arena
        self._should_terminate = False
        self._time_limit = time_limit

        # Configure initial observables.
        self._configure_observables(observables_options)

    @property
    def root_entity(self):
        """Return the stored root entity instance."""
        # We define a concrete property here that returns the instance 
        # assigned during __init__. This satisfies the abstract property requirement.
        if not hasattr(self, '_root_entity_instance') or self._root_entity_instance is None:
            raise AttributeError("Root entity has not been initialized properly.")
        return self._root_entity_instance

    def _configure_observables(self, options: Optional[Dict] = None):
        """Configure observables with options."""
        if options is None:
            options = {}

        for obs_key in dir(self.observables):
            if (obs_key.startswith('_') or not isinstance(
                    getattr(self.observables, obs_key), MJCFFeature)):
                continue  # Skip non-feature attributes.

            observable_obj = getattr(self.observables, obs_key)
            # Process configuration for this observable.
            observable_options = options.get(obs_key, {})
            if not observable_options and isinstance(options, dict):
                # Apply global options to all observables.
                observable_options = options

            # Apply computed options.
            if isinstance(observable_options, dict):
                for opt_key, opt_value in observable_options.items():
                    if opt_key in dir(observable_obj):
                        setattr(observable_obj, opt_key, opt_value)

    def get_reward(self, physics: 'mjcf.Physics') -> float:
        """Get the task reward."""
        factors = self.get_reward_factors(physics)
        return np.mean(factors)

    @abc.abstractmethod
    def get_reward_factors(self, physics: 'mjcf.Physics') -> np.ndarray:
        """Get reward terms and factors."""

    @abc.abstractmethod
    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Check for termination conditions."""

    def should_terminate_episode(self, physics: 'mjcf.Physics') -> bool:
        """Check if the episode should terminate."""
        time_limit = physics.time() >= self._time_limit
        self._should_terminate = self.check_termination(physics) or time_limit
        return self._should_terminate

    def get_discount(self, physics: 'mjcf.Physics') -> float:
        """Return a discount in [0, 1]."""
        return 1.0

    def _build_root_entity(self):
        """Build the root entity."""
        # Attach walker to the task's arena
        arena_attachment_frame = self.arena.attach(self.walker.mjcf_model)
        
        # Ensure consistent naming
        arena_attachment_frame.name = 'attachment_frame'
        
        # If using a ball, make sure the walker is properly constrained to it
        if hasattr(self.arena, 'ball'):
            print(f"Arena has a ball: {self.arena.ball}")
            # Create a physical connection between walker and ball
            walker_thorax = self.walker.mjcf_model.find('body', 'thorax')
            attachment_site = walker_thorax.add('site', name='ball_attachment', pos=[0, 0, 0], size=[0.01])
            
            ball_body = self.arena.ball.mjcf_model.find('body', 'ball')
            ball_attachment_site = ball_body.add('site', name='walker_attachment', pos=[0, 0, 0], size=[0.01])
            
            # Add a ball-and-socket joint between ant and ball
            arena_attachment_frame.add('weld', name='ant_ball_weld', 
                                      body1='walker/thorax', 
                                      body2='ball',
                                      relpose='0 0 0 1 0 0 0')
            
            print("Added weld constraint between ant and ball")
            
        # Important: Remove the free joint to prevent the walker from floating freely
        # This should be done after other attachments are set up
        freejoint = arena_attachment_frame.find('joint', 'free')
        if freejoint is not None:
            print("Removing free joint from attachment_frame")
            freejoint.remove()
        else:
            print("No freejoint found in attachment_frame - this is OK if already removed.")
        
        # Also check and remove any 'free' joint from the walker
        walker_freejoint = self.walker.mjcf_model.find_all('joint', 'free')
        if walker_freejoint:
            for joint in walker_freejoint:
                print(f"Removing free joint {joint} from walker")
                joint.remove()
        else:
            print("No 'free' joint found to remove - this is OK for the ant model")
            
        return self.arena


class Walking(composer.Task):
    """Walking task for the ant."""

    def __init__(
        self,
        walker: Type,
        arena: composer.Arena,
        force_actuators: bool = False,
        time_limit: float = 30.,
        joint_filter: float = 0.01,
        adhesion_filter: float = 0.007,
        add_ghost: bool = False,
        ghost_visible_legs: bool = False,
        ghost_offset=None,
        control_timestep: Optional[float] = None,
        custom_load_params: Optional[Dict[str, Any]] = None,
    ):
        """Initializes walking task.

        Args:
            walker: Ant Walker constructor. Ant will be at the origin.
            arena: Arena to put the ant in.
            force_actuators: Whether to use force or position actuators.
            time_limit: Time limit, in seconds.
            joint_filter: Timescale of filter for joint actuators. 0: disabled.
            adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
            add_ghost: Whether to add a ghost ant used for task definition, reference
                tracking, etc.
            ghost_visible_legs: Whether to show ghost ant's legs when add_ghost=True.
            ghost_offset: (x, y, z) offset for ghost position if add_ghost=True.
            control_timestep: Control timestep. If None, use 0.002s.
            custom_load_params: Optional custom params to pass to the ant constructor.
        """
        self._time_limit = time_limit
        self._walker_fn = walker

        # Build the walker
        walker_params = {
            'force_actuators': force_actuators,
            'joint_filter': joint_filter,
            'adhesion_filter': adhesion_filter,
        }
        if custom_load_params:
            walker_params.update(custom_load_params)
        
        self._walker = walker(**walker_params)
        self._arena = arena

        # Add walker to arena
        self._arena.add_free_entity(self._walker)
        
        # Set control timestep
        control_timestep = control_timestep or 0.002
        self.set_timesteps(control_timestep=control_timestep,
                          physics_timestep=self._walker._mjcf_root.option.timestep)

        # Add ghost if requested
        if add_ghost:
            ghost_params = dict(walker_params)
            # We don't need to filter the ghost.
            ghost_params['adhesion_filter'] = 0.0
            ghost_params['joint_filter'] = 0.0

            self._ghost = walker(name='ghost', **ghost_params)
            
            # Remove actuators from ghost
            for actuator in self._ghost.mjcf_model.find_all('actuator'):
                actuator.remove()

            # Make ghost disappear
            rgb_alpha = [1., 1., 1., 0.3]
            for geom in self._ghost.mjcf_model.find_all('geom'):
                geom.set_attributes(rgba=rgb_alpha, contype=0, conaffinity=0)
                # Maybe make legs invisible.
                if not ghost_visible_legs and geom.name:
                    for part in ['C1', 'C2', 'C3']:
                        if part in geom.name:
                            rgb_alpha[3] = 0.0
                            break
                    geom.set_attributes(rgba=rgb_alpha)
                # Reset alpha.
                rgb_alpha[3] = 0.3

            self._ghost_offset = ghost_offset or [0.03, 0, 0]
            self._arena.attach(self._ghost)
            self._ghost_body = self._ghost.mjcf_model.find('body', 'thorax')
            self._ghost_site = self._ghost_body.add(
                'site', name='ghost', pos=[0, 0, 0], size=[0.005])
            self._ghost_body.add(
                'camera',
                name='ghost_top_cam',
                pos=[0, 0, 0.4],
                xyaxes=[1, 0, 0, 0, 1, 0])
        else:
            self._ghost = None
            self._ghost_offset = None
            self._ghost_body = None
            self._ghost_site = None

        # Get root entity
        self.root_entity.mjcf_model.option.timestep = self._walker._mjcf_root.option.timestep
        self.root_entity.mjcf_model.option.gravity = self._walker._mjcf_root.option.gravity
        self.root_entity.mjcf_model.option.density = self._walker._mjcf_root.option.density
        self.root_entity.mjcf_model.option.viscosity = self._walker._mjcf_root.option.viscosity

        # Build observables
        self._task_observables = {}
        # Position
        self._task_observables['walker_position'] = observable.MJCFFeature('xpos',
                                                      self._walker.mjcf_model.find('body', 'thorax'))
        self._task_observables['walker_position'].enabled = True
        # Orientation
        self._task_observables['walker_orientation'] = observable.MJCFFeature('xmat',
                                                        self._walker.mjcf_model.find('body', 'thorax'))
        self._task_observables['walker_orientation'].enabled = True

        # Shadow walker observables
        if add_ghost:
            def ghost_rel_pos(physics):
                ghost_pos = physics.bind(self._ghost_body).xpos
                thorax_pos = physics.bind(self._walker.thorax).xpos
                return ghost_pos - thorax_pos

            self._task_observables['ghost_rel_pos'] = observable.Generic(ghost_rel_pos)
            self._task_observables['ghost_rel_pos'].enabled = True

    def initialize_episode_mjcf(self, random_state):
        """Initializes MJCF."""
        pass

    def initialize_episode(self, physics, random_state):
        """Initializes episode. Can be overridden."""
        self._walker.initialize_episode(physics, random_state)
        if self._ghost:
            self._ghost.initialize_episode(physics, random_state)
            physics.bind(self._ghost_body).pos = (
                physics.bind(self._walker.thorax).pos + self._ghost_offset)
            physics.bind(self._ghost_body).quat = physics.bind(
                self._walker.thorax).quat

    def before_step(self, physics, action, random_state):
        """Update ghost if we have one."""
        if self._ghost_body:
            physics.bind(self._ghost_body).pos = (
                physics.bind(self._walker.thorax).pos + self._ghost_offset)
            physics.bind(self._ghost_body).quat = physics.bind(
                self._walker.thorax).quat

    def action_spec(self, physics):
        """Returns action spec of the ant."""
        return self._walker.get_action_spec(physics)

    def get_reward_factors(self, physics):
        """Reward factors. Override this method for task-specific rewards."""
        return np.array([0.0])

    def get_reward(self, physics):
        """Calls get_reward_factors and aggregates them."""
        return np.sum(self.get_reward_factors(physics))

    def get_discount(self, physics):
        """Task-specific discount."""
        return 1.0

    def check_termination(self, physics) -> bool:
        """Check for termination conditions. Override for task-specific checks."""
        return False
        
    @property
    def root_entity(self):
        return self._arena
        
    @property
    def task_observables(self):
        return self._task_observables 