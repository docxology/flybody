"""Base task classes for ant tasks."""

import abc
from typing import Dict, Optional, Sequence, Union

import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import Arena
from dm_control.composer.observation import observable
from dm_control.composer.observation.observable import MJCFFeature
from dm_control.mjcf import Physics
from dm_control.utils import rewards


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


class Walking(Base):
    """Base class for walking tasks."""

    def __init__(self, **kwargs):
        """Constructor for walking task.
        
        Args:
            **kwargs: Arguments passed to the Base constructor.
        """
        super().__init__(**kwargs)
        
    def initialize_episode(self, physics, random_state):
        """Set the ant in its default, upright pose."""
        physics.bind(self._walker.root_body).quat = [1, 0, 0, 0]
        physics.bind(self._walker.root_body).pos = [0, 0, 0.1]
        
        # Initialize ghost, if any.
        if self._ghost:
            ghost_pos = physics.bind(self._walker.root_body).pos.copy()
            ghost_pos += self._ghost_offset
            physics.bind(self._ghost.root_body).pos = ghost_pos
            physics.bind(self._ghost.root_body).quat = [1, 0, 0, 0] 