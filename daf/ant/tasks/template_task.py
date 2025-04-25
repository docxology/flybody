"""Template task for the ant."""

from typing import Any, Callable, Dict, Optional

import numpy as np

from dm_control import composer
from dm_control.composer.observation import observable

from daf.ant.tasks.base import Walking


class AntTemplateTask(Walking):
    """Template task for prototyping with the ant model."""

    def __init__(
        self,
        walker,
        arena,
        force_actuators: bool = False,
        joint_filter: float = 0.01,
        adhesion_filter: float = 0.007,
        time_limit: float = 1.,
        mjcb_control: Optional[Callable] = None,
        observables_options: Optional[Dict[str, Any]] = None,
        action_corruptor: Optional[Callable] = None,
    ):
        """Initializes a template task for the ant model.

        Args:
            walker: Constructor for the ant walker.
            arena: Arena to use.
            force_actuators: Whether to use force or position actuators.
            joint_filter: Timescale of filter for joint actuators. 0: disabled.
            adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
            time_limit: Time limit, in seconds.
            mjcb_control: Optional callback to set MuJoCo mjcb_control.
            observables_options: Optional additional observables to enable or disable.
            action_corruptor: Optional function to add noise to actions.
        """
        super().__init__(
            walker=walker,
            arena=arena,
            force_actuators=force_actuators,
            joint_filter=joint_filter,
            adhesion_filter=adhesion_filter,
            time_limit=time_limit,
        )

        self._mjcb_control = mjcb_control
        self._action_corruptor = action_corruptor

        # Configure observables
        if observables_options:
            for name, options in observables_options.items():
                if hasattr(self._walker.observables, name):
                    for option_name, option_value in options.items():
                        setattr(
                            getattr(self._walker.observables, name),
                            option_name, option_value)

    def initialize_episode(self, physics, random_state):
        """Initializes an episode."""
        super().initialize_episode(physics, random_state)
        
        # Set MuJoCo callback if provided
        if self._mjcb_control:
            physics.set_control_callback(self._mjcb_control)

    def before_step(self, physics, action, random_state):
        """Called before physics step."""
        # Apply action corruption if provided
        if self._action_corruptor:
            action = self._action_corruptor(action, random_state)
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Returns default reward factors."""
        # Simple upright reward based on thorax orientation
        upright = physics.bind(self._walker.thorax).xmat[8]  # z-axis element
        upright_reward = np.clip(upright, 0, 1)
        
        return np.array([upright_reward])

    def check_termination(self, physics) -> bool:
        """Check for early termination."""
        return super().check_termination(physics) 