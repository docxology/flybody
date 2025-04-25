"""Template task for the ant."""

<<<<<<< HEAD
from typing import Any, Callable, Dict, Optional

import numpy as np

from dm_control import composer
from dm_control.composer.observation import observable
=======
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from dm_control import mjcf
from dm_control.composer import Arena, Entity, Task
from dm_control.mujoco import wrapper as mj_wrapper
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

from daf.ant.tasks.base import Walking


<<<<<<< HEAD
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
=======
class TemplateTask(Walking):
    """Template task class for the ant."""

    def __init__(
            self,
            walker,
            arena: Arena,
            force_actuators: bool = False,
            use_antennae: bool = True,
            mjcb_control: Optional[Callable] = None,
            joint_filter: float = 0.01,
            adhesion_filter: float = 0.007,
            observables_options: Optional[Dict] = None,
            action_corruptor: Optional[Callable] = None,
            **kwargs):
        """Template task.

        Args:
            walker: Constructor for the walker.
            arena: The arena to put the walker in.
            force_actuators: Whether to use force or position actuators.
            use_antennae: Whether to use the antennae.
            mjcb_control: Optional MuJoCo control callback.
            joint_filter: Timescale of filter for joint actuators. 0: disabled.
            adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
            observables_options: A dict mapping observable names to options. Or, 
                a dict mapping options to the same value for all observables.
            action_corruptor: A callable modifying action vector.
            **kwargs: Additional arguments passed to Task constructor.
        """
        self._action_corruptor = action_corruptor
        self._mjcb_control = mjcb_control
        
        super().__init__(
            walker=walker,
            arena=arena,
            use_antennae=use_antennae,
            force_actuators=force_actuators,
            joint_filter=joint_filter,
            adhesion_filter=adhesion_filter,
            observables_options=observables_options,
            **kwargs)

    def before_step(self, physics: 'mjcf.Physics', action: np.ndarray,
                   random_state: np.random.RandomState):
        """Before step callback.

        Args:
            physics: MuJoCo Physics.
            action: Action to apply.
            random_state: Random state.
        """
        # Corrupt action if requested.
        if self._action_corruptor is not None:
            action = self._action_corruptor(action)

        # Task-level control callback.
        if self._mjcb_control is not None:
            physics.forward()
            prev_callback = mj_wrapper.get_mjcb_control()
            mj_wrapper.set_mjcb_control(self._mjcb_control)
            try:
                physics.step()
            finally:
                mj_wrapper.set_mjcb_control(prev_callback)
          
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        # Template task has no reward factors.
        return np.ones(1)

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Returns False - no terminations in template task."""
        return False 
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
