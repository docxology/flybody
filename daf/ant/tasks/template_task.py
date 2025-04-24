"""Template task for the ant."""

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from dm_control import mjcf
from dm_control.composer import Arena, Entity, Task
from dm_control.mujoco import wrapper as mj_wrapper

from daf.ant.tasks.base import Walking


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