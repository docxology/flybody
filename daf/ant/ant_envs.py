"""Create examples of walking task environments for ant."""

from typing import Callable

import numpy as np

from dm_control import mujoco
from dm_control import composer
from dm_control.locomotion.arenas import floors

from daf.ant.ant import ant
from daf.ant.tasks.walk_on_ball import WalkOnBall
from daf.ant.tasks.template_task import TemplateTask
from daf.ant.tasks.arenas.ball import BallFloor


def walk_on_ball(force_actuators: bool = False,
                 use_antennae: bool = True,
                 random_state: np.random.RandomState | None = None):
    """Requires a tethered ant to walk on a floating ball.

    Args:
        force_actuators: Whether to use force or position actuators.
        use_antennae: Whether to use the antennae.
        random_state: Random state for reproducibility.

    Returns:
        Environment for ant walking on ball.
    """
    # Build an ant walker and arena.
    walker = ant.Ant
    arena = BallFloor(ball_pos=(-0.05, 0, -0.419),
                      ball_radius=0.454,
                      ball_density=0.0025,
                      skybox=False)
    # Build a task that rewards the agent for walking on a ball.
    time_limit = 2.
    task = WalkOnBall(walker=walker,
                      arena=arena,
                      force_actuators=force_actuators,
                      use_antennae=use_antennae,
                      joint_filter=0.01,
                      adhesion_filter=0.007,
                      time_limit=time_limit)

    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True)


def template_task(random_state: np.random.RandomState | None = None,
                  force_actuators: bool = False,
                  use_antennae: bool = True,
                  joint_filter: float = 0.01,
                  adhesion_filter: float = 0.007,
                  time_limit: float = 1.,
                  mjcb_control: Callable | None = None,
                  observables_options: dict | None = None,
                  action_corruptor: Callable | None = None):
    """An empty no-op walking task for testing.

    Args:
        random_state: Random state for reproducibility.
        force_actuators: Whether to use force or position actuators.
        use_antennae: Whether to use antennae. 
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
        time_limit: Episode time limit.
        mjcb_control: Optional MuJoCo control callback, a callable with
            arguments (model, data). For more information, see
            https://mujoco.readthedocs.io/en/stable/APIreference/APIglobals.html#mjcb-control
        observables_options (optional): A dict of dicts of configuration options
            keyed on observable names, or a dict of configuration options, which
            will propagate those options to all observables.
        action_corruptor (optional): A callable which takes an action as an
            argument, modifies it, and returns it. An example use case for
            this is to add random noise to the action.

    Returns:
        Template walking environment.
    """
    # Build an ant walker and arena.
    walker = ant.Ant
    arena = floors.Floor()
    # Build a no-op task.
    task = TemplateTask(walker=walker,
                        arena=arena,
                        force_actuators=force_actuators,
                        use_antennae=use_antennae,
                        joint_filter=joint_filter,
                        adhesion_filter=adhesion_filter,
                        observables_options=observables_options,
                        mjcb_control=mjcb_control,
                        action_corruptor=action_corruptor,
                        time_limit=time_limit)
    # Reset control callback, if any.
    mujoco.set_mjcb_control(None)
    return composer.Environment(time_limit=time_limit,
                                task=task,
                                random_state=random_state,
                                strip_singleton_obs_buffer_dim=True) 