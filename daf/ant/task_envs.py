"""Create examples of walking task environments for ant."""

from typing import Callable

import numpy as np

from dm_control import composer
from dm_control.locomotion.arenas import floors

from daf.ant.ant import ant
from daf.ant.tasks.walk_on_ball import AntWalkOnBall
from daf.ant.tasks.arenas.ball import BallFloor


def ant_walk_on_ball(force_actuators: bool = False,
                     random_state: np.random.RandomState | None = None):
    """Requires a tethered ant to walk on a floating ball.

    Args:
        force_actuators: Whether to use force or position actuators.
        random_state: Random state for reproducibility.

    Returns:
        Environment for ant walking on ball.
    """
    # Build an ant walker and arena
    walker = ant.Ant
    arena = BallFloor(ball_pos=(-0.05, 0, -0.419),
                      ball_radius=0.454,
                      ball_density=0.0025,
                      skybox=False)
    
    # Build a task that rewards the agent for making the ball move
    time_limit = 2.0
    task = AntWalkOnBall(walker=walker,
                        arena=arena,
                        force_actuators=force_actuators,
                        joint_filter=0.01,
                        adhesion_filter=0.007,
                        time_limit=time_limit)

    return composer.Environment(time_limit=time_limit,
                               task=task,
                               random_state=random_state,
                               strip_singleton_obs_buffer_dim=True)


def template_task(random_state: np.random.RandomState | None = None,
                 force_actuators: bool = False,
                 joint_filter: float = 0.01,
                 adhesion_filter: float = 0.007,
                 time_limit: float = 1.,
                 mjcb_control: Callable | None = None,
                 observables_options: dict | None = None,
                 action_corruptor: Callable | None = None):
    """Creates a task-less environment with an ant.

    Args:
        random_state: Random state for reproducibility.
        force_actuators: Whether to use force actuators or position actuators.
        joint_filter: Timescale of filter for joint actuators. 0: disabled.
        adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
        time_limit: Time limit for the episode.
        mjcb_control: Optional callback to set MuJoCo mjcb_control.
        observables_options: Optional additional observables to enable or disable.
        action_corruptor: Optional function to add noise to actions.

    Returns:
        Environment for template task.
    """
    from daf.ant.tasks.template_task import AntTemplateTask
    
    # Build an ant walker and arena
    walker = ant.Ant
    arena = floors.Floor()
    
    # Build a task that rewards the agent for whatever you specificy (template)
    task = AntTemplateTask(walker=walker,
                          arena=arena,
                          force_actuators=force_actuators,
                          joint_filter=joint_filter,
                          adhesion_filter=adhesion_filter,
                          time_limit=time_limit,
                          mjcb_control=mjcb_control,
                          observables_options=observables_options,
                          action_corruptor=action_corruptor)

    return composer.Environment(time_limit=time_limit,
                               task=task,
                               random_state=random_state,
                               strip_singleton_obs_buffer_dim=True) 