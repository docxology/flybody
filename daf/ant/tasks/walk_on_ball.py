"""Task for an ant walking on ball."""

import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.utils import rewards

from daf.ant.tasks.base import Walking


class WalkOnBall(Walking):
    """Task for walking on a ball."""

    def __init__(self,
                 walker,
                 arena,
                 force_actuators=False,
                 use_antennae=True,
                 joint_filter=0.01,
                 adhesion_filter=0.007,
                 time_limit=2.,
                 **kwargs):
        """Task for an ant walking on a ball.
        
        Args:
            walker: Constructor for the walker.
            arena: Arena with a ball.
            force_actuators: Whether to use force or position actuators.
            use_antennae: Whether to use the antennae.
            joint_filter: Timescale of filter for joint actuators. 0: disabled.
            adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
            time_limit: Maximum time before truncating episode.
            **kwargs: Additional arguments passed to Walking constructor.
        """
        super().__init__(
            walker=walker,
            arena=arena,
            use_antennae=use_antennae,
            force_actuators=force_actuators,
            joint_filter=joint_filter,
            adhesion_filter=adhesion_filter,
            time_limit=time_limit,
            **kwargs)

        # Add observation for ball angular velocity.
        self._walker.observables.add_observable('ball_angvel',
                                              self._get_ball_angvel)

    def _get_ball_angvel(self, physics: 'mjcf.Physics'):
        """Get ball angular velocity."""
        # Find the ball geom.
        entities = self._arena.mjcf_model.find_all('geom')
        for entity in entities:
            if 'ball' in entity.name:
                # Get the parent body.
                parent = entity.parent
                # Get angular velocity.
                angvel = physics.bind(parent).cvel[3:6]
                return angvel
        return np.zeros(3)

    def get_reward_factors(self, physics: 'mjcf.Physics') -> np.ndarray:
        """Get reward terms and factors."""
        # Ball rotation speed reward - encourage moving the ball.
        ball_angvel = self._walker.observables.ball_angvel(physics)
        angvel_norm = np.linalg.norm(ball_angvel)
        ball_rotation = rewards.tolerance(angvel_norm,
                                         bounds=(1.0, float('inf')),
                                         margin=1.0,
                                         value_at_margin=0.0,
                                         sigmoid='linear')

        # Height of thorax - encourage staying upright on the ball.
        thorax_height = self._walker.observables.thorax_height(physics)
        upright = rewards.tolerance(thorax_height,
                                  bounds=(0.25, 0.35),
                                  margin=0.1,
                                  value_at_margin=0.0,
                                  sigmoid='linear')

        return np.array([ball_rotation, upright])

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Check if the episode should terminate.
        
        Terminate if the ant falls off the ball.
        """
        # Check ant thorax height.
        thorax_height = self._walker.observables.thorax_height(physics)
        # If thorax is too low, ant has fallen off the ball.
        if thorax_height < 0.15:
            return True
        return False 