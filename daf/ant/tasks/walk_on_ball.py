"""Task of tethered ant walking on floating ball."""

from typing import Optional
import numpy as np

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.utils import rewards

from daf.ant.tasks.base import Walking


# Terminal velocities beyond which to terminate the episode
_TERMINAL_LINVEL = 10.0  # Linear velocity threshold
_TERMINAL_ANGVEL = 40.0  # Angular velocity threshold


class AntWalkOnBall(Walking):
    """Tethered ant walking on floating ball."""

    def __init__(self, claw_friction: Optional[float] = 1.0, **kwargs):
        """Task of tethered ant walking on floating ball.

        Args:
            claw_friction: Friction of claw geoms with floor.
            **kwargs: Arguments passed to the superclass constructor.
        """

        super().__init__(add_ghost=False, ghost_visible_legs=False, **kwargs)

        # Fuse ant's thorax with world.
        freejoint = self.root_entity.mjcf_model.find('attachment_frame', 'walker').find('joint', 'free')
        if freejoint is not None:
            freejoint.remove()
        else:
            print("Note: No freejoint found in attachment_frame - this is OK if already removed.")

        # Exclude "surprising" thorax-children collisions.
        for child in self._walker.mjcf_model.find('body', 'thorax').all_children():
            if child.tag == 'body':
                self._walker.mjcf_model.contact.add(
                    'exclude',
                    name=f'thorax_{child.name}',
                    body1='thorax',
                    body2=child.name)

        # Maybe change default claw friction.
        if claw_friction is not None:
            adhesion_collision = self._walker.mjcf_model.find('default', 'adhesion-collision')
            if adhesion_collision is not None:
                adhesion_collision.geom.friction = (claw_friction, )
            else:
                print("Warning: Could not find adhesion-collision default to set claw friction")

        # Enable task-specific observables.
        self._walker.observables.add_observable('ball_qvel', self.ball_qvel)

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        """Initialize the MJCF model for a new episode."""
        super().initialize_episode_mjcf(random_state)
        # Any custom initialization can go here

    def initialize_episode(self, physics: 'mjcf.Physics',
                           random_state: np.random.RandomState):
        """Initialize a new episode."""
        super().initialize_episode(physics, random_state)
        
        # Position the ant on top of the ball
        # First find the ball position and radius
        ball_body = physics.named.model.body_name2id['ball']
        ball_pos = physics.named.data.xpos[ball_body]
        ball_geom = physics.named.model.geom_bodyid == ball_body
        ball_radius = physics.named.model.geom_size[ball_geom][0][0]  # First dimension of size for sphere
        
        # Position the ant's thorax above the ball with a slight offset to create contact
        thorax_body = physics.named.model.body_name2id['walker/thorax']
        thorax_pos = physics.named.data.xpos[thorax_body].copy()
        
        # Set the thorax position to be on top of the ball
        thorax_pos[:] = ball_pos.copy()
        thorax_pos[2] = ball_pos[2] + ball_radius + 0.02  # Add a small offset above the ball
        
        # Apply the position to the ant's thorax
        physics.named.data.xpos[thorax_body] = thorax_pos
        
        # Also set the velocity to zero for stability
        thorax_vel_idx = physics.named.model.body_name2id['walker/thorax'] * 6
        physics.data.qvel[thorax_vel_idx:thorax_vel_idx+6] = 0
        
        # Forward the simulation a bit to establish contacts
        with physics.reset_context():
            # Do a short rollout to settle the ant on the ball
            for _ in range(5):
                physics.step()
        
        print(f"Initialized ant at position: {thorax_pos} on ball at {ball_pos} with radius {ball_radius}")

    def before_step(self, physics: 'mjcf.Physics', action,
                    random_state: np.random.RandomState):
        """Called before each physics step."""
        # Any custom pre-step logic can go here
        super().before_step(physics, action, random_state)

    def get_reward_factors(self, physics):
        """Calculate the factorized reward terms.
        
        The reward is based on making the ball roll in the negative y direction.
        """
        # Get the ball's angular velocity
        ball_qvel = physics.named.data.qvel['ball']
        
        # Target velocity is in the negative y direction (rolling backward)
        target_ball_qvel = [0., -5, 0]
        
        # Calculate reward based on how close the ball's velocity is to the target
        qvel = rewards.tolerance(ball_qvel - target_ball_qvel,
                                bounds=(0, 0),
                                sigmoid='linear',
                                margin=6,
                                value_at_margin=0.0)
        return np.hstack(qvel)

    def check_termination(self, physics: 'mjcf.Physics') -> bool:
        """Check if the episode should terminate early."""
        # Check if the ant is moving too fast (indicates instability)
        linvel = np.linalg.norm(self._walker.observables.velocimeter(physics))
        angvel = np.linalg.norm(self._walker.observables.gyro(physics))
        
        return (linvel > _TERMINAL_LINVEL or 
                angvel > _TERMINAL_ANGVEL or 
                super().check_termination(physics))

    @composer.observable
    def ball_qvel(self):
        """Observable for the ball's angular velocity."""
        def get_ball_qvel(physics: 'mjcf.Physics'):
            return physics.named.data.qvel['ball']
        return observable.Generic(get_ball_qvel) 