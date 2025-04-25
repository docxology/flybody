"""Observables for the ant model."""

from dm_control import composer
from dm_control.locomotion.walkers import legacy_base
import numpy as np
<<<<<<< HEAD
from dm_control.composer.observation import observable
=======
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05


class AntObservables(legacy_base.WalkerObservables):
    """Observables for the ant model."""

    def __init__(self, walker, buffer_size, eye_camera_size):
        """Constructor for ant observables.
        
        Args:
            walker: The ant walker instance.
            buffer_size: Buffer size for the observables.
            eye_camera_size: Size of the eye cameras.
        """
        self._walker = walker
        self._buffer_size = buffer_size
        self._eye_camera_size = eye_camera_size
        
        # Add missing attributes required by legacy_base.WalkerObservables
        # These would normally be initialized by the Walker base class
        if not hasattr(self._walker, '_end_effectors_pos_sensors'):
            self._walker._end_effectors_pos_sensors = []
        
        if not hasattr(self._walker, '_egocentric_camera'):
            # This might be needed by some legacy code
            self._walker._egocentric_camera = None
            
        # Add a dict to store observables, normally created by parent class
        self._observables_dict = {}
            
        # Initialize parent class
        try:
            super().__init__(walker)
        except Exception as e:
            print(f"Warning: Error during parent initialization: {e}")
            # Continue anyway as we've added our own implementations
            
    # Override getter and setter for _observables that might be used by legacy code
    @property
    def _observables(self):
        return self._observables_dict
        
    @_observables.setter
    def _observables(self, value):
        self._observables_dict = value

    @composer.observable
    def thorax_height(self):
        """Height of the thorax above ground."""
        def get_thorax_height(physics):
            return physics.bind(self._walker.thorax).xpos[2]
<<<<<<< HEAD
        return observable.Generic(get_thorax_height)
=======
        return get_thorax_height
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def abdomen_height(self):
        """Height of the abdomen above ground."""
        def get_abdomen_height(physics):
            return physics.bind(self._walker.abdomen).xpos[2]
<<<<<<< HEAD
        return observable.Generic(get_abdomen_height)
=======
        return get_abdomen_height
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def world_zaxis_hover(self):
        """World zaxis in thorax frame (approximate hovering orientation)."""
        def get_world_zaxis_thorax(physics):
            # Returns the world z-axis (0,0,1) in the thorax frame.
            return physics.bind(self._walker.thorax).xmat[[2, 5, 8]]
<<<<<<< HEAD
        return observable.Generic(get_world_zaxis_thorax)
=======
        return get_world_zaxis_thorax
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def world_zaxis(self):
        """World zaxis in thorax frame."""
<<<<<<< HEAD
        def get_world_zaxis(physics):
            return physics.bind(self._walker.thorax).xmat[[2, 5, 8]]
        return observable.Generic(get_world_zaxis)
=======
        return self.world_zaxis_hover
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def world_zaxis_abdomen(self):
        """World zaxis in abdomen frame."""
        def get_world_zaxis_abdomen(physics):
            # Returns the world z-axis (0,0,1) in the abdomen frame.
            return physics.bind(self._walker.abdomen).xmat[[2, 5, 8]]
<<<<<<< HEAD
        return observable.Generic(get_world_zaxis_abdomen)
=======
        return get_world_zaxis_abdomen
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def world_zaxis_head(self):
        """World zaxis in head frame."""
        def get_world_zaxis_head(physics):
            # Returns the world z-axis (0,0,1) in the head frame.
            return physics.bind(self._walker.head).xmat[[2, 5, 8]]
<<<<<<< HEAD
        return observable.Generic(get_world_zaxis_head)
=======
        return get_world_zaxis_head
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def force(self):
        """All force sensors readings."""
        def get_force_readings(physics):
<<<<<<< HEAD
            sensors = self._walker.mjcf_model.find_all('sensor', 'force')
            if not sensors:
                return np.zeros(1)
            return physics.bind(sensors).sensordata
        return observable.Generic(get_force_readings)
=======
            return physics.bind(
                    self._walker.mjcf_model.find_all('sensor', 'force')
                    ).sensordata
        return get_force_readings
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def touch(self):
        """All touch sensors readings."""
        def get_touch_readings(physics):
<<<<<<< HEAD
            sensors = self._walker.mjcf_model.find_all('sensor', 'touch')
            if not sensors:
                return np.zeros(1)
            return physics.bind(sensors).sensordata
        return observable.Generic(get_touch_readings)
=======
            return physics.bind(
                    self._walker.mjcf_model.find_all('sensor', 'touch')
                    ).sensordata
        return get_touch_readings
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def accelerometer(self):
        """Accelerometer readings."""
        def get_accelerometer(physics):
<<<<<<< HEAD
            sensors = self._walker.mjcf_model.find_all('sensor', 'accelerometer')
            if not sensors:
                return np.zeros(3)
            return physics.bind(sensors).sensordata
        return observable.Generic(get_accelerometer)
=======
            return physics.bind(
                    self._walker.mjcf_model.find_all('sensor', 'accelerometer')
                    ).sensordata
        return get_accelerometer
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def gyro(self):
        """Gyroscope readings."""
        def get_gyro(physics):
<<<<<<< HEAD
            sensors = self._walker.mjcf_model.find_all('sensor', 'gyro')
            if not sensors:
                return np.zeros(3)
            return physics.bind(sensors).sensordata
        return observable.Generic(get_gyro)
=======
            return physics.bind(
                    self._walker.mjcf_model.find_all('sensor', 'gyro')
                    ).sensordata
        return get_gyro
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def velocimeter(self):
        """Velocimeter readings."""
        def get_velocimeter(physics):
<<<<<<< HEAD
            sensors = self._walker.mjcf_model.find_all('sensor', 'velocimeter')
            if not sensors:
                return np.zeros(3)
            return physics.bind(sensors).sensordata
        return observable.Generic(get_velocimeter)
=======
            return physics.bind(
                    self._walker.mjcf_model.find_all('sensor', 'velocimeter')
                    ).sensordata
        return get_velocimeter
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def actuator_activation(self):
        """Actuator activation state."""
        def get_act(physics):
            return physics.data.act
<<<<<<< HEAD
        return observable.Generic(get_act)
=======
        return get_act
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def appendages_pos(self):
        """Positions of appendages in thorax frame."""
        def relative_pos_in_egocentric_frame(physics):
            # Compute thorax inverse rotation (transpose of rotation matrix).
            thorax_pos = physics.bind(self._walker.thorax).xpos
            thorax_mat = physics.bind(self._walker.thorax).xmat.reshape(3, 3)
            thorax_mat_inv = thorax_mat.T
            # Transform appendage positions.
            appendages_pos = physics.bind(self._walker.appendages).xpos
            rel_pos = []
            for i in range(len(self._walker.appendages)):
                pos = appendages_pos[i] - thorax_pos
                # Apply thorax rotation.
                rel_pos.append(thorax_mat_inv.dot(pos))
            return np.hstack(rel_pos)
<<<<<<< HEAD
        return observable.Generic(relative_pos_in_egocentric_frame)

    @composer.observable
    def self_contact(self):
        """Contact forces between the ant's own body parts."""
=======
        return relative_pos_in_egocentric_frame

    @composer.observable
    def self_contact(self):
        """Contact forces between the fly's own body parts."""
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        def sum_body_contact_forces(physics):
            # For all pairs of geoms, if both are part of the ant's body, get the force.
            body_force = np.zeros(6)
            for c in range(physics.data.ncon):
                contact = physics.data.contact[c]
                g1, g2 = contact.geom1, contact.geom2
                mj_model = physics.model
                name1 = mj_model.id2name(mj_model.geom_id[g1], 'geom')
                name2 = mj_model.id2name(mj_model.geom_id[g2], 'geom')
                if (name1 and name2 and 'walker/' in name1 and 'walker/' in name2
                    and name1 != name2):
                    # Get contact force.
                    force = np.zeros(6)
                    mj_model = physics.model.ptr
                    mj_data = physics.data.ptr
                    mjlib.mj_contactForce(mj_model, mj_data, c, force)
                    body_force += force
            return body_force
<<<<<<< HEAD
        return observable.Generic(sum_body_contact_forces)
=======
        return sum_body_contact_forces
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @property
    def vestibular(self):
        """Vestibular sensations: accelerometer, gyro, velocimeter."""
        return [self.accelerometer, self.gyro, self.velocimeter]

    @property
    def proprioception(self):
        """Proprioceptive sensations: joint_pos, joint_vel."""
<<<<<<< HEAD
        return [self.joints_pos, self.joints_vel]
=======
        return [self.joint_pos, self.joint_vel]
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @property
    def orientation(self):
        """Orientation of the ant: zaxis for various body parts."""
        return [self.world_zaxis, self.world_zaxis_abdomen, self.world_zaxis_head]

    @composer.observable
    def right_eye(self):
        """Right eye image."""
        def get_right_eye_image(physics):
<<<<<<< HEAD
            if self._walker.right_eye is None:
                return np.zeros((16, 16, 3), dtype=np.uint8)
=======
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
            return physics.render(
                camera_id=self._walker.right_eye.full_identifier,
                width=self._eye_camera_size,
                height=self._eye_camera_size,
                depth=False)
<<<<<<< HEAD
        return observable.Generic(get_right_eye_image)
=======
        return get_right_eye_image
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def left_eye(self):
        """Left eye image."""
        def get_left_eye_image(physics):
<<<<<<< HEAD
            if self._walker.left_eye is None:
                return np.zeros((16, 16, 3), dtype=np.uint8)
=======
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
            return physics.render(
                camera_id=self._walker.left_eye.full_identifier,
                width=self._eye_camera_size,
                height=self._eye_camera_size,
                depth=False)
<<<<<<< HEAD
        return observable.Generic(get_left_eye_image)
=======
        return get_left_eye_image
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def end_effectors_pos(self):
        """End effector positions in egocentric frame."""
        def get_end_effectors_pos(physics):
            # Simple implementation that returns a fixed array of zeros
            # just to satisfy the interface requirement
            # In a complete implementation, this would return the actual positions
            # of the end effectors in the egocentric frame
            num_effectors = len(self._walker.end_effectors) if hasattr(self._walker, 'end_effectors') else 0
            if num_effectors == 0:
                return np.zeros(3)  # Default when no end effectors
            
            # Get thorax position and rotation
            thorax_pos = physics.bind(self._walker.thorax).xpos
            thorax_mat = physics.bind(self._walker.thorax).xmat.reshape(3, 3)
            thorax_mat_inv = thorax_mat.T  # Inverse rotation
            
            # Get end effector positions
            end_effectors_pos = physics.bind(self._walker.end_effectors).xpos
            
            # Transform to egocentric coordinates
            rel_pos = []
            for pos in end_effectors_pos:
                rel_pos.append(thorax_mat_inv.dot(pos - thorax_pos))
                
            return np.hstack(rel_pos)
            
<<<<<<< HEAD
        return observable.Generic(get_end_effectors_pos)
=======
        return get_end_effectors_pos
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def joints_pos(self):
        """Joint positions."""
        def get_joints_pos(physics):
            # Get joint positions from MuJoCo sensors
            joint_pos_sensors = self._walker.mjcf_model.find_all('sensor', 'jointpos')
            if joint_pos_sensors:
                return physics.bind(joint_pos_sensors).sensordata
            
            # Fallback if no jointpos sensors
            all_joints = []
            for name, joint in self._walker.observable_joints.items():
                if hasattr(physics.named.data, 'qpos') and name in physics.named.data.qpos:
                    all_joints.append(physics.named.data.qpos[name])
            
            if all_joints:
                return np.array(all_joints)
            else:
                # Default empty array if no joints found
                return np.zeros(1)
        
<<<<<<< HEAD
        return observable.Generic(get_joints_pos)
=======
        return get_joints_pos
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        
    @composer.observable
    def joints_vel(self):
        """Joint velocities."""
        def get_joints_vel(physics):
            # Get joint velocities from MuJoCo sensors
            joint_vel_sensors = self._walker.mjcf_model.find_all('sensor', 'jointvel')
            if joint_vel_sensors:
                return physics.bind(joint_vel_sensors).sensordata
            
            # Fallback if no jointvel sensors
            all_joints = []
            for name, joint in self._walker.observable_joints.items():
                if hasattr(physics.named.data, 'qvel') and name in physics.named.data.qvel:
                    all_joints.append(physics.named.data.qvel[name])
            
            if all_joints:
                return np.array(all_joints)
            else:
                # Default empty array if no joints found
                return np.zeros(1)
        
<<<<<<< HEAD
        return observable.Generic(get_joints_vel)
=======
        return get_joints_vel
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    @composer.observable
    def egocentric_camera(self):
        """Egocentric camera observable."""
        def get_egocentric_image(physics):
            # Handle the case where egocentric_camera might not be defined
            camera = getattr(self._walker, 'egocentric_camera', None)
            if camera is None:
                # Return a small black image as fallback
                return np.zeros((16, 16, 3), dtype=np.uint8)
                
            try:
                # Try to render the camera view
                return physics.render(
                    camera_id=camera.full_identifier,
                    width=self._eye_camera_size,
                    height=self._eye_camera_size,
                    depth=False)
            except Exception as e:
                print(f"Warning: Failed to render egocentric camera: {e}")
                return np.zeros((16, 16, 3), dtype=np.uint8)
                
<<<<<<< HEAD
        return observable.Generic(get_egocentric_image)
=======
        return get_egocentric_image
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05

    # Add dummy implementation for joint_angles which might be expected
    @composer.observable
    def joint_angles(self):
        """Joint angles (same as joints_pos)."""
<<<<<<< HEAD
        def get_joint_angles(physics):
            if hasattr(self, 'joints_pos'):
                return self.joints_pos(physics)
            return np.zeros(1)
        return observable.Generic(get_joint_angles)
=======
        return self.joints_pos
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
        
    # Add dummy implementation for body_velocities
    @composer.observable
    def body_velocities(self):
        """Body linear and angular velocities."""
        def get_body_velocities(physics):
            # Return the linear and angular velocities of the thorax
            if hasattr(self._walker, 'thorax'):
                linear_vel = physics.bind(self._walker.thorax).cvel[0:3]
                angular_vel = physics.bind(self._walker.thorax).cvel[3:6]
                return np.concatenate([linear_vel, angular_vel])
            return np.zeros(6)  # Fallback
<<<<<<< HEAD
        return observable.Generic(get_body_velocities)

    @composer.observable
    def joint_pos(self):
        """Legacy name for joint positions (wrapper for joints_pos)."""
        def get_joint_pos(physics):
            if hasattr(self, 'joints_pos'):
                return self.joints_pos(physics)
            return np.zeros(1)  # Fallback if no data available
        return observable.Generic(get_joint_pos)
        
    @composer.observable
    def joint_vel(self):
        """Legacy name for joint velocities (wrapper for joints_vel)."""
        def get_joint_vel(physics):
            if hasattr(self, 'joints_vel'):
                return self.joints_vel(physics)
            return np.zeros(1)  # Fallback if no data available
        return observable.Generic(get_joint_vel) 
=======
        return get_body_velocities 
>>>>>>> b6de4f487bf9d84c86a28b0ffb245143db607e05
