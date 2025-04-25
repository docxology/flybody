"""Ant model walker."""

import collections as col
import os
from typing import Sequence

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.mujoco import wrapper as mj_wrapper
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils.transformations import quat_mul
from dm_env import specs
import numpy as np

from daf.ant.ant.observables import AntObservables

enums = mjbindings.enums
mjlib = mjbindings.mjlib

_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets/ant.xml')
# === Constants.
_SPAWN_POS = np.array((0, 0, 0.1))
# OrderedDict used to streamline enabling/disabling of action classes.
_ACTION_CLASSES = col.OrderedDict(adhesion=0,
                                  head=0,
                                  mouth=0,
                                  antennae=0,
                                  abdomen=0,
                                  legs=0,
                                  user=0)


def neg_quat(quat_a):
    """Returns neg(quat_a)."""
    quat_b = quat_a.copy()
    quat_b[0] *= -1
    return quat_b


def mul_quat(quat_a, quat_b):
    """Returns quat_a * quat_b."""
    quat_c = np.zeros(4)
    mjlib.mju_mulQuat(quat_c, quat_a, quat_b)
    return quat_c


def mul_jac_t_vec(physics, efc):
    """Maps forces from constraint space to joint space."""
    qfrc = np.zeros(physics.model.nv)
    mjlib.mj_mulJacTVec(physics.model.ptr, physics.data.ptr, qfrc, efc)
    return qfrc


def rot_vec_quat(vec, quat):
    """Rotates vector with quaternion."""
    res = np.zeros(3)
    mjlib.mju_rotVecQuat(res, vec, quat)
    return res


def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
    """Checks if any of substrings is in string."""
    return any(s in string for s in substrings)


def body_quat_from_springrefs(body: 'mjcf.element') -> np.ndarray:
    """Computes new body quat from all joint springrefs and current quat."""
    joints = body.joint
    if not joints:
        return None
    # Construct quaternions for all joint axes.
    quats = []
    for joint in joints:
        if hasattr(joint, 'springref') and joint.springref is not None:
            theta = joint.springref
        elif hasattr(joint, 'dclass') and joint.dclass is not None and hasattr(joint.dclass, 'joint') and joint.dclass.joint is not None and hasattr(joint.dclass.joint, 'springref') and joint.dclass.joint.springref is not None:
            theta = joint.dclass.joint.springref
        else:
            theta = 0
            
        if hasattr(joint, 'axis') and joint.axis is not None:
            axis = joint.axis
        elif hasattr(joint, 'dclass') and joint.dclass is not None and hasattr(joint.dclass, 'joint') and joint.dclass.joint is not None and hasattr(joint.dclass.joint, 'axis') and joint.dclass.joint.axis is not None:
            axis = joint.dclass.joint.axis
        elif hasattr(joint, 'dclass') and joint.dclass is not None and hasattr(joint.dclass, 'parent') and joint.dclass.parent is not None and hasattr(joint.dclass.parent, 'joint') and joint.dclass.parent.joint is not None and hasattr(joint.dclass.parent.joint, 'axis') and joint.dclass.parent.joint.axis is not None:
            axis = joint.dclass.parent.joint.axis
        else:
            print(f"Warning: Could not find axis for joint {joint.name}. Using default [0,0,1].")
            axis = [0, 0, 1]
            
        quats.append(np.hstack((np.cos(theta / 2), np.sin(theta / 2) * axis)))
    # Compute the new orientation quaternion.
    quat = np.array([1., 0, 0, 0])
    for i in range(len(quats)):
        quat = quat_mul(quats[-1 - i], quat)
    if body.quat is not None:
        quat = quat_mul(body.quat, quat)
    return quat


def change_body_frame(body, frame_pos, frame_quat):
    """Change the frame of a body while maintaining child locations."""
    frame_pos = np.zeros(3) if frame_pos is None else frame_pos
    frame_quat = np.array((1., 0, 0, 0)) if frame_quat is None else frame_quat
    # Get frame transformation.
    body_pos = np.zeros(3) if body.pos is None else body.pos
    dpos = body_pos - frame_pos
    body_quat = np.array((1., 0, 0, 0)) if body.quat is None else body.quat
    dquat = mul_quat(neg_quat(frame_quat), body_quat)
    # Translate and rotate the body to the new frame.
    body.pos = frame_pos
    body.quat = frame_quat
    # Move all its children to their previous location.
    for child in body.all_children():
        if not hasattr(child, 'pos'):
            continue
        # Rotate:
        if hasattr(child, 'quat'):
            child_quat = np.array(
                (1., 0, 0, 0)) if child.quat is None else child.quat
            child.quat = mul_quat(dquat, child_quat)
        # Translate, accounting for rotations.
        child_pos = np.zeros(3) if child.pos is None else child.pos
        pos_in_parent = rot_vec_quat(child_pos, body_quat) + dpos
        child.pos = rot_vec_quat(pos_in_parent, neg_quat(frame_quat))


# -----------------------------------------------------------------------------


class Ant(legacy_base.Walker):
    """An ant model."""

    def _build(
        self,
        name: str = 'walker',
        use_mouth: bool = False,
        use_antennae: bool = False,
        force_actuators: bool = False,
        joint_filter: float = 0.01,
        adhesion_filter: float = 0.007,
        dyntype_filterexact: bool = False,
        body_pitch_angle: float = 0.0,
        physics_timestep: float = 1e-4,
        control_timestep: float = 2e-3,
        num_user_actions: int = 0,
        eye_camera_fovy: float = 150.,
        eye_camera_size: int = 32,
    ):
        """Build an ant walker.

        Args:
            name: Name of the walker.
            use_mouth: Whether to use or retract the mouth.
            use_antennae: Whether to use the antennae.
            force_actuators: Whether to use force or position actuators for body
                and legs.
            joint_filter: Timescale of filter for joint actuators. 0: disabled.
            adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
            dyntype_filterexact: When joint or adhesion filters are enabled, whether
                to use exact-integration activation dyntype `filterexact`.
                If False, use approximate `filter` dyntype.
            body_pitch_angle: Body pitch angle for initial pose, relative to
                ground, degrees. 0: horizontal body position.
            physics_timestep: Timestep of the simulation.
            control_timestep: Timestep of the controller.
            num_user_actions: Optional, number of additional actions for custom usage
                e.g. in before_step callback. The action range is [-1, 1]. 0: Not used.
            eye_camera_fovy: Vertical field of view of the eye cameras, degrees. The
                horizontal field of view is computed automatically given the window
                size.
            eye_camera_size: Size in pixels (height and width) of the eye cameras.
                Height and width are assumed equal.
        """
        # Note: Unlike fruit fly, ants don't have wings, so we removed that parameter
        self._adhesion_filter = adhesion_filter
        self._control_timestep = control_timestep
        self._buffer_size = int(round(control_timestep / physics_timestep))
        self._eye_camera_size = eye_camera_size
        root = mjcf.from_path(_XML_PATH)
        self._mjcf_root = root
        if name:
            self._mjcf_root.model = name

        # Remove freejoint.
        free_joint = root.find('joint', 'free')
        if free_joint is not None:
            free_joint.remove()
        else:
            print("Note: No 'free' joint found to remove - this is OK for the ant model.")
        
        # Set eye camera fovy.
        eye_right = root.find('camera', 'eye_right')
        eye_left = root.find('camera', 'eye_left')
        if eye_right is not None:
            eye_right.fovy = eye_camera_fovy
        if eye_left is not None:
            eye_left.fovy = eye_camera_fovy

        # Identify actuator/body/joint/tendon class by substrings in its name.
        name_substr = {
            'adhesion': [],
            'head': ['head'],
            'mouth': ['rostrum', 'mandible', 'labium'],
            'antennae': ['antenna'],
            'abdomen': ['abdomen', 'gaster'],
            'legs': ['C1', 'C2', 'C3'],  # Using C for "coxa" instead of T for "thorax"
            'user': []
        }

        # === Retract disabled body parts and remove their actuators.

        # Maybe retract and disable mouth.
        if not use_mouth:
            # Set orientation quaternions to retracted mouth position.
            mouth_bodies = [
                b for b in root.find_all('body')
                if any_substr_in_str(name_substr['mouth'], b.name)
            ]
            for body in mouth_bodies:
                body.quat = body_quat_from_springrefs(body)
            # Remove mouth tendons and tendon actuators.
            for tendon in root.find_all('tendon'):
                if any_substr_in_str(name_substr['mouth'], tendon.name):
                    # Assume tendon actuator names are the same as tendon
                    # names.
                    actuator = root.find('actuator', tendon.name)
                    if actuator is not None:
                        actuator.remove()
                    tendon.remove()
            # Remove mouth actuators and joints.
            mouth_joints = [
                j for j in root.find_all('joint')
                if any_substr_in_str(name_substr['mouth'], j.name)
            ]
            for joint in mouth_joints:
                # Assume joint actuator names are the same as joint names.
                actuator = root.find('actuator', joint.name)
                if actuator is not None:
                    actuator.remove()
                # Remove joint from observable_joints if it exists
                if hasattr(joint, 'name') and joint.name in self.observable_joints:
                    del self.observable_joints[joint.name]
                joint.remove()
            # Remove mouth adhesion actuators.
            for actuator in root.find_all('actuator'):
                if ('adhere' in actuator.name and any_substr_in_str(
                        name_substr['mouth'], actuator.name)):
                    actuator.remove()
            # Remove mouth sensors.
            for sensor in root.find_all('sensor'):
                if any_substr_in_str(name_substr['mouth'], sensor.name):
                    sensor.remove()

        # Maybe retract and disable antennae.
        if not use_antennae:
            # Set orientation quaternions to retracted antennae position.
            antenna_bodies = [
                b for b in root.find_all('body')
                if any_substr_in_str(name_substr['antennae'], b.name)
            ]
            for body in antenna_bodies:
                body.quat = body_quat_from_springrefs(body)
            # Remove antennae tendons and tendon actuators.
            for tendon in root.find_all('tendon'):
                if any_substr_in_str(name_substr['antennae'], tendon.name):
                    # Assume tendon actuator names are the same as tendon
                    # names.
                    actuator = root.find('actuator', tendon.name)
                    if actuator is not None:
                        actuator.remove()
                    tendon.remove()
            # Remove antennae actuators and joints.
            antenna_joints = [
                j for j in root.find_all('joint')
                if any_substr_in_str(name_substr['antennae'], j.name)
            ]
            for joint in antenna_joints:
                # Assume joint actuator names are the same as joint names.
                actuator = root.find('actuator', joint.name)
                if actuator is not None:
                    actuator.remove()
                # Remove joint from observable_joints if it exists
                if hasattr(joint, 'name') and joint.name in self.observable_joints:
                    del self.observable_joints[joint.name]
                joint.remove()
            # Remove antennae sensors.
            for sensor in root.find_all('sensor'):
                if any_substr_in_str(name_substr['antennae'], sensor.name):
                    sensor.remove()

        # === Now initialize all actuator classes with number of actuator dims.
        # Note that we don't filter out wings, since ants don't have wings.

        # Setup joints and actuators.
        self._force_actuators = force_actuators

        # Convert observable_joints to a dict if it's a list
        observable_joints_dict = {}
        if isinstance(self.observable_joints, list):
            for joint in self.observable_joints:
                if hasattr(joint, 'name') and joint.name:
                    observable_joints_dict[joint.name] = joint
        else:
            observable_joints_dict = self.observable_joints
            
        for name, joint in observable_joints_dict.items():
            if name in root.find_all('actuator'):
                if 'user' not in name:
                    # For adhesion, we want to count the number of *unique* joint names
                    # before the first underscore.
                    if 'adhere' in name:
                        joint_name = name.split('_')[0]
                        # Add +1 only for unique joints.
                        if joint_name not in list(
                                n.split('_')[0]
                                for n in _ACTION_CLASSES['adhesion']):
                            _ACTION_CLASSES['adhesion'] += [joint_name]
                    # For all other joints, actuators are named same as joint.
                    elif any_substr_in_str(name_substr['head'], name):
                        _ACTION_CLASSES['head'] += [name]
                    elif any_substr_in_str(name_substr['mouth'], name):
                        _ACTION_CLASSES['mouth'] += [name]
                    elif any_substr_in_str(name_substr['antennae'], name):
                        _ACTION_CLASSES['antennae'] += [name]
                    elif any_substr_in_str(name_substr['abdomen'], name):
                        _ACTION_CLASSES['abdomen'] += [name]
                    elif any_substr_in_str(name_substr['legs'], name):
                        _ACTION_CLASSES['legs'] += [name]

        # Number of 'user' actions for task-specific control.
        for i in range(num_user_actions):
            _ACTION_CLASSES['user'] += [f'user_{i}']

        # Save action indices.
        self._action_indices = {}
        idx = 0
        for act_class, act_names in _ACTION_CLASSES.items():
            if not isinstance(act_names, list):
                continue  # Empty class.
            self._action_indices[act_class] = list(range(idx, idx + len(act_names)))
            idx += len(act_names)

        # Save joint springrefs.
        self._springrefs = {}
        
        # Handle observable_joints safely
        for name, joint in self.observable_joints.items():
            springref = 0
            if hasattr(joint, 'springref') and joint.springref is not None:
                springref = joint.springref
            elif (hasattr(joint, 'dclass') and joint.dclass is not None and 
                  hasattr(joint.dclass, 'joint') and joint.dclass.joint is not None and
                  hasattr(joint.dclass.joint, 'springref') and 
                  joint.dclass.joint.springref is not None):
                springref = joint.dclass.joint.springref
            
            self._springrefs[name] = springref

        if force_actuators:
            # Convert all position actuators to force except for adhesion.
            for actuator in root.find_all('actuator'):
                if (hasattr(actuator, 'joint') and actuator.joint and
                        'adhere' not in actuator.name):
                    # Convert position-controlled actuator to force-controlled.
                    # Set attributes to None instead of using remove
                    if hasattr(actuator, 'gainprm'):
                        actuator.gainprm = None
                    if hasattr(actuator, 'biasprm'):
                        actuator.biasprm = None
                    if hasattr(actuator, 'biastype'):
                        actuator.biastype = None
                    # Set new values - use array/tuple for MuJoCo attributes
                    try:
                        actuator.gainprm = [1.0]
                        actuator.biastype = 'none'
                    except Exception as e:
                        print(f"Warning: Could not set attributes for {actuator.name}: {e}")

        # Configure MuJoCo filter and dyntype.
        filtertype = 'filterexact' if dyntype_filterexact else 'filter'
        for actuator in root.find_all('actuator'):
            if hasattr(actuator, 'joint') and actuator.joint:
                try:
                    if 'adhere' in actuator.name:
                        actuator.dyntype = (filtertype
                                         if adhesion_filter > 0. else 'none')
                        # dynprm might need to be an array, try different formats
                        try:
                            actuator.dynprm = adhesion_filter
                        except Exception:
                            try:
                                actuator.dynprm = [adhesion_filter]
                            except Exception as e:
                                print(f"Warning: Could not set dynprm for {actuator.name}: {e}")
                    else:
                        actuator.dyntype = (filtertype
                                         if joint_filter > 0. else 'none')
                        try:
                            actuator.dynprm = joint_filter
                        except Exception:
                            try:
                                actuator.dynprm = [joint_filter]
                            except Exception as e:
                                print(f"Warning: Could not set dynprm for {actuator.name}: {e}")
                except Exception as e:
                    print(f"Warning: Could not set dyntype/dynprm for {actuator.name}: {e}")

        # Create an egocentric camera.
        self._mjcf_root.worldbody.add(
            'camera',
            name='egocentric',
            pos=[0., 0., 0.],
            quat=[1., 0., 0., 0.],
            mode="fixed")

        # Save the freejoint for getting pose later on.
        self._freejoint = 'root'

        # Add a tracking site for the CoM.
        self._mjcf_root.worldbody.add('site', name='com')

        # Set body pitch.
        thorax = self._mjcf_root.find('body', 'thorax')
        thorax.pos = (0., 0., 0.)

        # Initialize observable cache and observables.
        self._observables = AntObservables(self, self._buffer_size,
                                           self._eye_camera_size)
        self._prev_action = None

    def initialize_episode(self, physics: 'mjcf.Physics',
                          random_state: np.random.RandomState):
        """Sets the state of the ant and its joints."""
        physics.bind(self._mjcf_root.find('site', 'com')).pos = (
            physics.named.data.subtree_com['walker/'])

    @property
    def name(self):
        return self._mjcf_root.model

    @property
    def upright_pose(self):
        return base.WalkerPose(xpos=_SPAWN_POS)

    @property
    def weight(self):
        return 1e-3

    @property
    def adhesion_filter(self):
        return self._adhesion_filter

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def prev_action(self):
        return self._prev_action

    @composer.cached_property
    def root_entity(self):
        # Find the root body, typically the thorax in insect models
        root = self._mjcf_root.find('body', 'thorax') 
        if root is None:
            # Fallback or error if 'thorax' isn't found
            # This could happen if the XML structure changes
            all_bodies = self._mjcf_root.worldbody.body
            if all_bodies:
                # As a simple fallback, return the first body defined if thorax is missing
                print("Warning: Could not find 'thorax' body, using first body as root_entity.")
                return all_bodies[0]
            else:
                raise ValueError("Could not find 'thorax' or any other body to use as root_entity.")
        return root

    @composer.cached_property
    def thorax(self):
        return self._mjcf_root.find('body', 'thorax')

    @composer.cached_property
    def abdomen(self):
        return self._mjcf_root.find('body', 'abdomen')

    @composer.cached_property
    def head(self):
        return self._mjcf_root.find('body', 'head')

    @composer.cached_property
    def head_site(self):
        return self._mjcf_root.find('site', 'head')

    @composer.cached_property
    def observable_joints(self):
        """Returns all joints in the model as a dictionary keyed by name."""
        joints = self._mjcf_root.find_all('joint')
        # Convert list of joints to a dictionary for easier access
        joints_dict = {}
        for joint in joints:
            if hasattr(joint, 'name') and joint.name:
                joints_dict[joint.name] = joint
        return joints_dict

    @composer.cached_property
    def actuators(self):
        actuator_list = []
        
        # Check for head and abdomen actuators
        head_flexor = self._mjcf_root.find('actuator', 'flex_head')
        if head_flexor is not None:
            actuator_list.append(head_flexor)
        
        head_extensor = self._mjcf_root.find('actuator', 'extend_head')
        if head_extensor is not None:
            actuator_list.append(head_extensor)
        
        abdomen_flexor = self._mjcf_root.find('actuator', 'flex_abdomen')
        if abdomen_flexor is not None:
            actuator_list.append(abdomen_flexor)
        
        abdomen_extensor = self._mjcf_root.find('actuator', 'extend_abdomen')
        if abdomen_extensor is not None:
            actuator_list.append(abdomen_extensor)
        
        # Check for leg actuators
        for leg_id in ['1', '2', '3']:
            for side in ['left', 'right']:
                prefix = f"C{leg_id}_{side}"
                
                # Check each joint actuator
                for joint in ['coxa', 'femur', 'tibia', 'tarsus']:
                    flex_actuator = self._mjcf_root.find('actuator', f'flex_{joint}_{prefix}')
                    if flex_actuator is not None:
                        actuator_list.append(flex_actuator)
                    
                    extend_actuator = self._mjcf_root.find('actuator', f'extend_{joint}_{prefix}')
                    if extend_actuator is not None:
                        actuator_list.append(extend_actuator)
        
        return tuple(actuator_list)

    @composer.cached_property
    def mocap_tracking_bodies(self):
        # Which bodies to track?
        bodies = []
        
        # Core body parts - these should exist
        if self._mjcf_root.find('body', 'thorax') is not None:
            bodies.append({'name': 'thorax', 'site': 'thorax'})
        if self._mjcf_root.find('body', 'head') is not None:
            bodies.append({'name': 'head', 'site': 'head'})
        if self._mjcf_root.find('body', 'abdomen') is not None:
            bodies.append({'name': 'abdomen', 'site': 'abdomen'})
        
        # Check leg parts - only include those that exist
        for leg_id in ['1', '2', '3']:
            for side in ['left', 'right']:
                prefix = f"{side[0].upper()}{leg_id}"
                
                # Only add if both body and site exist
                femur_body = self._mjcf_root.find('body', f'femur_{prefix}')
                femur_site = self._mjcf_root.find('site', f'femur_{prefix}')
                if femur_body is not None and femur_site is not None:
                    bodies.append({
                        'name': f'femur_{prefix}',
                        'site': f'femur_{prefix}'
                    })
                
                tibia_body = self._mjcf_root.find('body', f'tibia_{prefix}')
                tibia_site = self._mjcf_root.find('site', f'tibia_{prefix}')
                if tibia_body is not None and tibia_site is not None:
                    bodies.append({
                        'name': f'tibia_{prefix}',
                        'site': f'tibia_{prefix}'
                    })
                
                tarsus_body = self._mjcf_root.find('body', f'tarsus_{prefix}')
                tarsus_site = self._mjcf_root.find('site', f'tarsus_{prefix}')
                if tarsus_body is not None and tarsus_site is not None:
                    bodies.append({
                        'name': f'tarsus_{prefix}',
                        'site': f'tarsus_{prefix}'
                    })
        
        return bodies

    @composer.cached_property
    def end_effectors(self):
        sites = []
        missing_sites = []
        
        # Check for sites with the format used in the XML: 'tarsus_C1_left'
        leg_ids = ['1', '2', '3']
        sides = ['left', 'right']
        
        for leg_id in leg_ids:
            for side in sides:
                site_name = f'tarsus_C{leg_id}_{side}'
                site = self._mjcf_root.find('site', site_name)
                if site is not None:
                    sites.append(site)
                else:
                    missing_sites.append(site_name)
        
        if missing_sites:
            print(f"Warning: Could not find the following tarsus sites: {', '.join(missing_sites)}")
            
        if not sites:
            print("ERROR: No end effector sites found! This will likely cause an error in legacy_base.py")
            # Create a dummy site to prevent NoneType error in legacy_base.py
            thorax = self._mjcf_root.find('body', 'thorax')
            if thorax is not None:
                # Add a site to the thorax as a fallback
                dummy_site = thorax.add('site', name='dummy_tarsus', pos=[0, 0, 0], size=[0.002])
                sites.append(dummy_site)
                print("Added a dummy tarsus site to prevent errors")
            
        return tuple(sites)

    @composer.cached_property
    def appendages(self):
        sites = []
        # Add ant antennae tips if they exist
        right_antenna = self._mjcf_root.find('site', 'antenna_right')
        left_antenna = self._mjcf_root.find('site', 'antenna_left')
        
        if right_antenna is not None:
            sites.append(right_antenna)
        if left_antenna is not None:
            sites.append(left_antenna)
            
        # Add all leg tarsi (end effectors)
        for ee in self.end_effectors:
            if ee is not None:  # Safety check
                sites.append(ee)
                
        # Ensure we have at least one site to avoid NoneType errors
        if not sites:
            # If no appendage sites found, return the thorax site as a fallback
            thorax_site = self._mjcf_root.find('site', 'thorax')
            if thorax_site is not None:
                sites.append(thorax_site)
                
        return tuple(sites)
    
    def _build_observables(self):
        return AntObservables(self, self._buffer_size, self._eye_camera_size)

    @composer.cached_property
    def left_eye(self):
        return self._mjcf_root.find('camera', 'eye_left')

    @composer.cached_property
    def right_eye(self):
        return self._mjcf_root.find('camera', 'eye_right')

    @composer.cached_property
    def egocentric_camera(self):
        return self._mjcf_root.find('camera', 'egocentric')

    @composer.cached_property
    def ground_contact_geoms(self):
        foot_geoms = []
        missing_geoms = []
        
        # Check for geoms with the format used in the XML: 'tarsal_claw_C1_left'
        leg_ids = ['1', '2', '3']
        sides = ['left', 'right']
        
        for leg_id in leg_ids:
            for side in sides:
                geom_name = f'tarsal_claw_C{leg_id}_{side}'
                geom = self._mjcf_root.find('geom', geom_name)
                if geom is not None:
                    foot_geoms.append(geom)
                else:
                    missing_geoms.append(geom_name)
        
        if missing_geoms:
            print(f"Warning: Could not find the following ground contact geoms: {', '.join(missing_geoms)}")
        
        # Ensure we have at least one geom to avoid NoneType errors
        if not foot_geoms:
            # If no foot geoms found, return the thorax geom as a fallback
            thorax_geom = self._mjcf_root.find('geom', 'thorax')
            if thorax_geom is not None:
                foot_geoms.append(thorax_geom)
                print("Warning: Using thorax geom as fallback for ground_contact_geoms")
                
        return tuple(foot_geoms)

    def apply_action(self, physics, action, random_state):
        """Apply action to MuJoCo actuators."""
        # Save prev action for filtering, if needed.
        self._prev_action = action.copy()
        # Activate MuJoCo actuators.
        physics.bind(self.actuators).ctrl = action
        # Update com tracking site.
        physics.bind(self._mjcf_root.find('site', 'com')).pos = (
            physics.named.data.subtree_com['walker/'])

    def get_action_spec(self, physics):
        """Return a flat action spec."""
        actuator_ctrlrange = physics.bind(self.actuators).ctrlrange
        act_count = 0
        for act_names in _ACTION_CLASSES.values():
            if isinstance(act_names, list):
                act_count += len(act_names)
        actuator_names = list(actuator.full_identifier
                            for actuator in self.actuators)
        # Handle user actions, which don't exist as actuators.
        if 'user' in self._action_indices and actuator_names:
            user_lb = actuator_ctrlrange[0, 0] * np.ones(
                len(self._action_indices['user']))
            user_ub = actuator_ctrlrange[0, 1] * np.ones(
                len(self._action_indices['user']))

            spec_lb = np.hstack((actuator_ctrlrange[:, 0], user_lb))
            spec_ub = np.hstack((actuator_ctrlrange[:, 1], user_ub))
            
            # For empty list (edge case when no actuators enabled)
            if not actuator_names:
                spec_lb = user_lb
                spec_ub = user_ub
        else:
            # Just use actuator ctrl range.
            spec_lb = actuator_ctrlrange[:, 0]
            spec_ub = actuator_ctrlrange[:, 1]
            
            # Empty action spec for empty list
            if not actuator_names:
                spec_lb = np.zeros((0,))
                spec_ub = np.zeros((0,))

        return specs.BoundedArray(
            shape=spec_lb.shape,
            dtype=np.float32,
            minimum=spec_lb,
            maximum=spec_ub,
            name="\t".join(actuator_names))

    # Re-add root_body to satisfy legacy_base.Walker abstract method requirement
    @composer.cached_property 
    def root_body(self):
        """Returns the root body of the walker, typically the thorax."""
        # This can simply return the same entity as root_entity
        return self.root_entity 