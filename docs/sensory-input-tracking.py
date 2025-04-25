#!/usr/bin/env python
# coding: utf-8

# # An example of tracking sensory input and action commands during episode rollout
# 
# During episode rollout, it's trivial to access and record any quantity involved in the MuJoCo physics simulation, whether this quantity is part of the model observables or not.

# # Imports

# In[ ]:


import os
import requests
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageDraw
import mediapy
from tqdm import tqdm

import tensorflow as tf
import tensorflow_probability as tfp
from acme import wrappers

from flybody.download_data import figshare_download
from flybody.fly_envs import flight_imitation
from flybody.agents.utils_tf import TestPolicyWrapper


# In[2]:


# Prevent tensorflow from stealing all the gpu memory.
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

render_kwargs = {'width': 640, 'height': 480}


# ## Download the WPG base pattern and a trained flight policy
# 
# This cell will download the required data to local `flybody-data` directory. The `flybody` supplimentary data can also be accessed at <https://doi.org/10.25378/janelia.25309105>

# In[10]:


figshare_download(['flight-imitation-dataset', 'trained-policies'])

wpg_path = 'flybody-data/datasets_flight-imitation/wing_pattern_fmech.npy'
ref_flight_path = 'flybody-data/datasets_flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5'
flight_policy_path = 'flybody-data/trained-fly-policies/flight'


# # Create flight imitation task environment

# In[8]:


env = flight_imitation(
    ref_path=ref_flight_path,
    wpg_pattern_path=wpg_path,
    terminal_com_dist=float('inf'),
)
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

_ = env.reset()
pixels = env.physics.render(camera_id=1, **render_kwargs)
PIL.Image.fromarray(pixels)


# In[9]:


env.observation_spec()


# In[10]:


env.action_spec()


# # Load a pre-trained flight policy

# In[13]:


# Load flight policy.
policy = tf.saved_model.load(flight_policy_path)
policy = TestPolicyWrapper(policy)


# # Rollout an episode and record:
# 1. Observable sensory inputs
# 2. Action commands
# 3. Unobserved quantities from physics simulator (fly body position and orientation)
# 4. (also record video)

# In[ ]:


n_steps = 150

timestep = env.reset()

# Allocate.
joints_pos = np.zeros((n_steps, 25))  # Proprioception: observable joint angles.
vel = np.zeros((n_steps, 3))  # Velocimeter.
zaxis = np.zeros((n_steps, 3))  # Gravity direction.
root_qpos = np.zeros((n_steps, 7))  # Fly's position and orientation.
actions = np.zeros((n_steps, 6))  # Wing action commands.

frames = []

for i in tqdm(range(n_steps)):

    # Record some of the sensory inputs.
    joints_pos[i] = timestep.observation['walker/joints_pos']
    vel[i] = timestep.observation['walker/velocimeter']
    zaxis[i] = timestep.observation['walker/world_zaxis']
    root_qpos[i] = env.physics.data.qpos[:7].copy()
    
    frames.append(env.physics.render(camera_id=1, **render_kwargs))

    # Step simulation.
    action = policy(timestep.observation)
    timestep = env.step(action)

    # Record wing action commands.
    actions[i] = action[3:9]


# ## Show rollout video

# In[ ]:


mediapy.show_video(frames)


# In[26]:


# Print observable joint names and indices.
[(i, joint.name) for i, joint in enumerate(env.task.walker.observable_joints)]


# # Plot example sensory inputs from the rollout

# ### Proprioception sensory inputs: head, abdomen, wing joint angles

# In[27]:


time_axis = np.arange(n_steps) * env.control_timestep() * 1000  # ms

plt.figure(figsize=(6, 10))
plt.suptitle('Proprioception sensory inputs: head, abdomen, wing joint angles')
plt.subplot(4, 1, 1)  # Head joints.
plt.plot(time_axis, joints_pos[:, :3], label=['head_abduct', 'head_twist', 'head'])
plt.ylabel('Head joint angles (rad)')
plt.legend()
plt.subplot(4, 1, 2)  # Abdomen joints.
plt.plot(time_axis, joints_pos[:, 9:23])
plt.ylabel('Abdomen joint angles (rad)')
plt.subplot(4, 1, 3)  # Left wing.
plt.plot(time_axis, joints_pos[:, 3:6], label=['yaw', 'roll', 'pitch'])
plt.ylabel('Left wing angles (rad)')
plt.legend()
plt.subplot(4, 1, 4)  # Right wing.
plt.plot(time_axis, joints_pos[:, 6:9])
plt.xlabel('Time (ms)')
plt.ylabel('Right wing angles (rad)')
plt.tight_layout()


# ### Egocentric velocity vector

# In[28]:


plt.plot(time_axis, vel, label=['vx', 'vy', 'vz'])
plt.xlabel('Time (ms)')
plt.ylabel('x,y,z components of egocentric\nvelocity vector (cm/s)')
plt.legend()


# ### Egocentric gravity direction vector

# In[29]:


plt.plot(time_axis, zaxis, label=['x', 'y', 'z'])
plt.xlabel('Time (ms)')
plt.ylabel('x,y,z components of egocentric\ngravity direction unit vector')
plt.legend()


# ### Wing action commands

# In[30]:


# Unitless, between (-1, 1).
plt.subplot(2, 1, 1)
plt.plot(time_axis, actions[:, :3], label=['yaw', 'roll', 'pitch'])
plt.ylabel('Left wing control\n(unitless)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time_axis, actions[:, 3:])
plt.xlabel('Time (ms)')
plt.ylabel('Right wing control')


# ### Fly body position and orientation in global coordinates

# Beyond what is directly observable to the fly model, any other simulation-related quantity can be accessed directly from the simulator.
# 
# The full state of the MuJoCo physics simulation is encapsulated in a single `MjData` datastructure. For more information see:
# </br>
# https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjdata

# In[31]:


plt.subplot(2, 1, 1)
plt.plot(time_axis, root_qpos[:, :3], label=['x', 'y', 'z'])
plt.ylabel('x,y,z position of root joint\nin world coordinates (cm)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time_axis, root_qpos[:, 3:], label=['w', 'x', 'y', 'z'])
plt.xlabel('Time (ms)')
plt.ylabel('Components of root joint\nquaternion in world\ncoordinates (unitless)');


# In[ ]:




