# Flybody Perception System

This document details the perception mechanisms within the Flybody simulation environment, outlining how observations are generated and structured. Understanding this is crucial for integrating alternative perception modules or analyzing sensory processing.

## Core Components

1.  **MuJoCo Model (`fruitfly.xml`):** The foundation is the detailed MuJoCo XML model (`flybody/fruitfly/assets/fruitfly.xml`). This file defines the physical structure of the fly, including its body parts, joints, and importantly, the placement and types of sensors.
2.  **`FruitFly` Walker (`flybody/fruitfly/fruitfly.py`):** This class wraps the MuJoCo model, providing methods to interact with it and defining interfaces for actuators and observables.
3.  **`FruitFlyObservables` (`flybody/fruitfly/fruitfly.py`):** This class, inheriting from `dm_control.locomotion.walkers.legacy_base.WalkerObservables`, defines how raw physics states and sensor readings are processed into meaningful observations exposed to the agent or environment.
4.  **Task (`flybody/tasks/*.py`):** Specific tasks (e.g., `FlightImitationWBPG`, `WalkImitation`) can define additional, task-relevant observables.
5.  **`dm_control.composer.Environment`:** This framework orchestrates the interaction between the walker, arena, and task, ultimately providing the `TimeStep` object which contains the `observation` dictionary.

## Sensor Modalities

The `fruitfly.xml` model defines several types of MuJoCo sensors:

*   **Vestibular/Inertial Sensors:**
    *   `accelerometer`: Measures linear acceleration (attached to thorax).
    *   `gyro`: Measures angular velocity (attached to thorax).
    *   `velocimeter`: Measures linear velocity (attached to thorax).
*   **Proprioception:**
    *   `jointpos`: Measures the angular position of actuated joints.
    *   `jointvel`: Measures the angular velocity of actuated joints.
    *   `tendonpos`, `tendonvel`: Measure position and velocity of tendons (used in legs).
    *   `actuatorfrc`: Measures the force produced by actuators.
*   **Exteroception (Contact):**
    *   `touch`: Sensors placed on various body parts (e.g., leg tips, abdomen) measuring contact forces.
    *   `force`, `torque`: Sensors associated with joints or specific sites.
*   **Vision:**
    *   `eye_right`, `eye_left`: Two egocentric `camera` sensors simulating the fly's compound eyes. Their field of view (`fovy`) and resolution (`width`, `height`) are configurable in the `FruitFly` constructor.

## Observation Processing (`FruitFlyObservables`)

The `FruitFlyObservables` class defines specific named observables derived from the sensors and physics state. Key observables include:

*   **Pose/Orientation:**
    *   `thorax_height`, `abdomen_height`: Vertical position of key body parts.
    *   `world_zaxis`, `world_zaxis_abdomen`, `world_zaxis_head`: Orientation of the Z-axis of specific body frames relative to the world frame (useful for posture).
    *   `orientation`: Provides `world_zaxis`, `world_zaxis_abdomen`, `world_zaxis_head` as a group.
*   **Vestibular:**
    *   `accelerometer`, `gyro`, `velocimeter`: Direct readings from the corresponding sensors.
    *   `vestibular`: Provides `accelerometer`, `gyro`, `velocimeter` as a group.
*   **Proprioception:**
    *   `joint_pos`, `joint_vel`: Readings from all enabled `jointpos` and `jointvel` sensors.
    *   `proprioception`: Provides `joint_pos`, `joint_vel` as a group.
    *   `actuator_activation`: The current activation state (control input) of the actuators.
    *   `appendages_pos`: Egocentric coordinates of key appendage sites (e.g., leg tips).
*   **Contact/Force:**
    *   `force`, `touch`: Readings from force and touch sensors.
    *   `self_contact`: A measure derived from contact forces between the fly's own body parts.
*   **Vision:**
    *   `right_eye`, `left_eye`: Provides the rendered image data (NumPy arrays) from the corresponding camera sensors.

## Task-Specific Observations

Tasks often add observations crucial for their specific goals. For example:

*   **`FlightImitationWBPG` / `WalkImitation`:**
    *   `ref_displacement`: The 3D vector from the agent's Center of Mass (CoM) to the reference trajectory's CoM.
    *   `ref_root_quat`: The relative quaternion representing the orientation difference between the agent's root body and the reference trajectory's root body.
    *   Future reference trajectory points: While not always exposed as direct observables, the task logic often uses future steps of the reference trajectory (e.g., `_future_steps` parameter) to guide rewards or internal state.

## Observation Space Structure

The final observation provided to an agent (e.g., via `env.step()` or `env.reset()`) is typically a dictionary (often an `OrderedDict`) where keys are the names of the enabled observables (e.g., `'walker/joint_pos'`, `'task/ref_displacement'`, `'walker/right_eye'`) and values are NumPy arrays containing the corresponding data.

The exact set of available observables depends on:
1.  The configuration of the `FruitFly` walker (e.g., `use_legs`, `use_wings`).
2.  The specific `Task` being used.
3.  The `observables_options` potentially passed to the `Task` or `Environment`.

## Integration Points for Alternative Perception

To replace or augment the perception system:

1.  **Modify `FruitFlyObservables`:** Add new observables derived from existing sensors or physics state, or modify how existing observables are calculated.
2.  **Add Custom Sensors:** Modify the `fruitfly.xml` to include new MuJoCo sensor types if needed.
3.  **Wrap the Environment:** Create a wrapper around the `composer.Environment` that modifies the observation dictionary after `env.step()`/`env.reset()`, inserting custom perceptual data or replacing existing entries.
4.  **Task-Level Modification:** Implement custom perception logic within a new or modified `Task` class, adding task-specific observables. 