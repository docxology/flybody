# Flybody Action System

This document details the action and control mechanisms within the Flybody simulation environment. It describes how agent actions are defined, processed, and applied to the simulated fruit fly model. This is essential for developing alternative action selection or motor control modules.

## Core Components

1.  **MuJoCo Model (`fruitfly.xml`):** The XML model (`flybody/fruitfly/assets/fruitfly.xml`) defines the actuators associated with the fly's joints and other controllable elements (like adhesion).
2.  **`FruitFly` Walker (`flybody/fruitfly/fruitfly.py`):** This class initializes the actuators based on the XML and constructor arguments (e.g., `force_actuators`). It defines the action space specification (`get_action_spec`) and applies the processed actions to the MuJoCo physics simulation (`apply_action`).
3.  **Task (`flybody/tasks/*.py`):** Specific tasks can intercept, interpret, or modify the agent's raw action *before* it is applied to the walker's actuators. This allows for implementing higher-level control schemes or integrating pattern generators.
4.  **`dm_control.composer.Environment`:** The composer environment receives the action from the agent (e.g., in `env.step(action)`) and passes it down through the task and walker hierarchy.

## Actuator Types

The `fruitfly.xml` model defines several types of MuJoCo actuators:

*   **`motor` (Joint Control):** These are the primary actuators for controlling joint movements.
    *   **Position Control:** When `force_actuators=False` (in `FruitFly` constructor), these actuators typically function as position servos, aiming for a target joint angle. The `kp` (stiffness) parameter in the XML determines how strongly they drive towards the target.
    *   **Force/Torque Control:** When `force_actuators=True`, these actuators directly apply a torque (for hinge joints) or force (for slide joints) proportional to the control signal.
    *   **Wings:** Wing actuators typically operate in force control mode regardless of the `force_actuators` setting for other body parts.
*   **`adhesion`:** Special actuators used to simulate sticky feet, primarily for walking tasks. They allow the feet (tarsi) geoms to stick to surfaces.
*   **`tendon`:** Some leg joints are actuated via tendons, which are driven by their own actuators (often position-controlled).

## Actuator Configuration

*   **Enabling/Disabling:** The `FruitFly` constructor (`use_legs`, `use_wings`, `use_mouth`, etc.) determines which body parts (and their associated joints/actuators) are active. Disabled parts are often retracted, and their actuators/joints/sensors are removed from the model during initialization.
*   **Filtering:** `joint_filter` and `adhesion_filter` parameters in the `FruitFly` constructor allow applying low-pass filters to the actuator control signals. This smooths the action application over time, simulating muscle activation dynamics or damping jerky movements. MuJoCo's `filter` or `filterexact` dyntypes can be used.

## Action Space (`get_action_spec`)

The `FruitFly.get_action_spec` method defines the structure and bounds of the action space exposed to the agent. Key characteristics:

*   **Structure:** It is typically a flat NumPy array (`dm_env.specs.BoundedArray`).
*   **Components:** The array concatenates the control signals for all *enabled* actuators, grouped by category (e.g., `adhesion`, `head`, `wings`, `legs`). The order is determined by the `_ACTION_CLASSES` dictionary and the order of actuators within the XML.
*   **Bounds:** Control values are usually scaled to a standard range, typically `[-1, 1]`. The `apply_action` method internally maps these scaled values back to the actual force/position ranges defined for each actuator in the XML.
*   **User Actions:** The `num_user_actions` parameter allows adding extra dimensions to the action array. These dimensions are not directly tied to specific actuators but can be used by the `Task` for custom control logic (see Task-Specific Handling).

## Action Application Flow

1.  **Agent:** An external agent (e.g., a reinforcement learning policy) outputs an action array conforming to the `action_spec`.
2.  **Environment (`env.step(action)`):** The action is passed to the environment.
3.  **Task (`task.before_step(physics, action, ...)`):** The task receives the action *first*. It can:
    *   Pass the action through unmodified.
    *   Interpret parts of the action (especially `user` actions) to modulate internal controllers or pattern generators.
    *   Modify the action components before they reach the walker.
    *   **Example (`FlightImitationWBPG`):** Reads the `user` action value, uses it to set the frequency of a `WingBeatPatternGenerator (wbpg)`, gets target wing *positions* from the `wbpg`, calculates the necessary *forces* to achieve those positions, and *adds* these forces to the wing actuator components of the `action` array.
4.  **Walker (`walker.apply_action(physics, action, ...)`):** The (potentially modified) action is received by the walker.
    *   It retrieves the previous action (potentially needed for filtering).
    *   It maps the scaled action values (`[-1, 1]`) back to the physical control ranges defined in the XML.
    *   It applies filtering if `joint_filter` or `adhesion_filter` are enabled.
    *   It writes the final control values to the `physics.data.ctrl` array, which MuJoCo uses during the next physics steps.
5.  **Physics Simulation:** MuJoCo simulates the physics using the applied `ctrl` values.

## Integration Points for Alternative Action/Control

To implement custom action selection or motor control:

1.  **Task-Level Control:** Implement the core logic within a custom `Task` class's `before_step` method. This is suitable for hierarchical control, pattern generators, or interpreting high-level actions.
    *   Define `num_user_actions` in the `FruitFly` constructor to receive high-level commands.
    *   Modify the `action` array within `before_step` before calling `super().before_step`.
2.  **Modify `FruitFly.apply_action`:** Change how actions are processed *after* the task but *before* being written to `physics.data.ctrl`. This could involve custom filtering, mapping, or reflex implementations.
3.  **Replace Actuators:** Modify the `fruitfly.xml` to use different actuator types (e.g., velocity control) if needed, potentially requiring adjustments to `apply_action` or task logic.
4.  **Environment Wrapper:** Wrap the `composer.Environment` to intercept actions before they even reach the task, although task-level modifications are generally preferred for integrating with the existing structure. 