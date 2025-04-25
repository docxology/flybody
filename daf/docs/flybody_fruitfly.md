# FlyBody Fruit Fly Model (`fruitfly/`)

This document describes the `flybody/fruitfly` directory, which contains the definition, assets, and build process for the simulated fruit fly model used in the FlyBody framework.

## Overview

This directory defines the physical structure, actuators, sensors, and visual appearance of the fruit fly model within the MuJoCo physics simulator.

## Directory Structure

```
flybody/fruitfly/
├── __init__.py             # Package initializer
├── fruitfly.py             # Main Python class defining the fly model logic
├── assets/                 # Contains the core XML and mesh files
│   ├── fruitfly.xml        # The final compiled MuJoCo XML model
│   ├── drosophila.xml      # Base XML structure for the fly
│   ├── drosophila_defaults.xml # Default physics/visual settings
│   └── meshes/             # Directory containing .stl or .obj mesh files for fly parts
└── build_fruitfly/         # Scripts and templates for building the fly model
    ├── make_fruitfly.py    # Python script to assemble the final fruitfly.xml
    └── fruitfly.xml        # (Likely a template or intermediate build file)
```

## Key Components

- **`fruitfly.py`**: Defines the `FruitFly` class, which inherits from `legacy_base.Walker` (likely from `dm_control`). This class programmatically builds the fly model, configures its parts (legs, wings, mouth, antennae based on arguments like `use_legs`, `use_wings`), sets up actuators (position/force control, filters), defines observables, and manages action application.
    - **Observables**: Defines various sensor readings available to an agent, such as `thorax_height`, `world_zaxis`, `force`, `touch`, `accelerometer`, `gyro`, `velocimeter`, `joints_pos`, `joints_vel`, `appendages`, `left_eye`, `right_eye`.
    - **Action Mapping**: Manages how agent actions map to MuJoCo control inputs (`apply_action`, `_ctrl_indices`, `_action_indices`).
    - **Configuration**: Handles options like `force_actuators`, `joint_filter`, `adhesion_filter`, `body_pitch_angle`, `eye_camera_fovy`, etc.

- **`assets/fruitfly.xml`**: The primary MuJoCo XML file loaded by the simulation. It defines the entire kinematic tree (bodies, joints, geoms), physics properties (mass, inertia, friction, damping), visual meshes, actuators (motors, adhesion), and sensors (touch, force, IMU, cameras).

- **`assets/meshes/`**: Contains the 3D model files (likely STL) that define the visual geometry of each part of the fly (thorax, head, leg segments, wings, etc.).

- **`build_fruitfly/make_fruitfly.py`**: This script takes base XML files (like `drosophila.xml`, `drosophila_defaults.xml`) and programmatically modifies and assembles them to generate the final `assets/fruitfly.xml`. It might involve adding/removing components, setting parameters, defining sensors, and ensuring consistency.

## Model Features

- **Detailed Kinematics**: Models the major body segments (thorax, head, abdomen) and appendages (legs with multiple joints, wings, mouthparts, antennae).
- **Physics Simulation**: Leverages MuJoCo for realistic physics simulation, including contacts, friction, and joint dynamics.
- **Actuation**: Includes various actuators:
    - General actuators for joints (position or force control).
    - Adhesion actuators for simulating sticky feet.
- **Sensors**: Equipped with a rich set of sensors:
    - Proprioceptive: Joint position and velocity sensors.
    - Vestibular: Accelerometer, Gyroscope, Velocimeter (simulating inertial measurement units).
    - Exteroceptive: Touch sensors on feet, Force sensors on leg segments.
    - Visual: Left and right eye cameras with configurable field-of-view (`fovy`) and resolution.
- **Configurability**: The model can be configured to enable/disable parts like legs or wings, switch actuator types, and adjust parameters, allowing for different simulation scenarios (e.g., walking vs. flight). 