# Architecture Overview

AdaptiveRL is structured around a minimal, modular pipeline consisting of four primary components:

```text
src/adaptive_rl/
├── environments/      # 3D kinematic drone, 6-DOF rigid-body quadrotor, & disturbed drone (Gymnasium)
├── algorithms/        # RL algorithm registry and wrappers (Stable-Baselines3 PPO & SAC)
├── planners/          # Classical motion-planning baselines (A* 3D lattice planner)
├── training/          # Unified RL training loop (RLTrainer/PPOTrainer), checkpointing, and artifact export
├── evaluation/        # Multi-episode deterministic evaluator, policy comparison, and metrics
├── benchmarking/      # Controlled ablations and cross-algorithm benchmarking (PPO vs SAC)
├── experiments/       # Preregistered online adaptation vs fixed shift runner (Protocol v2.0)
├── manifest.py        # Reproducibility metadata manifest generator (Git, OS, hardware, hashes)
├── config.py          # Strongly typed Pydantic configuration schemas
└── cli.py             # User-facing Typer CLI application
```

## Component Details

### 1. Environments (`adaptive_rl.environments`)

| Property | 3-DOF | 6-DOF |
|---|---|---|
| Observation dimensions | 29 | 36 |
| Action dimensions | 3 | 4 |
| Control | Acceleration | Motor thrust |
| Attitude | Not modeled | Quaternion |
| Angular velocity | Not modeled | 3-axis |

- **`DroneNavigation3DEnv` (`drone`, `drone_3d`)**: Point-mass 3-DOF kinematic translation with semi-implicit Euler stepping and linear aerodynamic drag ($c_{\text{drag}} = 0.05$). Continuous 3D acceleration commands $a \in [-1.0, 1.0]^3$ and 29-dimensional observation space.
- **`Drone6DOFEnv` (`drone-6dof`, `drone_6dof`)**: Rigid-body 6-DOF quadrotor flight dynamics featuring:
  - Unit quaternion attitude kinematics ($\mathbf{q} \in \mathbb{R}^4, \|\mathbf{q}\|=1.0$).
  - Euler rigid-body rotational equations with gyroscopic cross-coupling and rotational damping.
  - 4-rotor "X" configuration mapping motor thrusts to collective thrust and body roll, pitch, and yaw torques.
  - Runge-Kutta 4th-order (RK4) integration with 5 internal substeps and per-substep quaternion re-normalization.
  - Continuous swept-sphere and swept-box collision checking preventing high-speed obstacle tunneling.
  - 4 continuous motor thrust actions $a \in [-1.0, 1.0]^4 \to [0, F_{\max}]^4$ and 36-dimensional observation space.
  - Fully inspectable decomposed reward terms (progress, step, effort, angular rate, tilt, terminal).
- **`DroneDisturbed3DEnv` (`drone_disturbed`, `drone_disturbance`)**: 3D drone navigation subject to steady prevailing wind, stochastic Ornstein-Uhlenbeck (OU) gusts, moving dynamic obstacles with boundary reflection, and step-indexed disturbance recovery tracking.

### 2. Visualization & Flight Deck (`adaptive_rl.gui`)
- **Interactive 3D Arena Plotly Visualizer (`build_arena_3d_figure`, `compute_quadrotor_3d_geometry`)**:
  - **Volumetric 3D Quadrotor Airframe**:
    - Central aerodynamic fuselage pod with upper/lower chamfered contours and forward nose.
    - 4 volumetric cylindrical arms arranged in the exact physical X-configuration.
    - 4 structural motor mountings and cylindrical rotor hubs.
    - 4 aerodynamic twisted dual-blade propellers with distinct front/rear rotor coloring and semi-transparent rotor disc blur.
    - Port and starboard tubular landing gear skids connected by angled structural struts.
    - Forward-facing optical sensor gimbal and camera assembly.
    - Dual rear telemetry antenna masts.
    - Illuminated navigation chevron indicating aircraft heading vector ($+\mathbf{X}_B$).
  - **Physical Dimension Consistency**:
    - Visualizer dimensions are strictly aligned with the physics engine: default `arm_span = 0.4 m` ($2 \times \text{arm\_length}$) and `rotor_radius = 0.09 m` ($0.45 \times \text{arm\_length}$).
    - Rotor hubs sit at radial distance $\sqrt{d^2 + d^2} = 0.2\text{ m}$ where $d = 0.2 / \sqrt{2} \approx 0.1414\text{ m}$, exactly matching `DroneDynamics6DOF(arm_length=0.2)`.
  - **Attitude Transformation**:
    - Exact rigid-body body-to-world transformation applied to all airframe vertex meshes: $\mathbf{p}_W = \mathbf{p}_{\text{center}} + R(\mathbf{q}) \mathbf{p}_B$.
  - **LiDAR Sensing Representation**:
    - 16-ray LiDAR beam visualization with configurable world-fixed (`body_frame_lidar=False`, default for backward compatibility) or body-fixed (`body_frame_lidar=True`) coordinate frame semantics.
  - **Flight Deck Telemetry & Scrubber**:
    - Scrubbing slider providing step-indexed inspection of 3D positions, Euler attitude angles (roll, pitch, yaw), linear velocities, body angular rates, goal distance, and obstacle clearance.
    - Reliable outcome differentiation: distinctly categorizes and displays `SUCCESS` (green), `COLLISION` (red with collision boundary/obstacle type and collision count), `TIMEOUT` (amber), `MAX_STEPS` (amber-orange), and `UNKNOWN`.
- **Streamlit Application (`app.py`)**:
  - Full aerospace mission-control styling: dark graphite background, navy telemetry panels, and high-contrast instruments.
  - Dynamics model selector supporting both 3-DOF Kinematic Drone and 6-DOF Quadrotor.
  - **Custom Destination Selector & Target Validation (`validate_target_coordinates`)**:
    - Real-time numerical X/Y/Z coordinate entry with units in metres.
    - Pre-flight validation enforcing arena boundary limits, finite float checks, boundary collision margins ($d_{\text{wall}} > r_{\text{collision}} = 0.8\text{ m}$), and initial separation ($d_{\text{start}} > r_{\text{target}} = 1.5\text{ m}$).
    - Direct parameter propagation to `DroneNavigation3DEnv(goal_pos=...)` and `Drone6DOFEnv(goal_pos=...)`, affecting observation vectors, progress rewards, obstacle procedural generation, and success criteria.
    - 3D visual waypoint marker displaying exact coordinate text, arrival radius ring, altitude beacon column, and ground pad.
    - Session configuration fingerprinting (SHA-256 of environment, target coordinates, random seed, obstacle count, and policy checkpoint) invalidating stale simulation flight states upon any parameter modification.
    - Generalization advisory notice for pretrained models navigating to out-of-distribution targets.
  - Checkpoint integrity inspection detecting verified vs unverified shape-matched models and displaying parameter fingerprints.

### 3. Architecture Distinction: Three.js Browser Demo vs Python 6-DOF Flight Deck
- **GitHub Pages Showcase (`docs/index.html`)**:
  - A lightweight client-side Three.js interactive demonstration designed for universal browser access without requiring a Python runtime or GPU dependencies.
  - Operates on a simplified **3-DOF point-mass kinematic model** implemented entirely in JavaScript. It does *not* execute Python code, PyTorch neural networks, or rigid-body 6-DOF equations.
- **Python Flight Deck (`app.py` & `adaptive_rl`)**:
  - The genuine Python engineering simulator executing the full rigid-body 6-DOF flight dynamics engine (`Drone6DOFEnv`) with quaternion attitude kinematics, Euler rotational equations, motor RPM/thrust mapping, and RK4 integration.
  - Features genuine PyTorch PPO and SAC inference, swept collision detection, and volumetric Plotly 3D airframe rendering.
- **Educational Simulation Scope**:
  - The 6-DOF rigid-body equations are theoretical and educational. The simulator has not been validated against physical flight telemetry from physical drone hardware. Direct deployment to autopilots (e.g. PX4 or ArduPilot) requires hardware-in-the-loop validation and cascaded low-level PID attitude controllers.

### 2. Algorithms & Training (`adaptive_rl.algorithms`, `adaptive_rl.training`)
- **PPO (`PPOAlgorithm`)**: On-policy actor-critic algorithm wrapped from Stable-Baselines3.
  - Multi-Layer Perceptron (`MlpPolicy`).
  - Generalized Advantage Estimation (GAE) with $\gamma=0.99, \lambda=0.95$.
- **SAC (`SACAlgorithm`)**: Off-policy maximum-entropy actor-critic algorithm wrapped from Stable-Baselines3.
  - Replay buffer size $100{,}000$, soft target update $\tau=0.005$, learning rate $3 \times 10^{-4}$, batch size $256$.
- **Trainer (`RLTrainer` / `PPOTrainer`)**: Generalized trainer dispatching algorithms by configuration (`config.algorithm.name`), orchestrating callbacks, periodic checkpointing, and structured metadata logging.

### 3. Evaluation & Benchmarking (`adaptive_rl.evaluation`, `adaptive_rl.benchmarking`)
- Evaluates trained checkpoints across deterministic seed sets.
- Tracks binary outcomes: Target Reached (Success), Obstacle Collision, Boundary Violation, Timeout.
- Measures trajectory metrics: path length, path efficiency, obstacle surface clearance, max velocity/acceleration.
- Provides comparative benchmarking workflow (`adaptive-rl benchmark compare-algorithms`) under strictly fair, identical environment configurations and evaluation seeds.
- Serializes evaluation reports to `artifacts/evaluation.json`, `artifacts/evaluation.csv`, and `artifacts/algorithm_comparison.json`.

### 4. Classical Planning Baseline (`adaptive_rl.planners.astar3d`)
- Implements `AStar3DPlanner`: a deterministic classical 3D motion-planning baseline operating on a spatial lattice.
- Independent of PyTorch, trained weights, and reinforcement learning dependencies.
- **Search Space & Neighborhood**: Configurable 3D lattice resolution (default: 0.5 m) supporting 6-connected (axis-aligned) or 26-connected (diagonal) motion.
- **Heuristic**: Admissible and consistent Euclidean distance $h(n) = \|pos(n) - goal\|$.
- **Analytical Clearance Geometry**:
  - Every segment $[p_0, p_1]$ is verified using exact segment-to-sphere analytical distance:
    $$t^* = \text{clamp}\left(\frac{(C - p_0) \cdot (p_1 - p_0)}{\|p_1 - p_0\|^2}, 0, 1\right)$$
    $$\text{dist}(segment, C) = \|p_0 + t^* (p_1 - p_0) - C\|$$
    $$\text{clearance} = \text{dist}(segment, C) - (R_{\text{obs}} + R_{\text{coll}})$$
  - Enforces arena boundary clearance: $\min(p_0[i], p_1[i]) - R_{\text{coll}} > 0$ and $\max(p_0[i], p_1[i]) + R_{\text{coll}} < \text{bounds}[i]$.
- **Empty-Arena Behavior**: Returns direct optimal two-point straight line $[start, goal]$ with 100% path efficiency without intermediate lattice waypoints.
- **Fair Benchmarking Adapter**: `compare_with_planner` evaluates PPO, AStar3D, and Random Policy under identical procedural seeds and exports structured comparison reports to `artifacts/evaluation_planner_comparison.json`.
- **CLI Command**:
  ```bash
  adaptive-rl evaluate --model artifacts/models/drone_ppo_demo_final.zip --compare-planner astar
  ```
- **Limitations & Feasibility Paradigms**:
  - **Dynamic Feasibility (PPO & Random Policy)**: Evaluated through closed-loop physics simulation in `DroneNavigation3DEnv`. The drone must generate continuous acceleration control commands to navigate under inertia, velocity drag, and actuation limits. Success requires dynamically executing the trajectory without colliding.
  - **Geometric Feasibility (A* Planner)**: Evaluated as open-loop 3D spatial path planning. Success denotes finding an obstacle-free, boundary-clearing geometric path from start to goal on the discretized spatial lattice. The path is not executed through drone attitude dynamics or closed-loop tracking control.

### 5. Perception Subsystem & Phase V Status (`adaptive_rl.perception`)

The perception subsystem models exteroceptive sensing, coordinate transformations, and sensor observations:

* **Distinction Between Sensing Paradigms**:
  - **Legacy 16-Ray LiDAR**: Embedded directly within 3-DOF (29-D) and 6-DOF (36-D) Gym observation vectors for vector-based RL training (PPO & SAC).
  - **Modular Phase V Perception**: Autonomous sensor architecture producing immutable, timestamped 3D point clouds (`PointCloud3D`) adhering to ADR-002, ADR-003, and ADR-006 with zero simulation ground-truth leakage.

* **Implementation Status**:
  - **Phase V.1 (Simulated 3D LiDAR & PointCloud3D Foundation)**: `IMPLEMENTED`. Vectorized ray-sphere and ray-box intersection, Gaussian noise, statistical dropout, and caller-managed timing.
  - **Phase V.1.1 (Sensor Contract Hardening)**: `IMPLEMENTED` (Verified locally). Strict SO(3) Frobenius norm validation, positive exit intersections for origins inside obstacles, and input finiteness enforcement.
  - **Phase V.2 (3D Obstacle Detection & Clustering)**: `NOT IMPLEMENTED` (Planned).
  - **Phase V.3 (Dynamic Obstacle Tracking & Kalman Filtering)**: `NOT IMPLEMENTED` (Planned).
  - **Phase V.4 (Multi-Sensor Fusion & Calibration)**: `NOT IMPLEMENTED` (Planned).
  - **Phase V.5 (Perception-Aware Planner Integration)**: `NOT IMPLEMENTED` (Planned). Navigation planners currently operate on direct ground-truth obstacles, not detected perception tracks.
