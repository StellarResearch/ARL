# ARL (Adaptive Reinforcement Learning) 

AdaptiveRL is an educational reinforcement-learning project in which a Proximal Policy Optimization (PPO) agent learns to navigate a simulated 3D drone through an obstacle-filled environment toward a target coordinate.

🌐 **Project Website & Interactive Showcase:** [https://stellarresearch.github.io/ARL/](https://stellarresearch.github.io/ARL/)

---

## Table of Contents

- [What Is AdaptiveRL?](#what-is-adaptiverl)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Linux Support](#linux-support)
- [Prerequisites](#prerequisites)
- [Step 1 — Install System Requirements](#step-1--install-system-requirements)
- [Step 2 — Download ARL](#step-2--download-arl)
- [Step 3 — Check Python](#step-3--check-python)
- [Step 4 — Create a Virtual Environment](#step-4--create-a-virtual-environment)
- [Step 5 — Upgrade Packaging Tools](#step-5--upgrade-packaging-tools)
- [Step 6 — Install AdaptiveRL](#step-6--install-adaptiverl)
- [Step 7 — Verify Installation](#step-7--verify-installation)
- [Step 8 — Run Tests](#step-8--run-tests)
- [Step 9 — Train the PPO Agent](#step-9--train-the-ppo-agent)
- [Training Output & Artifacts](#training-output--artifacts)
- [Step 10 — Evaluate the Trained Agent](#step-10--evaluate-the-trained-agent)
- [Step 11 — Run the Command-Line Drone Demo](#step-11--run-the-command-line-drone-demo)
- [Step 12 — Launch the 3D GUI](#step-12--launch-the-3d-gui)
- [GUI Walkthrough](#gui-walkthrough)
- [Quick Start](#quick-start)
- [Technical Details](#technical-details)
- [Reward Function](#reward-function)
- [PPO Explanation](#ppo-explanation)
- [Measured Benchmark Results](#measured-benchmark-results)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Leaving the Environment](#leaving-the-environment)
- [Updating an Existing Installation](#updating-an-existing-installation)
- [Clean Reinstall](#clean-reinstall)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## What Is AdaptiveRL?

AdaptiveRL is an educational reinforcement-learning project in which a PPO agent learns to navigate a simulated 3D drone through an obstacle-filled environment toward a target.

> [!IMPORTANT]
> **THIS IS AN EDUCATIONAL SIMULATION.**
>
> This project is an academic computer-science demonstration featuring:
> 1. A fast **3-DOF kinematic point-mass simulation** (`DroneNavigation3DEnv`) with 3 acceleration actions.
> 2. A physics-rich **6-DOF rigid-body quadrotor simulation** (`Drone6DOFEnv`) featuring unit quaternion attitude, Euler rotational equations, four individual motor thrusts, rotor torque mapping, Runge-Kutta 4th-order (RK4) integration, and full 3D quadrotor airframe visualization.
>
> It is **NOT**:
> - a real drone autopilot
> - a real-world flight controller
> - PX4 Autopilot or ArduPilot
> - calibrated against physical drone hardware telemetry
> - production hardware software

---

## Features

- **Dual Simulation Dynamics**:
  - **3-DOF Kinematic Drone** (`drone`, `drone_3d`): Fast point-mass kinematics with semi-implicit Euler integration, continuous 3D acceleration actions ($[-1, 1]^3$), and 29-dimensional observation space.
  - **6-DOF Quadrotor** (`drone-6dof`, `drone_6dof`): Theoretical rigid-body flight dynamics featuring unit quaternion attitude, Euler rotational dynamics, gyroscopic terms, 4-motor thrust control ($[-1, 1]^4 \to [0, F_{\max}]^4$), Runge-Kutta 4th-order (RK4) integration, continuous swept collision checking, and 36-dimensional observation space.

### 3-DOF Kinematic vs. 6-DOF Rigid-Body Simulation Comparison

| Property | 3-DOF Kinematic | 6-DOF Rigid-Body |
|---|---|---|
| Observation dimensions | 29 dimensions | 36 dimensions (includes quat, angular rates) |
| Action dimensions | 3 (accelerations $[-1, 1]^3$) | 4 (motor commands $[-1, 1]^4 \to [0, F_{\max}]^4$) |
| RL Algorithms | PPO & SAC | PPO & SAC |
| Attitude | Not modeled (upright point mass) | Unit quaternion $\mathbf{q} \in \mathbb{R}^4, \|\mathbf{q}\|=1.0$ |
| Angular velocity | Not modeled | 3-axis body angular velocity $\boldsymbol{\omega} \in \mathbb{R}^3$ |
| Integration | Semi-implicit Euler ($\Delta t = 0.05\text{ s}$) | Runge-Kutta 4th-order (5 substeps, $\Delta t_{\text{sub}} = 0.01\text{ s}$) |
| Motor Dynamics | Not modeled | Optional ($\tau_{\text{motor}} = 0.03\text{ s}$, default disabled) |
| Ground Effect | Not modeled | Optional (Cheeseman-Bennett, default disabled) |
| Collision Checking | Point-sphere / box check | Continuous swept-sphere & swept-box checking |
| Legacy LiDAR Sensing | 16-ray spherical rangefinder | 16-ray spherical rangefinder (world/body-frame) |
| Input Validation | Finite bounds | Strict (rejects NaN/Inf actions & motor thrusts) |

- **Real Volumetric 3D Quadrotor Geometry & Visualization**:
  - Full volumetric mesh geometry implemented with Plotly `go.Mesh3d` and NumPy (replacing wireframe-only lines):
    - Central aerodynamic fuselage pod with thickness and chamfered contours.
    - Four solid cylindrical structural arms in X configuration.
    - Four cylindrical motor mountings and rotor hubs.
    - Aerodynamic propeller blades with visible 3D volume, thickness, and counter-rotating dynamics.
    - Port/starboard landing gear skids and angled struts.
    - Forward optical sensor gimbal and rear telemetry antennas.
  - Exact physical dimensions synchronized directly with physics dynamics: `arm_span = 0.4 m` ($2 \times \text{arm\_length}$), `rotor_radius = 0.09 m`, and diagonal offset $d = 0.2 / \sqrt{2} \approx 0.1414\text{ m}$.
  - Distinct front (red) and rear (blue) arm/rotor visual identification plus forward heading nosecone pointer (+X_B direction).
  - Exact body-to-world rigid-body coordinate transformation: $\mathbf{p}_W = \mathbf{p}_{\text{drone}} + R(\mathbf{q}) \mathbf{p}_B$ preserved across all roll, pitch, and yaw attitudes.

### Modular Perception Architecture (Phase V Status)

- **Legacy 16-Ray LiDAR vs. Modular Phase V**:
  - **Legacy 16-Ray LiDAR**: Embedded directly within 3-DOF (29-D) and 6-DOF (36-D) Gym observation vectors for vector-based RL training (PPO & SAC).
  - **Modular Phase V Perception**: Standalone sensor subsystem (`adaptive_rl.perception.sensors`) producing immutable, strongly typed `PointCloud3D` observations with zero simulation ground-truth leakage.
- **Perception Implementation Status**:
  - **Phase V.1 Simulated 3D LiDAR**: `IMPLEMENTED`. Vectorized ray-sphere and ray-box intersection, Gaussian range noise, statistical dropout, and caller-managed timing.
  - **Phase V.1.1 Sensor Contract Hardening**: `IMPLEMENTED` & locally verified. Orthonormal SO(3) validation using explicit Frobenius norm ($\|R R^T - I\|_F \le 10^{-5}$), strictly positive exit intersections for origins inside or on the boundary of spherical obstacles ($t_{\text{exit}} = -b + \sqrt{\Delta}$), PointCloud3D validation, and non-finite input guards.
  - **Phase V.2 3D Obstacle Detection**: `NOT IMPLEMENTED` (Planned).
  - **Phase V.3 Dynamic Obstacle Tracking**: `NOT IMPLEMENTED` (Planned).
  - **Phase V.4 Multi-Sensor Fusion**: `NOT IMPLEMENTED` (Planned).
  - **Phase V.5 Perception-Aware Navigation & Planner Integration**: `NOT IMPLEMENTED` (Planned). Current motion planners and RL policies do not integrate with detected perception tracks.
- **Reinforcement Learning (PPO & SAC)**:
  - On-policy PPO and off-policy SAC using Stable-Baselines3.
  - Unified configuration pipelines (`configs/ppo_drone_6dof.yaml`, `configs/sac_drone_6dof.yaml`, `configs/drone_6dof.yaml`).
- **Interactive Streamlit Flight Deck (`app.py`)**:
  - Dynamics model selector (3-DOF Kinematic Drone vs. 6-DOF Quadrotor).
  - **User-Defined 3D Destination / Custom Target Selector**:
    - Users can set custom destination coordinates (`Target X`, `Target Y`, `Target Z` in metres) or restore environment defaults.
    - Pre-flight validation strictly checks finite numbers, arena boundary limits (`30.0 × 30.0 × 15.0 m`), boundary collision safety margins (`0.8 m`), and launch point separation (`> 1.5 m`). Invalid inputs block simulation launch without silent clamping.
    - The physics environments (`DroneNavigation3DEnv` and `Drone6DOFEnv`) directly instantiate `goal_pos`, ensuring observations, relative goal vectors, progress rewards, obstacle generation clearance, and arrival detection use the exact user-defined waypoint.
    - Trajectory visualization renders an emerald diamond destination waypoint with exact `[X, Y, Z]` metres annotations, a 3D arrival radius boundary (`1.5 m`), altitude beacon column, and ground projection pad.
    - Run configuration fingerprinting (SHA-256) invalidates stale flight data when target coordinates, random seed, obstacle count, or policy checkpoint change.
    - Five canonical flight outcomes distinctly handled: `SUCCESS`, `COLLISION`, `TIMEOUT`, `MAX_STEPS`, and `UNKNOWN`.
    - Custom targets are fully integrated into RL training (`EnvironmentConfig.parameters["goal_pos"]`) and comparative policy evaluation.
  - Trajectory playback scrubber updating quadrotor position, attitude, arm orientations, LiDAR (body-fixed or world-fixed), velocity, and telemetry (altitude, speed, Euler roll/pitch/yaw in degrees, angular velocity, goal distance).
  - Explicit simulation playback mode indicator distinguishing educational simulation playback from live physical hardware.
- **16-Ray Simulated LiDAR**: Spherical rangefinder calculating analytical distances to obstacles and arena boundaries with configurable noise and dropout.
- **Procedurally Generated Spherical Obstacles**: Placed with guaranteed clearance from launch and goal points.
- **Continuous Swept Collision Detection**: Spherical segment and bounding box collision checks preventing high-speed obstacle tunneling.
- **Target Detection**: Automatic goal-arrival termination within a calibrated target radius.
- **Reward Shaping & Decomposed Inspection**: Multi-component reward encouraging progress while penalizing collisions, excessive control effort, angular velocity, and tilt.
- **Deterministic Evaluation**: Reusable evaluation pipeline with reproducible seed control.
- **Autonomous Navigation & Path Planning (Phase III)**:
  - Deterministic 3D A* path planner (`AStarPlanner3D`) over a 26-connectivity grid with vehicle radius and safety margin obstacle inflation, start/goal clearance verification, and line-of-sight shortcutting.
  - Dynamic obstacle simulation (`DynamicObstacleSphere3D`) with reflective and periodic kinematic propagation.
  - Multi-waypoint mission manager (`MissionManager`) with state-machine sequencing, reactive replanning, and autonomous Return-to-Home (RTH).
- **Optimal Robust Flight Control (LQR)**:
  - Continuous-time Algebraic Riccati Equation (ARE) state-feedback controller (`QuadrotorLQRController`) for 12-state linearized dynamics with analytical motor mixer inversion.
- **Actuator & Aerodynamic Sim-to-Real Fidelity**:
  - Continuous first-order motor lag ($\tau_{\text{motor}} = 0.033\text{ s}$) integrated across RK4 substeps.
  - Aerodynamic wind vector drag continuously evaluated against quadrotor relative airspeed.
  - Cheeseman-Bennett ground effect model boosting rotor thrust when operating in close proximity to the ground plane ($z < 4 R_{\text{rotor}}$).
- **Sensor Realism & Non-Linear State Estimation**:
  - Physically grounded 6-axis IMU model (`IMUSensor`) measuring true specific force, angular rates, and bias random-walk drift.
  - Non-linear SO(3) attitude complementary filter (`ComplementaryFilter`) combining high-rate gyro integration with accelerometer gravity leveling and bias estimation.
- **Domain Randomization Engine**:
  - Configurable physical parameter randomization (`DomainRandomizer`) for Sim-to-Real policy transfer with canonical JSON config serialization and reproducible seed control.
- **Autonomous Flight Validation Suite (18 Scenarios)**:
  - 18 standardized flight scenarios covering hover hold, step translation, slalom navigation, tight canyon passage, dynamic intruder avoidance, sequential waypoints, RTH, blocked dead-ends, and fault degradation.
  - Disambiguated flight outcome taxonomy (`SUCCESS`, `COLLISION`, `BOUNDARY_BREACH`, `TIMEOUT`, `MAX_STEPS`, `CRASH_FAULT`, `RECOVERED_DEGRADED`, `BLOCKED_NO_PATH`).
  - Cryptographic Flight Recorder (`FlightRecord` / `FlightRecorder`) with dual SHA-256 fingerprinting (`config_fingerprint` and `record_digest`) and Zip Slip archive path-traversal security verification.
  - PPO & SAC Benchmarking Suite (`FlightBenchmarkSuite`) with strict policy attribution, rejection of silent random action fallback, and $N \ge 10$ inferential confidence interval thresholds.
- **Interactive Streamlit Flight Deck (`app.py`)**:
  - Dedicated **"🚀 Flight Scenarios & Benchmarks"** tab featuring 3D planned vs. flown trajectory visualization, commanded vs. actual motor thrust telemetry, synchronized 4-panel time series, and scenario execution.
- **Comprehensive Documentation**: Full mathematical specifications in [docs/AUTONOMOUS_NAVIGATION.md](file:///home/aryan/ml_workspace/ARL/docs/AUTONOMOUS_NAVIGATION.md) and [docs/FLIGHT_VALIDATION.md](file:///home/aryan/ml_workspace/ARL/docs/FLIGHT_VALIDATION.md).
- **Automated Test Suite**: 493 automated tests across kinematics, 6-DOF physics invariants, 3D A* planning, LQR control, actuator lag, ground effect, IMU estimation, domain randomization, and cryptographic provenance.

---

## System Architecture

```text
                           User
                            │
                            ▼
             ┌──────────────────────────────┐
             │     CLI / Streamlit GUI      │
             │ (Aerospace Flight Deck & 3D) │
             └──────────────┬───────────────┘
                            │
                            ▼
             ┌──────────────────────────────┐
             │   RL Algorithms & Planners   │
             │   (PPO, SAC, & 3D A* Lattice)│
             └──────────────┬───────────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│     DroneNavigation3DEnv     │ │        Drone6DOFEnv          │
│       (3-DOF Kinematic)      │ │    (6-DOF Rigid-Body Quad)   │
├──────────────────────────────┤ ├──────────────────────────────┤
│ • 29-D Observation Vector    │ │ • 36-D Observation Vector    │
│ • 3-D Acceleration Action    │ │ • 4-D Motor Thrust Action    │
│ • Semi-implicit Euler Drag   │ │ • RK4 (5 Substeps) Dynamics  │
│ • Spherical Obstacles        │ │ • Quaternion & Body Torques  │
│ • 16-Ray World-Fixed LiDAR   │ │ • World / Body-Frame LiDAR   │
└──────────────┬───────────────┘ └──────────────┬───────────────┘
               │                                │
               └──────────────┬─────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │    Closed-Loop Feedback      │
               │ Reward Shaping + Diagnostics │
               └──────────────────────────────┘
```

### The Reinforcement Learning Loop
1. **Airframe state observation**:
   - In **3-DOF mode**, the agent receives a 29-dimensional continuous state vector: normalized position, velocity, relative goal vector, goal distance ratio, and 16 spherical LiDAR range readings.
   - In **6-DOF mode**, the agent receives a 36-dimensional state vector: normalized position, linear velocity, unit quaternion attitude $(\mathbf{q} \in \mathbb{R}^4, \|\mathbf{q}\|=1.0)$, body angular rates $(\mathbf{\omega} \in \mathbb{R}^3)$, target position, relative goal vector, distance ratio, and 16-ray 3D LiDAR readings (configurable as world-fixed or body-fixed).
2. **Policy forward pass**: The observation vector is processed by an on-policy actor-critic network (**PPO**) or off-policy maximum-entropy network (**SAC**).
3. **Action generation**:
   - In **3-DOF mode**, the policy outputs continuous 3D acceleration $(a_x, a_y, a_z) \in [-1.0, 1.0]^3$.
   - In **6-DOF mode**, the policy outputs four normalized motor commands $(u_1, u_2, u_3, u_4) \in [-1.0, 1.0]^4$, mapped to individual physical rotor thrusts $[0, F_{\max}]^4$.
4. **Physical dynamic integration**:
   - In **3-DOF mode**, the kinematic engine updates linear velocity with aerodynamic drag damping and computes updated 3D position over $\Delta t = 0.1\text{ s}$.
   - In **6-DOF mode**, collective thrust and roll/pitch/yaw body torques drive rigid-body translational and rotational equations of motion integrated via Runge-Kutta 4th-order (RK4) over 5 internal substeps with continuous swept-sphere collision detection.
5. **Reward calculation**: Step returns combine Euclidean progress toward the goal, step penalties, actuator effort penalties, attitude tilt and rate penalties, and terminal bonuses/penalties (+100 for goal arrival, -100 for collision).
6. **Policy optimization**: Transitions update policy networks via PPO clipped surrogate gradients or SAC dual Q-learning with automatic entropy adjustment.

---

## Linux Support

AdaptiveRL is developed and tested primarily on Linux.

The project uses standard Python packaging and should work on Linux distributions providing Python 3.10+. The commands below use the distribution's package manager only to install Python and virtual-environment support.

Examples of compatible Linux distributions include:
- Ubuntu (20.04 LTS, 22.04 LTS, 24.04 LTS)
- Debian (11 Bullseye, 12 Bookworm)
- Linux Mint (20.x, 21.x)
- Fedora (38, 39, 40+)
- Arch Linux / Manjaro
- openSUSE (Leap, Tumbleweed)
- Pop!_OS

---

## Prerequisites

Before installing AdaptiveRL, make sure your system meets the following requirements:

- **Operating System**: Linux (x86_64 or aarch64)
- **Python**: Version 3.10, 3.11, or 3.12
- **Package Manager**: `pip`
- **Version Control**: `git`
- **Virtual Environment**: `python3-venv` (or equivalent package for your distribution)
- **Disk Space**: ~1 GB free space for Python dependencies (PyTorch, Stable-Baselines3, Plotly, Streamlit)
- **Network**: Internet connection for initial dependency installation

Check your current tools in a terminal:
```bash
python3 --version
git --version
python3 -m pip --version
```

---

## Step 1 — Install System Requirements

Use your distribution's package manager to install Python 3, pip, venv, and git.

### Ubuntu / Debian / Linux Mint / Pop!_OS
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

### Fedora
```bash
sudo dnf install -y python3 python3-pip git
```

### Arch Linux / Manjaro
```bash
sudo pacman -Syu
sudo pacman -S --needed python python-pip git
```

### openSUSE
```bash
sudo zypper install python3 python3-pip git
```

### Generic Fallback
If Python 3.10+ and Git are already installed on your system, you can skip this step and proceed to Step 2.

---

## Step 2 — Download ARL

Clone the canonical repository using Git and navigate into the project directory:

```bash
git clone https://github.com/StellarResearch/ARL.git
cd ARL
```

Verify that you are in the repository:
```bash
git status
ls
```

You should see files and directories similar to:
```text
app.py  artifacts  configs  CONTRIBUTING.md  docs  LICENSE  Makefile  pyproject.toml  README.md  src  tests
```

---

## Step 3 — Check Python

Verify that your system `python3` meets the minimum version requirement (Python 3.10 or newer):

```bash
python3 --version
```

- If your version is **3.10, 3.11, or 3.12**, you are ready to proceed.
- If your version is lower than 3.10, you will need to install a newer Python version through your distribution's package repositories or using a tool such as `pyenv`.
- Do **not** attempt to overwrite or force-replace your operating system's default Python symlink, as this can break system utilities.

---

## Step 4 — Create a Virtual Environment

Create an isolated Python virtual environment inside the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify that the virtual environment is active:
```bash
python --version
pip --version
```

Seeing `(.venv)` at the beginning of your terminal prompt confirms that the virtual environment is active.

---

## Step 5 — Upgrade Packaging Tools

Upgrade the core packaging tools (`pip`, `setuptools`, and `wheel`) inside your virtual environment to ensure trouble-free package resolution:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Step 6 — Install AdaptiveRL

Install AdaptiveRL in editable mode (`-e`) with all dependencies:

```bash
python -m pip install -e ".[all]"
```

- `-e` installs the repository in editable development mode, allowing local code modifications to take effect immediately without reinstallation.
- `[all]` installs the complete stack: runtime requirements, reinforcement learning libraries (Gymnasium, PyTorch, Stable-Baselines3), GUI tools (Streamlit, Plotly), plotting (Matplotlib), and developer utilities (pytest, ruff, mypy).

### Smaller Installation Options

If you only need specific components, smaller dependency sets are available:

- **Core only** (CLI, YAML config parsing, kinematics):
  ```bash
  python -m pip install -e .
  ```
- **Reinforcement Learning** (Gymnasium, Stable-Baselines3, PyTorch):
  ```bash
  python -m pip install -e ".[rl]"
  ```
- **Browser GUI** (Streamlit, Plotly):
  ```bash
  python -m pip install -e ".[gui]"
  ```
- **Benchmark Plotting** (Matplotlib, for `adaptive-rl benchmark budgets --plot`):
  ```bash
  python -m pip install -e ".[plot]"
  ```
- **Development & Testing** (pytest, ruff, mypy):
  ```bash
  python -m pip install -e ".[dev]"
  ```
- **Full Installation (Recommended)**:
  ```bash
  python -m pip install -e ".[all]"
  ```

---

## Step 7 — Verify Installation

Verify that the CLI entry point and environment definitions are registered properly:

```bash
# 1. Check CLI options
adaptive-rl --help

# 2. Check package version
adaptive-rl version

# 3. Inspect the drone environment specification and observation dimensions
adaptive-rl env inspect drone

# 4. Validate the demonstration configuration YAML
adaptive-rl config validate configs/drone_ppo_demo.yaml
```

If all four commands complete with green status messages, the installation is working correctly.

---

## Step 8 — Run Tests

Run the automated test suite using `pytest`:

```bash
python -m pytest
```

For detailed per-test execution traces:
```bash
python -m pytest -v
```

The repository includes automated unit and integration tests verifying:
- 3D kinematics equations and aerodynamic drag
- 29-dimensional observation space bounds
- Analytical 16-ray LiDAR raycasts and obstacle clearance
- Farama Gymnasium compliance checker
- Deterministic random seeding
- PPO policy training and checkpoint persistence
- Random policy baseline evaluation
- Obstacle-density experiment scaling
- Plotly 3D visualizer and chart generation
- CLI commands and argument parsing

---

## Step 9 — Train the PPO Agent

Training means the reinforcement learning agent interacts with the simulated 3D drone environment, accumulates experiences, and updates its neural network policy using PPO.

### Fast Demonstration Training (Recommended for Evaluation)
```bash
adaptive-rl train --config configs/drone_ppo_demo.yaml
```
- **Budget**: 25,000 timesteps
- **Estimated time**: ~20–35 seconds on a modern x86_64 CPU *(training time depends on your CPU and system configuration)*

### Full Training Run
```bash
adaptive-rl train --config configs/drone_ppo.yaml
```
- **Budget**: 50,000 timesteps
- **Estimated time**: ~40–70 seconds on CPU

---

## Training Output & Artifacts

When a training run completes, artifacts are automatically written to disk:

- **Trained Model Checkpoint**:  
  `artifacts/models/{experiment_name}_final.zip`  
  *(e.g., `artifacts/models/drone_ppo_demo_final.zip`)*
- **Training Metadata & Loss/Reward Log**:  
  `artifacts/metadata/{experiment_name}_training.json`  
  *(contains total timesteps, `training_time_seconds` for PPO optimization only, broader `duration_seconds` through model serialization, mean reward, and per-episode return lists)*
- **Periodic Checkpoints** (if configured):  
  `artifacts/checkpoints/{experiment_name}/`

---

## Step 10 — Evaluate the Trained Agent

Evaluate the saved policy weights across multiple deterministic test episodes:

```bash
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20
```

### Compare Against the Random Action Baseline
To scientifically prove that the agent learned purposeful navigation rather than succeeding by random chance, include the `--compare-random` flag:

```bash
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20 \
  --compare-random
```

This prints a formatted comparison table displaying:
- **Success Rate (%)**: Percentage of episodes reaching within 1.5m of the target.
- **Collision Rate (%)**: Percentage of episodes colliding with obstacles or arena walls (with separate obstacle and boundary collision breakdown).
- **Mean Reward**: Average cumulative episodic return ($\pm$ standard deviation).
- **Mean Steps**: Average flight duration before termination or truncation.
- **Trajectory Quality**: Path length (m), straight-line distance (m), and path efficiency.
- **Safety & Dynamics**: Minimum obstacle surface clearance (m), maximum velocity (m/s), and maximum acceleration (m/s²).

The evaluation report is saved to `artifacts/evaluation.json` (and optionally to CSV via `--output-csv`).

View all evaluation options:
```bash
adaptive-rl evaluate --help
```

### Run the PPO Learning-Curve Budget Benchmark

Train a fresh PPO model at each timestep budget and evaluate every model under identical evaluation conditions (PPO only):

```bash
adaptive-rl benchmark budgets \
  --config configs/drone_ppo_demo.yaml \
  --budgets 5000,10000,25000 \
  --training-seed 42 \
  --eval-seeds 42,43,44,45,46 \
  --episodes 20 \
  --deterministic
```

Machine-readable results are written to `artifacts/benchmarks/learning_curve_budget.json` and `learning_curve_budget.csv`, with one saved model and a `budget_result.json` per completed budget. Each budget's artifacts are staged and published together only after training and evaluation complete. The run-level JSON reports `status`, `completed_budgets`, `failed_budget`, and `error`, so a partially completed run is never mistaken for a complete one; the command exits non-zero when a budget fails and keeps the artifacts of every budget that finished. Use a fresh output directory for each run; existing reports or requested budget artifacts are not overwritten.

Each budget starts from the same configuration, training seed, and evaluation settings, but trains a fresh model and environment lifecycle. This is one training trajectory per budget, not an estimate of between-run training variance. Evaluation group seeds identify statistical groups; reset seeds for their episodes are derived independently of the requested episode count. JSON records the source commit and dirty-tree state when available, effective configuration fingerprint, evaluation split, environment-configuration fingerprint, seeds, and runtime/library versions. Repeatability is intended within a fixed software/hardware stack, but bit-for-bit training results are not guaranteed across devices, library versions, or nondeterministic accelerator kernels.

Add `--plot` to render `learning_curve_budget.png`. Plotting uses the optional Matplotlib extra:

```bash
python -m pip install -e ".[plot]"
```

`--plot-x-axis trained` (the default) plots against the timesteps PPO actually collected, while `--plot-x-axis requested` plots against the requested budget; the two differ when a budget is not aligned to a rollout boundary (budget `65` trains to `128` with `n_steps: 64`).

Metric semantics and the full JSON/CSV schema are documented in [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md).

---

## Step 11 — Run the Command-Line Drone Demo

Run a single deterministic flight demonstration in the terminal:

```bash
adaptive-rl demo-drone \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --seed 42
```

The terminal outputs real-time step telemetry:
- **Step number**
- **Altitude ($Z$ coordinate in meters)**
- **Ground speed ($\text{m/s}$)**
- **Distance to goal ($\text{m}$)**
- **Proximity to nearest obstacle ($\text{m}$)**
- **Step reward**
- **Final outcome banner (`SUCCESS` or `FAILED / COLLISION`)**

---

## Step 12 — Launch the 3D GUI

Launch the interactive presentation flight deck in your default web browser:

```bash
adaptive-rl gui
```

Alternatively, you can launch Streamlit directly:
```bash
streamlit run app.py
```

- **Local Address**: `http://localhost:8501`
- **To Stop the GUI Server**: Press `Ctrl+C` in the terminal.

If port 8501 is already occupied by another application, pass a custom port:
```bash
adaptive-rl gui --port 8502
```

---

## GUI Walkthrough

The browser GUI is organized into 5 dedicated demonstration panels:

### 1. 🎮 Live 3D Flight Demo
- **3D Flight Arena**: Renders bounding arena wireframes, red obstacle spheres, green diamond target, initial launch position, and continuous 3D flight trajectory.
- **LiDAR Visualization**: Visualizes 16-ray spherical rangefinder beams cast from the drone.
- **Trajectory Playback Scrubber**: Scrub forward and backward in time to inspect obstacle clearances, velocity vectors, and altitude changes step-by-step.
- **Controls**: Choose policy checkpoint, adjust procedural obstacle count (0 to 8), toggle LiDAR beams, and set deterministic seeds.

### 2. 📈 Train PPO
- **Interactive Controls**: Select training budget (5k, 10k, 25k, 50k steps), learning rate, and random seed.
- **CPU Execution**: Executes real Stable-Baselines3 PPO training with live status indicators.
- **Training Curve**: Displays an interactive Plotly chart of raw episode returns and 10-episode moving averages.
- **Safety**: Automatically disables simultaneous training jobs to prevent resource contention.

### 3. 📊 Benchmark Evaluation & Baseline
- **Head-to-Head Comparison**: Evaluates the trained PPO agent against the untrained Random Policy baseline under identical random seeds.
- **Metric Summaries**: Displays Success Rate, Collision Rate, Mean Return, and Episode Length.
- **Visual Comparison**: Interactive grouped bar chart comparing survival and collision rates.
- **Export**: Saves formal benchmark reports to `artifacts/evaluation.json`.

### 4. 🎯 Difficulty Experiment (Obstacle Density)
- **Difficulty Scaling**: Evaluates the policy across **4, 6, and 8 obstacles** under controlled seeds.
- **Empirical Demonstration**: Shows how increasing obstacle density constrains safe flight corridors, leading to higher collision rates.
- **Honest Display**: Displays "Not evaluated" until executed by the user; no placeholder or fake numbers.
- **Export**: Exports results to `artifacts/obstacle_density_experiment.json`.

### 5. 📘 About & Architecture
- **Step-by-Step "How It Works"**: Beginner- and professor-friendly 7-step explanation of the RL control loop.
- **Technical Specifications**: Full 29-dimensional observation breakdown, 3D continuous acceleration space, and kinematic state formulas.
- **Honest Academic Scope**: Explicit documentation of point-mass kinematics, geometric raycasts, and simulation boundaries.

---

## Quick Start

For a new Linux user on Ubuntu or Debian, here is the complete sequence from a fresh terminal:

```bash
# 1. Install system prerequisites
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# 2. Clone repository and enter directory
git clone https://github.com/StellarResearch/ARL.git
cd ARL

# 3. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Upgrade packaging tools and install AdaptiveRL
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[all]"

# 5. Verify installation and run tests
adaptive-rl --help
adaptive-rl env inspect drone
python -m pytest

# 6. Train demonstration model (~25s on CPU)
adaptive-rl train --config configs/drone_ppo_demo.yaml

# 7. Evaluate PPO vs Random baseline
adaptive-rl evaluate \
  --config configs/drone_ppo_demo.yaml \
  --model artifacts/models/drone_ppo_demo_final.zip \
  --episodes 20 \
  --compare-random

# 8. Launch interactive 3D browser GUI
adaptive-rl gui
```

---

## Technical Details

### Action Space
```text
Box(-1.0, 1.0, shape=(3,), dtype=float32)
```
The action represents a continuous 3D acceleration command $(u_x, u_y, u_z) \in [-1.0, 1.0]^3$. The environment scales these values to physical commanded acceleration:
$$\mathbf{a}_t = \mathbf{u}_t \cdot a_{\max} \quad (a_{\max} = 4.0\text{ m/s}^2)$$

### Observation Space (29 Dimensions)
```text
Box(-1.0, 1.0, shape=(29,), dtype=float32)
```
The 29-dimensional continuous state vector is structured as follows:

| Index Range | Dimension Count | Description | Normalization |
|---|---|---|---|
| `[0:3]` | 3 | Drone current position $\mathbf{p} = [x, y, z]$ | Divided by arena bounds $[30.0, 30.0, 15.0]$ |
| `[3:6]` | 3 | Drone current velocity $\mathbf{v} = [v_x, v_y, v_z]$ | Divided by max velocity $v_{\max} = 8.0\text{ m/s}$ |
| `[6:9]` | 3 | Target waypoint position $\mathbf{g} = [g_x, g_y, g_z]$ | Divided by arena bounds |
| `[9:12]` | 3 | Relative target vector $\mathbf{g} - \mathbf{p}$ | Divided by arena bounds |
| `[12:13]` | 1 | Euclidean distance to target $\|\mathbf{g} - \mathbf{p}\|$ | Divided by arena diagonal ($\approx 45.0\text{ m}$) |
| `[13:29]` | 16 | 16-ray spherical LiDAR rangefinder readings | Normalized in $[0.0, 1.0]$ ($20.0\text{ m}$ max range) |

**Total dimensions**: $3 + 3 + 3 + 3 + 1 + 16 = \mathbf{29}$

### Environment Parameters

| Parameter | Value | Description |
|---|---|---|
| Flight Arena Bounds | $30.0\text{ m} \times 30.0\text{ m} \times 15.0\text{ m}$ | Bounded rectangular flight volume |
| Simulation Time Step ($\Delta t$) | $0.1\text{ s}$ | Discrete time delta per step |
| Maximum Velocity ($v_{\max}$) | $8.0\text{ m/s}$ | Physical speed cap |
| Maximum Acceleration ($a_{\max}$) | $4.0\text{ m/s}^2$ | Maximum thrust acceleration |
| Aerodynamic Linear Drag ($c_{\text{drag}}$) | $0.05$ | Velocity damping coefficient |
| Maximum Steps per Episode | $200$ steps | Episode truncation limit ($20.0\text{ s}$ of flight) |
| Default Obstacle Count | $4$ spheres | Procedurally generated with start/goal clearance |
| Obstacle Radius | $2.0\text{ m}$ | Radius of procedural spherical obstacles |
| Target Arrival Radius | $1.5\text{ m}$ | Distance threshold for goal success |
| Drone Collision Radius | $0.8\text{ m}$ | Physical drone clearance radius |
| LiDAR Rangefinder | $16$ rays, $20.0\text{ m}$ range | Spherical Fibonacci distribution |

---

## Reward Function

The reward function at step $t$ actively guides the policy toward the goal while penalizing collisions and slow trajectories:

$$R_t = w_{\text{prog}} (d_{t-1} - d_t) + r_{\text{step}} - w_{\text{act}} \|\mathbf{a}_t\|^2 + R_{\text{terminal}}$$

- **Distance Progress ($w_{\text{prog}} = 2.0$)**: Rewards reducing Euclidean distance to the target: $2.0 \times (d_{t-1} - d_t)$.
- **Time Step Penalty ($r_{\text{step}} = -0.05$)**: Small penalty on every step to encourage finding direct, time-efficient paths.
- **Control Regularization ($w_{\text{act}} = 0.01$)**: Penalizes excessive acceleration commands (energy conservation).
- **Goal Reached ($R_{\text{terminal}} = +100.0$)**: Awarded when the drone enters within $1.5\text{ m}$ of the target coordinate.
- **Collision ($R_{\text{terminal}} = -100.0$)**: Incurred when the drone's collision radius strikes an obstacle or arena boundary.

---

## PPO Explanation

Proximal Policy Optimization (PPO) is an on-policy actor-critic reinforcement learning algorithm.

Instead of hand-coding navigation heuristics, the agent learns through trial and error:
1. **Observe**: The agent reads the current 29-dimensional sensor vector.
2. **Act**: The policy network outputs a continuous 3D acceleration command.
3. **Receive Reward**: The environment scores the action based on distance progress and obstacle proximity.
4. **Update Policy**: PPO uses a clipped surrogate objective function that prevents destructively large policy updates, ensuring stable and reliable convergence.

---

## Measured Benchmark Results

> [!NOTE]
> **Previously measured repository benchmark**  
> Results vary depending on your hardware, CPU speed, software library versions, random seeds, and training configuration. The values below were empirically measured on an x86_64 CPU under controlled seeds ($42$).

### 1. PPO vs Untrained Random Action Baseline (20 Test Episodes, Seed 42)

| Policy Evaluated | Success Rate (%) | Collision Rate (%) | Mean Reward | Mean Steps | Outcome |
|---|---|---|---|---|---|
| **Random Policy Baseline** | **0.0%** | **100.0%** | **-101.24** | **71.2** | Collided in 100% of test episodes |
| **Trained PPO Policy (25k steps)** | **5.0%** | **35.0%** | **-3.34** | **141.0** | **65% survival rate**, +97.9 reward delta |

*Finding: An unguided random agent collides 100% of the time within ~71 steps. PPO dramatically reduces collisions to 35% and doubles flight duration, validating goal-directed attraction and obstacle avoidance.*

> [!NOTE]
> **Provenance & Benchmark Metric Disclosure:**
> An earlier preliminary development table in `docs/RESULTS.md` recorded 40.0% success and +24.8 reward measured across 10 unseeded demonstration runs without fixed procedural seed control. In contrast, the formal benchmark above was measured across 20 procedural test seeds (Seed 42) with strict terminal collision conditions, demonstrating the realistic difficulty of dense 3D navigation. Both sets of empirical data are preserved for scientific transparency.

### 2. Obstacle-Density Scaling (10 Test Episodes per Condition, Seed 42)

| Condition | Obstacle Count | Collision Rate (%) | Mean Return | Mean Flight Steps |
|---|---|---|---|---|
| **Low Density** | 4 Obstacles | **20.0%** | **+5.00** | 169.0 steps |
| **Medium Density** | 6 Obstacles | **40.0%** | **-15.20** | 136.2 steps |
| **High Density** | 8 Obstacles | **70.0%** | **-32.43** | 88.2 steps |

*Finding: As obstacle density increases from 4 to 8, the collision rate rises from 20% to 70% and mean return drops, demonstrating how environmental complexity restricts safe flight paths.*

### 3. Unseen-Environment Generalization Benchmark

To verify that the drone policy learns genuine spatial obstacle avoidance rather than memorizing fixed obstacle layouts, AdaptiveRL partitions random seeds into disjoint deterministic intervals:
- **Training split (`train`)**: Seeds $[0, 1000)$ ($0 \le \text{seed} < 1000$, 1000 unique environments).
- **Unseen test split (`test`)**: Seeds $[1000, 1200)$ ($1000 \le \text{seed} < 1200$, 200 unique held-out environments).

#### Running the Benchmark
```bash
# 1. Train with the designated training split
adaptive-rl train --config configs/drone_ppo.yaml --split train

# 2. Evaluate specifically on held-out unseen test environments
adaptive-rl evaluate --config configs/drone_ppo.yaml --model artifacts/models/drone_ppo_final.zip --split test

# 3. Run the complete generalization benchmark (evaluates both splits, computes gaps, exports JSON)
adaptive-rl evaluate-generalization --model artifacts/models/drone_ppo_final.zip
```

#### Generalization Gaps
- **Success Gap**: $\Delta_{\text{success}} = \text{train\_success\_rate} - \text{test\_success\_rate}$
- **Reward Gap**: $\Delta_{\text{reward}} = \text{train\_mean\_reward} - \text{test\_mean\_reward}$

Benchmark results are automatically exported to `artifacts/generalization_benchmark.json`:
```json
{
  "train": {
    "seeds": [0, 1, 2, "..."],
    "success_rate": 0.85,
    "collision_rate": 0.10,
    "mean_reward": 82.50
  },
  "test": {
    "seeds": [1000, 1001, 1002, "..."],
    "success_rate": 0.70,
    "collision_rate": 0.25,
    "mean_reward": 61.20
  },
  "generalization_gap": {
    "success": 0.15,
    "reward": 21.30
  }
}
```

---

## Project Structure

```text
ARL/
├── app.py                      # Interactive 5-tab Streamlit presentation GUI
├── pyproject.toml              # Build specification, dependencies, and CLI entry point
├── README.md                   # Complete project documentation and Linux guide
├── LICENSE                     # MIT License
├── Makefile                    # Make shortcuts for install, test, and lint
├── CONTRIBUTING.md             # Developer guidelines and contribution workflow
├── configs/
│   ├── drone_ppo.yaml          # Full training configuration (50,000 steps)
│   └── drone_ppo_demo.yaml     # Fast demonstration configuration (25,000 steps)
├── docs/
│   ├── COLLEGE_DEMO.md         # 5–10 minute demonstration script & viva defense Q&A
│   ├── DEMO.md                 # CLI & GUI execution walkthrough
│   └── EXPERIMENT.md           # Experimental methodology & verified empirical metrics
├── src/adaptive_rl/
│   ├── __init__.py             # Package version declaration
│   ├── cli.py                  # Typer CLI implementation
│   ├── config.py               # Pydantic configuration schemas and YAML loader
│   ├── benchmarking/
│   │   └── learning_curve.py   # PPO budget sweep, evaluation, and JSON/CSV/plot exports
│   ├── algorithms/
│   │   ├── base.py             # BaseAlgorithm abstract interface
│   │   ├── ppo.py              # Stable-Baselines3 PPO wrapper
│   │   └── random_policy.py    # Uniform-random action policy baseline
│   ├── environments/
│   │   ├── base.py             # AdaptiveRLEnv abstract base class
│   │   ├── drone.py            # DroneNavigation3DEnv, kinematics, and LiDAR raycaster
│   │   └── registry.py         # Gymnasium environment factory registry
│   ├── evaluation/
│   │   ├── evaluator.py        # Evaluator, baseline comparison, and density experiment
│   │   ├── generalization.py   # Unseen-environment generalization benchmark & seed partition protocol
│   │   └── metrics.py          # EvaluationMetrics dataclass
│   ├── gui/
│   │   ├── __init__.py         # GUI exports
│   │   └── visualizer.py       # Plotly 3D flight deck and interactive chart builders
│   └── training/
│       ├── callbacks.py        # Episode metric logging and checkpoint callbacks
│       └── trainer.py          # PPOTrainer training orchestrator
└── tests/                      # Automated unit and integration tests
    ├── test_cli.py
    ├── test_configuration.py
    ├── test_drone.py
    ├── test_evaluation.py
    ├── test_gui.py
    ├── test_learning_curve_benchmark.py
    └── test_training.py
```

---

## Troubleshooting

### `python3: command not found`
Python 3 is not installed or not in your system `$PATH`.  
- **Ubuntu/Debian**: `sudo apt install python3`  
- **Fedora**: `sudo dnf install python3`  
- **Arch**: `sudo pacman -S python`

### `python3 -m venv` fails with an error
Some Debian/Ubuntu systems package `venv` separately. Install the package:
```bash
sudo apt install python3-venv
```

### `pip` problems or `pip: command not found`
Always run pip as a module under your active Python interpreter:
```bash
python -m pip install <package>
```
instead of invoking the global `pip` binary directly.

### `adaptive-rl: command not found`
This indicates your virtual environment is not currently active, or the package was not installed in editable mode. Run:
```bash
source .venv/bin/activate
python -m pip install -e ".[all]"
```

### GUI does not launch
Verify that Streamlit is installed in your environment:
```bash
python -m pip show streamlit
```
If missing, reinstall GUI dependencies:
```bash
python -m pip install -e ".[gui]"
```
Then launch:
```bash
adaptive-rl gui
# or
streamlit run app.py
```

### Port 8501 already in use
If another service is using port 8501, specify a different port:
```bash
adaptive-rl gui --port 8502
```
Or with Streamlit:
```bash
streamlit run app.py --server.port 8502
```

---

## Leaving the Environment

When you are finished working with AdaptiveRL, deactivate the virtual environment:

```bash
deactivate
```

To resume working later, enter the directory and reactivate the environment:
```bash
cd ARL
source .venv/bin/activate
```

---

## Updating an Existing Installation

To pull the latest changes from Git and update dependencies:

```bash
cd ARL
git pull
source .venv/bin/activate
python -m pip install -e ".[all]"
```

---

## Clean Reinstall

If your virtual environment becomes corrupted or has conflicting packages, you can safely remove and rebuild it:

```bash
cd ARL
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[all]"
```

*(This will not delete your saved models or training logs in `artifacts/`.)*

---

## Limitations

- **Simulation Only**: Designed strictly as an academic computer science and reinforcement learning demonstration project.
- **No Hardware Validation**: This is a theoretical simulation model and has not been tuned or validated against physical UAV flight hardware telemetry.
- **Analytical Rangefinder**: LiDAR sensing calculates geometric ray-sphere and ray-box intersections.
- **No Real Drone Autopilot**: Not intended for direct deployment onto real drone flight hardware without a low-level attitude controller.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feature/my-improvement`.
3. Create and activate a virtual environment.
4. Install all dependencies: `python -m pip install -e ".[all]"`.
5. Run the test suite: `python -m pytest`.
6. Make your changes and add relevant unit tests.
7. Run tests, linter, and typechecker:
   ```bash
   ruff check src/ tests/ app.py
   ruff format --check src/ tests/ app.py
   mypy src/
   pytest -v tests/
   ```
8. Commit your changes and open a Pull Request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for the full text.
