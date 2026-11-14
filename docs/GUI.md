# ARL Flight Deck GUI Guide

The **AdaptiveRL (ARL) Flight Deck** is an interactive browser-based scientific dashboard built with [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/). It provides real-time 3D flight trajectory playback, policy training, empirical baseline benchmarking, clutter stress testing, and cryptographic experiment manifest verification.

---

## 1. Quick Launch

Launch the dashboard from your activated virtual environment:

```bash
# Using the ARL CLI entrypoint:
adaptive-rl gui

# Or using Streamlit directly:
streamlit run app.py
```

By default, the dashboard runs locally at `http://localhost:8501`.

### Custom Port or Remote Host
```bash
adaptive-rl gui --port 8502 --host 0.0.0.0
```

---

## 2. Flight Deck Tabs & Capabilities

### Tab 1: 🚁 Live 3D Flight Deck
- **Dynamics Model Selection**:
  - **6-DOF Rigid-Body Quadrotor**: Simulates unit quaternion attitude $\mathbf{q} = [q_w, q_x, q_y, q_z]$, Euler rotational dynamics, individual motor thrusts, and Runge-Kutta 4th-order (RK4) integration with 5 substeps.
  - **3-DOF Kinematic Drone**: Fast point-mass translational model with semi-implicit Euler integration and continuous 3D acceleration commands.
- **Interactive 3D Arena (Plotly)**:
  - Bounded 30m × 30m × 15m arena with coordinate axes in the ENU world frame (+X East, +Y North, +Z Up).
  - Procedural spherical obstacles with safety exclusion zones.
  - Target beacon with vertical guidance column and launch origin marker.
  - Continuous flight trajectory ribbon.
  - Detailed procedural quadrotor airframe: carbon fuselage, 4 arms in X-configuration (R1/R2 front with red accents, R3/R4 rear with blue accents), 4 rotor hubs, circular blade paths, landing gear struts, and forward heading nosecone.
  - 16-ray spherical LiDAR sensor with distance color-coding (Cyan > 10m, Amber 4–10m, Crimson < 4m) and contact hit points.
- **Flight Trajectory Scrubber**:
  - Scrub forward or backward through flight timesteps to inspect quadrotor orientation, linear velocity, altitude, and LiDAR readings at any recorded moment.
- **Organized Telemetry Strip**:
  - **Navigation**: Altitude ($Z$), Distance to Target, Relative Target Vector ($\Delta X, \Delta Y, \Delta Z$).
  - **Attitude**: Roll ($\phi$), Pitch ($\theta$), Yaw ($\psi$) in degrees, and Angular Velocity Magnitude ($|\boldsymbol{\omega}|$).
  - **Kinematics & Velocity**: Linear Speed ($|\mathbf{v}|$), Velocity Vector Components ($v_x, v_y, v_z$).
  - **Perception & Sensors**: Nearest Obstacle Distance ($m$), LiDAR Clearance Status.
  - **Mission Outcome & Return**: Flight State (`SUCCESS`, `COLLISION`, `NAVIGATING`), Step Reward, Cumulative Return.

---

### Tab 2: ⚡ Train Policy
- Execute real reinforcement learning training directly on CPU using **Stable-Baselines3**.
- Select training budget (5,000 to 50,000 timesteps), Adam learning rate, and random seed.
- Monitor training progress with live status notifications and wall-clock duration timing.
- View interactive Plotly learning curves of raw episodic returns and moving averages.
- Saves model weights to `artifacts/models/{name}_final.zip` and logs to `artifacts/metadata/{name}_training.json`.

---

### Tab 3: 📊 Benchmark & Evaluation
- Statistically evaluate trained policies against an untrained **Random Action Baseline** under identical random seeds.
- Select test episode budgets (5 to 50 episodes) and evaluation mode.
- View structured performance tables:
  - Success Rate (%)
  - Collision Rate (%)
  - Mean Return ($\pm$ standard deviation)
  - Mean Flight Duration (steps)
- Generates side-by-side comparative bar charts and exports reports to `artifacts/evaluation.json`.

---

### Tab 4: 🎯 Clutter Stress Testing
- Measure policy robustness as environmental clutter increases.
- Evaluates the policy under identical seeds across **4, 6, and 8 procedural obstacles**.
- Generates degradation tables and collision escalation curves.
- Saves verified experiment data to `artifacts/obstacle_density_experiment.json`.

---

### Tab 5: 🔬 Research Evidence Dashboard
- Inspect experiment provenance and cryptographic integrity tracking based on `adaptive_rl.manifest`:
  - **Git Commit Provenance**: Full Git commit SHA, branch name, and dirty working tree status.
  - **Host System Telemetry**: CPU core count, primary compute device, GPU models, OS name, Python runtime version.
  - **Software Stack**: Exact pinned versions of PyTorch, Stable-Baselines3, Farama Gymnasium, NumPy, Typer, and Pydantic.
  - **Cryptographic SHA-256 Verification**: Computes real-time cryptographic checksums of disk artifacts (model checkpoints, training logs) and validates them against the logged manifest record. Displays verified badges only upon successful matches.
  - **Traceable Benchmarks**: Reads and visualizes real measured benchmark files (`artifacts/evaluation.json`, `artifacts/obstacle_density_experiment.json`) without fallback to mock or fabricated data.

---

### Tab 6: 📘 Architecture & Specifications
- Side-by-side mathematical comparison of 3-DOF and 6-DOF simulation models.
- Explicit equations of motion: semi-implicit Euler integration vs. RK4 with quaternion kinematics.
- Breakdown of the 29-dimensional (3-DOF) and 36-dimensional (6-DOF) observation spaces.
- Transparent limitations: educational simulation platform, theoretical rigid-body physics, not validated on physical flight hardware.

---

## 3. Supported Model Checkpoints

The dashboard automatically scans `artifacts/models/` for `.zip` model checkpoints.

Each checkpoint is inspected for:
- Algorithm type (`ppo` or `sac`)
- Observation and action space dimensions
- Environment compatibility (3-DOF vs. 6-DOF)

If an incompatible or corrupted checkpoint is selected, the flight deck displays an informative warning banner and safely defaults to the random baseline.

---

## 4. Honest Simulation Scope & Safety Notice

> [!IMPORTANT]
> The ARL Flight Deck visualizes numerical physics simulations (Euler / RK4). It does **not** represent live telemetry from physical UAV hardware. ARL policies are designed for algorithmic benchmarking and must not be flashed onto physical drone electronic speed controllers (ESCs) or autopilots (PX4 / ArduPilot) without low-level cascaded PID attitude stabilization and hardware fail-safes.
