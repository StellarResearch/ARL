# Autonomous Navigation, Robust Control & High-Fidelity Dynamics (Phase III)

## Overview

ARL Phase III advances the 6-DOF quadrotor simulation into a scientifically rigorous, physically verifiable platform for autonomous 3D navigation, optimal robust flight control, and high-fidelity sim-to-real transfer.

This document details:
1. **Remediated Baseline & Statistical Benchmark Protocols** (strict policy attribution, $N \ge 10$ inferential threshold, unambiguous outcome taxonomy).
2. **Deterministic 3D Autonomous Navigation Architecture** (bounded spatial grid 3D A*, obstacle inflation, line-of-sight shortcutting, dynamic obstacles, reactive replanning, and mission state machine).
3. **Continuous-Time Algebraic Riccati Equation (LQR) Control** (12-state linearized dynamics, analytical mixer inversion, stability margins).
4. **Actuator & Aerodynamic Sim-to-Real Fidelity** (first-order motor lag $\tau_{\text{motor}}$, continuous sub-step aerodynamic wind, Cheeseman-Bennett ground effect).
5. **Sensor Realism & Non-Linear Attitude Estimation** (6-axis IMU with specific force, gyro random walk drift, and SO(3) complementary filtering).
6. **Domain Randomization Engine** (bounded physical property sampling, cryptographic configuration fingerprinting, and deterministic reproducibility).
7. **Flight Deck GUI Telemetry Integration** (3D flight trajectory visualization, actuator command-versus-actual analysis, and scenario suite).

---

## 1. Correctness Remediation & Statistical Integrity

### 1.1 Strict Policy Attribution & Prevention of Silent Random Fallback
In prior milestones, running a benchmark when no checkpoint existed could silently evaluate random actions under the nominal algorithm label (`PPO` or `SAC`). 
Phase III establishes strict cryptographic policy provenance:
- Evaluating an RL algorithm without a valid trained model or checkpoint raises `ValueError` unless explicitly marked as a random policy baseline (`algorithm="RANDOM"`).
- Checkpoint integrity is validated via SHA-256 fingerprinting (`checkpoint_hash`).
- Metadata attributes `policy_source` explicitly: `"trained_policy"`, `"checkpoint"`, `"heuristic_pid"`, `"lqr"`, or `"random_sample"`.

### 1.2 Statistical Sample Size & Confidence Intervals ($N \ge 10$)
To prevent publication of deceptive conclusions from underpowered sample sets:
- **Full Benchmark Protocol:** Requires sample size $N \ge 10$ evaluation episodes across unique, recorded seeds. Confidence intervals (95% CI via Student's $t$-distribution) are computed only when $N \ge 10$.
- **Quick Smoke Testing:** Runs with $N < 10$ are explicitly tagged `is_smoke_test = True`. Descriptive statistics (mean, std, min, max) are provided, while inferential confidence intervals are suppressed (`confidence_interval_95 = None`) to prevent misleading scientific claims.
- **Seed Validation:** Duplicate seeds in an evaluation batch are detected and rejected.

### 1.3 Unambiguous Flight Outcome Taxonomy
Simulation termination and truncation signals are mapped with strict precedence:
1. `COLLISION` (or `CRASH_FAULT` if under active actuator failure): Drone collides with terrain, obstacles, or ground. Takes precedence over step bounds.
2. `BOUNDARY_BREACH`: Drone leaves the defined flight arena.
3. `SUCCESS`: Drone reaches within target arrival tolerance ($\le 0.3\text{ m}$) with controlled velocity ($\le 0.5\text{ m/s}$).
4. `TIMEOUT`: Simulated elapsed time exceeds scenario `max_duration_seconds`.
5. `MAX_STEPS`: Episode step count exceeds maximum steps without timeout or collision.
6. `RECOVERED_DEGRADED`: Actuator fault injected, but vehicle maintained stability and completed scenario or survived without crash.
7. `BLOCKED_NO_PATH`: Navigation planner correctly identified goal is obstructed and halted safely.

---

## 2. Deterministic 3D Autonomous Navigation Architecture

The navigation system lives in `adaptive_rl.navigation` and cleanly separates geometric path planning from low-level attitude and thrust control.

```
       +-----------------------+
       |   MissionManager      | <--- Waypoint sequencing, RTH, Mode switching
       +-----------+-----------+
                   |
                   v
       +-----------------------+
       |   AStarPlanner3D      | <--- 26-connectivity grid, obstacle inflation,
       +-----------+-----------+      Euclidean distance metric, LoS shortcutting
                   | Collision-free Waypoints
                   v
       +-----------------------+
       | Flight Controller     | <--- QuadrotorPIDController / QuadrotorLQRController
       +-----------+-----------+
                   | Desired Motor Thrusts [f1, f2, f3, f4]
                   v
       +-----------------------+
       | DroneDynamics6DOF     | <--- RK4 integration, Motor Lag, Wind, Ground Effect
       +-----------------------+
```

### 2.1 3D A* Path Planning (`AStarPlanner3D`)
- **Spatial Grid Representation:** Discretizes the 3D bounding box $[-10, 10] \times [-10, 10] \times [0, 15]\text{ m}$ into uniform voxels (default grid resolution $\Delta = 0.5\text{ m}$).
- **Connectivity:** 26-connected neighbor graph (orthogonal, planar diagonal, and spatial cubic diagonal transitions) with true Euclidean edge costs:
  $$c(u, v) = \|p_u - p_v\|_2$$
- **Admissible Heuristic:** Euclidean distance to goal:
  $$h(u) = \|p_u - p_{\text{goal}}\|_2$$
- **Obstacle Inflation:** Every spherical obstacle with radius $r_{\text{obs}}$ is inflated by:
  $$r_{\text{effective}} = r_{\text{obs}} + r_{\text{vehicle}} + \delta_{\text{margin}}$$
  where nominal $r_{\text{vehicle}} = 0.25\text{ m}$ and configurable safety margin $\delta_{\text{margin}} = 0.20\text{ m}$. Any voxel whose distance to obstacle center is $\le r_{\text{effective}}$ is marked impassable.
- **Line-of-Sight (LoS) Path Simplification:** The raw grid path undergoes deterministic ray-cast shortcutting (`simplify_path`). Intermediate nodes are pruned if a continuous, collision-free straight corridor exists between earlier and later waypoints, reducing waypoint count by 60–80% while preserving clearance.

### 2.2 Dynamic Obstacles (`DynamicObstacleSphere3D`)
- Encapsulates moving spherical obstacles with configurable velocity vectors and boundary reflection or periodic sinusoidal trajectory models.
- Step-wise kinematic propagation allows testing reactive replanning under non-stationary environments.

### 2.3 Mission Manager (`MissionManager`)
- Coordinates multi-waypoint missions, dynamic obstacle tracking, emergency stop, and automated Return-to-Home (RTH).
- Sequences waypoints when vehicle enters the capture radius ($r_{\text{arrival}} \le 0.4\text{ m}$).

---

## 3. Optimal Robust Flight Control (LQR)

`QuadrotorLQRController` in `adaptive_rl.controllers.lqr` implements linear quadratic optimal state feedback based on the continuous-time Algebraic Riccati Equation (ARE).

### 3.1 State and Input Formulation
The continuous 12-dimensional state vector $\mathbf{x}$ and 4-dimensional control input $\mathbf{u}$ are defined as:
$$\mathbf{x} = [x, y, z, \dot{x}, \dot{y}, \dot{z}, \phi, \theta, \psi, p, q, r]^T \in \mathbb{R}^{12}$$
$$\mathbf{u} = [\Delta T, \tau_x, \tau_y, \tau_z]^T \in \mathbb{R}^4$$
where $\Delta T = T_{\text{total}} - m g$ is deviation from hover collective thrust, and $\tau = [\tau_x, \tau_y, \tau_z]$ are body torques.

### 3.2 Linearized Dynamics Matrices
Linearized around hover equilibrium ($\mathbf{x}_0 = \mathbf{0}, \mathbf{u}_0 = \mathbf{0}$):
$$\dot{\mathbf{x}} = A \mathbf{x} + B \mathbf{u}$$

The Jacobian matrices:
$$A = \begin{bmatrix}
0_{3 \times 3} & I_{3 \times 3} & 0_{3 \times 3} & 0_{3 \times 3} \\
0_{3 \times 3} & -\frac{D_v}{m} I_{3 \times 3} & \begin{bmatrix} 0 & g & 0 \\ -g & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} & 0_{3 \times 3} \\
0_{3 \times 3} & 0_{3 \times 3} & 0_{3 \times 3} & I_{3 \times 3} \\
0_{3 \times 3} & 0_{3 \times 3} & 0_{3 \times 3} & -\frac{D_\omega}{I} I_{3 \times 3}
\end{bmatrix}, \quad
B = \begin{bmatrix}
0_{3 \times 1} & 0_{3 \times 3} \\
\begin{bmatrix} 0 \\ 0 \\ 1/m \end{bmatrix} & 0_{3 \times 3} \\
0_{3 \times 1} & 0_{3 \times 3} \\
0_{3 \times 1} & I^{-1}
\end{bmatrix}$$

### 3.3 ARE Solution & Control Allocation
Solving the continuous Algebraic Riccati Equation:
$$A^T P + P A - P B R^{-1} B^T P + Q = 0$$
yields the optimal constant gain matrix:
$$K = R^{-1} B^T P \in \mathbb{R}^{4 \times 12}$$

Control law:
$$\mathbf{u} = -K (\mathbf{x} - \mathbf{x}_{\text{target}})$$

The total thrust $T = m g + \Delta T$ and body torques $\tau$ are mapped into individual rotor thrusts $[f_1, f_2, f_3, f_4]^T$ using analytical inversion of the X-configuration mixer matrix:
$$\begin{bmatrix} f_1 \\ f_2 \\ f_3 \\ f_4 \end{bmatrix} = M^{-1} \begin{bmatrix} T \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix}$$
Individual thrusts are clamped to $[0, f_{\text{max}}]$ before normalization into gym action space $[-1, 1]^4$.

---

## 4. Actuator & Aerodynamic Sim-to-Real Fidelity

### 4.1 First-Order Motor Dynamics (Actuator Lag)
Real brushless DC motors and Electronic Speed Controllers (ESCs) exhibit electro-mechanical response delay. The actual thrust $f_i(t)$ tracks commanded thrust $f_{i,\text{cmd}}(t)$ according to:
$$\dot{f}_i(t) = \frac{1}{\tau_{\text{motor}}} \left( f_{i,\text{cmd}}(t) - f_i(t) \right)$$
where default time constant $\tau_{\text{motor}} = 0.033\text{ s}$ ($\sim 30\text{ Hz}$ bandwidth).
In `DroneDynamics6DOF`, motor dynamics are integrated continuously across the 5 internal RK4 substeps ($dt_{\text{sub}} = 0.01\text{ s}$):
$$f_i(t + dt) = f_i(t) + \frac{dt}{\tau_{\text{motor}}} (f_{i,\text{cmd}} - f_i(t))$$

### 4.2 Aerodynamic Wind Model
Ambient atmospheric wind is modeled as continuous vector $\mathbf{v}_{\text{wind}} = [w_x, w_y, w_z]^T\text{ m/s}$. The aerodynamic drag opposing vehicle motion is governed by relative airspeed:
$$\mathbf{v}_{\text{rel}} = \mathbf{v}_{\text{quad}} - \mathbf{v}_{\text{wind}}$$
$$\mathbf{F}_{\text{drag}} = -D_v \|\mathbf{v}_{\text{rel}}\|_2 \mathbf{v}_{\text{rel}} - D_{v,\text{linear}} \mathbf{v}_{\text{rel}}$$
This wind force is evaluated at every RK4 substep, providing realistic drift and control disturbance.

### 4.3 Cheeseman-Bennett Ground Effect Model
When operating close to the ground plane ($z < 4 R_{\text{rotor}}$, where $R_{\text{rotor}} \approx 0.1\text{ m}$), rotor downwash interacts with the ground surface, producing an aerodynamic cushion that increases effective thrust.
Using the classical Cheeseman-Bennett formulation:
$$\frac{T_{\text{IGE}}}{T_{\text{OGE}}} = \frac{1}{1 - \left( \frac{R_{\text{rotor}}}{4 z} \right)^2}$$
Clamped to a maximum multiplier of $1.30\times$ to avoid singularity at $z \to 0$:
$$T_{\text{actual}} = \eta_{\text{IGE}}(z) \cdot T_{\text{commanded}}, \quad \eta_{\text{IGE}}(z) = \min\left(1.30, \frac{1}{1 - (R_{\text{rotor}} / (4 \max(z, 0.05)))^2}\right)$$

---

## 5. Sensor Realism & Non-Linear Attitude Estimation

### 5.1 6-Axis Inertial Measurement Unit (`IMUSensor`)
`IMUSensor` in `adaptive_rl.sensors.imu` simulates physical MEMS accelerometers and rate gyroscopes:
1. **Specific Force (Accelerometer):**
   $$\mathbf{a}_{\text{meas}} = R(q)^T (\mathbf{a}_{\text{inertial}} - \mathbf{g}) + \mathbf{b}_{\text{acc}} + \boldsymbol{\eta}_{\text{acc}}$$
   At rest on a level surface ($\mathbf{a}_{\text{inertial}} = \mathbf{0}, \mathbf{g} = [0, 0, -9.81]^T$), the accelerometer measures $+9.81\text{ m/s}^2$ upward along body $+Z$.
2. **Angular Velocity (Gyroscope):**
   $$\boldsymbol{\omega}_{\text{meas}} = \boldsymbol{\omega}_{\text{true}} + \mathbf{b}_{\text{gyro}} + \boldsymbol{\eta}_{\text{gyro}}$$
3. **Bias Random-Walk Dynamics:**
   $$\dot{\mathbf{b}}(t) \sim \mathcal{N}(0, \sigma_{\text{drift}}^2)$$

### 5.2 SO(3) Complementary Filter (`ComplementaryFilter`)
`ComplementaryFilter` in `adaptive_rl.sensors.estimation` estimates attitude quaternion $\hat{q}$ and position/velocity:
- High-frequency attitude is propagated via gyroscope kinematic integration:
  $$\dot{\hat{q}} = \frac{1}{2} \hat{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega}_{\text{meas}} - \hat{\mathbf{b}} + \boldsymbol{\omega}_{\text{corr}} \end{bmatrix}$$
- Low-frequency gravity leveling compares estimated body gravity direction $\hat{\mathbf{z}} = R(\hat{q})^T [0, 0, 1]^T$ against normalized accelerometer reading $\mathbf{a}_{\text{norm}}$:
  $$\mathbf{e}_{\text{tilt}} = \mathbf{a}_{\text{norm}} \times \hat{\mathbf{z}}$$
  $$\boldsymbol{\omega}_{\text{corr}} = k_p \mathbf{e}_{\text{tilt}}$$
- Accelerometer bias is dynamically estimated via integral tracking:
  $$\dot{\hat{\mathbf{b}}} = -k_i \mathbf{e}_{\text{tilt}}$$

---

## 6. Domain Randomization Engine

`DomainRandomizer` in `adaptive_rl.environments.randomization` supports Sim-to-Real policy transfer by randomizing vehicle physical parameters within specified bounds:
- Mass: $m \in [0.85, 1.15]\text{ kg}$ ($\pm 15\%$)
- Arm length: $L \in [0.18, 0.22]\text{ m}$ ($\pm 10\%$)
- Inertia: $I_{xx}, I_{yy} \in [0.007, 0.0095]\text{ kg}\cdot\text{m}^2$, $I_{zz} \in [0.012, 0.017]\text{ kg}\cdot\text{m}^2$
- Motor thrust coefficient: $k_{\text{thrust}} \in [0.85, 1.15]\times$
- Drag coefficients: $D_v \in [0.05, 0.20]$, $D_\omega \in [0.01, 0.03]$
- Motor time constant: $\tau_{\text{motor}} \in [0.02, 0.05]\text{ s}$

Parameters can be sampled reproducibly from explicit random seeds and serialized/reloaded via canonical JSON configuration files.

---

## 7. Autonomous Scenario Suite (18 Scenarios)

The test suite includes 18 standardized scenarios:
1. `hover_equilibrium`: Steady-state hover hold, verifying zero position drift and attitude stability.
2. `straight_line_navigation`: 3D waypoint navigation from origin to `[10, 10, 5]`.
3. `vertical_ascent_descent`: Vertical climb from 1.0 m to 8.0 m followed by commanded descent to 2.0 m.
4. `horizontal_translation`: Fast planar translation along North/East axes with bounded tilt.
5. `position_holding_disturbance`: Position hold under sustained continuous aerodynamic wind.
6. `obstacle_avoidance`: 3D A* guided trajectory navigating around a static spherical obstacle.
7. `boundary_avoidance`: Flight containment within bounded spatial flight corridor.
8. `target_approach_arrival`: Precision terminal velocity deceleration and arrival hovering within target sphere.
9. `attitude_disturbance_recovery`: Stabilization and recovery from high initial pitch/roll disturbance.
10. `actuator_fault_degradation`: In-flight 50% thrust drop on Motor 1 at $t = 1.0\text{ s}$ (verifies true aerodynamic crash under unallocated control).
11. `multi_obstacle_navigation`: Complex 3D slalom weaving through multiple distributed spherical obstacles.
12. `narrow_passage_navigation`: Precision traversal through a tight obstacle canyon.
13. `dynamic_obstacle_crossing`: Reactive obstacle avoidance against a moving spherical intruder crossing the flight path.
14. `multiple_sequential_waypoints`: Continuous closed-loop tracking of 4 consecutive 3D spatial waypoints.
15. `blocked_route_dead_end`: Safe identification of an unpassable cul-de-sac and graceful retreat.
16. `return_to_home_mission`: Outbound flight to a distant target coordinate followed by automated RTH sequence.
17. `turbulent_canyon_wind`: Traversal of a narrow canyon under continuous 3.0 m/s lateral wind disturbance.
18. `degraded_actuator_loiter`: Loitering under minor (10%) motor thrust degradation, verifying degraded recovery.

---

## 8. CLI Usage & Reproducible Commands

Run scenarios, benchmarks, and tests directly via the `adaptive-rl` CLI:

```bash
# Run full test suite
pytest -v

# Run CI verification smoke suite
adaptive-rl test-ci

# Run single autonomous flight scenario (Scenario 11 Slalom with LQR controller)
python -c "
from adaptive_rl.flight_testing import FlightScenarioRunner, get_scenario
runner = FlightScenarioRunner()
record = runner.run_scenario(get_scenario('slalom_run'), controller_type='lqr')
print('Outcome:', record.summary.outcome)
"

# Run benchmark with explicit algorithm and DOFs
adaptive-rl benchmark-flight --algo PPO --dof 6 --seeds 10,11,12,13,14,15,16,17,18,19

# Launch interactive Flight Deck GUI (with Tab 6 3D Navigation and Telemetry)
streamlit run app.py
```
