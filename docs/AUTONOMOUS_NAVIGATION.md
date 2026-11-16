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

### 1.3 Authoritative, Mutually Exclusive Flight Outcome Taxonomy
Simulation termination and truncation signals are mapped with strict deterministic precedence to ensure mutual exclusivity and prevent false-positive claims:

1. `CRASH_FAULT`: An active actuator fault was injected and the aircraft collided with terrain, obstacles, arena boundaries, ground floor ($z \le 0.8\text{ m}$), or exceeded maximum recovery tilt ($\theta \ge 35^\circ$).
2. `COLLISION`: Aircraft collided with a static or dynamic spherical obstacle. Strict precedence: takes priority over `TIMEOUT` or `MAX_STEPS` even if collision occurs on the final allowed step.
3. `BOUNDARY_BREACH`: Aircraft exited the arena boundaries ($x, y, z$). Strict precedence over step/time limits.
4. `SUCCESS`: Aircraft reached within target arrival tolerance ($r \le r_{\text{target}}$) with velocity settled below threshold ($v \le 0.5\text{ m/s}$), visited all sequential waypoints, and completed with zero collisions. **A high-velocity fly-through ($> 0.5\text{ m/s}$) is strictly rejected from SUCCESS.**
5. `RECOVERED_DEGRADED`: An active actuator fault was injected, but the aircraft maintained level attitude ($\theta < 35^\circ$), kept altitude ($z > 0.8\text{ m}$), reached the target sphere, and settled ($v \le 0.5\text{ m/s}$). **Surviving without completing the mission and settling at the goal is classified as `MAX_STEPS` or `TIMEOUT`, NEVER `RECOVERED_DEGRADED`.**
6. `BLOCKED_NO_PATH`: Navigation planner provably determined that no safe collision-free path exists, or dynamic obstacles obstructed the flight corridor, and commanded a safe hover hold.
7. `TIMEOUT`: Simulated elapsed wall-clock duration exceeded scenario `max_duration_seconds`.
8. `MAX_STEPS`: Episode step count reached `max_steps` without collision, arrival, or failure.

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
- **Connectivity & Corner-Cutting Prevention:** 26-connected neighbor graph (orthogonal, planar diagonal, and spatial cubic diagonal transitions) with true Euclidean edge costs:
  $$c(u, v) = \|p_u - p_v\|_2$$
  **Continuous Edge Collision Checking:** In prior implementations, diagonal neighbor transitions validated endpoint voxels without validating the connecting edge, allowing paths to cut diagonally through obstacle corners. In Phase III hardening, every candidate neighbor expansion performs continuous swept line segment checking $\mathbf{p}(t) = (1-t)\mathbf{p}_u + t\mathbf{p}_v$ with fine interpolation step $\le \frac{\Delta}{2}$. Any neighbor whose connecting edge intersects the inflated obstacle boundary is strictly rejected.
- **Admissible Heuristic:** Euclidean distance to goal:
  $$h(u) = \|p_u - p_{\text{goal}}\|_2$$
- **Obstacle Inflation:** Every spherical obstacle with radius $r_{\text{obs}}$ is inflated by:
  $$r_{\text{effective}} = r_{\text{obs}} + r_{\text{vehicle}} + \delta_{\text{margin}}$$
  where nominal $r_{\text{vehicle}} = 0.25\text{ m}$ and configurable safety margin $\delta_{\text{margin}} = 0.20\text{ m}$. Any voxel or segment within $r_{\text{effective}}$ of an obstacle center is impassable.
- **Line-of-Sight (LoS) Simplification:** The raw grid path undergoes deterministic ray-cast shortcutting (`simplify_path`). Every prospective segment connecting non-adjacent waypoints is verified with continuous collision checks. If an unvalidated shortcut occurs, the simplification fallback guarantees that no unvalidated or intersecting segment is ever appended to the path. If no safe continuation exists, the planner safely reports `NO_PATH` rather than returning an unsafe path.
  *Note on Guarantees:* Path collision safety is guaranteed conditional on grid resolution $\Delta$, the safety inflation margin $\delta_{\text{margin}}$, valid bounded vehicle tracking error, and static obstacle environments.

### 2.2 Dynamic Obstacles & Continuous Swept Collision Detection (`DynamicObstacleSphere3D`)
- **Kinematic Horizon Prediction:** Dynamic obstacles propagate according to linear, oscillating, or bouncing velocity models. `predict_occupied_spheres(...)` forecasts the swept volume of moving obstacles over a lookahead horizon $[t, t + H]$ (e.g., $H = 2.0\text{ s}$) using analytical upper bounds on displacement ($v_{\max} H$) and spacing $\le 1.2 R$ to guarantee zero corridor gaps.
- **Continuous Swept Collision with Boundary Reflection Decomposition:**
  Dynamic obstacles reflecting off arena boundaries follow piecewise linear trajectories with sharp directional changes. Standard linear chord interpolation between $[t_0, t_1]$ cuts across the arena interior, missing boundary reflections and leading to false negatives.
  - `get_bounce_times(t0, t1)` analytically computes all exact reflection timestamps $\tau_{\text{bounce}} \in (t_0, t_1)$ where the obstacle reflects from arena boundaries.
  - The interval is partitioned into exact linear pieces $[t_a, t_b]$ at reflection events, checking swept relative clearance on each piece with interpolated vehicle endpoints.
- **Rigorous Trajectory Curvature Envelopes:**
  - **Vehicle Curvature Deviation:** When 6-DOF intermediate RK4 substeps are provided, the local curvature deviation bound is $\varepsilon_{\text{veh}} = \frac{1}{8} a_{\max} (\Delta \tau)^2$ across each substep. When only step endpoints are provided, the bound defensively evaluates against the full step duration: $\varepsilon_{\text{veh}} = \frac{1}{8} a_{\max} (\Delta t)^2$.
  - **Oscillating Obstacle Curvature:** For harmonic oscillating obstacles with period $T$ and amplitude $A$, maximum acceleration is $\omega^2 \|A\|_2$ (where $\omega = 2\pi / T$), yielding curvature deviation $\varepsilon_{\text{obs}} = \frac{1}{8} \tau_{\text{step}}^2 \omega^2 \|A\|_2$.
- **Defensible Acceleration Envelope (`max_translational_acceleration`):**
  Analytical maximum translational acceleration is derived directly from the physical parameters of `DroneDynamics6DOF`:
  $$a_{\max}(v) = \frac{4 f_{\max}}{m} + g + \frac{D_v v + D_q v^2}{m}$$
  where $v$ is bounded by the operational envelope: $\max(v_{\text{nominal}}, v_{\text{state}}, v_{\text{substeps}})$, or derived from terminal velocity $v_{\text{term}}$ under full motor thrust and gravity when no bound is specified. Arbitrary empirical fallbacks are strictly prohibited; calculations fail closed if physical parameters are invalid.

### 2.3 Mission Manager (`MissionManager`)
- Coordinates multi-waypoint missions, predictive dynamic obstacle avoidance, and fail-safe recovery.
- **Receding-Horizon Local Replanning:** Monitors forward path clearance against predicted dynamic obstacle positions. If an intruder is projected to obstruct the flight corridor, `MissionManager` triggers local replanning around the predicted swept volumes.
- **Fail-Safe Hover Hold:** If the dynamic corridor is completely obstructed and no valid replan is found, `MissionManager` commands an immediate zero-velocity hover hold at the current position, maintaining collision clearance until the obstacle clears the route.

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

**Numerical Solver & Honest Fallback Identification:**
`QuadrotorLQRController` solves the ARE directly using `scipy.linalg.solve_continuous_are`. The controller sets `solver_status = "ARE_SOLVED"` and `is_are_solution = True`. To prevent silent approximations, if numerical ARE solving fails (or `scipy` is unavailable), the controller strictly raises an exception unless the caller explicitly sets `allow_fallback=True`.

Control law:
$$\mathbf{u} = -K (\mathbf{x} - \mathbf{x}_{\text{target}})$$

### 3.4 Analytical Control Allocation Mixer Inversion Derivation
In `DroneDynamics6DOF`, the physical relationship between individual rotor thrusts $[f_1, f_2, f_3, f_4]^T$ and total thrust/body torques is governed by rotor positions $(x_i, y_i)$ and reaction torque constants:
- Rotor 1 (Front-Right): $(+d, -d)$, CCW ($+c_\tau$)
- Rotor 2 (Front-Left): $(+d, +d)$, CW ($-c_\tau$)
- Rotor 3 (Rear-Left): $(-d, +d)$, CCW ($+c_\tau$)
- Rotor 4 (Rear-Right): $(-d, -d)$, CW ($-c_\tau$)

where $d = L / \sqrt{2}$. The forward control allocation matrix $M \in \mathbb{R}^{4 \times 4}$ is:
$$\begin{bmatrix} T \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix} = \begin{bmatrix} 1 & 1 & 1 & 1 \\ -d & d & d & -d \\ -d & -d & d & d \\ c_\tau & -c_\tau & c_\tau & -c_\tau \end{bmatrix} \begin{bmatrix} f_1 \\ f_2 \\ f_3 \\ f_4 \end{bmatrix}$$

Because the row vectors of $M$ are mutually orthogonal:
- $\mathbf{m}_1 \cdot \mathbf{m}_2 = -d + d + d - d = 0$
- $\mathbf{m}_1 \cdot \mathbf{m}_3 = -d - d + d + d = 0$
- $\mathbf{m}_1 \cdot \mathbf{m}_4 = c_\tau - c_\tau + c_\tau - c_\tau = 0$
- $\mathbf{m}_2 \cdot \mathbf{m}_3 = d^2 - d^2 + d^2 - d^2 = 0$
- $\mathbf{m}_2 \cdot \mathbf{m}_4 = -d c_\tau - d c_\tau + d c_\tau + d c_\tau = 0$
- $\mathbf{m}_3 \cdot \mathbf{m}_4 = -d c_\tau + d c_\tau + d c_\tau - d c_\tau = 0$

The matrix is directly invertible analytically:
$$M^{-1} = \begin{bmatrix}
\frac{1}{4} & -\frac{1}{4d} & -\frac{1}{4d} & \frac{1}{4c_\tau} \\
\frac{1}{4} & \frac{1}{4d} & -\frac{1}{4d} & -\frac{1}{4c_\tau} \\
\frac{1}{4} & \frac{1}{4d} & \frac{1}{4d} & \frac{1}{4c_\tau} \\
\frac{1}{4} & -\frac{1}{4d} & \frac{1}{4d} & -\frac{1}{4c_\tau}
\end{bmatrix}$$

Thus, the exact inverted motor thrusts are:
$$f_1 = \frac{T}{4} - \frac{\tau_x}{4d} - \frac{\tau_y}{4d} + \frac{\tau_z}{4c_\tau}$$
$$f_2 = \frac{T}{4} + \frac{\tau_x}{4d} - \frac{\tau_y}{4d} - \frac{\tau_z}{4c_\tau}$$
$$f_3 = \frac{T}{4} + \frac{\tau_x}{4d} + \frac{\tau_y}{4d} + \frac{\tau_z}{4c_\tau}$$
$$f_4 = \frac{T}{4} - \frac{\tau_x}{4d} + \frac{\tau_y}{4d} - \frac{\tau_z}{4c_\tau}$$

Multiplying $M \cdot M^{-1} = I_{4 \times 4}$ holds to machine precision ($< 10^{-15}$). Individual thrusts are clamped to $[0, f_{\text{max}}]$ and mapped into normalized actions in $[-1, 1]^4$.

### 3.5 Real Environment Observation Adapter
`QuadrotorLQRController` natively accepts:
- 36-dimensional normalized Gymnasium observations from `Drone6DOFEnv`, decoding normalized position, velocity, attitude quaternion, angular rates, and target coordinates.
- 13-dimensional true physical state vectors $[p, v, q, \omega]$.
- Direct `DroneState6DOF` instances.
Unsupported observation dimensions are strictly rejected with descriptive `ValueError`.

---

## 4. Actuator & Aerodynamic Sim-to-Real Fidelity

### 4.1 First-Order Motor Dynamics (Actuator Lag)
Real brushless DC motors and Electronic Speed Controllers (ESCs) exhibit electro-mechanical response delay. The actual thrust $f_i(t)$ tracks commanded thrust $f_{i,\text{cmd}}(t)$ according to:
$$\dot{f}_i(t) = \frac{1}{\tau_{\text{motor}}} \left( f_{i,\text{cmd}}(t) - f_i(t) \right)$$
where default time constant $\tau_{\text{motor}} = 0.03\text{ s}$ ($\sim 33\text{ Hz}$ bandwidth).

**Exact Analytical Step Response vs Forward Euler:**
In prior implementations, motor thrust was integrated using simple forward Euler discretization $f(t+h) = f(t) + \frac{h}{\tau}(f_{\text{cmd}} - f(t))$, which suffers from numerical instability when $h > \tau$ and does not accurately match continuous-time physics. In Phase III hardening, `DroneDynamics6DOF` integrates actuator dynamics across each RK4 substep using the exact analytical first-order step response:
$$\alpha = \exp\left(-\frac{h}{\tau_{\text{motor}}}\right)$$
$$f_i(t + h) = f_i(t) \alpha + f_{i,\text{cmd}}(t) (1 - \alpha)$$
This analytical form is unconditionally stable for any substep size $h > 0$ and preserves physical fidelity to machine precision.

### 4.2 Aerodynamic Wind Model
Ambient atmospheric wind is modeled as continuous vector $\mathbf{v}_{\text{wind}} = [w_x, w_y, w_z]^T\text{ m/s}$. The aerodynamic drag opposing vehicle motion is governed by relative airspeed:
$$\mathbf{v}_{\text{rel}} = \mathbf{v}_{\text{quad}} - \mathbf{v}_{\text{wind}}$$
$$\mathbf{F}_{\text{drag}} = -D_v \mathbf{v}_{\text{rel}} - D_q \|\mathbf{v}_{\text{rel}}\|_2 \mathbf{v}_{\text{rel}}$$
where $D_v$ is linear damping and $D_q$ is quadratic aerodynamic drag coefficient. Both linear and quadratic drag components are evaluated at every RK4 substep, providing realistic drift and control disturbance under opposing, cross, or tailwind conditions.

### 4.3 Cheeseman-Bennett Ground Effect Model
When operating close to the ground plane ($z < 4 R_{\text{rotor}}$, where $R_{\text{rotor}} \approx 0.09\text{ m}$), rotor downwash interacts with the ground surface, producing an aerodynamic cushion that increases effective thrust.
Using the classical Cheeseman-Bennett formulation:
$$\frac{T_{\text{IGE}}}{T_{\text{OGE}}} = \frac{1}{1 - \left( \frac{R_{\text{rotor}}}{4 z_{\text{eff}}} \right)^2}$$
where $z_{\text{eff}} = \max(0.5 R_{\text{rotor}}, z)$ and clamped to a maximum multiplier of $1.30\times$ to avoid singularity at $z \to 0$:
$$T_{\text{actual}} = \eta_{\text{IGE}}(z) \cdot T_{\text{commanded}}, \quad \eta_{\text{IGE}}(z) = \min\left(1.30, \max\left(1.0, \frac{1}{1 - (R_{\text{rotor}} / (4 z_{\text{eff}}))^2}\right)\right)$$

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

### 5.3 Sensor Observability Limits & Dead-Reckoning Drift
In any strapdown inertial navigation system without external aiding:
1. **Yaw (Heading) Unobservability:** Gravity leveling observes roll and pitch because the gravity vector $[0, 0, -g]^T$ produces a non-zero projection onto body $X$ and $Y$ when tilted. However, rotations about the vertical world axis $Z$ (yaw) leave the gravity vector invariant ($\partial \mathbf{g} / \partial \psi = \mathbf{0}$). Consequently, yaw is mathematically unobservable from accelerometer readings alone. Without a 3-axis magnetometer or external optical/GNSS reference, gyro bias drift causes uncorrected yaw random-walk drift over extended durations.
2. **Position Dead-Reckoning Divergence:** Estimating position requires double integration of transformed acceleration:
   $$\mathbf{p}(t) = \mathbf{p}(0) + \mathbf{v}(0) t + \int_0^t \int_0^s (R(\hat{q}) \mathbf{a}_{\text{meas}} + \mathbf{g}) d\tau ds$$
   Any residual attitude error $\delta \theta$ tilts the subtracted gravity vector, introducing a false horizontal acceleration $g \sin(\delta \theta) \approx g \delta \theta$. Double integration of this term produces unbounded quadratic position error drift $\mathcal{O}(g \delta \theta \cdot t^2)$. This is an inherent physical property of unaided inertial odometry, correctly reflected in the simulation telemetry.

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

### 6.1 Cryptographic Experiment Configuration Fingerprinting & Deep Provenance
To guarantee exact reproducibility and provenance across arbitrary user callbacks, controllers, and fault injectors, `adaptive_rl.flight_testing` implements multi-layer cryptographic digest tracking:
- **Physics Simulation Fingerprint (`PhysicsSimulationFingerprint`):** Full parameter coverage of mass, gravity, arm geometry, rotor dynamics, damping, inertia, and simulation timesteps.
- **Deep Callable Provenance (`extract_callable_provenance`):**
  - **Descriptor Inspection:** Unwraps and fingerprints `staticmethod` (`.__func__`), `classmethod` (`.__func__`), and `property` (`.fget`, `.fset`, `.fdel`) descriptors across the entire class MRO hierarchy.
  - **Recursive Module Dependency Resolution:** Recursively resolves submodules and nested attribute chains (`pkg.submod.helper`) referenced in bytecode `co_names`. Changes to underlying helper functions in imported submodules alter the provenance digest deterministically.
  - **Safe Recursion and Cycle Handling:** Detects circular call graphs and self-recursive functions, breaking cycles via deterministic lexical cycle references (`recursion_ref`) without stack overflow.
  - **Fails-Closed Incomplete Tracking:** Dynamic or non-serializable objects (e.g. opaque native handles, threading locks) are safely flagged as `provenance_complete = False` with structured error taxonomy, preventing non-deterministic memory address leakage into cryptographic digests.
- **Dual Integrity Verification:** `FlightRecord` computes two separate digests: `config_fingerprint` (covering experiment preconditions) and `record_digest` (covering post-execution metrics, trajectories, and outcomes). Tampering with either pre-flight configuration or post-flight telemetry is strictly detected.

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
