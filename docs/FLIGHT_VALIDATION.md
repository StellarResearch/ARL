# Autonomous Flight Validation, Dynamics Verification & Empirical Benchmarking

## Overview

AdaptiveRL provides an aerospace-grounded flight dynamics simulation and empirical reinforcement learning platform. This document specifies the mathematical assumptions, coordinate frames, reference control architecture, 10 standard autonomous flight scenarios, statistical benchmarking protocols, and cryptographic provenance guarantees.

---

## 1. Physical Assumptions and Coordinate Frames

### 1.1 Coordinate Frame Conventions

```
World Frame (ENU):
  +X_W: East
  +Y_W: North
  +Z_W: Upward (aligned with gravity opposing vector; g = 9.81 m/s^2 acts along -Z_W)

Body Frame (Fixed to quadrotor center of mass):
  +X_B: Forward
  +Y_B: Left
  +Z_B: Upward (aligned with collective motor thrust vector)
```

The transformation from body frame to world frame is given by the rotation matrix $R(q) \in \text{SO}(3)$ derived from the unit attitude quaternion $q = [q_w, q_x, q_y, q_z]$:

$$v_W = R(q) v_B, \quad v_B = R(q)^T v_W$$

```python
from adaptive_rl.environments import body_to_world_vector, world_to_body_vector

# Transform body velocity or thrust vector to world ENU
v_world = body_to_world_vector(v_body, quaternion)
v_body_back = world_to_body_vector(v_world, quaternion)
```

### 1.2 Equations of Motion

1. **Translational Dynamics (World Frame):**
   $$m \ddot{p} = \begin{bmatrix} 0 \\ 0 \\ -m g \end{bmatrix} + R(q) \begin{bmatrix} 0 \\ 0 \\ T_{\text{total}} \end{bmatrix} - D_v v$$
   where $m$ is the quadrotor mass (1.0 kg nominal), $g = 9.81\text{ m/s}^2$, $D_v$ is the linear translational damping coefficient ($0.1\text{ N}\cdot\text{s/m}$), and $T_{\text{total}} = \sum_{i=1}^4 f_i$.

2. **Rotational Dynamics (Body Frame):**
   $$I \dot{\omega} = \tau - \omega \times (I \omega) - D_\omega \omega$$
   where $I = \text{diag}(I_{xx}, I_{yy}, I_{zz})$ is the diagonal moment of inertia tensor, $\omega = [p, q, r]^T$ is the body angular velocity, and $D_\omega$ is the aerodynamic rotational damping coefficient ($0.02\text{ N}\cdot\text{m}\cdot\text{s/rad}$).

3. **Quaternion Kinematics:**
   $$\dot{q} = \frac{1}{2} q \otimes \begin{bmatrix} 0 \\ \omega \end{bmatrix}$$
   Numerical Runge-Kutta 4th-order (RK4) integration is executed over 5 substeps per environment step ($dt = 0.05\text{ s}$, $dt_{\text{sub}} = 0.01\text{ s}$). Quaternion re-normalization ($q \leftarrow q / \|q\|$) is enforced after every RK4 substep to prevent numerical integration drift.

### 1.3 Rotor Configuration & Motor Thrust Mixer

The quadrotor employs an **X-configuration** with arm length $L = 0.2\text{ m}$ ($d = L / \sqrt{2} \approx 0.1414\text{ m}$):

- **Rotor 1 (Front-Right):** Position $[+d, -d, 0]^T$, Counter-Clockwise (reaction torque: $+c_\tau f_1$)
- **Rotor 2 (Front-Left):** Position $[+d, +d, 0]^T$, Clockwise (reaction torque: $-c_\tau f_2$)
- **Rotor 3 (Rear-Left):** Position $[-d, +d, 0]^T$, Counter-Clockwise (reaction torque: $+c_\tau f_3$)
- **Rotor 4 (Rear-Right):** Position $[-d, -d, 0]^T$, Clockwise (reaction torque: $-c_\tau f_4$)

The physical allocation matrix $M$ maps motor thrusts $[f_1, f_2, f_3, f_4]^T$ to collective forces and body torques:

$$\begin{bmatrix} T_{\text{total}} \\ \tau_x \\ \tau_y \\ \tau_z \end{bmatrix} = \begin{bmatrix} 1 & 1 & 1 & 1 \\ -d & d & d & -d \\ -d & -d & d & d \\ c_\tau & -c_\tau & c_\tau & -c_\tau \end{bmatrix} \begin{bmatrix} f_1 \\ f_2 \\ f_3 \\ f_4 \end{bmatrix}$$

Because the rows of $M$ are mutually orthogonal, the analytical inverse is:

$$f_1 = \frac{T}{4} - \frac{\tau_x}{4d} - \frac{\tau_y}{4d} + \frac{\tau_z}{4 c_\tau}$$
$$f_2 = \frac{T}{4} + \frac{\tau_x}{4d} - \frac{\tau_y}{4d} - \frac{\tau_z}{4 c_\tau}$$
$$f_3 = \frac{T}{4} + \frac{\tau_x}{4d} + \frac{\tau_y}{4d} + \frac{\tau_z}{4 c_\tau}$$
$$f_4 = \frac{T}{4} - \frac{\tau_x}{4d} + \frac{\tau_y}{4d} - \frac{\tau_z}{4 c_\tau}$$

Actuator limits clamp each thrust: $f_i \in [0, f_{\max}]$ (where $f_{\max} = 5.0\text{ N}$).

---

## 2. Reference Flight Controller

A classical cascaded PID reference controller (`QuadrotorPIDController` / `ReferenceFlightController` in `adaptive_rl.controllers.pid`) is provided for independent physical validation, axis-isolation verification, and benchmark baseline comparison:

1. **Outer Loop (Position & Velocity Control):**
   Computes desired world acceleration:
   $$a_{\text{des}} = K_{p,\text{pos}} (p^* - p) + K_{d,\text{pos}} (v^* - v) + K_{i,\text{pos}} \int (p^* - p) dt$$
   Clamped to maximum horizontal acceleration and safe tilt limits ($\phi_{\max} = 35^\circ$).

2. **Attitude Setpoint Computation:**
   Translates required horizontal accelerations into target roll $\phi^*$ and pitch $\theta^*$ angles, and collective thrust magnitude $T^* = m \sqrt{a_x^{*2} + a_y^{*2} + (g + a_z^*)^2}$.

3. **Inner Loop (Attitude & Rate Control):**
   Computes attitude quaternion tracking error $q_e = q^* \otimes q^{-1}$ and generates restoring body torques $[\tau_x, \tau_y, \tau_z]$ via proportional-derivative feedback on orientation and angular rates.

4. **Actuator Allocation:**
   Executes analytical mixer inversion and clamps individual thrusts to $[0, f_{\max}]$. Normalizes actions to $[-1.0, 1.0]^4$ matching the RL environment contract.

---

## 3. The 10 Standard Autonomous Flight Scenarios

AdaptiveRL provides 10 reproducible autonomous flight validation scenarios (`adaptive_rl.flight_testing.scenarios`):

| # | Scenario Identifier | Description | Success Criteria |
|---|---------------------|-------------|------------------|
| 1 | `hover_equilibrium` | Hold steady hover at $[15, 15, 7.5]\text{ m}$ for 200 steps | Pos error $< 0.5\text{ m}$, zero collisions |
| 2 | `straight_line_navigation` | Diagonal transit from $[5, 5, 3]\text{ m}$ to $[25, 25, 12]\text{ m}$ | Arrival within $1.2\text{ m}$, zero collisions |
| 3 | `vertical_ascent_descent` | Climb from $z=2\text{ m}$ to $z=12\text{ m}$, then descend to $z=5\text{ m}$ | Multi-waypoint arrival within $1.0\text{ m}$ |
| 4 | `horizontal_translation` | Execute $14\text{ m} \times 14\text{ m}$ square pattern at constant $z=6\text{ m}$ | Complete 4 corner waypoints |
| 5 | `position_holding_disturbance` | Hold position against $+1.2\text{ N}$ lateral aerodynamic step force | Recover and maintain pos error $< 1.0\text{ m}$ |
| 6 | `obstacle_avoidance` | Navigate through a field of 2 spherical obstacles to goal | No obstacle impact ($d_{\text{clearance}} > 0$) |
| 7 | `boundary_avoidance` | Fly toward perimeter boundary $[28, 28, 13.5]\text{ m}$ | Arrest flight before wall breach |
| 8 | `target_approach_arrival` | Approach waypoint $[20, 15, 6]\text{ m}$ and settle ($v < 0.2\text{ m/s}$) | Precision settling within $0.8\text{ m}$ |
| 9 | `attitude_disturbance_recovery`| Recover to level flight from initial $25^\circ$ tilt and $1.5\text{ rad/s}$ rate | Settle into level hover within 250 steps |
| 10| `actuator_fault_degradation`| Maintain controlled flight under 40% partial thrust loss on Motor 1 | Controlled flight without catastrophic flip |

---

## 4. Measurable Flight Metrics

All metrics are calculated strictly from actual simulation state and time-series sensor observations:

- **Position Error:** Final Euclidean distance, Mean error, Maximum error, Root-Mean-Square Error (RMS).
- **Velocity Error:** Mean velocity error, Peak speed, RMS velocity error.
- **Attitude Error:** Mean tilt angle (°), Maximum tilt angle (°), Euler angles $(\phi, \theta, \psi)$.
- **Angular Velocity:** Mean and peak angular velocity magnitude $\|\omega\|$.
- **Obstacle Clearance:** Minimum Euclidean clearance to nearest obstacle surface ($d - r_{\text{obs}} - r_{\text{drone}}$).
- **Control Effort:** Normalized actuator effort integral $\sum_{t} \sum_{i=1}^4 u_{i,t}^2$.
- **Flight Duration:** Steps and physical simulation seconds.
- **Execution Wall Time:** Real execution time measured via high-precision monotonic clock in milliseconds.

---

## 5. Reproducible PPO & SAC Benchmarking Matrix

AdaptiveRL provides an automated comparative benchmarking suite supporting all 4 standard matrix combinations:

1. **PPO + 3-DOF Kinematic:** 29-D observation, continuous 3-D velocity command space.
2. **PPO + 6-DOF Rigid-Body:** 36-D observation, continuous 4-D individual motor thrust space.
3. **SAC + 3-DOF Kinematic:** Off-policy maximum-entropy actor-critic on 3-DOF navigation.
4. **SAC + 6-DOF Rigid-Body:** Off-policy maximum-entropy actor-critic on 6-DOF quadrotor dynamics.

### Execution Profiles

- **`quick` Profile:** Evaluates 2 episodes per combination on deterministic seeds `[101, 102]`. Designed for CI verification, smoke testing, and local development (runs in $< 1\text{ s}$).
- **`full` Profile:** Evaluates multiple seed groups (`[1001, 1002, 1003, 1004, 1005]`) with statistical confidence intervals across seed summaries.

### Statistical Integrity & Minimum Sample Thresholds

- Adheres strictly to `adaptive_rl.evaluation.statistics`.
- Two-sided 95% Student's $t$ confidence intervals are computed across independent seed summaries whenever $N \ge 2$.
- When $N < 2$, inferential significance claims are rejected and labeled with an explicit statistical limitation note: *"Sample size N is insufficient for inferential statistics or confidence intervals. Only descriptive episode values are reported."*
- Descriptive episode-level metrics are never conflated with cross-seed inferential means.

---

## 6. Cryptographic Flight Recorder & Telemetry Provenance

Each completed flight experiment produces an immutable, cryptographically verifiable `FlightRecord`:

- **Git Provenance:** Commit SHA-1, branch name, dirty repository working tree flag.
- **Host & Runtime Environment:** OS name/release, Python version, CPU cores, GPU hardware.
- **Pinned Dependencies:** Exact versions of `adaptive-rl`, `torch`, `stable-baselines3`, `gymnasium`, `numpy`, `typer`, `pydantic`.
- **Physics Simulation Fingerprint:** SHA-256 digest of canonical, key-sorted physics parameters ($m, g, L, d, r_{\text{rotor}}, f_{\max}, c_\tau, D_v, D_\omega, I, dt, \text{substeps}$).
- **Tamper Evidence:** Modifying any behavior-affecting physics parameter or simulation configuration in a saved record invalidates the SHA-256 fingerprint upon reload:

```python
from adaptive_rl.flight_testing import FlightRecorder

# Load and verify cryptographic provenance
record = FlightRecorder.load_record_json(
    "artifacts/flight_records/flight_val_1794725045_record.json"
)
assert record.verify_fingerprint() is True
```

- **Security Hardening (Zip Slip Prevention):** Model checkpoint loaders inspect internal zip entry paths and immediately reject archives containing illegal directory traversal sequences (`..`, `/`, `\`).

---

## 7. How to Execute

### 7.1 CLI Commands

Execute all 10 autonomous flight validation scenarios:
```bash
adaptive-rl validate-flight --scenario all
```

Execute a specific scenario:
```bash
adaptive-rl validate-flight --scenario hover_equilibrium
adaptive-rl validate-flight --scenario obstacle_avoidance
```

Run the quick PPO & SAC benchmark suite (all 4 combinations):
```bash
adaptive-rl benchmark-flight --profile quick
```

Run the full statistical benchmark suite:
```bash
adaptive-rl benchmark-flight --profile full
```

### 7.2 Interactive Streamlit Flight Deck

Launch the 3D Flight Deck GUI:
```bash
adaptive-rl gui
# or
streamlit run app.py
```

Navigate to the **"🚀 Flight Scenarios & Benchmarks"** tab:
1. Select any of the 10 flight scenarios.
2. Select the Reference PID Controller or an available RL checkpoint.
3. Click **"Execute Autonomous Flight Scenario"**.
4. Inspect real-time 3D flight playback, outcome badges, and synchronized 4-panel telemetry time series (Altitude, Speed, Euler angles, Obstacle clearance).
5. Download structured JSON records or time-series CSV telemetry.
