# 6-DOF Rigid-Body Quadrotor Dynamics Environment (`drone-6dof`)

## 1. Overview and Scope

The `drone-6dof` environment simulates the rigid-body flight dynamics of a quadrotor in 3D space, incorporating full rotational attitude kinematics (unit quaternions), Euler rigid-body rotational equations, individual motor thrusts, aerodynamic damping, procedural 3D spherical obstacles, and 16-ray LiDAR sensing.

> [!IMPORTANT]
> **Theoretical Model Disclaimer**:
> This environment implements a canonical rigid-body flight dynamics model intended for educational, algorithmic, and reinforcement learning research. It does **not** represent a validated digital twin of any specific commercial or physical UAV hardware, and has not been tuned or validated against real-world flight telemetry.

---

## 2. Comparison: 3-DOF Kinematic vs. 6-DOF Rigid-Body Simulation

| Feature / Property | 3-DOF Model (`drone`, `drone_3d`) | 6-DOF Model (`drone-6dof`, `drone_6dof`) |
| :--- | :--- | :--- |
| **Physical Model** | Point-mass kinematic translation | Rigid-body translation & rotational dynamics |
| **Attitude State** | None (assumed upright point mass) | Unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]$ |
| **Angular Rates** | None | Body angular velocity $\boldsymbol{\omega} = [p, q, r]$ (rad/s) |
| **Actuation** | 3 continuous accelerations $a \in [-1, 1]^3$ | 4 individual motor thrusts $F_i \in [0, F_{\max}]$ |
| **Torque Generation** | None | Rotor-generated roll, pitch, and yaw torques |
| **Numerical Integration** | Semi-implicit Euler ($\Delta t = 0.05$ s) | Runge-Kutta 4th-Order (RK4, 5 substeps, $\Delta t_{\text{sub}} = 0.01$ s) |
| **Quaternion Normalization** | Not applicable | Enforced after every substep ($\|\mathbf{q}\| = 1.0$) |
| **Observation Dimensions** | 29 dimensions | 36 dimensions (includes quat, angular velocity) |
| **Collision Detection** | Point sphere / box check | Continuous swept-sphere & swept-box collision checking |
| **Computational Cost** | Ultra-fast (~100,000 steps/sec) | Physics-rich (~2,000–5,000 steps/sec) |
| **Algorithms Supported** | PPO & SAC | PPO & SAC |

### 2.1 Physics Features: Default-Enabled vs. Optional

To maintain full backward compatibility and predictable training dynamics, advanced aeromechanical features are optional:

| Feature | Parameter | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **RK4 Substeps** | `substeps: int` | `5` (Default) | Internal numerical integration stages per environment step ($\Delta t_{\text{sub}} = 0.01\text{ s}$). Must be a positive integer $\ge 1$. |
| **First-Order Motor Lag** | `enable_motor_dynamics: bool` | `False` (Optional) | When `True`, couples continuous actuator response with time constant `motor_time_constant` ($\tau = 0.03\text{ s}$, finite positive float). When `False`, commanded thrusts apply instantaneously at substeps. |
| **Cheeseman-Bennett Ground Effect** | `enable_ground_effect: bool` | `False` (Optional) | When `True`, increases effective rotor thrust when within proximity of ground plane ($z < 2L$). |
| **Prevailing Wind Vector** | `wind_vector: Sequence[float]` | `[0.0, 0.0, 0.0]` (Optional) | Constant aerodynamic flow velocity $[v_{w,x}, v_{w,y}, v_{w,z}]$ in world frame. |
| **Quadratic Drag** | `quadratic_drag_coeff: float` | `0.0` (Optional) | Nonlinear aerodynamic drag proportional to $\|v\| v$. |
| **Input Validation** | Strict Contract | Enforced | Rejects non-finite (NaN, $\pm\infty$) actions and direct motor thrusts with `ValueError`. Preserves saturation clipping for finite out-of-bound inputs. |

---

## 3. Coordinate Frames & Conventions

### 3.1 World Frame: ENU (East-North-Up)
* $+X_W$: East
* $+Y_W$: North
* $+Z_W$: Upward against gravity.
* Gravitational acceleration: $\mathbf{g} = [0, 0, -9.81]^T\text{ m/s}^2$ (or configurable non-negative scalar).

### 3.2 Body Frame
Fixed to the quadrotor center of mass (CoM):
* $+X_B$: Forward along the longitudinal axis.
* $+Y_B$: Left along the lateral axis.
* $+Z_B$: Upward along the normal axis (aligned with collective thrust vector).

### 3.3 Attitude Representation: Unit Quaternion
Attitude is represented by a unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T \in \mathbb{R}^4$ with $\|\mathbf{q}\| = 1$, parameterizing the rotation from body frame to world frame:
$$\mathbf{v}_W = R(\mathbf{q}) \mathbf{v}_B$$

The direction cosine matrix $R(\mathbf{q})$ is given by:
$$R(\mathbf{q}) = \begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}$$

Euler angles (roll $\phi$, pitch $\theta$, yaw $\psi$) are converted using numerically safe trigonometry:
$$\begin{aligned}
\phi &= \text{atan2}(2(q_w q_x + q_y q_z), 1 - 2(q_x^2 + q_y^2)) \\
\theta &= \arcsin(\text{clamp}(2(q_w q_y - q_z q_x), -1, 1)) \\
\psi &= \text{atan2}(2(q_w q_z + q_x q_y), 1 - 2(q_y^2 + q_z^2))
\end{aligned}$$

---

## 4. Rotor Layout & Actuation

The quadrotor operates in an "X" configuration with arm length $L = 0.2\text{ m}$. With $d = \frac{L}{\sqrt{2}} \approx 0.1414\text{ m}$:

| Rotor Index | Position $(x_B, y_B)$ | Direction | Reaction Torque |
| :--- | :--- | :--- | :--- |
| **Rotor 1 (Front-Right)** | $(+d, -d)$ | CCW | $+c_\tau F_1 \hat{z}_B$ |
| **Rotor 2 (Front-Left)** | $(+d, +d)$ | CW | $-c_\tau F_2 \hat{z}_B$ |
| **Rotor 3 (Rear-Left)** | $(-d, +d)$ | CCW | $+c_\tau F_3 \hat{z}_B$ |
| **Rotor 4 (Rear-Right)** | $(-d, -d)$ | CW | $-c_\tau F_4 \hat{z}_B$ |

### Forces and Torques
Each rotor generates thrust $F_i \in [0, F_{\max}]$ directed along $+Z_B$:
1. **Total Collective Thrust**:
   $$T = \sum_{i=1}^4 F_i$$
2. **Body Torques** $\boldsymbol{\tau} = [\tau_x, \tau_y, \tau_z]^T$:
   $$\tau_x = d (-F_1 + F_2 + F_3 - F_4) \quad \text{(Roll torque about } +X_B\text{)}$$
   $$\tau_y = d (-F_1 - F_2 + F_3 + F_4) \quad \text{(Pitch torque about } +Y_B\text{)}$$
   $$\tau_z = c_\tau (+F_1 - F_2 + F_3 - F_4) \quad \text{(Yaw torque about } +Z_B\text{)}$$
   where $c_\tau = 0.01\text{ m}$ is the aerodynamic torque-to-thrust ratio.

---

## 5. Equations of Motion

### 5.1 Translational Dynamics (World Frame)
$$m \ddot{\mathbf{p}} = \begin{bmatrix} 0 \\ 0 \\ -mg \end{bmatrix} + R(\mathbf{q}) \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix} - D_v \mathbf{v}$$
where:
* $m = 1.0\text{ kg}$ is total mass.
* $D_v = 0.1\text{ N}\cdot\text{s/m}$ is linear aerodynamic drag.
* $\mathbf{v} = \dot{\mathbf{p}}$ is world-frame linear velocity.

### 5.2 Rotational Dynamics (Euler Rigid-Body Equations, Body Frame)
$$I \dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (I \boldsymbol{\omega}) - D_\omega \boldsymbol{\omega}$$
where:
* $\boldsymbol{\omega} = [p, q, r]^T$ is the body angular velocity.
* $I = \text{diag}(0.01, 0.01, 0.02)\text{ kg}\cdot\text{m}^2$ is the moment of inertia tensor.
* $D_\omega = 0.02\text{ N}\cdot\text{m}\cdot\text{s/rad}$ is rotational aerodynamic damping.

### 5.3 Quaternion Kinematics
$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega} \end{bmatrix}$$

---

## 6. Numerical Integration & Validation Tolerances

Stepping employs **Runge-Kutta 4th-order (RK4)** integration:
* Environment step $\Delta t = 0.05\text{ s}$ with 5 substeps ($\Delta t_{\text{sub}} = 0.01\text{ s}$).
* Re-normalization enforced after every substep ($\|\mathbf{q}\| \leftarrow 1.0$).
* Continuous swept-sphere and swept-box collision checking between steps prevents tunneling.

### Verified Test Tolerances (`tests/test_physics_validation_6dof.py`):
1. **Hover Equilibrium**: Acceleration $\|d\mathbf{v}\| \le 10^{-9}\text{ m/s}^2$; vertical drift $< 10^{-4}$ m over 100 steps.
2. **Free Fall**: Acceleration equals $[0, 0, -g]$ within $10^{-6}\text{ m/s}^2$.
3. **Pure Torques**: Roll, pitch, and yaw torques uncoupled within $10^{-9}\text{ N}\cdot\text{m}$.
4. **Aerodynamic Drag**: Velocity decay matches analytical exponential $v_0 e^{-(D_v/m)t}$ within $0.1\%$.
5. **Orthogonality**: $R R^T = I$ and $\det(R) = 1.0$ within $10^{-6}$.
6. **Quaternion Stability**: Maintained $\|\mathbf{q}\| = 1.0 \pm 10^{-5}$ over 50,000 steps and 100,000 steps.

---

## 7. Action and Observation Spaces

### 7.1 Action Space
`Box(-1.0, 1.0, shape=(4,), dtype=float32)`:
$$F_i = \frac{a_i + 1}{2} F_{\max} \in [0, 5.0]\text{ N}$$
Hover thrust corresponds to $F_i = \frac{mg}{4} = 2.4525\text{ N}$ ($a_i \approx -0.019$).

### 7.2 Observation Space
`Box(-1.0, 1.0, shape=(36,), dtype=float32)`:
* `[0:3]`: Normalized position $\mathbf{p} / \text{bounds}$ in $[0, 1]^3$
* `[3:6]`: Normalized velocity $\mathbf{v} / v_{\max}$ in $[-1, 1]^3$
* `[6:10]`: Attitude quaternion $[q_w, q_x, q_y, q_z]$ in $[-1, 1]^4$
* `[10:13]`: Normalized angular velocity $\boldsymbol{\omega} / \omega_{\max}$ in $[-1, 1]^3$
* `[13:16]`: Normalized target position $\mathbf{g} / \text{bounds}$ in $[0, 1]^3$
* `[16:19]`: Relative target vector $(\mathbf{g} - \mathbf{p}) / \text{bounds}$ in $[-1, 1]^3$
* `[19]`: Normalized Euclidean distance to target in $[0, 1]$
* `[20:36]`: 16-ray 3D LiDAR normalized distances in $[0, 1]$

---

## 8. Reward Function & Decomposition

The reward function guides the agent toward the goal while penalizing collisions, excessive control effort, angular velocity, and unlevel attitude:

$$R_t = R_{\text{terminal}} + w_{\text{progress}} \Delta d + r_{\text{step}} - w_{\text{effort}} \|\mathbf{a}\|^2 - w_{\text{rate}} \|\boldsymbol{\omega}\|^2 - w_{\text{tilt}} (1 - R_{33})$$

Where:
* $R_{\text{terminal}} = +100.0$ (goal reached) or $-100.0$ (collision).
* $w_{\text{progress}} = 2.0$ (reward for closing distance to goal).
* $r_{\text{step}} = -0.05$ (step penalty).
* $w_{\text{effort}} = 0.01$ (penalizes excessive motor commands).
* $w_{\text{rate}} = 0.05$ (penalizes high body angular rates).
* $w_{\text{tilt}} = 0.1$ (penalizes tilt deviation from level flight, where $R_{33} = \hat{z}_B \cdot \hat{z}_W$).

All components are exposed in `info["reward_components"]` for evaluation and analysis.

---

## 9. Physical/Visual Geometry Synchronization & Sensing Frames

### 9.1 Exact Rigid-Body Dimensions
All visual airframe representations and collision volumes are strictly derived from the underlying physics dynamics object (`DroneDynamics6DOF`):
* Arm length: $L = 0.2\text{ m}$ (radial distance from CoM to rotor centers).
* Diagonal rotor offset: $d = \frac{L}{\sqrt{2}} \approx 0.1414\text{ m}$.
* Rotor body positions:
  $$\mathbf{r}_1 = (+d, -d, 0), \quad \mathbf{r}_2 = (+d, +d, 0), \quad \mathbf{r}_3 = (-d, +d, 0), \quad \mathbf{r}_4 = (-d, -d, 0)$$
* Total arm span: $2L = 0.4\text{ m}$.
* Propeller disc radius: $r_{\text{prop}} = 0.45 L = 0.09\text{ m}$.
* Body-to-world rigid transformation: $\mathbf{p}_W = \mathbf{p}_{\text{drone}} + R(\mathbf{q}) \mathbf{p}_B$ is enforced identically across fuselage, arms, motor mounts, propeller blades, landing gear, and forward sensor nosecone.

### 9.2 LiDAR Reference Frames
The environment supports configurable LiDAR beam reference frames:
* **World-Fixed (`body_frame_lidar=False`, default)**: Rays are distributed on the unit sphere in the world inertial frame.
* **Body-Fixed (`body_frame_lidar=True`)**: Rays rotate with the quadrotor attitude matrix $R(\mathbf{q}) \mathbf{d}_{\text{ray}}$, accurately modeling an onboard gimbal-less sensor pod.
* Both modes report active reference frame semantics in step info (`info["lidar_reference_frame"]`).

> [!NOTE]
> **Distinction from Phase V Modular Perception**:
> The 16-ray LiDAR readings in the observation vector are a legacy range feature designed specifically for direct vector-based RL policies (PPO/SAC). In contrast, the modular Phase V perception foundation (`adaptive_rl.perception.sensors.SimulatedLiDAR3D`) produces high-density, timestamped, immutable `PointCloud3D` observations adhering to ADR-002, ADR-003, and ADR-006 with arbitrary ray configurations (e.g., 32 azimuth × 8 elevation beams), rigorous SO(3) extrinsics validation, and zero simulation ground-truth leakage.

---

## 10. Outcome Categorization & Telemetry Semantics

Simulation episodes evaluate Gymnasium termination/truncation signals into five canonical, mutually exclusive outcomes:
1. **`SUCCESS`**: Target arrival radius reached ($d_{\text{goal}} \le r_{\text{target}}$) with `terminated=True, is_success=True`.
2. **`COLLISION`**: Obstacle or arena boundary collision detected with `terminated=True, collision_occurred=True` (or non-terminal collision when `terminate_on_collision=False`).
3. **`TIMEOUT`**: Wall-clock or environment duration timeout signal (`info["timeout"]=True`).
4. **`MAX_STEPS`**: Maximum episode step budget exhausted without goal or collision (`truncated=True, step_count >= max_steps`).
5. **`UNKNOWN`**: Unexpected non-terminal truncation or unclassified exit.
