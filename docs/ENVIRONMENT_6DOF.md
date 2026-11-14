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
