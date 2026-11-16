# ADR-003: Coordinate-Frame and Timestamp Conventions

## Status
ACCEPTED (Architecture Specification for Phase V)

## Context
A major source of catastrophic failures in autonomous robotics is coordinate-frame confusion (e.g., mixing NED and ENU conventions, or applying sensor extrinsics with inverted signs).

The ARL repository contains established coordinate and kinematic conventions in `DroneDynamics6DOF` (`src/adaptive_rl/environments/drone_6dof.py`), `IMUSensor` (`src/adaptive_rl/sensors/imu.py`), and `AStarPlanner3D` (`src/adaptive_rl/navigation/planner.py`).
Per Non-Negotiable Engineering Rule 1, 7, and Section 5.B:
- We must inspect existing code before selecting conventions.
- We must not silently introduce a second coordinate convention.

## Decision
We formally standardize the coordinate frame hierarchy across the perception subsystem, strictly aligning with existing ARL dynamics and navigation conventions.

### 1. Frame Definitions
1. **World Frame ($\mathcal{F}_W$) — East-North-Up (ENU):**
   - Origin: Fixed inertially at the arena ground reference point $(0, 0, 0)$.
   - $+X_W$: East
   - $+Y_W$: North
   - $+Z_W$: Up (vertical altitude against gravity; floor is at $Z=0$).
   - Gravity vector in world frame: $\mathbf{g}_W = [0, 0, -9.81]^T\text{ m/s}^2$.
   - Units: Metres ($\text{m}$).

2. **Body Frame ($\mathcal{F}_B$) — Forward-Left-Up (FLU):**
   - Origin: Rigidly attached to the quadrotor center of mass (CoM).
   - $+X_B$: Forward (aircraft nose, aligned with forward flight heading chevron).
   - $+Y_B$: Left (port arm direction).
   - $+Z_B$: Up (normal to the 4-rotor plane; collective motor thrust vector points along $+Z_B$).
   - Units: Metres ($\text{m}$).

3. **Sensor Frame ($\mathcal{F}_S$) — Sensor-Fixed FLU (Default Range Frame):**
   - Origin: Optical / acoustic center of the sensor emitter/receiver.
   - Default Alignment: Forward-Left-Up (+X Boresight, +Y Left, +Z Up).
   - *Optical Convention Adapter:* For camera sensors that follow standard computer vision conventions (+Z optical boresight, +X right, +Y down), an explicit rotation matrix $R_{\text{opt}}^{\text{flu}}$ is provided to convert optical points into sensor-body FLU prior to downstream fusion.

### 2. Transformation Pipeline
Every observed 3D point $\mathbf{p}_S \in \mathcal{F}_S$ is mapped to the World Frame $\mathbf{p}_W \in \mathcal{F}_W$ via:

$$\mathbf{p}_B = \mathbf{p}_S^B + R_S^B \mathbf{p}_S$$
$$\mathbf{p}_W = \mathbf{p}_B^W(t) + R_B^W(\mathbf{q}(t)) \mathbf{p}_B$$

Where:
- $\mathbf{p}_S^B \in \mathbb{R}^3$: Extrinsic translation vector from vehicle CoM to sensor origin in body frame.
- $R_S^B \in \mathrm{SO}(3)$: Extrinsic rotation matrix mapping sensor vectors to body coordinates.
- $\mathbf{p}_B^W(t) \in \mathbb{R}^3$: Vehicle position in the world frame at measurement timestamp $t$.
- $R_B^W(\mathbf{q}(t)) \in \mathrm{SO}(3)$: Vehicle attitude rotation matrix derived from unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]$ (scalar-first):

$$R_B^W(\mathbf{q}) = \begin{bmatrix}
1 - 2(q_y^2 + q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1 - 2(q_x^2 + q_y^2)
\end{bmatrix}$$

This exactly matches `quaternion_to_rotation_matrix` in `src/adaptive_rl/environments/drone_6dof.py`.

### 3. Timestamp Synchronization Conventions
- **Simulation Time:** All components synchronize against the simulation master clock $t_{\text{sim}} \in [0, \infty)$ in elapsed seconds.
- **Sensor Timestamps:** Every sensor reading, detection, and track contains an explicit floating-point timestamp $t_{\text{meas}}$.
- **Measurement Latency Modeling:** A configurable sensor transport and processing latency $\tau_{\text{lat}} \ge 0$ can be injected:
  $$t_{\text{delivered}} = t_{\text{meas}} + \tau_{\text{lat}}$$
  When the perception tracker consumes a delayed measurement at $t_{\text{delivered}}$, it queries vehicle state history at the measurement epoch $t_{\text{meas}}$ for spatial projection, preventing attitude smearing.

## Alternatives Considered
1. **Adopting NED (North-East-Down):**
   *Rejected:* All existing environment files, Plotly visualizers, collision bounds, floor altitude checks ($z \ge 0.8\text{ m}$), and LQR controllers in ARL use ENU ($+Z$ Up). Switching to NED would invert $Z$, causing severe breakage across the entire repository.
2. **Implicit Frame Assumptions (Untyped Arrays):**
   *Rejected:* Passing coordinates without declaring `frame_id` leads to bugs where body-frame points are evaluated directly against world-frame boundary boxes.

## Technical Reasoning
- Reuses verified mathematical functions (`quaternion_to_rotation_matrix`, `world_to_body_vector`, `body_to_world_vector`) already implemented and tested in `drone_6dof.py`.
- Preserves 100% compatibility with Phase III/IV navigation and flight testing.
- Cleanly separates sensor calibration parameters ($R_S^B, \mathbf{p}_S^B$) from vehicle dynamics.

## Consequences
- **Positive:**
  - Absolute clarity on axis directions and sign conventions.
  - Zero coordinate-frame discrepancy between physics, sensors, tracking, and planning.
- **Negative:**
  - Requires point clouds to undergo extrinsic and pose rotations before world-frame tracking.
- **Neutral:**
  - Any external camera data following OpenCV (+Z forward, +Y down) must be transformed through the defined optical adapter.

## Known Limitations
- High angular velocities ($\|\boldsymbol{\omega}\| > 5\text{ rad/s}$) during a long sensor sweep can introduce geometric distortion (motion blur / rolling shutter). In Phase V.0, instantaneous exposure at $t_{\text{meas}}$ is assumed.

## Future Reconsideration Criteria
- If integrating with standard PX4 / MAVLink telemetry streams (which use NED in certain submessages), dedicated ENU $\leftrightarrow$ NED transformation adapters will be added at the communication boundary.
