# ADR-008: Failure Handling, Stale Observations, and Unknown-Space Safety Policy

## Status
ACCEPTED (Architecture Specification for Phase V)

## Context
In real-world aerial robotics and safety-critical perception systems, sensors drop frames, communication channels experience latency, algorithms crash or encounter numerical errors (NaNs), and unobserved regions lurk behind obstacles.

A naive autonomous navigation system might:
1. Treat missing sensor data as "no obstacles present" and fly directly into hazards.
2. Treat unknown or unobserved space as guaranteed free space (optimistic assumption failure).
3. Continue propagating stale dead-reckoning tracks indefinitely without expanding uncertainty.
4. Attempt to execute trajectory replans while sensor data is corrupted, causing vehicle divergence.

Phase V must establish rigorous, fail-closed safety behaviors and define an explicit policy for unknown space.

## Decision
We define an authoritative **Perception Fail-Safe Matrix**, an **Unknown-Space Exploration Policy**, and explicit state transitions in `MissionManager`.

### 1. Authoritative Perception Fail-Safe Matrix & Localization Decoupling
The perception subsystem establishes a deterministic, hierarchical response to sensor and localization anomalies, clearly distinguishing high-level mission states (`MissionManager`) from low-level flight controller fail-safes:

| Failure Mode | Detection Criterion | Immediate Low-Level Flight Action | Mission State Transition |
| :--- | :--- | :--- | :--- |
| **Non-Finite Sensor Data** | `np.isnan(points)` or `np.isinf(points)` | Drop corrupted returns immediately; flag `valid_mask = False`; **never** map invalid returns to $0.0\text{ m}$ (phantom collision) or `max_range` (obstacle masking). | Maintain state; if corrupted frames persist $> 3$ consecutive frames $\to$ `ABORTED` |
| **Covariance Anomaly (ADR-005)** | Asymmetry, negative eigenvalue, or non-finite | If $\|P-P^T\| \le 10^{-7} \to$ symmetrize. If $-10^{-7} \le \lambda_{\min} < 0 \to$ eigen-clamp. If $\lambda_{\min} < -10^{-7} \to$ **fatal filter divergence**; track cannot be repaired by Joseph form; purge track from active memory. | Invalidate track; re-evaluate world model clearance |
| **Sensor Transport Latency** | $t_{\text{now}} - t_{\text{meas}} > 0.20\text{ s}$ | Query historical pose buffer at measurement epoch $t_{\text{meas}}$ for spatial projection; inflate measurement noise covariance $R$. | Continue; if latency $> 0.50\text{ s}$ log critical latency alert |
| **Brief Exteroceptive Dropout** (Healthy Localization) | No sensor frame for $\Delta t \le 0.30\text{ s}$ ($\le 6$ steps at 20 Hz) | Vehicle pose $[p, v, q]$ healthy. Coast active tracks forward via Kalman filter motion model; monotonically inflate spatial covariance $\Sigma_p(t)$; enforce braking horizon speed limit. | Maintain `EXECUTING` (no abrupt abort) |
| **Sustained Exteroceptive Dropout** (Healthy Localization) | Dropout duration $0.30\text{ s} < \Delta t \le 2.0\text{ s}$ | Vehicle pose $[p, v, q]$ healthy. Freeze forward path progression; command $\mathbf{v}_{\text{cmd}} = \mathbf{0}$, $\mathbf{p}_{\text{cmd}} = \mathbf{p}_{\text{hold}}$. **Stationary hover hold is physically feasible.** | Transition to `WAITING` (Safe Hover Hold) |
| **Terminal Exteroceptive Outage** (Healthy Localization) | Dropout duration exceeds $T_{\text{outage}} = 2.0\text{ s}$ | Terminate active mission execution; prohibit further path planning. If a verified landing corridor exists, command controlled descent ($v_z = -0.5\text{ m/s}$); otherwise hold hover until operator intervention or battery failsafe. | Transition from `WAITING` to `FAILED_BLOCKED` |
| **Loss of Horizontal Position Aiding** (Velocity Aided) | External position reference lost; velocity observable (e.g. optical flow/airspeed) | **Stationary position hover CANNOT be guaranteed.** Command zero-velocity hold ($\mathbf{v}_{\text{cmd}} = \mathbf{0}$); horizontal drift will slowly accumulate. If unaided duration $> 3.0\text{ s}$, initiate controlled landing descent. | Transition to `WAITING` (Degraded Velocity Hold) $\to$ `FAILED_BLOCKED` |
| **Loss of Altitude Aiding** | Barometer / rangefinder reference lost | **Altitude hold CANNOT be maintained.** Vertical double-integration drifts quadratically. Flight controller aborts waypoint navigation and commands collective thrust ramp for immediate landing descent. | Transition to `FAILED_BLOCKED` (Low-Level Emergency) |
| **Loss of Velocity Observability** (Pure Strapdown IMU) | Velocity and position references both lost | **Neither position hover nor zero-velocity hold can be guaranteed.** Quadratic divergence in position ($\mathcal{O}(t^2)$). Low-level autopilot commands immediate descent at current attitude. | Transition immediately to `FAILED_BLOCKED` |
| **Complete Localization Failure** (Attitude Corrupted) | $|\|\mathbf{q}\| - 1.0| \ge 0.05$ or non-finite pose | Fail closed immediately; never substitute ground truth; disarm motors or command minimum autorotation thrust to prevent high-speed flyaway into walls. | Transition immediately to `ABORTED` |
| **Excessive Track Uncertainty** | $\sigma_{\max} = \sqrt{\lambda_{\max}(\Sigma_p)} > 2.5\text{ m}$ | Treat track as large spatial exclusion sphere ($r_{\text{eff}} = \hat{r}_{\text{obs}} + 2.5\text{ m}$); trigger replanning. If no safe alternative corridor exists $\to$ hover hold. | Trigger `REPLANNING`; if blocked $\to$ `WAITING` |
| **Stale Track Stagnation** | Track unobserved for $> K_{\text{max\_coast}}$ (5 frames) | Purge track from active memory; clear associated world-model exclusion volume. | Re-evaluate world model clearance |

#### Policy on Emergency Controlled Descent
In an unmapped or obstacle-rich flight arena, commanding vertical descent without active exteroceptive sensing carries inherent collision risk with obstacles directly beneath the airframe.
Controlled vertical descent ($v_z = -0.5\text{ m/s}$) is permitted **only** when:
1. A verified obstacle-free vertical corridor was logged prior to sensor loss, OR
2. Downward proximity sensing (ultrasonic/optical rangefinder) remains operational, OR
3. The emergency descent is explicitly modeled as a high-risk fail-safe action of last resort to prevent unbounded horizontal drift and arena wall impact.

### 2. Unknown-Space Policy: Delay-Aware Braking Horizon Constraint
In 3D aerial navigation, space traversed by the sensor beam is verified `FREE`. Space intersecting obstacle returns is verified `OCCUPIED`. All other space (behind obstacles, outside sensor FOV, or beyond maximum range) is `UNKNOWN`.

- **Policy Principle: Unknown space is NOT guaranteed free space.**
- **Initial Global Path Planning:**
  At mission start ($t=0$), before the drone has explored the arena, global path planning operates under an **Optimistic but Validated** assumption: it plans an initial candidate route through unobserved space, but enforces a maximum forward lookahead clearance requirement.

#### Rigorous Response Delay Classification Budget
When an obstacle appears at the boundary of verified visible free space, the aircraft cannot decelerate instantaneously. The total system response delay $\tau_{\text{delay}}$ comprises five distinct physical and computational components, classified by evidence status:

| Delay Component | Parameter Symbol | Nominal Value | Evidence Classification | Technical Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Sensor Measurement Interval** | $\tau_{\text{sensor}}$ | $0.050\text{ s}$ | `DERIVED` | Derived from configured sensor update rate ($f_{\text{sensor}} = 20\text{ Hz} \implies \Delta t = 0.050\text{ s}$). Represents worst-case arrival latency within discrete clock. |
| **Point-Cloud Processing Latency** | $\tau_{\text{proc}}$ | $0.020\text{ s}$ | `PROVISIONAL TARGET` | Engineering target for spatial clustering and detection pipeline execution on standard benchmark hardware in Phase V.2. |
| **Tracking & Replanning Latency** | $\tau_{\text{replan}}$ | $0.010\text{ s}$ | `PROVISIONAL TARGET` | Engineering target for Kalman filter state update and local clearance check. |
| **Attitude Tilt Reorientation Delay** | $\tau_{\text{attitude}}$ | $0.100\text{ s}$ | `ASSUMED` | Estimated duration required to reorient quadrotor from cruise pitch $\theta_{\text{cruise}}$ to maximum reverse braking pitch $\theta_{\text{brake}}$ under angular rate limits ($\dot{\theta}_{\max} \approx 3.0\text{ rad/s}$). **Not yet experimentally measured on 6-DOF simulator.** |
| **Motor Thrust Lag** | $\tau_{\text{motor}}$ | $0.030\text{ s}$ | `DERIVED` | Derived directly from physical first-order brushless ESC/motor lag parameter $\tau_{\text{motor}} = 0.030\text{ s}$ in `DroneDynamics6DOF`. |
| **Total Response Delay** | $\tau_{\text{delay}}$ | $\approx 0.210\text{ s}$ | `ASSUMED COMPOSITE BUDGET` | Upper-bounded composite response delay budget. |

> [!CAUTION]
> Neither the total delay budget ($\tau_{\text{delay}} = 0.21\text{ s}$) nor the braking deceleration ($a_{\text{brake\_min}}$) has been experimentally demonstrated on physical flight hardware. They represent a provisional engineering budget to be validated in Phase V.5.

#### Conservative Bounded-Acceleration Stopping Distance Formulation
If the quadrotor is accelerating forward (e.g. during waypoint departure or speed transitions) at the moment an obstacle is detected, assuming constant velocity during the response delay $\tau_{\text{delay}}$ will dangerously underestimate the stopping distance.

Let:
- $v$: Current forward velocity at detection epoch.
- $a_{\text{forward\_max}}$: Upper bound on forward acceleration during the response delay interval ($\tau_{\text{delay}}$).
- $a_{\text{brake\_min}}$: Guaranteed lower bound on achievable horizontal braking deceleration.
- $\tau_{\text{delay}}$: Upper-bounded total system response delay.

1. **Velocity at Initiation of Active Braking:**
   $$v_{\text{brake\_entry}} = v + a_{\text{forward\_max}} \cdot \tau_{\text{delay}}$$
2. **Displacement During Response Delay:**
   $$d_{\text{delay}} = v \cdot \tau_{\text{delay}} + \frac{1}{2} a_{\text{forward\_max}} \cdot \tau_{\text{delay}}^2$$
3. **Displacement During Active Deceleration:**
   $$d_{\text{braking}} = \frac{v_{\text{brake\_entry}}^2}{2 a_{\text{brake\_min}}} = \frac{(v + a_{\text{forward\_max}} \cdot \tau_{\text{delay}})^2}{2 a_{\text{brake\_min}}}$$
4. **General Conservative Stopping Distance:**
   $$d_{\text{stop}}(v) = v \cdot \tau_{\text{delay}} + \frac{1}{2} a_{\text{forward\_max}} \cdot \tau_{\text{delay}}^2 + \frac{(v + a_{\text{forward\_max}} \cdot \tau_{\text{delay}})^2}{2 a_{\text{brake\_min}}}$$

#### Justification for the Constant-Speed Simplification
The simplified delay-aware formula:
$$d_{\text{stop}}(v) = v \cdot \tau_{\text{delay}} + \frac{v^2}{2 a_{\text{brake\_min}}}$$
is valid **if and only if** $a_{\text{forward\_max}} = 0$, which is strictly restricted to flight phases where:
1. The quadrotor has completed its waypoint acceleration phase and is in steady-state cruise.
2. The mission manager commands zero or negative forward acceleration setpoints.
3. Aerodynamic tailwinds are bounded by $v_{\text{wind\_tail}} \le 0$.

Under this justified assumption ($a_{\text{forward\_max}} = 0$), requiring $d_{\text{stop}}(v) \le d_{\text{available}}$ (where $d_{\text{available}} = \max\left(0, d_{\text{visible}} - d_{\text{margin}}\right)$) yields the quadratic inequality:
$$\frac{1}{2 a_{\text{brake\_min}}} v^2 + \tau_{\text{delay}} v - d_{\text{available}} \le 0$$
Solving for the positive root gives the **Delay-Aware Braking Horizon Bound**:
$$v_{\text{max\_allowed}} = a_{\text{brake\_min}} \left( \sqrt{\tau_{\text{delay}}^2 + \frac{2 d_{\text{available}}}{a_{\text{brake\_min}}}} - \tau_{\text{delay}} \right)$$

#### Protocol for Measuring $a_{\text{brake\_min}}$ in Simulation (Phase V.5)
The value of $a_{\text{brake\_min}}$ will be evaluated empirically in `Drone6DOFEnv` across adversarial Monte Carlo test runs:
- Maximum gross vehicle weight ($m = 1.15\text{ kg}$ via domain randomization).
- Degraded motor thrust ($k_{\text{thrust}} = 0.90$).
- Conservative pitch tilt limit ($\theta_{\text{brake\_limit}} \le 30^\circ$).
- Sustained tailwind disturbance ($v_{\text{wind}} = 3.0\text{ m/s}$).
- First-order motor lag $\tau_{\text{motor}} = 0.03\text{ s}$ and RK4 integration.
The minimum sustained horizontal deceleration observed across these worst-case runs defines $a_{\text{brake\_min}}$. Prior to those empirical benchmarks, any numerical figure is classified as a `PROVISIONAL TARGET`.

- **Blind Corridor Prohibition:**
  If an upcoming waypoint segment has zero sensor coverage and verified visible distance $d_{\text{available}} = 0$, $v_{\text{max\_allowed}}$ evaluates to $0.0\text{ m/s}$. The mission transitions to `WAITING` and commands a local yaw sensor sweep or stationary hover hold until visibility is acquired.

- **Blind Corridor Prohibition:**
  If an upcoming waypoint segment has zero sensor coverage and verified visible distance $d_{\text{available}} = 0$, $v_{\text{max\_allowed}}$ evaluates to $0.0\text{ m/s}$. The mission transitions to `WAITING` and commands a local yaw sensor sweep or stationary hover hold until visibility is acquired.

### 3. MissionManager State Machine Additions
The authoritative state machine from Phase IV (`IDLE`, `PLANNING`, `EXECUTING`, `WAITING`, `REPLANNING`, `ARRIVED`, `FAILED_BLOCKED`, `ABORTED`) is extended with perception-aware transitions:
- `EXECUTING` $\to$ `WAITING`: Triggered when sustained sensor dropout occurs ($\Delta t > 0.30\text{ s}$), dynamic track uncertainty blocks forward corridor, or visible stopping distance drops below safety buffer.
- `WAITING` $\to$ `REPLANNING`: Triggered when sensor stream resumes (within $2.0\text{ s}$) or moving obstacle clears the corridor.
- `WAITING` $\to$ `FAILED_BLOCKED`: Triggered when sensor dropout duration exceeds $T_{\text{outage}} = 2.0\text{ s}$ or wait duration exceeds `max_wait_time` (5.0 s) without route clearance, commanding controlled descent.
- Any state $\to$ `ABORTED`: Immediate fail-closed transition if persistent non-finite sensor data ($> 3$ frames) or corrupted estimator attitude occurs.

### 4. Software Safety vs. Real Aircraft Certification
ARL is an open-source research and engineering simulation platform. While these fail-safe policies are modeled on aerospace defense principles:
- **Simulation Scope:** These rules govern software simulation, test harnesses, and algorithmic fail-safe state machines.
- **Airworthiness Disclaimer:** This architecture does not constitute physical aircraft airworthiness certification under FAA/EASA standards (such as RTCA DO-178C for software, DO-254 for airborne electronic hardware, or SORA / ASTM F38 for unmanned aircraft). Physical flight requires redundant hardware sensors, dedicated safety microcontrollers, independent flight termination systems (FTS), and hardware-in-the-loop (HITL) physical stress testing.

## Alternatives Considered
1. **Pure Optimistic Planning (Unknown == Free Everywhere):**
   *Rejected:* Causes high-speed collisions when flying around corners into unobserved static or dynamic obstacles.
2. **Pure Pessimistic Planning (Unknown == Impassable Everywhere):**
   *Rejected:* The aircraft would refuse to take off or move forward, because all space outside its immediate 10-meter sensor FOV is initially unobserved.
3. **Silent Drop-Frame Interpolation:**
   Linearly interpolating lost sensor frames without growing uncertainty.
   *Rejected:* Masks physical sensor disconnects; leads to phantom collisions when moving obstacles maneuver during dropped frames.

## Technical Reasoning
- **Fail-Closed Philosophy:** If information is missing or corrupted, the system must always transition to a lower-risk state (slow down, hover hold, or land) rather than guessing or continuing at high speed.
- **Physical Feasibility:** The braking velocity limit guarantees that stopping distance never exceeds the verified sensor horizon.

## Consequences
- **Positive:**
  - Robust against sensor dropouts, network latency, and numerical instabilities.
  - Transparent state machine behavior that can be deterministically verified in automated tests.
  - Prevents blind collisions in cluttered environments.
- **Negative:**
  - In highly occluded environments, the aircraft will fly more conservatively and slow down at corners.
- **Neutral:**
  - Requires tuning braking deceleration $a_{\text{brake}}$ based on drone actuation limits.

## Known Limitations
- If both GPS/position references and visual/LiDAR odometry drop out simultaneously, the vehicle's position estimate will drift due to IMU dead-reckoning divergence (ADR-003). In this event, immediate emergency landing is commanded.

## Future Reconsideration Criteria
- If 3D frontier exploration or active perception (gimbal gaze planning) is introduced in future phases, unknown-space boundary tracking will be integrated directly into path search heuristics.
