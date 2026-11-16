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

### 1. Authoritative Perception Fail-Safe Matrix
The following deterministic, hierarchical responses are mandated for all perception anomalies:

| Failure Mode | Detection Criterion | Immediate Algorithmic Action | Mission State Transition |
| :--- | :--- | :--- | :--- |
| **Non-Finite Sensor Data** | `np.isnan(data).any()` or `np.isinf(data).any()` | Drop corrupted returns immediately; flag `valid_mask = False`; **never** map invalid returns to $0.0\text{ m}$ (phantom collision) or `max_range` (obstacle masking). | Maintain state; if corrupted frames persist $> 3$ consecutive frames $\to$ `ABORTED` |
| **Invalid Covariance Matrix** | Non-symmetric or $\min(\text{eig}(P)) \le 0$ | Apply Joseph stabilization $P = (I-KH)P(I-KH)^T + KRK^T$; enforce symmetry $\frac{1}{2}(P+P^T)$; clamp eigenvalues $\ge 10^{-8}$; if uncorrectable, invalidate track. | Invalidate track; re-evaluate world model clearance |
| **Sensor Transport Latency** | $t_{\text{now}} - t_{\text{meas}} > 0.20\text{ s}$ | Query historical pose buffer at measurement epoch $t_{\text{meas}}$ for spatial projection; inflate measurement noise covariance $R$. | Continue; if latency $> 0.50\text{ s}$ log critical latency alert |
| **Brief Sensor Dropout** | No frame for $\Delta t \le 0.30\text{ s}$ ($\le 6$ steps at 20 Hz) | Coast active tracks forward via Kalman filter motion model; monotonically inflate spatial covariance $\Sigma_p(t)$; enforce braking horizon speed limit. | Maintain `EXECUTING` (no abrupt abort) |
| **Sustained Sensor Dropout** | Dropout duration $0.30\text{ s} < \Delta t \le 2.0\text{ s}$ | Freeze forward path progression; command immediate deceleration to zero-velocity stationary hover hold. | Transition to `WAITING` (Safe Hover Hold) |
| **Terminal Sensor Outage** | Dropout duration exceeds $T_{\text{outage}} = 2.0\text{ s}$ | Terminate active mission execution; prohibit further path planning; command controlled vertical descent ($v_z = -0.5\text{ m/s}$) to safe landing. | Transition from `WAITING` to `FAILED_BLOCKED` |
| **Loss of Localization Aiding** | Reference position/altitude drops out | Attitude remains stabilized via IMU gravimetric leveling; horizontal position drifts open-loop; command immediate zero-velocity hover hold; if unaided duration $> 3.0\text{ s} \to$ controlled descent. | Transition to `WAITING` $\to$ `FAILED_BLOCKED` |
| **Excessive Track Uncertainty** | $\sigma_{\max} = \sqrt{\lambda_{\max}(\Sigma_p)} > 2.5\text{ m}$ | Treat track as large spatial exclusion sphere ($r_{\text{eff}} = \hat{r}_{\text{obs}} + 2.5\text{ m}$); trigger replanning. If no safe alternative corridor exists $\to$ hover hold. | Trigger `REPLANNING`; if blocked $\to$ `WAITING` |
| **Stale Track Stagnation** | Track unobserved for $> K_{\text{max\_coast}}$ (5 frames) | Purge track from active memory; clear associated world-model exclusion volume. | Re-evaluate world model clearance |
| **Corrupted Vehicle Ego-Pose** | $|\|\mathbf{q}\| - 1.0| \ge 0.05$ or non-finite pose | Fail closed immediately; never substitute ground truth; cut autonomous planning. | Transition immediately to `ABORTED` |

### 2. Unknown-Space Policy: Delay-Aware Braking Horizon Constraint
In 3D aerial navigation, space traversed by the sensor beam is verified `FREE`. Space intersecting obstacle returns is verified `OCCUPIED`. All other space (behind obstacles, outside sensor FOV, or beyond maximum range) is `UNKNOWN`.

- **Policy Principle: Unknown space is NOT guaranteed free space.**
- **Initial Global Path Planning:**
  At mission start ($t=0$), before the drone has explored the arena, global path planning operates under an **Optimistic but Validated** assumption: it plans an initial candidate route through unobserved space, but enforces a maximum forward lookahead clearance requirement.
- **Physical Response Delay Decomposition:**
  When an obstacle appears at the boundary of verified visible free space, the aircraft cannot decelerate instantaneously. The total system response delay $\tau_{\text{delay}}$ comprises:
  $$\tau_{\text{delay}} = \tau_{\text{sensor}} + \tau_{\text{proc}} + \tau_{\text{replan}} + \tau_{\text{attitude}} + \tau_{\text{motor}}$$
  where:
  - $\tau_{\text{sensor}} = 1 / f_{\text{sensor}} \approx 0.050\text{ s}$ (sensor measurement period at 20 Hz).
  - $\tau_{\text{proc}} \approx 0.020\text{ s}$ (point-cloud clustering and detection pipeline latency).
  - $\tau_{\text{replan}} \approx 0.010\text{ s}$ (tracker update and mission replanning check).
  - $\tau_{\text{attitude}} \approx 0.100\text{ s}$ (time required to tilt quadrotor to maximum braking pitch attitude $\theta_{\text{brake}}$).
  - $\tau_{\text{motor}} \approx 0.030\text{ s}$ (first-order brushless motor thrust lag).
  Nominal demonstrated total response delay is $\tau_{\text{delay}} \approx 0.21\text{ s}$.

- **Delay-Aware Braking Horizon Inequality:**
  During delay $\tau_{\text{delay}}$, the quadrotor continues traveling forward at speed $v$. Under demonstrated conservative horizontal braking deceleration $a_{\min} > 0$ (derived from maximum pitch tilt $\theta_{\text{brake}}$ and aerodynamic drag), the total stopping distance is:
  $$d_{\text{stop}}(v) = v \cdot \tau_{\text{delay}} + \frac{v^2}{2 a_{\min}}$$
  To guarantee that the aircraft can come to a complete stationary hover within verified visible free space $d_{\text{available}} = \max\left(0, d_{\text{visible}} - d_{\text{margin}}\right)$, the velocity must satisfy:
  $$\frac{1}{2 a_{\min}} v^2 + \tau_{\text{delay}} v - d_{\text{available}} \le 0$$
  Solving the quadratic inequality for the positive velocity root yields the **Delay-Aware Braking Horizon Bound**:
  $$v_{\text{max\_allowed}} = a_{\min} \left( \sqrt{\tau_{\text{delay}}^2 + \frac{2 d_{\text{available}}}{a_{\min}}} - \tau_{\text{delay}} \right)$$

  > [!NOTE]
  > The elementary formula $v = \sqrt{2 a d}$ is strictly an idealized zero-delay ($\tau_{\text{delay}} = 0$) special case. Assuming $\tau = 0$ on physical aircraft causes substantial stopping-distance undershoot and collisions. Physical stopping guarantees are valid only when calibrated against experimentally validated $\tau_{\text{delay}}$ and $a_{\min}$ bounds.

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
