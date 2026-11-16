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

### 1. Perception Fail-Safe Matrix
The following deterministic responses are mandated for all perception anomalies:

| Failure Mode | Detection Criterion | Immediate Action | Mission State Transition |
| :--- | :--- | :--- | :--- |
| **Non-Finite Sensor Data** | `np.isnan(data).any()` or `np.isinf(data).any()` | Reject frame; log critical warning; zero out corrupted returns | Keep prior state; if sustained $> 3$ frames $\to$ `ABORTED` |
| **Invalid Covariance Matrix** | Non-symmetric or $\min(\text{eig}(P)) \le 0$ | Joseph symmetrization; clamp eigenvalue $\ge 10^{-8}$; if uncorrectable, invalidate track | Flag track invalid; replan route around remaining valid tracks |
| **Sensor Transport Latency** | $t_{\text{now}} - t_{\text{meas}} > \tau_{\text{max\_latency}}$ ($0.2\text{ s}$) | Use historical drone pose at $t_{\text{meas}}$ for projection; inflate measurement noise $R$ | Continue; if latency $> 0.5\text{ s}$ trigger warning |
| **Sensor Signal Disconnection / Dropout** | No new sensor frames received for $\Delta t > 0.3\text{ s}$ | Coast existing tracks; grow spatial uncertainty $\Sigma_p(t)$; brake vehicle velocity | Transition to `WAITING` (Safe Hover Hold) |
| **Sustained Sensor Loss (Timeout)** | Sensor dropout exceeds $T_{\text{timeout}} = 2.0\text{ s}$ | Terminate active mission; do not attempt path planning | Transition to `FAILED_BLOCKED` or emergency controlled descent |
| **Excessive Track Uncertainty** | $\sigma_{\max} = \sqrt{\lambda_{\max}(\Sigma_p)} > 2.5\text{ m}$ | Treat track as large spatial exclusion sphere ($r_{\text{eff}} = r_{\text{obs}} + 2.5\text{ m}$) | Trigger replan; if corridor impassable $\to$ `WAITING` |
| **Stale Track Stagnation** | Track unobserved for $> K_{\text{max\_coast}}$ (5 frames) | Purge track from active tracker memory | Re-evaluate world model clearance |
| **Unnormalized Quaternion from Estimator** | $|\|\mathbf{q}\| - 1.0| \ge 0.05$ | Reject estimate; fail-closed immediate emergency hover | `ABORTED` (matches Phase IV avionics safety guard) |

### 2. Unknown-Space Policy: Risk-Weighted Semi-Conservative Planning
In 3D aerial navigation, space traversed by the sensor beam is verified `FREE`. Space intersecting obstacle returns is verified `OCCUPIED`. All other space (behind obstacles, outside sensor FOV, or beyond maximum range) is `UNKNOWN`.

- **Policy Principle: Unknown space is NOT guaranteed free space.**
- **Initial Global Path Planning:**
  At mission start ($t=0$), before the drone has explored the arena, global path planning operates under an **Optimistic but Validated** assumption: it plans an initial candidate route through unobserved space, but enforces a maximum forward lookahead clearance requirement.
- **Receding-Horizon Local Execution Guard:**
  The quadrotor is strictly forbidden from entering unobserved space at high speed. The forward flight speed is governed by the **Sensor Horizon Braking Bound**:
  $$v_{\text{max\_allowed}} = \sqrt{2 a_{\text{brake}} (d_{\text{sensor\_horizon}} - d_{\text{safety\_stop}})}$$
  If a planned route passes into an unobserved occluded region (e.g. around a sharp corner or behind a large obstacle), the MissionManager automatically commands velocity reduction, ensuring the aircraft can come to a complete stationary hover within the visible, verified free space if an obstacle is suddenly revealed.
- **Blind Corridor Prohibition:**
  If an upcoming waypoint segment has zero sensor coverage and exceeds the maximum stopping distance, the mission transitions to `WAITING` and commands a local yaw sensor sweep or hover hold until visibility is acquired.

### 3. MissionManager State Machine Additions
The authoritative state machine from Phase IV (`IDLE`, `PLANNING`, `EXECUTING`, `WAITING`, `REPLANNING`, `ARRIVED`, `FAILED_BLOCKED`, `ABORTED`) is extended with perception-aware transitions:
- `EXECUTING` $\to$ `WAITING`: Triggered when sensor dropout occurs, dynamic track uncertainty blocks forward corridor, or unexpected obstacle detected within braking margin.
- `WAITING` $\to$ `REPLANNING`: Triggered when sensor stream resumes or obstacle moves out of the flight corridor.
- `WAITING` $\to$ `FAILED_BLOCKED`: Triggered when wait duration exceeds `max_wait_time` (e.g. 5.0 s) without sensor restoration or path clearance.
- Any state $\to$ `ABORTED`: Immediate transition if non-finite sensor data or catastrophic estimator failure occurs.

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
