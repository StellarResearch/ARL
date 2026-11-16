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

### 1. Authoritative Perception Fail-Safe Decision Table & Localization Decoupling
The perception subsystem establishes a deterministic, hierarchical response to sensor, estimator, and environmental anomalies. It strictly distinguishes high-level mission state management (`MissionManager`) from low-level autopilot flight safety actions (`FlightController`).

The following decision table serves as the **Single Authoritative Source of Truth** for all failure modes across the ARL autonomy stack:

| Failure Mode | Observable State Variables | Remaining Control Capabilities | Required Mission State Transition | Permitted Low-Level Flight Response | Prohibited Response | Required Telemetry | Recovery Conditions | Timeout / Unresolved Dependency | Validation Scenario |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Brief Exteroceptive Dropout** | Full ego-motion $[\mathbf{p}, \mathbf{v}, \mathbf{q}, \boldsymbol{\omega}]$ healthy | Full 3D position hold, velocity damping, attitude control | Maintain `EXECUTING` | Coast active tracks forward via Kalman filter; inflate spatial covariance $\Sigma_p(t)$; enforce delay-aware braking horizon speed limit | **PROHIBITED** to abort abruptly; **PROHIBITED** to accelerate forward | `failsafe_event: "brief_dropout"`, `dropout_duration_sec` | Sensor frame received within $\Delta t \le 0.30\text{ s}$ with valid timestamps | $\Delta t \le 0.30\text{ s}$ ($\le 6$ steps at 20 Hz) | Scenario P07 |
| **2. Sustained Exteroceptive Dropout** | Full ego-motion $[\mathbf{p}, \mathbf{v}, \mathbf{q}, \boldsymbol{\omega}]$ healthy | Full 3D position hold, velocity damping, attitude control | `EXECUTING` $\to$ `WAITING` | Freeze forward path progression; command $\mathbf{v}_{\text{cmd}} = \mathbf{0}$, $\mathbf{p}_{\text{cmd}} = \mathbf{p}_{\text{hold}}$. **Stationary hover hold is physically feasible.** | **PROHIBITED** to continue forward flight into unobserved space; **PROHIBITED** to hover indefinitely | `failsafe_event: "sustained_dropout"`, `hover_hold_engaged: True` | Sensor stream restored within $2.0\text{ s}$ with verified corridor clearance | $0.30\text{ s} < \Delta t \le 2.0\text{ s}$; bounded by battery reserve | Scenario P07 |
| **3. Terminal Exteroceptive Outage** | Full ego-motion $[\mathbf{p}, \mathbf{v}, \mathbf{q}, \boldsymbol{\omega}]$ healthy | Full 3D position hold, velocity damping, attitude control | `WAITING` $\to$ `FAILED_BLOCKED` | Terminate mission. If verified obstacle-free landing corridor exists, command controlled descent ($v_z = -0.5\text{ m/s}$); otherwise hold hover until battery limit | **PROHIBITED** to plan new paths; **PROHIBITED** to execute blind high-speed return-to-base | `failsafe_event: "terminal_sensor_outage"`, `descent_engaged: bool` | Ground intervention and manual sensor reset | Outage exceeds $T_{\text{outage}} = 2.0\text{ s}$; battery reserve limit | Scenario P07 |
| **4. Loss of Horizontal Position Aiding** | Altitude $z$, vertical velocity $v_z$, attitude $\mathbf{q}$, angular velocity $\boldsymbol{\omega}$. Horizontal position $\mathbf{p}_{xy}$ is **UNOBSERVABLE** | Altitude hold, attitude stabilization, short-term horizontal velocity damping. **Position hold is IMPOSSIBLE.** | `EXECUTING` $\to$ `WAITING` $\to$ `FAILED_BLOCKED` | Command zero velocity setpoint $\mathbf{v}_{\text{cmd}, xy} = \mathbf{0}$ to damp translation kinetic energy. If unaided $> 3.0\text{ s}$, initiate controlled descent | **PROHIBITED TO CLAIM STATIONARY HOVER.** Vehicle drifts open-loop with wind and aerodynamic trim imbalances | `failsafe_event: "horizontal_aiding_loss"`, `drift_rate_est`, `unaided_sec` | Horizontal position aiding restored ($\sigma_{xy} < 1.0\text{ m}$) within $3.0\text{ s}$ | $T_{\text{unaided}} = 3.0\text{ s}$ (`PROVISIONAL TARGET`) | Scenario P11 (provisional) |
| **5. Loss of Altitude Aiding** | Horizontal position $\mathbf{p}_{xy}$, velocity $\mathbf{v}_{xy}$, attitude $\mathbf{q}$. Altitude $z$ and vertical velocity $v_z$ are **UNOBSERVABLE** | Horizontal position hold and attitude control. **Vertical rate/position closed loop is UNAVAILABLE.** | `EXECUTING` $\to$ `FAILED_BLOCKED` | Hold horizontal position; command collective thrust ramp to sub-hover thrust ($T \approx 0.90 mg$) for open-loop settling descent | **PROHIBITED TO CLAIM GUARANTEED VERTICAL DESCENT RATE** (e.g. claiming exact $-0.5\text{ m/s}$ when $v_z$ is unobservable) | `failsafe_event: "altitude_aiding_loss"`, `raw_baro_health: "fault"` | Backup rangefinder/altimeter acquisition within $1.5\text{ s}$ | $2.0\text{ s}$ maximum before ground impact mitigation | Scenario P12 (provisional) |
| **6. Loss of Velocity Observability** | Attitude $\mathbf{q}$ (gravimetric leveling) and angular rates $\boldsymbol{\omega}$ only. Position and velocity are **UNOBSERVABLE** | Attitude stabilization ($\phi = \theta = 0$, rate damping) only. **Neither position hold nor velocity damping is feasible.** | `EXECUTING` $\to$ `FAILED_BLOCKED` | Level airframe wings ($\phi_{\text{cmd}} = \theta_{\text{cmd}} = 0$); command sub-hover descent thrust ($T \approx 0.85 mg$) to settle immediately | **PROHIBITED TO CLAIM STATIONARY HOVER OR SAFE COLLISION AVOIDANCE** | `failsafe_event: "velocity_unobservable"` | Velocity observability restored ($\sigma_v < 0.5\text{ m/s}$) | $1.0\text{ s}$ (`PROVISIONAL TARGET`) | Scenario P13 (provisional) |
| **7. Attitude Estimator Failure** | **NONE.** Vehicle orientation cannot be determined | **ZERO.** Quadrotor cannot maintain torque balance or point thrust vector | Transition immediately to `ABORTED` | `NO VALIDATED SAFE ACTION SPECIFIED — SYSTEM CAPABILITY LIMITATION` | **PROHIBITED to claim safe hover, controlled descent, or motor disarm as safe. PROHIBITED to claim autorotation.** | `failsafe_event: "attitude_estimator_crash"`, `quaternion_health: "corrupted"` | None airborne; ground power-cycle only | Immediate ($0.0\text{ s}$) | Scenario P14 (provisional) |
| **8. Complete Localization Failure** | **NONE** from software state estimator | Software stack has zero flight control capability | Transition immediately to `ABORTED` | `NO VALIDATED SAFE ACTION SPECIFIED — SYSTEM CAPABILITY LIMITATION` | **PROHIBITED to claim controlled flight, safe landing, or safe hover** | `failsafe_event: "total_localization_failure"` | None airborne | Immediate ($0.0\text{ s}$) | Scenario P15 (provisional) |
| **9. Battery Reserve Violation** | All flight sensors and state estimators $[\mathbf{p}, \mathbf{v}, \mathbf{q}, \boldsymbol{\omega}]$ healthy; battery telemetry active | Full flight control available, but flight time is strictly finite ($t_{\text{remaining}} \le t_{\text{land}}$) | `EXECUTING` or `WAITING` $\to$ `FAILED_BLOCKED` | Inhibit further waiting or exploration; command immediate direct vertical descent or shortest-path return-to-base (RTB) | **PROHIBITED TO MAINTAIN INDEFINITE HOVER.** Hover consumes peak power, causing BMS low-voltage cutoff and uncontrolled crash | `failsafe_event: "battery_reserve_violation"`, `remaining_soc_pct`, `time_to_depletion_sec` | Ground landing and recharge only | Dynamic battery reserve threshold ($\text{SoC} \le 20\%$ or $t_{\text{reserve}} \le 60\text{ s}$) | Scenario P16 (provisional) |
| **10. Operator Communication Loss** | All onboard flight sensors and autonomy state estimators healthy | Full onboard autonomous navigation and flight control available | Autonomous contingency (`EXECUTING` or `WAITING` $\to$ `FAILED_BLOCKED`) | Continue pre-planned route if verified obstacle-free; if path blocked, enter bounded `WAITING` ($t \le 5.0\text{ s}$), then execute autonomous back-tracking to last comms link or vertical descent | **PROHIBITED to drift unbounded or terminate flight prematurely when onboard autonomy is healthy** | `failsafe_event: "c2_link_loss"`, `autonomous_contingency_active: True` | Datalink re-establishment with valid heartbeat within timeout | $T_{\text{c2\_timeout}} = 5.0\text{ s}$ (`PROVISIONAL TARGET`) | Scenario P17 (provisional) |
| **11. Covariance Anomaly (ADR-005)** | Sensor and state estimates valid; covariance exhibiting numerical defect | Vehicle retains full flight control; tracking pipeline updates affected | Maintain state, or trigger `REPLANNING` / `WAITING` if track purged | Apply Joseph stabilization; symmetrize $P_z$; clamp tiny eigenvalues ($-10^{-7} \le \lambda < 0$); purge divergent tracks ($\lambda_{\min} < -10^{-7}$) | **PROHIBITED to use Joseph form as automatic repair for non-finite or diverged matrices** | `covariance_health: "symmetrized" \| "eigen_clamped" \| "divergent_invalidated"` | Filter reinitialization or fresh track acquisition | 1 frame update | Scenario P09 |
| **12. Non-Finite Sensor Data** | Localization healthy; sensor returns contain NaN or Inf values | Full flight control available | Maintain state; if corrupted frames persist $> 3$ frames $\to$ `ABORTED` | Drop corrupted points; flag `valid_mask = False` | **PROHIBITED to map invalid returns to $0.0\text{ m}$ (phantom obstacle) or `max_range` (obstacle masking)** | `failsafe_event: "corrupted_sensor_frame"` | Clean finite frame received within 3 consecutive frames | 3 consecutive frames ($0.15\text{ s}$) | Scenario P09 |
| **13. Excessive Track Uncertainty** | Localization healthy; track position uncertainty $\sigma_{\max} > 2.5\text{ m}$ | Full flight control available | Trigger `REPLANNING`; if blocked $\to$ `WAITING` | Treat track as large spatial exclusion sphere ($r_{\text{eff}} = \hat{r}_{\text{obs}} + 2.5\text{ m}$); trigger dynamic replan | **PROHIBITED to fly through high-uncertainty corridor** | `failsafe_event: "excessive_track_uncertainty"`, `track_id`, `sigma_max` | Track uncertainty reduces below $2.5\text{ m}$ via fresh measurements | Bounded by waiting timeout ($5.0\text{ s}$) | Scenario P03 |
| **14. Stale Track Stagnation** | Localization healthy; track unobserved for $> K_{\text{max\_coast}}$ (5 frames) | Full flight control available | Re-evaluate world model clearance | Purge track from active memory; clear associated world-model exclusion volume | **PROHIBITED to propagate unobserved tracks indefinitely without covariance inflation** | `failsafe_event: "stale_track_purged"`, `track_id` | New detection re-initiates track lifecycle | 5 frames ($0.25\text{ s}$) | Scenario P07 |

---

### 1.1 Critical Prerequisites for Emergency Controlled Descent
In an unmapped or obstacle-rich flight arena, commanding vertical descent without active exteroceptive sensing carries inherent collision risk with obstacles directly beneath the airframe. The autonomy stack strictly distinguishes four descent regimes:

1. **Verified Corridor Descent (Nominal Fail-Safe):**
   - **Prerequisites:** A verified obstacle-free vertical corridor was logged prior to sensor loss, downward rangefinder sensing is operational, and altitude/velocity estimation is healthy.
   - **Action:** Execute controlled constant-rate vertical descent ($v_z = -0.5\text{ m/s}$) to landing.
2. **Unverified Terrain Descent (High-Risk Fail-Safe of Last Resort):**
   - **Prerequisites:** Exteroceptive sensing is lost, terrain beneath the aircraft is unmapped, but vehicle localization and attitude control remain healthy.
   - **Action:** Bounded emergency descent executed strictly to prevent unbounded horizontal drift, arena wall breach, or battery exhaustion. This action is classified as a **High-Risk Emergency Action**, *not* a guaranteed safe landing.
3. **Loss of Altitude Observability Descent:**
   - **Prerequisites:** Barometer/altimeter failed; vertical state $z, v_z$ unobservable.
   - **Action:** Closed-loop rate control is impossible. Command a feedforward sub-hover collective thrust ramp ($T \approx 0.90 mg$) to settle gently without tumbling.
4. **Loss of Attitude Control:**
   - **Prerequisites:** Attitude estimator crashed or quaternion non-finite.
   - **Action:** Controlled descent is **physically impossible**. Quadrotor will tumble; cannot be recovered by software.

### 1.2 Battery Reserve Dependency & Bounded Waiting Policy
A multicopter in stationary hover consumes near-peak electrical power ($P_{\text{hover}} \approx 150\text{--}250\text{ W}$ for typical 1 kg quadrotors), rapidly depleting onboard battery reserves.
- **Indefinite Hover Prohibition:** Under no circumstances may the aircraft remain in `WAITING` (hover hold) indefinitely. Hovering until total battery exhaustion triggers battery management system (BMS) low-voltage shutdown, resulting in motor cut-off and an uncontrolled free-fall crash.
- **Bounded Waiting Timeout:** The duration of any `WAITING` state is strictly bounded by:
  $$t_{\text{wait}} \le \min\left(t_{\text{max\_wait}}, \; t_{\text{battery\_reserve}}\right)$$
  where:
  - $t_{\text{max\_wait}} = 5.0\text{ s}$: `PROVISIONAL TARGET` (engineering timeout for obstacle clearance or sensor recovery).
  - $t_{\text{battery\_reserve}}$: Dynamic estimate of remaining flight time before reaching the mandatory 20% landing reserve.
  If $t_{\text{wait}}$ exceeds this bound, the mission transitions immediately from `WAITING` to `FAILED_BLOCKED` and initiates emergency landing.

### 1.3 Multicopter Aerodynamics vs. Helicopter Autorotation
> [!IMPORTANT]
> A multi-rotor aerial vehicle (quadrotor) **CANNOT AUTOROTATE**.
> In full-scale helicopters, variable-pitch rotor blades can be transitioned to negative/low collective pitch during an engine failure, drawing kinetic energy from upward airflow to maintain rotor RPM, followed by a positive collective flare before touchdown.
> In contrast, quadrotors utilize fixed-pitch propellers. If collective motor thrust is dropped to zero or minimum, the fixed-pitch blades lack collective pitch control; rotor speed decays rapidly, aerodynamic control torques vanish, and the vehicle enters an uncontrolled tumbling descent. Multicopter descent after power loss is a ballistic or drag-stabilized tumble, never an aerodynamic autorotation. Documentation must never claim autorotation capability for ARL vehicles.

### 1.4 Motor Disarm & Flight Termination Disclaimer
Commanding motor disarm while airborne causes instantaneous loss of control authority, transforming the quadrotor into an unguided ballistic projectile resulting in structural destruction upon ground impact.
- Motor disarm is permitted **only** after touchdown has been verified by ground contact sensors, or as an absolute flight termination command to prevent high-speed perimeter breaches.
- In the absence of an independently verified hardware ballistic parachute or certified Flight Termination System (FTS) interface in the repository:
  $$\text{NO VALIDATED SAFE ACTION SPECIFIED — SYSTEM CAPABILITY LIMITATION}$$
  The software autonomy stack acknowledges this physical hardware boundary rather than inventing unverified safety claims.

### 1.5 Elimination of Conflicting Timeout Rules
To ensure strict determinism across all architectural specifications:
1. Brief sensor dropout ($\Delta t \le 0.30\text{ s}$): Maintain `EXECUTING`; coast tracks forward.
2. Sustained sensor dropout ($0.30\text{ s} < \Delta t \le 2.0\text{ s}$): Transition to `WAITING` (safe hover hold).
3. Terminal sensor outage ($\Delta t > 2.0\text{ s}$) OR waiting timeout ($t_{\text{wait}} > 5.0\text{ s}$): Transition from `WAITING` to `FAILED_BLOCKED`; command controlled descent.
4. Estimator crash, non-finite state, or fatal covariance divergence ($\lambda_{\min} < -10^{-7}$): Transition immediately to `ABORTED`.
All other documentation files (`PHASE_V_PERCEPTION_ARCHITECTURE.md`, `PHASE_V_VALIDATION_PLAN.md`) reference these exact thresholds.

### 2. Unknown-Space Policy: Delay-Aware Braking Horizon Constraint
In 3D aerial navigation, space traversed by the sensor beam is verified `FREE`. Space intersecting obstacle returns is verified `OCCUPIED`. All other space (behind obstacles, outside sensor FOV, or beyond maximum range) is `UNKNOWN`.

- **Policy Principle: Unknown space is NOT guaranteed free space.**
- **Initial Global Path Planning:**
  At mission start ($t=0$), before the drone has explored the arena, global path planning operates under an **Optimistic but Validated** assumption: it plans an initial candidate route through unobserved space, but enforces a maximum forward lookahead clearance requirement.

#### 2.1 General Allowed-Speed Expression: Full Mathematical Derivation
When an obstacle appears at the boundary of verified visible free space, the aircraft cannot decelerate instantaneously. During the upper-bounded system response delay $\tau_{\text{delay}}$, the vehicle may be actively accelerating forward with acceleration $a_{\text{forward\_max}} \ge 0$.

Let:
- $v$: Initial forward ground speed at the detection epoch ($v \ge 0$, in $\text{m/s}$).
- $\tau_{\text{delay}}$: Upper-bounded total system response delay ($\tau_{\text{delay}} > 0$, in $\text{s}$).
- $a_{\text{forward\_max}}$: Maximum possible forward acceleration during response delay ($a_f \ge 0$, in $\text{m/s}^2$).
- $a_{\text{brake\_min}}$: Guaranteed lower bound on achievable horizontal braking deceleration ($a_b > 0$, in $\text{m/s}^2$).
- $D$: Verified available visible free distance minus safety margin ($D = d_{\text{available}} \ge 0$, in $\text{m}$).

1. **Forward Velocity at Initiation of Active Braking:**
   $$v_{\text{brake\_entry}} = v + a_{\text{forward\_max}} \cdot \tau_{\text{delay}}$$
2. **Displacement During Response Delay:**
   $$d_{\text{delay}} = v \cdot \tau_{\text{delay}} + \frac{1}{2} a_{\text{forward\_max}} \cdot \tau_{\text{delay}}^2$$
3. **Displacement During Active Deceleration:**
   $$d_{\text{braking}} = \frac{v_{\text{brake\_entry}}^2}{2 a_{\text{brake\_min}}} = \frac{(v + a_{\text{forward\_max}} \cdot \tau_{\text{delay}})^2}{2 a_{\text{brake\_min}}}$$
4. **Total Conservative Stopping Distance Inequality:**
   $$d_{\text{stop}}(v) = d_{\text{delay}} + d_{\text{braking}} = v \cdot \tau_{\text{delay}} + \frac{1}{2} a_{\text{forward\_max}} \cdot \tau_{\text{delay}}^2 + \frac{(v + a_{\text{forward\_max}} \cdot \tau_{\text{delay}})^2}{2 a_{\text{brake\_min}}} \le D$$

To solve for the maximum permissible initial speed $v_{\text{max\_allowed}}$, we express the inequality as a quadratic polynomial in $v$:
$$\frac{1}{2 a_b} v^2 + \tau \left(1 + \frac{a_f}{a_b}\right) v + \left[ \frac{1}{2} a_f \tau^2 \left(1 + \frac{a_f}{a_b}\right) - D \right] \le 0$$
Multiplying through by $2 a_b > 0$:
$$v^2 + 2 \tau (a_b + a_f) v + \left[ a_f \tau^2 (a_b + a_f) - 2 a_b D \right] \le 0$$
Completing the square on $v$:
$$\left[ v + \tau (a_b + a_f) \right]^2 - \tau^2 (a_b + a_f)^2 + a_f \tau^2 (a_b + a_f) - 2 a_b D \le 0$$
Simplifying the constant terms:
$$-\tau^2 (a_b + a_f)^2 + a_f \tau^2 (a_b + a_f) = \tau^2 (a_b + a_f) \left[ -(a_b + a_f) + a_f \right] = - a_b \tau^2 (a_b + a_f)$$
Substituting back:
$$\left[ v + \tau (a_b + a_f) \right]^2 - a_b \left[ \tau^2 (a_b + a_f) + 2 D \right] \le 0$$
$$\left[ v + \tau (a_b + a_f) \right]^2 \le a_b \left[ (a_b + a_f) \tau^2 + 2 D \right]$$
Taking the positive square root yields the **General Delay-Aware Speed Limit**:
$$v_{\text{max\_allowed}} = \sqrt{a_{\text{brake\_min}} \left[ (a_{\text{brake\_min}} + a_{\text{forward\_max}}) \tau_{\text{delay}}^2 + 2 D \right]} - (a_{\text{brake\_min}} + a_{\text{forward\_max}}) \tau_{\text{delay}}$$

**Physical Units & Dimensional Consistency:**
- $[a_b \cdot ((a_b + a_f) \tau^2 + 2 D)] = \frac{\text{m}}{\text{s}^2} \cdot \left[ \frac{\text{m}}{\text{s}^2} \cdot \text{s}^2 + \text{m} \right] = \frac{\text{m}^2}{\text{s}^2}$
- Square root term has units of $\sqrt{\text{m}^2/\text{s}^2} = \text{m/s}$.
- Second term $(a_b + a_f) \tau$ has units of $\frac{\text{m}}{\text{s}^2} \cdot \text{s} = \text{m/s}$.
- Dimensionally homogeneous and consistent in $\text{m/s}$.

**Domain Restrictions & Minimum Available Distance ($D_{\text{min\_req}}$):**
For $v_{\text{max\_allowed}} \ge 0$, we require:
$$\sqrt{a_b \left[ (a_b + a_f) \tau^2 + 2 D \right]} \ge (a_b + a_f) \tau$$
Squaring both sides:
$$a_b (a_b + a_f) \tau^2 + 2 a_b D \ge (a_b + a_f)^2 \tau^2$$
$$2 a_b D \ge (a_b + a_f) \tau^2 \left[ (a_b + a_f) - a_b \right] = a_f (a_b + a_f) \tau^2$$
Dividing by $2 a_b$:
$$D \ge D_{\text{min\_req}} = \frac{a_f (a_b + a_f) \tau^2}{2 a_b} = \frac{1}{2} a_f \tau^2 + \frac{(a_f \tau)^2}{2 a_b}$$

**Physical Significance of $D_{\text{min\_req}}$:**
$D_{\text{min\_req}}$ is the exact distance traveled if the aircraft starts from rest ($v = 0$) and accelerates forward at rate $a_f$ throughout the response delay $\tau$.
- **Feasible Regime ($D \ge D_{\text{min\_req}}$):** A valid non-negative initial speed exists: $v_{\text{max\_allowed}} \ge 0$.
- **Infeasible Regime ($D < D_{\text{min\_req}}$):** The available visible corridor is smaller than the minimum stopping distance from rest under forward acceleration $a_f$. In this regime, the radical evaluates to less than $(a_b + a_f) \tau$, producing a negative root.
  **Autonomous Action in Infeasible Regime:** The aircraft MUST command $v_{\text{max\_allowed}} = 0.0\text{ m/s}$, inhibit forward acceleration ($a_f \leftarrow 0$), and immediately enter `WAITING` (safe hover hold) or execute a yaw sensor sweep.

#### 2.2 Constant-Speed Simplification ($a_{\text{forward\_max}} = 0$)
When $a_{\text{forward\_max}} = 0$, substituting $a_f = 0$ into the general formula gives:
$$v_{\text{max\_allowed}} = \sqrt{a_b \left[ a_b \tau^2 + 2 D \right]} - a_b \tau = a_{\text{brake\_min}} \left( \sqrt{\tau_{\text{delay}}^2 + \frac{2 D}{a_{\text{brake\_min}}}} - \tau_{\text{delay}} \right)$$
> [!IMPORTANT]
> The simplified formula is valid **if and only if** $a_{\text{forward\_max}} = 0$. This condition is strictly restricted to flight phases where:
> 1. The quadrotor has completed its waypoint acceleration phase and is in steady-state cruise.
> 2. The flight controller commands zero or negative forward acceleration setpoints.
> 3. Aerodynamic tailwinds are bounded ($v_{\text{wind\_tail}} \le 0$).

#### 2.3 Response Delay Classification Budget
The composite system response delay $\tau_{\text{delay}}$ comprises five distinct physical and computational components, classified strictly by evidence level:

| Delay Component | Parameter Symbol | Nominal Value | Evidence Classification | Technical Justification |
| :--- | :--- | :--- | :--- | :--- |
| **Sensor Measurement Interval** | $\tau_{\text{sensor}}$ | $0.050\text{ s}$ | `DERIVED FROM IMPLEMENTATION` | Derived from configured sensor update rate ($f_{\text{sensor}} = 20\text{ Hz} \implies \Delta t = 0.050\text{ s}$). Represents worst-case arrival latency within discrete clock. |
| **Point-Cloud Processing Latency** | $\tau_{\text{proc}}$ | $0.020\text{ s}$ | `PROVISIONAL TARGET` | Engineering target for spatial clustering and detection pipeline execution on standard benchmark hardware in Phase V.2. |
| **Tracking & Replanning Latency** | $\tau_{\text{replan}}$ | $0.010\text{ s}$ | `PROVISIONAL TARGET` | Engineering target for Kalman filter state update and local clearance check. |
| **Attitude Tilt Reorientation Delay** | $\tau_{\text{attitude}}$ | $0.100\text{ s}$ | `ASSUMED` | Estimated duration required to reorient quadrotor from cruise pitch $\theta_{\text{cruise}}$ to maximum reverse braking pitch $\theta_{\text{brake}}$ under angular rate limits ($\dot{\theta}_{\max} \approx 3.0\text{ rad/s}$). **Not yet experimentally measured on 6-DOF simulator.** |
| **Motor Thrust Lag** | $\tau_{\text{motor}}$ | $0.030\text{ s}$ | `DERIVED FROM IMPLEMENTATION` | Derived directly from physical first-order brushless ESC/motor lag parameter $\tau_{\text{motor}} = 0.030\text{ s}$ in `DroneDynamics6DOF`. |
| **Total Response Delay** | $\tau_{\text{delay}}$ | $\approx 0.210\text{ s}$ | `ASSUMED COMPOSITE BUDGET` | Upper-bounded composite response delay budget. |

> [!CAUTION]
> Neither the total delay budget ($\tau_{\text{delay}} = 0.21\text{ s}$) nor the braking deceleration ($a_{\text{brake\_min}}$) has been experimentally demonstrated on physical flight hardware. They represent a provisional engineering budget to be validated in Phase V.5.

#### 2.4 Timing Breakdown, Measurement Age & Concurrency
To avoid double-counting latency terms, the timing architecture is explicitly sequenced:
1. **Measurement Sample Age vs. Sampling Interval:**
   The sensor captures frames at discrete epochs $t_k = k \cdot \Delta t$ ($50\text{ ms}$ interval). If an obstacle appears immediately after epoch $t_k$, it is not observed until $t_{k+1}$. The maximum age of the latest measurement at acquisition is $\tau_{\text{sensor}} = 50\text{ ms}$.
2. **Sequential Computational Pipeline:**
   Upon frame arrival at $t_{k+1}$, point-cloud clustering consumes $\tau_{\text{proc}} \le 20\text{ ms}$, followed by tracking and 4D lattice clearance check consuming $\tau_{\text{replan}} \le 10\text{ ms}$. These computational steps execute sequentially, yielding total software processing latency $\tau_{\text{proc}} + \tau_{\text{replan}} \le 30\text{ ms}$.
3. **Actuator Execution Latency:**
   Once a braking command is issued, attitude reorientation ($\tau_{\text{attitude}} \approx 100\text{ ms}$) and motor response ($\tau_{\text{motor}} \approx 30\text{ ms}$) begin. Because aerodynamic braking force builds as pitch tilts, attitude reorientation and motor dynamics act in series-parallel. Treating them as additive dead time ($\tau_{\text{attitude}} + \tau_{\text{motor}} = 130\text{ ms}$) is a conservative upper bound.
4. **Pipelining Concurrency:**
   In future multi-threaded avionics architectures (Phase V.4), point-cloud clustering of frame $k$ may overlap with path tracking of frame $k-1$. In the current single-threaded baseline, all steps execute sequentially.

#### 2.5 Actuator First-Order Dynamics vs. Pure Dead Time
In `DroneDynamics6DOF` (`src/adaptive_rl/environments/drone_6dof.py`), motor dynamics are modeled as a coupled first-order differential equation:
$$\dot{f}_i(t) = \frac{f_{\text{cmd}, i}(t) - f_i(t)}{\tau_{\text{motor}}}$$
integrated analytically at each Runge-Kutta 4 (RK4) evaluation substep:
$$f_i(t + h) = f_i(t) e^{-h / \tau_{\text{motor}}} + f_{\text{cmd}, i} \left(1 - e^{-h / \tau_{\text{motor}}}\right)$$
where $h = \Delta t / \text{substeps} = 0.05 / 5 = 0.01\text{ s}$, and $\tau_{\text{motor}} = 0.030\text{ s}$.
- A first-order lag is **NOT** a pure dead-time delay. At $t = \tau_{\text{motor}}$, the thrust reaches $1 - e^{-1} \approx 63.2\%$ of commanded thrust; at $t = 3 \tau_{\text{motor}}$, it reaches $95.0\%$.
- Modeling $\tau_{\text{motor}}$ as an additive pure dead time in $\tau_{\text{delay}}$ is a **conservative equivalent-delay approximation** that overestimates stopping distance, ensuring safety margins.
- In Phase V.5 simulation benchmarks, motor dynamics are simulated directly via RK4 substeps rather than approximated as dead time.

#### 2.6 Attitude Reorientation Modeling
The assumed attitude delay $\tau_{\text{attitude}} = 0.100\text{ s}$ represents the time required to tilt the quadrotor from cruise pitch $\theta_{\text{cruise}}$ to maximum reverse braking pitch $\theta_{\text{brake}}$:
$$t_{\text{attitude}} = \frac{|\Delta \theta|}{\dot{\theta}_{\max}} = \frac{|\theta_{\text{brake}} - \theta_{\text{cruise}}|}{\dot{\theta}_{\max}}$$
Assuming a nominal pitch change $|\Delta \theta| \approx 17^\circ \approx 0.30\text{ rad}$ and maximum angular pitch rate $\dot{\theta}_{\max} = 3.0\text{ rad/s}$ under rate limits:
$$t_{\text{attitude}} = \frac{0.30\text{ rad}}{3.0\text{ rad/s}} = 0.100\text{ s}$$
> [!NOTE]
> This linear expression assumes a bang-bang or rectangular pitch rate profile. In reality, angular acceleration $\ddot{\theta}$ is limited by rotor torque and rotational inertia ($I_{yy} = 0.01\text{ kg}\cdot\text{m}^2$). This value is classified strictly as `ASSUMED` and must be calibrated on the 6-DOF simulator.

#### 2.7 Aerodynamic Wind, Drag, and Forward Acceleration
Tailwind velocity must NOT be confused with forward acceleration:
- Let $\mathbf{v}_g$ denote vehicle ground-relative velocity, $\mathbf{v}_w$ denote ambient atmospheric wind velocity, and $\mathbf{v}_a = \mathbf{v}_g - \mathbf{v}_w$ denote air-relative velocity (airspeed).
- Aerodynamic drag acts opposing airspeed: $\mathbf{F}_{\text{drag}} = -c_{\text{drag}} \|\mathbf{v}_a\| \mathbf{v}_a$.
- A constant tailwind ($\mathbf{v}_w$ in flight direction) reduces air-relative velocity during braking, diminishing aerodynamic drag and reducing net deceleration.
- Wind gusts introduce transient gust accelerations $a_{\text{gust}} = \frac{1}{m} \frac{\partial \mathbf{F}_{\text{drag}}}{\partial \mathbf{v}_w} \dot{\mathbf{v}}_w$. When navigating in gusty environments, $a_{\text{forward\_max}}$ must encompass the maximum anticipated gust acceleration ($a_{\text{forward\_max}} \ge a_{\text{cmd\_max}} + a_{\text{gust\_max}}$).

#### 2.8 Simulation Protocol for Empirical Measurement of $a_{\text{brake\_min}}$ (Phase V.5)
In Phase V.5, braking performance will be evaluated empirically in `Drone6DOFEnv` across adversarial Monte Carlo test runs:
1. **Test Envelope & Initial Conditions:**
   - Initial forward velocities: $v_0 \in \{1.0, 2.0, 3.0, 4.0, 5.0\}\text{ m/s}$.
   - Mass domain randomization: $m \in [0.90, 1.15]\text{ kg}$ (nominal $1.0\text{ kg}$).
   - Motor thrust degradation: $k_{\text{thrust}} \in [0.90, 1.00]$ (accounting for battery voltage sag).
   - Maximum braking pitch limit: $\theta_{\text{brake\_limit}} \le 30^\circ$ ($0.524\text{ rad}$).
   - Sustained adverse tailwind: $v_{\text{wind}} = 3.0\text{ m/s}$ in flight heading direction.
   - Numerical integration: RK4 with step size $h = 0.01\text{ s}$ (5 substeps per 0.05 s environment step).
2. **Deceleration Measurement & Extraction:**
   Upon issuance of the emergency braking command at $t_{\text{cmd}}$, ground speed $v_g(t)$ is recorded at each 100 Hz simulation substep. The sustained braking deceleration is extracted as:
   $$a_{\text{brake}}(t) = -\frac{d v_g(t)}{dt}$$
3. **Statistical Reporting & Sample Bound:**
   - Evaluations must span $M \ge 10$ distinct, recorded seeds.
   - The minimum observed deceleration in the finite sample is reported strictly as:
     $$\text{observed\_min\_deceleration} = \min_{m, t} a_{\text{brake}}^{(m)}(t)$$
   - A statistically defensible lower bound is computed via one-sided normal tolerance limits or Student's $t$ lower confidence limit:
     $$a_{\text{brake\_min\_empirical}} = \bar{a}_{\text{brake}} - k_{\text{tol}} s_a$$
   - Prior to these empirical benchmarks, any numerical figure (e.g. $2.0\text{ m/s}^2$) is strictly classified as a `PROVISIONAL TARGET`.

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
