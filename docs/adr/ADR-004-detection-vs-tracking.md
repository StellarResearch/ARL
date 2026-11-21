# ADR-004: Obstacle Detection versus Persistent Tracking

## Status
ACCEPTED (Architecture Specification for Phase V)

## Context
A critical failure mode in robotic perception is treating an instantaneous sensor detection as a confirmed, perfectly known physical obstacle.
In noisy sensor environments:
1. False positive returns (e.g., ground clutter, sensor artifacts, multipath reflections) would trigger unnecessary emergency replanning or false blocked aborts if treated as obstacles.
2. Missed detections (e.g., absorption, glancing angles, temporary occlusions) would cause the vehicle to believe an obstacle has vanished, leading to immediate flight path collisions.
3. Raw detections provide only instantaneous spatial position without velocity, preventing predictive 4D planning against moving obstacles.

Therefore, a strict conceptual and algorithmic boundary must separate **Instantaneous Obstacle Detection** from **Persistent Obstacle Tracking**.

> [!IMPORTANT]
> The data contracts and state machines described below are **ARCHITECTURAL SPECIFICATIONS** for implementation in Phase V.2 and Phase V.3 (`adaptive_rl.perception.detection` and `adaptive_rl.perception.tracking`). They are not present in the current Phase V.0 codebase.

## Decision
We decouple detection from tracking into two distinct processing stages with formal data structures:

### 1. Instantaneous Detection Contract (`ObstacleDetection3D`)
Generated independently on each sensor frame without temporal memory:
```python
@dataclass(frozen=True)
class ObstacleDetection3D:
    detection_id: str  # Ephemeral UUID for this frame
    timestamp: float  # Sensor measurement epoch
    frame_id: str  # Typically "world" after projection
    centroid: np.ndarray  # [x, y, z] center of detected cluster
    bounding_radius: float  # Enclosing spherical radius
    covariance_position: np.ndarray  # 3x3 position uncertainty matrix
    point_count: int  # Number of sensor returns supporting this detection
    confidence: float  # Detection confidence score in [0.0, 1.0]
    sensor_id: str  # Originating sensor
```
*Note on Array Immutability:* In Phase V.2 implementations, factory methods will set `covariance_position.flags.writeable = False` and `centroid.flags.writeable = False` to prevent in-place mutation.

### 2. Persistent Track Contract (`TrackedObstacle3D`)
Maintained over time by `DynamicObstacleTracker3D`:
```python
class TrackState(Enum):
    TENTATIVE = "tentative"  # Newly observed, unconfirmed
    CONFIRMED = "confirmed"  # Statistically verified, passed to planner
    COASTED = "coasted"  # Temporarily missed, forward-predicted
    DELETED = "deleted"  # Dead track, purged from memory


@dataclass
class TrackedObstacle3D:
    track_id: int  # Persistent unique integer ID across time
    state: TrackState
    state_vector: np.ndarray  # [x, y, z, vx, vy, vz]^T (6D state)
    covariance: np.ndarray  # 6x6 state covariance matrix P
    bounding_radius: float  # Estimated physical obstacle radius
    first_observed_time: float
    last_observed_time: float
    consecutive_hits: int
    consecutive_misses: int
    total_updates: int
```

### 3. Track Lifecycle & Confirmation Logic
- **Birth / Tentative:** When a detection cannot be associated with any existing track, a new `TENTATIVE` track is initialized with state $[x, y, z, 0, 0, 0]^T$ and large initial velocity variance. Tentative tracks are **never** passed to the flight planner.
- **Confirmation ($M$-of-$N$ Rule):** A tentative track transitions to `CONFIRMED` only after being associated with valid detections in at least $M=3$ out of the last $N=5$ consecutive sensor frames.
- **Coasting (Miss Handling):** If a confirmed track receives no detection in a frame, it transitions to `COASTED`. Its state and covariance are propagated forward via its motion model ($P_{k|k-1} = F P_{k-1} F^T + Q$).
- **Death / Deletion:** A track is deleted if it remains unobserved for more than $K_{\text{max\_coast}} = 5$ frames (0.25 s at 20 Hz), or if its position variance exceeds safety limits ($\text{Tr}(P_{\text{pos}}) > \sigma_{\text{max\_var}}$).

### 4. Tracking Algorithm: Constant Velocity (CV) 3D Kalman Filter
We select a **Linear 3D Constant Velocity Kalman Filter** with **Gated Nearest Neighbor (GNN) / Hungarian Data Association**:

- **State Formulation:** $\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T \in \mathbb{R}^6$
- **Measurement Formulation:** $\mathbf{z} = [x, y, z]^T \in \mathbb{R}^3$, with measurement matrix:
  $$H = \begin{bmatrix} I_{3 \times 3} & 0_{3 \times 3} \end{bmatrix} \in \mathbb{R}^{3 \times 6}$$
- **State Transition Matrix:**
  $$F(\Delta t) = \begin{bmatrix} I_{3 \times 3} & \Delta t \cdot I_{3 \times 3} \\ 0_{3 \times 3} & I_{3 \times 3} \end{bmatrix}$$
- **Process Noise Matrix ($Q$):** Continuous white-noise acceleration model discretized over $\Delta t$:
  $$Q = q_{\text{accel}} \begin{bmatrix} \frac{\Delta t^3}{3} I_{3 \times 3} & \frac{\Delta t^2}{2} I_{3 \times 3} \\ \frac{\Delta t^2}{2} I_{3 \times 3} & \Delta t \cdot I_{3 \times 3} \end{bmatrix}$$
- **Data Association:** Detections are gated using Mahalanobis distance:
  $$d_M^2 = (\mathbf{z} - H \hat{\mathbf{x}})^T S^{-1} (\mathbf{z} - H \hat{\mathbf{x}}) \le \gamma_{\text{gate}}$$
  where $S = H P H^T + R$. Cost matrix entries exceeding $\gamma_{\text{gate}}$ (e.g. $\chi^2_3(0.99) \approx 11.34$) are set to $\infty$. The optimal assignment is solved via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).

## Alternatives Considered
1. **Extended Kalman Filter (EKF) with Non-linear Spherical Models:**
   *Rejected for Phase V:* The measurement equation in Cartesian world coordinates is strictly linear ($H = [I, 0]$), and obstacle motion in ARL is piecewise linear or smooth oscillation. An EKF adds non-linear Jacobian calculation and linearization errors without providing benefits over Cartesian linear KF.
2. **Constant Acceleration (CA) Model (9D State $[p, v, a]$):**
   *Rejected:* Estimating acceleration from noisy position measurements requires high frame rates and suffers from severe noise amplification ($1/\Delta t^2$). In testing, CA models exhibit high velocity ringing during maneuvers.
3. **Interacting Multiple Model (IMM) / Joint Probabilistic Data Association (JPDA):**
   *Deferred:* IMM and JPDA provide superior tracking during rapid obstacle maneuvers and dense clutter, but introduce significant computational complexity and tuning parameters. A linear CV KF provides a mathematically transparent, defensible, and robust baseline that can be upgraded in Phase V.3 if empirical validation warrants it.

## Technical Reasoning
- **Linearity and Global Optimality:** For linear Gaussian systems, the standard Kalman filter is the provably minimum mean-square error (MMSE) estimator.
- **Low Computational Overhead:** $6 \times 6$ matrix operations take $< 0.1\text{ ms}$ in Python/NumPy, easily running 50+ tracks simultaneously without lagging the 20 Hz flight control loop.
- **Built-in Velocity Estimation:** The planner receives smoothed 3D velocities ($[v_x, v_y, v_z]$) and associated velocity covariances directly from the filter state vector, enabling predictive 4D spatiotemporal obstacle avoidance.

## Consequences
- **Positive:**
  - False positive detections are filtered out before reaching the planner.
  - Temporary occlusions and missed detections are bridged smoothly by the coasting predictor.
  - Persistent obstacle IDs enable stable temporal reasoning and velocity estimation.
- **Negative:**
  - $M$-of-$N$ confirmation introduces an initial detection latency of $M \cdot \Delta t$ (e.g., 3 frames $\approx 0.15\text{ s}$).
  - Maneuvering obstacles (e.g., sharp boundary bounces) will temporarily experience higher innovation errors until the filter converges.
- **Neutral:**
  - Requires tuning process noise parameter $q_{\text{accel}}$ and measurement covariance $R$.

## Known Limitations
- When dynamic obstacles bounce off arena boundaries (sharp velocity reversal), the linear CV assumption is temporarily violated. The process noise $Q$ must be sufficiently open to prevent filter divergence during bounces.

## Future Reconsideration Criteria
- If obstacle trajectory maneuvers cause significant track loss during Phase V.3 testing, an IMM filter combining CV and turn-rate / bouncing modes will be evaluated.
