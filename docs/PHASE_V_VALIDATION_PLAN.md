# ARL Phase V: Scientific Validation Plan & Experimental Protocol

## Executive Summary

Phase V evolves AdaptiveRL from omniscient simulation navigation to perception-aware autonomous flight. Because autonomous perception introduces stochastic uncertainties, measurement latency, clustering approximations, and tracking errors, a rigorous and reproducible scientific validation framework is essential.

This document establishes the **Phase V Scientific Validation Plan**:
1. Formal mathematical metric definitions across detection, tracking, uncertainty calibration, navigation, and robustness.
2. An authoritative ground-truth evaluation protocol enforcing strict isolation.
3. Standardized experimental scenarios (Scenarios P01 through P10).
4. Statistical significance protocols ($N \ge 10$ inferential threshold, Student's $t$ confidence intervals).
5. Provisional acceptance criteria and phase gates for Phases V.1 through V.6.

> [!IMPORTANT]
> All quantitative performance targets specified herein are **PROVISIONAL TARGETS** established as engineering design goals. In accordance with Non-Negotiable Engineering Rules 8 and 9, they do not represent measured baseline data and must be verified empirically in subsequent phases.

---

## 1. Scientific Metric Definitions

```mermaid
flowchart TD
    subgraph METRICS["Perception-Aware Evaluation Taxonomy"]
        DET["1. Detection Metrics<br/>(Precision, Recall, Localization RMSE)"]
        TRK["2. Tracking Metrics<br/>(CLEAR MOTA, MOTP, Velocity RMSE, IDSW)"]
        UNC["3. Uncertainty Metrics<br/>(NEES, Covariance Coverage, Calibration)"]
        NAV["4. Navigation & Safety Metrics<br/>(Clearance, Collisions, Replan Rate)"]
        ROB["5. Adversarial Robustness Metrics<br/>(Dropout, Latency, Noise Breakdown)"]
    end
```

### 1.1 Exteroceptive Detection Metrics (Component-Level)
Evaluated at the component level on isolated point clouds and static sensor frames without closed-loop flight dynamics:

1. **One-to-One Bipartite Detection Association:**
   To evaluate detection quality rigorously and prevent over-counting when spatial clustering produces fragmented clusters, associations between candidate detections $\mathcal{D} = \{\hat{\mathbf{c}}_j\}_{j=1}^{N_{\text{det}}}$ and evaluation-eligible ground-truth obstacles $GT_{\text{eval}} = \{\mathbf{c}_i^*\}_{i=1}^{N_{\text{gt}}}$ are resolved via **one-to-one bipartite matching** (e.g., Hungarian algorithm / Munkres assignment or distance-sorted greedy assignment with mutual exclusion):
   - Pairwise distance matrix: $C_{i, j} = \|\hat{\mathbf{c}}_j - \mathbf{c}_i^*\|_2$.
   - Association gating: A candidate match $(i, j)$ is eligible only if $C_{i, j} \le d_{\text{match\_thresh}} = r_{\text{obs}, i} + 0.5\text{ m}$. Inadmissible pairs receive cost $C_{i, j} = \infty$.
   - Mutual exclusion constraint: Each ground-truth obstacle $i \in GT_{\text{eval}}$ matches at most ONE detection ($TP \le 1$ per obstacle), and each detection matches at most one ground-truth obstacle.
   - Classification rules:
     - **True Positive ($TP$):** An eligible ground-truth obstacle $i \in GT_{\text{eval}}$ paired with a candidate detection $j$ within the gating threshold.
     - **False Positive ($FP$):** Any candidate detection $j$ that remains unassigned, OR any duplicate/redundant detection that lies in the vicinity of a ground-truth obstacle that is already matched to another detection. Multiple detections of the same physical obstacle contribute exactly one $TP$ and $(N_{\text{det}, i} - 1)$ $FP$s.
     - **False Negative ($FN$):** Any evaluation-eligible ground-truth obstacle $i \in GT_{\text{eval}}$ that has no assigned detection within $d_{\text{match\_thresh}}$.

2. **Precision & Recall:**
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

3. **Centroid Localization Error (RMSE):**
   Evaluated strictly over the matched pairs ($TP$):
   $$\text{RMSE}_{\text{det}} = \sqrt{\frac{1}{TP} \sum_{k=1}^{TP} \|\hat{\mathbf{c}}_k - \mathbf{c}_k^*\|_2^2}$$

4. **Bounding Radius Error:**
   $$e_r = \frac{1}{TP} \sum_{k=1}^{TP} (\hat{r}_k - r_k^*)$$

---

### 1.2 Multi-Object Tracking (MOT) Metrics
Evaluated over continuous flight trajectories using the classical CLEAR MOT standard (Bernardin & Stiefelhagen, 2008), adapted for exteroceptive aerial range sensing:

1. **Evaluation-Eligible Ground-Truth Set ($GT_{\text{eval}, t}$):**
   An obstacle $i$ belongs to the ground-truth evaluation set $GT_{\text{eval}, t}$ at time $t$ if and only if it satisfies physical sensor observability constraints:
   - **Range constraint:** Distance from sensor origin to obstacle surface does not exceed sensor maximum operational range:
     $$\|\mathbf{c}_{i, t}^* - \mathbf{p}_{\text{drone}, t}\| \le d_{\text{max\_range}} + r_{\text{obs}, i}$$
   - **Angular Field-of-View (FOV) constraint:** Obstacle volume intersects the active sensor horizontal and vertical angular aperture $[\theta_{\min}, \theta_{\max}] \times [\phi_{\min}, \phi_{\max}]$.
   - **Partial Visibility & Occlusion Threshold ($\rho_{\text{visible}}$):**
     Real-world LiDAR sensing does not view obstacle centroids directly; it receives returns from exposed surface facets. Let $\rho_{\text{visible}}(i, t) \in [0, 1]$ denote the fraction of the obstacle's forward-facing surface area (or simulated ray-cast probe intersections) that is unoccluded by intervening geometry and lies strictly within the active sensor FOV and range:
     $$\rho_{\text{visible}}(i, t) \ge \rho_{\min} = 0.15 \quad (15\%\text{ minimum visible surface threshold})$$
   - **Observability Exclusion Rule:** Obstacles outside the sensor FOV, beyond maximum range, or occluded such that $\rho_{\text{visible}}(i, t) < 0.15$ lack sufficient observable surface returns for reliable clustering and are strictly excluded from $GT_{\text{eval}, t}$. Tracker recall and MOTA are not penalized for physically unobservable targets. If an active tracker coasts or associates a marginal detection with $\rho_{\text{visible}} < 0.15$, the match is credited as a valid association without penalizing the tracker when missed.

2. **Multiple Object Tracking Accuracy (MOTA):**
   $$\text{MOTA} = 1 - \frac{\sum_t (FN_t + FP_t + \text{IDSW}_t)}{\sum_t |GT_{\text{eval}, t}|}$$
   where:
   - $FN_t$: Missed targets within observable set $GT_{\text{eval}, t}$ at time step $t$.
   - $FP_t$: Phantom / spurious confirmed tracks at time step $t$.
   - $\text{IDSW}_t$: Identity switches (a persistent target is assigned a different track ID).
   - $|GT_{\text{eval}, t}|$: Number of observable ground-truth obstacles at time step $t$.

3. **Multiple Object Tracking Precision (MOTP):**
   Measures the average spatial localization accuracy of successfully tracked targets:
   $$\text{MOTP} = \frac{\sum_{t, i} d(\mathbf{p}_{i, t}^*, \hat{\mathbf{p}}_{i, t})}{\sum_t |M_t|}$$
   where $d(\mathbf{p}_{i, t}^*, \hat{\mathbf{p}}_{i, t})$ is Euclidean distance and $|M_t|$ is the number of valid target-track matches at step $t$.

4. **Velocity Estimation RMSE:**
   $$\text{RMSE}_v = \sqrt{\frac{1}{\sum_t |M_t|} \sum_{t, i} \|\hat{\mathbf{v}}_{i, t} - \mathbf{v}_{i, t}^*\|_2^2}$$

5. **Track Continuity Metrics:**
   - **Mostly Tracked (MT):** Percentage of ground-truth trajectories tracked for $\ge 80\%$ of their observable lifespan.
   - **Mostly Lost (ML):** Percentage of ground-truth trajectories tracked for $< 20\%$ of their observable lifespan.
   - **Track Fragmentation:** Count of times an interrupted track re-acquires a target after coasting.

---

### 1.3 Uncertainty & Covariance Calibration Metrics
To ensure that the estimated covariance matrix $P(t)$ (and its spatial position submatrix $\Sigma_{p}$) accurately reflects physical prediction dispersion rather than arbitrary values:

1. **Normalized Estimation Error Squared (NEES):**
   For a 3D position error $\tilde{\mathbf{p}}_k = \hat{\mathbf{p}}_k - \mathbf{p}_k^*$ and estimated position covariance $\Sigma_{p, k}$:
   $$\epsilon_{p, k} = \tilde{\mathbf{p}}_k^T \Sigma_{p, k}^{-1} \tilde{\mathbf{p}}_k$$
   Under linear Gaussian estimation assumptions, $\epsilon_{p, k}$ follows a Chi-squared distribution with 3 degrees of freedom ($\epsilon_{p, k} \sim \chi_3^2$).

2. **Average NEES ($\bar{\epsilon}_p$) Across Independent Monte Carlo Runs:**
   Let $\epsilon_{p, k}^{(m)}$ be the position NEES at step $k$ for Monte Carlo run $m \in \{1, \dots, M\}$.
   If the $M$ runs are independent (distinct evaluation episodes seeded with unique initial conditions and random noise realizations):
   $$Y_{p, k} = \sum_{m=1}^M \epsilon_{p, k}^{(m)} \sim \chi^2_{3M}$$
   The Average NEES at time step $k$ is:
   $$\bar{\epsilon}_{p, k} = \frac{1}{M} \sum_{m=1}^M \epsilon_{p, k}^{(m)}$$
   The two-sided $95\%$ probability acceptance interval for statistical filter consistency is:
   $$\left[ \frac{\chi^2_{3M}(0.025)}{M}, \; \frac{\chi^2_{3M}(0.975)}{M} \right]$$
   - If $\bar{\epsilon}_{p, k} > \frac{\chi^2_{3M}(0.975)}{M}$: Filter is **overconfident** (true error exceeds covariance; high risk of collision).
   - If $\bar{\epsilon}_{p, k} < \frac{\chi^2_{3M}(0.025)}{M}$: Filter is **underconfident** (covariance is overly inflated; navigable corridors blocked).

   > [!IMPORTANT]
   > Independence is established across $M$ distinct Monte Carlo simulation episodes with unique randomized seeds. Sequential time samples along a single flight trajectory are temporally autocorrelated and must not be treated as independent identically distributed samples for Chi-squared bounds.

3. **Uncertainty Coverage: Exact Ellipsoid vs. Isotropic Spherical-Bound Metrics:**
   We distinguish between exact statistical confidence ellipsoid containment and the conservative isotropic bounding sphere used in path planning:

   - **Metric A: Exact Confidence Ellipsoid Coverage ($\text{Coverage}_{\text{Ellipsoid}}$):**
     Tests containment within the true anisotropic Mahalanobis confidence ellipsoid of radius $k_{\sigma}$:
     $$\text{Coverage}_{\text{Ellipsoid}}(k_{\sigma}) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\left( \tilde{\mathbf{p}}_k^T \Sigma_{p, k}^{-1} \tilde{\mathbf{p}}_k \le k_{\sigma}^2 \right)$$
     Because the Mahalanobis distance squared follows a $\chi^2_3$ distribution under linear Gaussian assumptions, the theoretical coverage probability is governed exactly by the $\chi^2_3$ cumulative distribution function:
     $$P(\epsilon \le k_{\sigma}^2) = F_{\chi^2_3}(k_{\sigma}^2) = 2 \Phi(k_{\sigma}) - 1 - \sqrt{\frac{2}{\pi}} k_{\sigma} e^{-k_{\sigma}^2 / 2}$$
     - For $k_{\sigma} = 3.0$: Theoretical containment is $F_{\chi^2_3}(9.0) \approx 0.9707$ ($97.07\%$ in 3D, not $98.9\%$, which applies only in 2D).
     - For $95\%$ containment: $k_{\sigma} = \sqrt{\chi^2_3(0.95)} \approx 2.795$.
     - For $99\%$ containment: $k_{\sigma} = \sqrt{\chi^2_3(0.99)} \approx 3.368$.
     Evaluation criterion: $|\text{Coverage}_{\text{Ellipsoid}}(k_{\sigma}) - F_{\chi^2_3}(k_{\sigma}^2)| \le \delta_{\text{tol}}$ across calibrated tracks.

   - **Metric B: Conservative Isotropic Spherical-Bound Coverage ($\text{Coverage}_{\text{SphericalBound}}$):**
     In collision avoidance and trajectory generation (ADR-005), evaluating arbitrary rotated ellipsoids is computationally expensive. The planner encloses the uncertainty ellipsoid inside a conservative isotropic sphere of radius $r_{\text{margin}} = k_{\sigma} \sqrt{\lambda_{\max}(\Sigma_{p, k})}$:
     $$\text{Coverage}_{\text{SphericalBound}}(k_{\sigma}) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}\left( \|\tilde{\mathbf{p}}_k\|_2 \le k_{\sigma} \sqrt{\lambda_{\max}(\Sigma_{p, k})} \right)$$
     **Mathematical Relationship & Surrogate Role:**
     By Rayleigh quotient bounds, for any symmetric positive-definite covariance $\Sigma_{p, k}$:
     $$\tilde{\mathbf{p}}^T \Sigma_{p, k}^{-1} \tilde{\mathbf{p}} \ge \frac{\|\tilde{\mathbf{p}}\|_2^2}{\lambda_{\max}(\Sigma_{p, k})}$$
     Therefore, the exact confidence ellipsoid is strictly contained within the isotropic bounding sphere:
     $$\left\{ \tilde{\mathbf{p}} : \tilde{\mathbf{p}}^T \Sigma_{p, k}^{-1} \tilde{\mathbf{p}} \le k_{\sigma}^2 \right\} \subseteq \left\{ \tilde{\mathbf{p}} : \|\tilde{\mathbf{p}}\|_2 \le k_{\sigma} \sqrt{\lambda_{\max}(\Sigma_{p, k})} \right\}$$
     Consequently:
     $$P\left(\|\tilde{\mathbf{p}}\|_2 \le k_{\sigma} \sqrt{\lambda_{\max}(\Sigma_{p, k})}\right) \ge F_{\chi^2_3}(k_{\sigma}^2)$$
     Metric B is an **outer planning surrogate** providing a conservative safety upper bound, *not* an exact Chi-squared level set. For anisotropic covariances ($\lambda_{\max} > \lambda_{\min}$), the spherical coverage will strictly exceed $F_{\chi^2_3}(k_{\sigma}^2)$ (e.g. for $k_{\sigma}=3$, empirical spherical coverage typically reaches $99.5\%+$). It must not be reported as if it were the theoretical 3D Gaussian containment probability.

---

### 1.4 Navigation, Safety & Mission Execution Metrics (System-Level)
Evaluated on closed-loop flight in the 6-DOF simulation environment:

1. **Authoritative Outcome Classification:**
   Mapped strictly to the 8 mutually exclusive outcomes from Phase III/IV:
   `COLLISION`, `BOUNDARY_BREACH`, `SUCCESS`, `CRASH_FAULT`, `RECOVERED_DEGRADED`, `BLOCKED_NO_PATH`, `TIMEOUT`, `MAX_STEPS`.

2. **Continuous Swept Clearance ($d_{\text{clearance}}$):**
   Evaluated analytically between quadrotor flight segments and moving obstacle envelopes (split at boundary reflection events).
   - Minimum Clearance: $d_{\min} = \min_{t} d_{\text{clearance}}(t)$.
   - Clearance Violation: $d_{\min} < r_{\text{vehicle}} = 0.25\text{ m}$.

3. **Receding-Horizon Replanning Frequency:**
   Number of dynamic replan events triggered per meter of flight path.

4. **Path Efficiency & Extra Distance:**
   $$\eta_{\text{path}} = \frac{L_{\text{optimal\_direct}}}{L_{\text{actual}}}$$

5. **Flight Control Stability:**
   - RMS attitude tilt error: $\text{RMS}(\theta_{\text{tilt}})$.
   - Maximum body angular rate: $\max \|\boldsymbol{\omega}\|$.
   - Actuator saturation ratio: Percentage of time motor thrusts are clamped to $[0, f_{\max}]$.

---

## 2. Standardized Experimental Scenarios (P01 – P10)

Ten standardized perception benchmark scenarios are defined to test the perception pipeline incrementally:

| Scenario ID | Name | Description | Key Focus Area |
| :--- | :--- | :--- | :--- |
| **P01** | `static_isolated_detection` | Single stationary sphere in clear arena. Quadrotor hovers at fixed range. | Detection precision, centroid error, clustering stability. |
| **P02** | `dynamic_linear_crossing` | Spherical intruder crosses flight path perpendicularly at $1.5\text{ m/s}$. | CV Kalman filter convergence, velocity RMSE, MOTA. |
| **P03** | `dynamic_oscillating_intruder` | Obstacle executing harmonic oscillation ($A=3.0\text{ m}, T=5.0\text{ s}$). | Tracking under continuous acceleration, NEES consistency. |
| **P04** | `boundary_bounce_reflection` | Obstacle reflecting off arena boundary walls at $2.0\text{ m/s}$. | Association recovery across sharp velocity sign inversion. |
| **P05** | `multi_obstacle_slalom` | 4 moving obstacles crossing staggered flight corridors simultaneously. | Multi-target data association, gating, identity switches (IDSW). |
| **P06** | `narrow_canyon_traversal` | Drone navigates through tight corridor between static walls. | Swept clearance under covariance inflation; false blockage checks. |
| **P07** | `sensor_blackout_coasting` | Sensor feed dropped for $1.0\text{ s}$ during dynamic obstacle crossing. | Track coasting, uncertainty expansion, fail-safe hover hold. A 1.0 s blackout tests that the vehicle enters and maintains `WAITING` (hover hold) with 100% reliability, zero collisions, and resumes when the stream restores within 2.0 s. Outage exceeding 2.0 s triggers terminal controlled descent to landing. Stationary hover hold is strictly conditioned on healthy vehicle localization (position aiding available; exteroceptive sensor dropout only). |
| **P08** | `divergent_prediction_evasion` | Dynamic obstacle shifts velocity unexpectedly at $t=2.5\text{ s}$. | Online divergence detection, receding-horizon evasive replan. |
| **P09** | `heavy_sensor_noise_stress` | LiDAR range noise increased to $\sigma = 0.20\text{ m}$ ($6\times$ nominal). | Filter robustness, NEES calibration, false positive rejection. |
| **P10** | `dense_clutter_perception_flight`| Full autonomous mission: 3 sequential waypoints, 3 dynamic + 3 static obstacles. | End-to-end perception-aware navigation; zero ground truth. |

### 2.1 Provisional Avionics & Fail-Safe Injection Scenarios (P11 – P17)
To validate the Authoritative Fail-Safe Decision Table (ADR-008) under adverse conditions, seven dedicated failure-injection test scenarios are provisionally scheduled for Phase V.6:

| Scenario ID | Name | Injected Anomaly Condition | Expected System-Level & Fail-Safe Response |
| :--- | :--- | :--- | :--- |
| **P11** | `horizontal_aiding_loss_adversarial` | External horizontal position aiding dropped at $t=2.0\text{ s}$; velocity remains observable. | Vehicle commands velocity damping setpoint $\mathbf{v}_{\text{cmd}, xy} = \mathbf{0}$; does **not** claim stationary position hold; transitions to `WAITING` then initiates controlled descent if unaided duration $> 3.0\text{ s}$. |
| **P12** | `altitude_sensor_failure` | Barometer / rangefinder reference corrupted with non-finite or frozen readings. | Autonomous flight terminates closed-loop vertical control; commands collective sub-hover thrust ramp ($T \approx 0.90 mg$) to settle; transitions to `FAILED_BLOCKED`. |
| **P13** | `strapdown_dead_reckoning_divergence`| Position and velocity aiding lost simultaneously; pure strapdown IMU dead-reckoning. | Quadrotor levels wings ($\phi = \theta = 0$); engages open-loop sub-hover descent thrust ($T \approx 0.85 mg$); transitions immediately to `FAILED_BLOCKED`. |
| **P14** | `attitude_filter_singularity_injection` | Quaternion corrupted ($|\|\mathbf{q}\|-1| \ge 0.05$) or gyro rate saturation injected. | Immediate fail-closed transition to `ABORTED`; triggers `NO VALIDATED SAFE ACTION SPECIFIED — SYSTEM CAPABILITY LIMITATION`. Prohibits autorotation claims. |
| **P15** | `total_estimator_breakdown` | Full state estimator divergence across all channels. | Immediate transition to `ABORTED`; all active path planning and tracking loops halted. |
| **P16** | `battery_exhaustion_during_waiting` | Battery State of Charge (SoC) drops to $\le 20\%$ reserve threshold while hovering in `WAITING`. | Bounded waiting rule triggers: $t_{\text{wait}} \le \min(t_{\text{max\_wait}}, t_{\text{battery\_reserve}})$; terminates hover immediately; commands emergency landing rather than hovering to power exhaustion. |
| **P17** | `c2_datalink_blackout` | Command and control heartbeat lost for $> 5.0\text{ s}$. | Autonomous contingency engages: continues route if corridor verified free; enters bounded `WAITING` ($5.0\text{ s}$) then executes autonomous back-tracking to last known comms link or vertical descent. |

---

## 3. Ground-Truth Evaluation Harness Architecture

The evaluation harness operates as an isolated observer:

```python
@dataclass
class PerceptionStepRecord:
    timestamp: float
    drone_true_position: np.ndarray
    drone_est_position: np.ndarray
    true_obstacles: List[Dict[str, Any]]        # True IDs, positions, velocities
    raw_detections: List[ObstacleDetection3D]   # Instantaneous detections
    tracked_obstacles: List[TrackedObstacle3D]  # Active filter tracks with P
    min_physical_clearance: float
    is_collision: bool
    mission_state: str

class PerceptionEvaluationHarness:
    """Offline evaluator computing scientific perception and tracking metrics."""

    def __init__(self, match_distance_threshold: float = 1.0) -> None:
        self.match_threshold = match_distance_threshold
        self.records: List[PerceptionStepRecord] = []

    def log_step(self, record: PerceptionStepRecord) -> None:
        """Record simulation step for offline metric computation."""
        self.records.append(record)

    def compute_summary_report(self) -> Dict[str, Any]:
        """Compute full CLEAR MOT, NEES, and navigation metrics over episode."""
        ...
```

---

## 4. Statistical Rigor, Seed Scheduling & Reproducibility Nuance

### 4.1 Avoidance of Temporal Autocorrelation Pooling Fallacy & Episode Aggregation
To ensure valid inferential statistics, benchmark analysis strictly differentiates **within-episode temporal dynamics** from **across-episode statistical replicates**:

1. **The Temporal Autocorrelation Fallacy:**
   Sequential state errors $\tilde{\mathbf{p}}_k$ and measurement residuals along a single flight trajectory are governed by autoregressive dynamical equations ($\tilde{\mathbf{p}}_{k+1} = (I - K_{k+1} H) F \tilde{\mathbf{p}}_k + \dots$). Consecutive time steps are strongly autocorrelated and violate the independent identically distributed (i.i.d.) assumption of classical sampling theory.
   Pooling hundreds or thousands of consecutive time steps across an episode directly into a single sample variance or confidence interval calculation commits the **temporal autocorrelation fallacy**, artificially inflating degrees of freedom, severely underestimating the standard error, and yielding falsely narrow confidence intervals.

2. **Episode-Level Summary Statistics:**
   For any evaluation metric $X$ (such as detection precision, recall, MOTA, velocity RMSE, NEES consistency, or clearance):
   - A single summary metric $X^{(m)}$ is computed for each independent simulation episode $m \in \{1, \dots, M\}$:
     $$X^{(m)} = \text{MetricFunction}\left(\{\mathbf{x}_{k}^{(m)}, \hat{\mathbf{x}}_{k}^{(m)}\}_{k=1}^{K_m}\right)$$
   - Only the episode-level summary statistics $\{X^{(1)}, X^{(2)}, \dots, X^{(M)}\}$ are treated as independent observations.

3. **Inferential Sample Size ($M \ge 10$):**
   - Full benchmark evaluations mandate $M \ge 10$ independent episodes across distinct, recorded seeds and randomized initial conditions (obstacle trajectories, sensor noise, wind disturbance).
   - The across-episode sample mean and sample standard deviation are:
     $$\bar{X} = \frac{1}{M} \sum_{m=1}^M X^{(m)}, \quad s_X = \sqrt{\frac{1}{M-1} \sum_{m=1}^M \left(X^{(m)} - \bar{X}\right)^2}$$
   - Two-sided $95\%$ Confidence Intervals are computed via Student's $t$-distribution with $M - 1$ degrees of freedom:
     $$\text{CI}_{95}(X) = \bar{X} \pm t_{0.025, M-1} \frac{s_X}{\sqrt{M}}$$
     For $M = 10$, $t_{0.025, 9} \approx 2.262$.

4. **Smoke Test Marker:**
   Runs with $M < 10$ are explicitly tagged `is_smoke_test = True`; inferential confidence intervals are suppressed to prevent misleading scientific conclusions.

### 4.2 Deterministic Seed Scheduling & Reproducibility Nuance
1. **Deterministic Seed Scheduling:**
   Standard seed schedules (`seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`) guarantee **seeded repeatability** on identical runtime environments.
2. **Repeatability vs. Cross-Platform Reproducibility:**
   - **Seeded Repeatability:** Executing a scenario on the *same machine, OS, Python version, and BLAS/LAPACK library* with a fixed seed produces bit-exact identical trajectories.
   - **Cross-Platform Reproducibility:** Due to floating-point operation re-ordering in multi-threaded BLAS/LAPACK (e.g. OpenBLAS vs MKL) and compiler SIMD optimizations, slight variations in least-significant bits can accumulate over thousands of integration steps. Statistical reporting mandates aggregate confidence intervals over independent episodes rather than assuming cross-platform bit-exact equality.

---

## 5. Provisional Phase Acceptance Criteria

All quantitative targets listed below are **PROVISIONAL ENGINEERING GOALS** to be calibrated against baseline empirical evaluations in subsequent phases:

| Phase | Milestone Name | Key Provisional Acceptance Criteria | Status |
| :--- | :--- | :--- | :--- |
| **Phase V.0** | Architecture & Specification | Complete specifications, 8 ADRs, validation plan, zero code regression | `CURRENT` |
| **Phase V.1** | Simulated Sensor Foundation | Ray casting throughput $\ge 50\text{ Hz}$ benchmark target ($< 20\text{ ms}$ per sweep) for 20 Hz simulation; range noise matches target $\sigma \pm 10\%$; dropout model verified | `PROVISIONAL TARGET` |
| **Phase V.2** | 3D Obstacle Detection | Detection Precision $\ge 95\%$; Recall $\ge 90\%$; Centroid RMSE $< 0.15\text{ m}$ on isolated obstacles | `PROVISIONAL TARGET` |
| **Phase V.3** | Dynamic Obstacle Tracking | MOTA $\ge 0.85$; Velocity RMSE $< 0.25\text{ m/s}$; Zero ID switches on linear crossing; NEES within $95\%$ bounds | `PROVISIONAL TARGET` |
| **Phase V.4** | Sensor Fusion & Extrinsics | Projected point cloud spatial error $< 0.05\text{ m}$ under $30^\circ/\text{s}$ vehicle yaw rate | `PROVISIONAL TARGET` |
| **Phase V.5** | Perception-Aware Navigation | Success Rate $\ge 90\%$ on Scenarios P01–P06; Zero ground-truth leakage; Minimum clearance $> 0.20\text{ m}$ | `PROVISIONAL TARGET` |
| **Phase V.6** | Adversarial Validation | $100\%$ fail-safe hover hold on sensor blackout (conditioned on healthy localization); Success $\ge 85\%$ across full 10-scenario suite ($M \ge 10$) | `PROVISIONAL TARGET` |
