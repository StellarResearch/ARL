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

1. **Detection Association:**
   A detection $j$ at $\hat{\mathbf{c}}_j$ matches ground-truth obstacle $i$ at $\mathbf{c}_i^*$ if:
   $$\|\hat{\mathbf{c}}_j - \mathbf{c}_i^*\|_2 \le d_{\text{match\_thresh}} = r_{\text{obs}, i} + 0.5\text{ m}$$
   Unmatched detections are False Positives ($FP$); unmatched ground-truth obstacles within the observable sensor FOV are False Negatives ($FN$); matched pairs are True Positives ($TP$).

2. **Precision & Recall:**
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

3. **Centroid Localization Error (RMSE):**
   $$\text{RMSE}_{\text{det}} = \sqrt{\frac{1}{TP} \sum_{k=1}^{TP} \|\hat{\mathbf{c}}_k - \mathbf{c}_k^*\|_2^2}$$

4. **Bounding Radius Error:**
   $$e_r = \frac{1}{TP} \sum_{k=1}^{TP} (\hat{r}_k - r_k^*)$$

---

### 1.2 Multi-Object Tracking (MOT) Metrics
Evaluated over continuous flight trajectories using the classical CLEAR MOT standard (Bernardin & Stiefelhagen, 2008), adapted for exteroceptive aerial range sensing:

1. **Evaluation-Eligible Ground-Truth Set ($GT_{\text{eval}, t}$):**
   An obstacle $i$ belongs to the ground-truth evaluation set $GT_{\text{eval}, t}$ at time $t$ if and only if it satisfies sensor observability constraints:
   - Range constraint: $\|\mathbf{c}_{i, t}^* - \mathbf{p}_{\text{drone}, t}\| \le d_{\text{max\_range}}$.
   - Angular field of view constraint: Center $\mathbf{c}_{i, t}^*$ lies within sensor horizontal and vertical FOV.
   - Occlusion constraint: Line of sight from sensor origin to obstacle center is not fully occluded by another obstacle (ray transmission $\ge 10\%$).
   Obstacles outside the sensor FOV or occluded behind solid structures are physically unobservable and are strictly excluded from $GT_{\text{eval}, t}$ so that tracker performance is not penalized for unobservable space.

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
To ensure that the estimated covariance matrix $P(t)$ accurately reflects physical prediction dispersion rather than arbitrary values:

1. **Normalized Estimation Error Squared (NEES):**
   For a 3D position error $\tilde{\mathbf{p}}_k = \hat{\mathbf{p}}_k - \mathbf{p}_k^*$ and estimated position covariance $\Sigma_{p, k}$:
   $$\epsilon_{p, k} = \tilde{\mathbf{p}}_k^T \Sigma_{p, k}^{-1} \tilde{\mathbf{p}}_k$$
   Under linear Gaussian estimation assumptions, $\epsilon_{p, k}$ follows a Chi-squared distribution with 3 degrees of freedom ($\epsilon_{p, k} \sim \chi_3^2$).

2. **Average NEES ($\bar{\epsilon}_p$) Across Independent Monte Carlo Runs:**
   Let $\epsilon_{p, k}^{(m)}$ be the position NEES at step $k$ for Monte Carlo run $m \in \{1, \dots, M\}$.
   If the $M$ runs are independent (distinct evaluation episodes seeded with unique initial conditions and random noise realizations):
   $$Y_{p, k} = \sum_{m=1}^M \epsilon_{p, k}^{(m)} \sim \chi^2_{3M}$$
   The Average NEES is:
   $$\bar{\epsilon}_{p, k} = \frac{1}{M} \sum_{m=1}^M \epsilon_{p, k}^{(m)}$$
   The two-sided $95\%$ probability acceptance interval for statistical filter consistency is:
   $$\left[ \frac{\chi^2_{3M}(0.025)}{M}, \; \frac{\chi^2_{3M}(0.975)}{M} \right]$$
   - If $\bar{\epsilon}_{p, k} > \frac{\chi^2_{3M}(0.975)}{M}$: Filter is **overconfident** (true error exceeds covariance; high risk of collision).
   - If $\bar{\epsilon}_{p, k} < \frac{\chi^2_{3M}(0.025)}{M}$: Filter is **underconfident** (covariance is overly inflated; navigable corridors blocked).

   > [!IMPORTANT]
   > Independence is established across $M$ distinct Monte Carlo simulation episodes with unique randomized seeds. Sequential time samples along a single flight trajectory are temporally autocorrelated and must not be treated as independent identically distributed samples for Chi-squared bounds.

3. **Empirical Confidence Ellipsoid Containment:**
   For a 3D Gaussian distribution, the containment probability $P$ inside a Mahalanobis radius $k_{\sigma}$ is governed by the CDF of $\chi_3^2$:
   $$P(\epsilon \le k_{\sigma}^2) = F_{\chi^2_3}(k_{\sigma}^2) = 2 \Phi(k_{\sigma}) - 1 - \sqrt{\frac{2}{\pi}} k_{\sigma} e^{-k_{\sigma}^2 / 2}$$
   - For $k_{\sigma} = 3.0$, $F_{\chi^2_3}(9.0) \approx 0.9707$ ($97.07\%$ containment, not $98.9\%$, which is the 2D value).
   - For $95\%$ containment: $k_{\sigma} = \sqrt{\chi^2_3(0.95)} \approx 2.795$.
   - For $99\%$ containment: $k_{\sigma} = \sqrt{\chi^2_3(0.99)} \approx 3.368$.
   The empirical containment metric verifies:
   $$\text{Coverage}(k_{\sigma}) = \frac{1}{N} \sum_{k=1}^N \mathbb{I}\left( \|\hat{\mathbf{p}}_k - \mathbf{p}_k^*\|_2 \le k_{\sigma} \sqrt{\lambda_{\max}(\Sigma_{p, k})} \right) \ge F_{\chi^2_3}(k_{\sigma}^2) - \delta_{\text{tol}}$$

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
| **P07** | `sensor_blackout_coasting` | Sensor feed dropped for $1.0\text{ s}$ during dynamic obstacle crossing. | Track coasting, uncertainty expansion, fail-safe hover hold. A 1.0 s blackout tests that the vehicle enters and maintains `WAITING` (hover hold) with 100% reliability, zero collisions, and resumes when the stream restores within 2.0 s. Outage exceeding 2.0 s triggers terminal controlled descent to landing. |
| **P08** | `divergent_prediction_evasion` | Dynamic obstacle shifts velocity unexpectedly at $t=2.5\text{ s}$. | Online divergence detection, receding-horizon evasive replan. |
| **P09** | `heavy_sensor_noise_stress` | LiDAR range noise increased to $\sigma = 0.20\text{ m}$ ($6\times$ nominal). | Filter robustness, NEES calibration, false positive rejection. |
| **P10** | `dense_clutter_perception_flight`| Full autonomous mission: 3 sequential waypoints, 3 dynamic + 3 static obstacles. | End-to-end perception-aware navigation; zero ground truth. |

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

In accordance with Phase III/IV standards:
1. **Inferential Sample Size ($N \ge 10$):**
   - Full benchmark evaluations mandate $N \ge 10$ independent episodes across distinct, recorded seeds.
   - 95% Confidence Intervals are computed via Student's $t$-distribution:
     $$\text{CI}_{95} = \bar{x} \pm t_{0.025, N-1} \frac{s}{\sqrt{N}}$$
2. **Smoke Test Marker:**
   Runs with $N < 10$ are explicitly tagged `is_smoke_test = True`; inferential confidence intervals are suppressed to prevent misleading scientific conclusions.
3. **Deterministic Seed Scheduling:**
   Standard seed schedules (`seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]`) guarantee **seeded repeatability** on identical runtime environments.
4. **Repeatability vs. Cross-Platform Reproducibility:**
   - **Seeded Repeatability:** Executing a scenario on the *same machine, OS, Python version, and BLAS/LAPACK library* with a fixed seed produces bit-exact identical trajectories.
   - **Cross-Platform Reproducibility:** Due to floating-point operation re-ordering in multi-threaded BLAS/LAPACK (e.g. OpenBLAS vs MKL) and compiler SIMD optimizations, slight variations in least-significant bits can accumulate over thousands of integration steps. Statistical reporting mandates aggregate confidence intervals rather than assuming cross-platform bit-exact equality.

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
| **Phase V.6** | Adversarial Validation | $100\%$ fail-safe hover hold on sensor blackout; Success $\ge 85\%$ across full 10-scenario suite ($N \ge 10$) | `PROVISIONAL TARGET` |
