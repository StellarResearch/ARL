# ADR-005: Uncertainty Representation and Covariance-Aware Planning

## Status
ACCEPTED (Architecture Specification for Phase V)

## Context
In Phase III and Phase IV, collision checking in `AStarPlanner3D` and `TimeAwareAStarPlanner3D` applied a fixed, static safety inflation margin:
$$r_{\text{effective}} = r_{\text{obs}} + r_{\text{vehicle}} + \delta_{\text{margin}}$$
where $r_{\text{vehicle}} = 0.25\text{ m}$ and $\delta_{\text{margin}} = 0.20\text{ m}$ (or $0.30\text{ m}$).

This deterministic formulation assumes that:
1. Obstacle position and radius are known with zero error.
2. Future obstacle positions can be projected infinitely forward in time with zero dispersion.

In perception-aware navigation, neither assumption holds. The estimated position of an obstacle is a stochastic random variable $\mathbf{p} \sim \mathcal{N}(\hat{\mathbf{p}}, \Sigma_p)$, and its velocity has uncertainty $\Sigma_v$. When predicting forward over a lookahead horizon $\tau \in [0, H]$, the position uncertainty grows monotonically. If planning ignores this growth, the drone will plan dangerously close to moving obstacles whose predicted locations have high dispersion. Conversely, if an arbitrarily large fixed margin is used everywhere, the planner experiences false blockage in navigable corridors.

## Decision
We specify a rigorous, covariance-aware spatial inflation model linking the tracking filter's uncertainty directly to the geometric collision checking routines in `PerceptionWorldModel3D` and `TimeAwareAStarPlanner3D`.

### 1. Covariance Representation
Every tracked obstacle maintains a full $6 \times 6$ state error covariance matrix:
$$P(t) = \begin{bmatrix} \Sigma_p(t) & \Sigma_{pv}(t) \\ \Sigma_{pv}(t)^T & \Sigma_v(t) \end{bmatrix} \in \mathbb{R}^{6 \times 6}$$
where $\Sigma_p(t) \in \mathbb{R}^{3 \times 3}$ is the spatial position covariance, and $\Sigma_v(t) \in \mathbb{R}^{3 \times 3}$ is the velocity covariance.

### 2. Analytical Forward Uncertainty Propagation
When evaluating clearance over lookahead horizon $\tau = t_{\text{eval}} - t_{\text{curr}} \ge 0$:
The predicted position covariance under the Constant Velocity motion model propagates as:
$$\Sigma_p(t + \tau) = \Sigma_p(t) + \tau \left(\Sigma_{pv}(t) + \Sigma_{pv}(t)^T\right) + \tau^2 \Sigma_v(t) + \frac{\tau^3}{3} q_{\text{accel}} I_{3 \times 3}$$

Notice that as lookahead $\tau$ increases:
- The uncertainty grows quadratically with velocity variance ($\tau^2 \Sigma_v$) and cubically with process noise ($\tau^3 q_{\text{accel}}$).
- If observations are interrupted (coasting), uncertainty inflates continuously, naturally commanding wider avoidance margins.

### 3. Directional and Isotropic Uncertainty Inflation
To ensure compatibility with fast spherical clearance queries without requiring expensive rotated ellipsoid distance solvers at every A* lattice expansion:
1. **Conservative Isotropic Bound (Primary Planning Bound):**
   The maximum semi-axis of the 3D uncertainty ellipsoid is given by the spectral radius (largest eigenvalue) of the spatial covariance matrix:
   $$\sigma_{\max}(\tau) = \sqrt{\lambda_{\max}(\Sigma_p(t + \tau))}$$
   Since $\Sigma_p$ is symmetric positive semi-definite (SPSD), its maximum eigenvalue is efficiently computed via analytical closed-form or `scipy.linalg.eigvalsh`.

2. **Perception-Aware Effective Obstacle Radius:**
   $$r_{\text{effective}}(t + \tau) = \hat{r}_{\text{obs}} + r_{\text{vehicle}} + \delta_{\text{nominal}} + k_{\sigma} \cdot \sigma_{\max}(\tau)$$
   where:
   - $\hat{r}_{\text{obs}}$: Estimated physical obstacle radius.
   - $r_{\text{vehicle}}$: Bounding sphere radius of quadrotor ($0.25\text{ m}$).
   - $\delta_{\text{nominal}}$: Nominal baseline aerodynamic/control buffer ($0.15\text{ m}$).
   - $k_{\sigma}$: Confidence scaling multiplier derived from Chi-squared distribution $\chi^2_3$.
     For 3D position error $\tilde{\mathbf{p}} \sim \mathcal{N}(\mathbf{0}, \Sigma_p)$, the squared Mahalanobis distance $\epsilon = \tilde{\mathbf{p}}^T \Sigma_p^{-1} \tilde{\mathbf{p}}$ follows a $\chi^2_3$ distribution. The containment probability $P$ inside the ellipsoid $\epsilon \le k_{\sigma}^2$ is:
     $$P(\epsilon \le k_{\sigma}^2) = F_{\chi^2_3}(k_{\sigma}^2) = 2 \Phi(k_{\sigma}) - 1 - \sqrt{\frac{2}{\pi}} k_{\sigma} e^{-k_{\sigma}^2 / 2}$$
     Evaluating this:
     - $k_{\sigma} = 3.0$ yields $F_{\chi^2_3}(9.0) \approx 0.9707$ ($97.07\%$ containment, not $98.9\%$, which is the 2D value $1 - e^{-4.5}$).
     - For $95\%$ containment: $k_{\sigma} = \sqrt{\chi^2_3(0.95)} = \sqrt{7.8147} \approx 2.795$.
     - For $99\%$ containment: $k_{\sigma} = \sqrt{\chi^2_3(0.99)} = \sqrt{11.3449} \approx 3.368$.
     Default configuration adopts $k_{\sigma} = 3.0$ ($97.07\%$ containment) with $k_{\sigma} = 3.37$ ($99.0\%$ containment) available for safety-critical canyon corridors.

3. **Maximum Uncertainty Clamping & Fail-Closed Guard:**
   To prevent runaway inflation in the event of numerical divergence or stale tracks:
   $$\sigma_{\max}(\tau) = \min\left(\sigma_{\max}(\tau), \sigma_{\text{clamp\_ceiling}}\right)$$
   If $\sigma_{\max}(\tau) > \sigma_{\text{abort\_threshold}}$ (e.g. $> 2.5\text{ m}$), the track is flagged as `UNCERTAINTY_EXCEEDED`, forcing the MissionManager into a safe hovering hold or commanding a deliberate sensor-clearing maneuver.

### 4. Rigorous Covariance Anomaly Taxonomy & Handling Policy
Covariance matrices must remain symmetric positive semi-definite (SPSD) to represent mathematically valid uncertainty. Numerical issues must not be treated as a single monolith, and Joseph stabilization must not be viewed as an automatic repair mechanism for already corrupted matrices.

#### 4.1 Mixed-Unit Dimensional Normalization
The full 6D tracking state vector combines spatial position and linear velocity:
$$\mathbf{x} = \begin{bmatrix} \mathbf{p} \\ \mathbf{v} \end{bmatrix} = \begin{bmatrix} x & y & z & v_x & v_y & v_z \end{bmatrix}^T \in \mathbb{R}^6$$
where position coordinates have physical units of meters ($\text{m}$), and velocity components have physical units of meters per second ($\text{m/s}$).

Consequently, the state covariance matrix $P \in \mathbb{R}^{6 \times 6}$ is partitioned as:
$$P = \begin{bmatrix} P_{pp} & P_{pv} \\ P_{pv}^T & P_{vv} \end{bmatrix}$$
where:
- Spatial position covariance $P_{pp} \in \mathbb{R}^{3 \times 3}$ has units of $\text{m}^2$.
- Velocity covariance $P_{vv} \in \mathbb{R}^{3 \times 3}$ has units of $(\text{m/s})^2 = \text{m}^2/\text{s}^2$.
- Cross-covariance $P_{pv} \in \mathbb{R}^{3 \times 3}$ has units of $\text{m}^2/\text{s}$.

> [!WARNING]
> Applying an unscaled absolute eigenvalue threshold (e.g., $\lambda_{\min} < -10^{-7}$) directly to the mixed-unit matrix $P$ is dimensionally inconsistent because an unscaled eigenvalue problem $\det(P - \lambda I) = 0$ equates $\text{m}^2$ with $(\text{m/s})^2$. An eigenvalue check that passes in one unit system may fail if units are changed (e.g. from meters to millimeters).

To ensure scale invariance and dimensional consistency, all numerical health checks (symmetry, positive semi-definiteness, and condition number) are evaluated in **Dimensionless Coordinates**:

Let $D \in \mathbb{R}^{6 \times 6}$ denote the diagonal scale normalization matrix:
$$D = \operatorname{diag}\left( L_{\text{char}}, L_{\text{char}}, L_{\text{char}}, \; V_{\text{char}}, V_{\text{char}}, V_{\text{char}} \right)$$
where characteristic physical scales are derived from verified subsystem parameters:
- $L_{\text{char}} = 10.0\text{ m}$: Characteristic spatial scale, derived from maximum LiDAR sensor range ($d_{\text{max\_range}} = 10.0\text{ m}$).
- $V_{\text{char}} = 5.0\text{ m/s}$: Characteristic velocity scale, derived from maximum vehicle operational cruise speed ($v_{\max} = 5.0\text{ m/s}$ in `DroneDynamics6DOF`).

The dimensionless state vector $\mathbf{z}$ and dimensionless covariance matrix $P_z$ are defined as:
$$\mathbf{z} = D^{-1} \mathbf{x}, \quad P_z = D^{-1} P D^{-T} = D^{-1} P D^{-1} \in \mathbb{R}^{6 \times 6}$$
Because $P_z$ is unitless, absolute tolerances $\epsilon_{\text{sym}} = 10^{-7}$ and $\epsilon_{\text{eigen}} = 10^{-7}$ applied to $P_z$ are mathematically rigorous, scale-invariant, and dimensionless.

For 3D measurement covariance $R \in \mathbb{R}^{3 \times 3}$ (pure position units $\text{m}^2$), the dimensionless measurement covariance is:
$$R_z = \frac{1}{L_{\text{char}}^2} R$$

#### 4.2 Authoritative Covariance Anomaly Taxonomy
The perception subsystem establishes a deterministic taxonomy covering 11 mutually exclusive numerical and structural anomaly classes:

| Anomaly Class | Quantitative Detection Criterion | Algorithmic Response | Telemetry & Lifecycle Action |
| :--- | :--- | :--- | :--- |
| **1. Wrong Shape** | `P.shape != (6, 6)` or `R.shape != (3, 3)` | Reject matrix immediately; fatal interface contract violation. | `covariance_health: "shape_error"`. Track invalidated and purged. |
| **2. Non-Finite Values** | `np.isnan(P).any()` or `np.isinf(P).any()` or `np.isnan(R).any()` | Reject update; arithmetic breakdown cannot be repaired. | `covariance_health: "non_finite"`. Track invalidated and purged. |
| **3. Excessive Asymmetry** | $\frac{\|P_z - P_z^T\|_{\infty}}{\max(\|P_z\|_{\infty}, 10^{-12})} > \epsilon_{\text{sym}} = 10^{-7}$ | Matrix has suffered catastrophic numerical corruption. Reject update. | `covariance_health: "excessive_asymmetry"`. Track invalidated. |
| **4. Small Numerical Asymmetry** | $\frac{\|P_z - P_z^T\|_{\infty}}{\max(\|P_z\|_{\infty}, 10^{-12})} \le \epsilon_{\text{sym}} = 10^{-7}$ | Floating-point truncation error. Enforce exact symmetry: $P_z \leftarrow \frac{1}{2}(P_z + P_z^T)$. | `covariance_health: "symmetrized"`. Track remains nominal. |
| **5. Valid Positive Semi-Definite** | $\lambda_{\min}(P_z) \ge 0$ | Nominal condition. Matrix represents physically valid dispersion. | `covariance_health: "valid_psd"`. Nominal execution. |
| **6. Tiny Negative Eigenvalues** | $-10^{-7} \le \lambda_{\min}(P_z) < 0$ | Eigen-decomposition $P_z = V \Lambda V^T$; clamp $\lambda_i \leftarrow \max(\lambda_i, 10^{-8})$; reconstruct $P_z \leftarrow V \Lambda_{\text{clamped}} V^T$. | `covariance_health: "eigen_clamped"`. Log diagnostic event. |
| **7. Materially Negative Eigenvalues** | $\lambda_{\min}(P_z) < -10^{-7}$ | **Fatal filter divergence.** Matrix is non-physical and cannot be rescued by clamping or Joseph stabilization. | `covariance_health: "divergent_invalidated"`. Track state set to `DELETED`; purged from world model. |
| **8. Invalid Measurement Covariance** | $\min(\operatorname{eig}(R_z)) \le 0$ or $\det(R_z) \le 0$ | Sensor measurement noise is non-positive. Reject measurement; do NOT execute Kalman update; coast track forward with motion model. | `covariance_health: "invalid_R"`. Track coasts without update. |
| **9. Ill-Conditioned / Singular Covariance** | $\kappa(P_z) = \frac{\lambda_{\max}(P_z)}{\max(\lambda_{\min}(P_z), 10^{-15})} > 10^{12}$ | Extreme uncertainty ratio causing numerical failure during matrix inversion of $S = H P H^T + R$. Regularize: $P_z \leftarrow P_z + 10^{-6} I$. | `covariance_health: "ill_conditioned_regularized"`. Log condition warning. |
| **10. Invalid State / Measurement Values** | $\|\mathbf{p}\| > d_{\text{arena\_max}}$ or $\|\mathbf{v}\| > 20\text{ m/s}$ or non-finite centroid | Physical plausibility bounds violated. Reject candidate measurement update. | `covariance_health: "state_out_of_bounds"`. Measurement dropped. |
| **11. Failed Innovation Inversion** | Cholesky factorization of $S = H P H^T + R$ fails (`LinAlgError`) | Numerical singularity during Kalman gain computation. Reject measurement; coast track with motion model. | `covariance_health: "inversion_failed"`. Track coasts. |

#### 4.3 Deterministic Order of Operations
To prevent race conditions and ensure predictable numerical hygiene, every Kalman filter prediction and correction cycle MUST execute checks and transformations in the following strict sequential order:

1. **Finite Value Verification:** Check `np.isfinite(P).all()` and `np.isfinite(R).all()`. If non-finite $\to$ Class 2 (fatal purge).
2. **Dimension & Shape Verification:** Validate array dimensions. If mismatch $\to$ Class 1 (fatal contract defect).
3. **Measurement Covariance Validation ($R$):** Check $R$ for positive definiteness $\min(\operatorname{eig}(R)) > 0$. If invalid $\to$ Class 8 (reject measurement; coast track).
4. **Dimensionless Coordinate Transformation:** Map physical covariance to dimensionless form:
   $$P_z = D^{-1} P D^{-1}$$
5. **Symmetry Assessment & Symmetrization:** Compute relative asymmetry $\Delta_{\text{sym}} = \|P_z - P_z^T\|_{\infty} / \max(\|P_z\|_{\infty}, 10^{-12})$.
   - If $\Delta_{\text{sym}} > 10^{-7} \implies$ Class 3 (fatal asymmetry; invalidate track).
   - If $\Delta_{\text{sym}} \le 10^{-7} \implies$ Class 4: enforce $P_z \leftarrow \frac{1}{2}(P_z + P_z^T)$.
6. **Spectral Decomposition & Eigenvalue Clamping:**
   Compute eigenvalues $\lambda_i$ and eigenvectors $V$ using `scipy.linalg.eigh(P_z)`.
   - If $\lambda_{\min} < -10^{-7} \implies$ Class 7: **fatal filter divergence**. Set track status to `DELETED`; purge from world model.
   - If $-10^{-7} \le \lambda_{\min} < 0 \implies$ Class 6: clamp $\lambda_i \leftarrow \max(\lambda_i, 10^{-8})$ and reconstruct $P_z \leftarrow V \Lambda_{\text{clamped}} V^T$.
   - If $\lambda_{\min} \ge 0 \implies$ Class 5: valid PSD.
7. **Condition Number Verification:** Compute $\kappa(P_z) = \lambda_{\max} / \lambda_{\min}$. If $\kappa(P_z) > 10^{12} \implies$ Class 9: regularize with $10^{-6} I$.
8. **Physical Coordinate Reconstruction:** Transform back to physical units:
   $$P \leftarrow D P_z D$$
9. **Joseph-Stabilized Measurement Update:**
   When incorporating measurement $R$:
   $$P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R_k K_k^T$$
   The Joseph formulation guarantees positive semi-definiteness algebraically **if and only if** prior $P_{k|k-1}$ and measurement noise $R_k$ are valid SPSD matrices. It prevents round-off subtraction degradation that occurs in the naive form $P = (I - KH)P$. It cannot fix non-finite inputs, invalid measurement covariances, or diverged matrices.
10. **Telemetry Emission:** Log `covariance_health` event and eigenvalue spectrum at every execution step.

## Alternatives Considered
1. **Exact Ellipsoid-to-Segment Distance Checking in 4D Lattice:**
   Evaluating oriented bounding ellipsoids $(p - c)^T \Sigma^{-1} (p - c) \le k_{\sigma}^2$ along swept line segments.
   *Rejected:* Requires solving iterative root-finding (polynomial degree 6) for every spatial edge expansion. In Phase IV, lattice expansions exceed 15,000 steps; iterative ellipsoid checks would cause planning latency to jump from 3 seconds to $> 60$ seconds, violating real-time execution constraints.
2. **Fixed Constant Heuristic Padding:**
   Increasing fixed $\delta_{\text{margin}}$ from $0.2\text{ m}$ to $1.0\text{ m}$ without computing covariance.
   *Rejected:* Completely uncalibrated; unnecessarily blocks narrow passages when tracking confidence is high, while remaining vulnerable when velocity uncertainty is high at long horizons.

## Technical Reasoning
- **Mathematical Soundness:** Using the spectral radius $\sqrt{\lambda_{\max}(\Sigma)}$ guarantees an outer bounding sphere that strictly encloses the true 3D confidence ellipsoid in all directions:
  $$\forall \mathbf{u} \in \mathbb{R}^3, \|\mathbf{u}\| = 1 \implies \mathbf{u}^T \Sigma_p \mathbf{u} \le \lambda_{\max}(\Sigma_p)$$
- **Computational Efficiency:** Reduces covariance-aware collision checking to a standard spherical clearance test, preserving all existing analytical $O(1)$ pre-filtering and swept-segment optimizations in `TimeAwareAStarPlanner3D`.

## Consequences
- **Positive:**
  - Forward-planned routes dynamically shy away from moving obstacles with high velocity uncertainty.
  - Missed detections immediately inflate safety margins, naturally inducing conservative behavior.
  - Zero performance degradation in 4D lattice search complexity.
- **Negative:**
  - An isotropic bounding sphere is conservative along minor uncertainty axes (e.g., if uncertainty is high only along the depth axis).
- **Neutral:**
  - Requires empirical validation in Phase V.4 and V.6 to calibrate process noise $q_{\text{accel}}$ so that the confidence ellipsoid matches actual prediction errors.

## Known Limitations
- The Gaussian assumption holds well for sensor measurement noise and unperturbed flight, but is approximate when an obstacle reflects from a wall (non-Gaussian multi-modal distribution).

## Future Reconsideration Criteria
- If narrow canyon navigation requires tighter bounds along minor uncertainty axes, oriented ellipsoid swept checks will be evaluated selectively on the final trajectory post-smoothing pass.
