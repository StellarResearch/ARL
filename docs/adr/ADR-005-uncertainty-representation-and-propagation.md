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

The perception subsystem establishes a strict anomaly taxonomy and handling hierarchy:

| Anomaly Class | Quantitative Criterion | Algorithmic Response | Telemetry & Lifecycle Action |
| :--- | :--- | :--- | :--- |
| **Wrong Shape** | `P.shape != (6, 6)` or `R.shape != (3, 3)` | Reject matrix immediately; fatal contract bug. | `covariance_health: "shape_error"`. Track invalidated. |
| **Non-Finite Values** | `np.isnan(P).any()` or `np.isinf(P).any()` | Reject update; arithmetic breakdown cannot be repaired. | `covariance_health: "non_finite"`. Track invalidated and purged. |
| **Non-Positive Measurement Noise** | $\min(\text{eig}(R)) \le 0$ or $\det(R) \le 0$ | Reject measurement; do not execute Kalman update. | `covariance_health: "invalid_R"`. Track coasts without update. |
| **Small Numerical Asymmetry** | $\|P - P^T\|_{\infty} \le 10^{-7}$ | Enforce exact symmetry: $P \leftarrow \frac{1}{2}(P + P^T)$. | `covariance_health: "symmetrized"`. Track remains nominal. |
| **Tiny Negative Eigenvalues** | $-10^{-7} \le \lambda_{\min}(P) < 0$ | Eigen-decomposition $P = V \Lambda V^T$; clamp $\lambda_i \leftarrow \max(\lambda_i, 10^{-8})$; reconstruct $P \leftarrow V \Lambda_{\text{clamped}} V^T$. | `covariance_health: "eigen_clamped"`. Log diagnostic event. |
| **Materially Negative Eigenvalues** | $\lambda_{\min}(P) < -10^{-7}$ | **Fatal filter divergence.** Matrix is non-physical and cannot be rescued by clamping or Joseph stabilization. | `covariance_health: "divergent_invalidated"`. Track state set to `DELETED`; purged from world model. |

#### Role of the Joseph Stabilized Formulation
The Joseph formulation:
$$P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R_k K_k^T$$
guarantees positive semi-definiteness algebraically **if and only if** prior covariance $P_{k|k-1}$ and measurement noise $R_k$ are valid SPSD matrices. It prevents round-off subtraction degradation that occurs in the naive form $P = (I - KH)P$. It cannot fix non-finite inputs, invalid measurement covariances, or diverged matrices. Every update step must log `covariance_health` telemetry.

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
