# ADR-007: Perception-Aware World Model & Planner Integration Strategy

## Status
ACCEPTED (Architecture Specification for Phase V)

## Context
In Phase III and Phase IV, `AStarPlanner3D` and `TimeAwareAStarPlanner3D` were coupled directly to lists of static and dynamic obstacle objects:
```python
def plan(
    self,
    start_position: np.ndarray,
    goal_position: np.ndarray,
    static_obstacles: Sequence[ObstacleSphere3D],
    dynamic_obstacles: Optional[Sequence[DynamicObstacleSphere3D]] = None,
    t_start: float = 0.0,
    ...
)
```
If Phase V were to simply pass raw point clouds or tracking objects directly into `plan(...)`, it would tightly couple the path planner to perception data structures, breaking existing planner unit tests, benchmark harnesses, and regression baselines.

Furthermore, dynamic path planning requires spatiotemporal queries:
- What is the clearance at point $\mathbf{p}$ at future time $t$?
- Is the transition between $(\mathbf{p}_1, t_1)$ and $(\mathbf{p}_2, t_2)$ collision-free under uncertainty?
- Is candidate space free, occupied, or unobserved (unknown)?

> [!IMPORTANT]
> The interfaces and data structures defined in this document are **ARCHITECTURAL SPECIFICATIONS** for implementation in Phase V.5 (`adaptive_rl.perception.world_model`). They are not present in the current Phase V.0 codebase.

## Decision
We define an abstract, decoupled **World Model Interface** (`WorldModelInterface`) in `adaptive_rl.perception.world_model` and refactor the planners to query this interface rather than iterating over raw obstacle collections.

### 1. Abstract World Model Contract (`WorldModelInterface`)
```python
class OccupancyState(Enum):
    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


class WorldModelInterface(ABC):
    """Abstract spatial and spatiotemporal environment query interface for planners."""

    @abstractmethod
    def query_clearance(
        self,
        position: np.ndarray,
        time_sec: float,
        vehicle_radius: float = 0.25,
        confidence_k: float = 3.0,
    ) -> float:
        """Compute signed minimum distance to nearest obstacle boundary at time_sec.

        Returns negative value if inside obstacle, positive if clear.
        confidence_k: Mahalanobis radius multiplier (k=3.0 yields 97.07% containment under chi^2_3).
        Incorporates covariance-based uncertainty inflation.
        """
        pass

    @abstractmethod
    def is_transition_valid(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        t_start: float,
        t_end: float,
        vehicle_radius: float = 0.25,
        confidence_k: float = 3.0,
    ) -> Tuple[bool, float]:
        """Continuous swept clearance check across space-time segment.

        confidence_k: Mahalanobis radius multiplier (k=3.0 yields 97.07% containment under chi^2_3).
        Returns:
            (is_valid, min_clearance_along_segment)
        """
        pass

    @abstractmethod
    def query_occupancy(
        self,
        position: np.ndarray,
        time_sec: float,
    ) -> OccupancyState:
        """Classify space as FREE, OCCUPIED, or UNKNOWN."""
        pass

    @abstractmethod
    def get_active_tracks(
        self,
        time_sec: float,
    ) -> Sequence[TrackedObstacle3D]:
        """Retrieve confirmed tracked obstacles forward-projected to time_sec."""
        pass
```

### 2. Concrete World Model Implementations
To preserve complete backward compatibility and enable direct head-to-head benchmarking:
1. **`PerceptionWorldModel3D` (Phase V Production Model):**
   - Fed continuously by `DynamicObstacleTracker3D` and static background map.
   - Evaluates dynamic obstacle clearance using tracker state estimates $[\hat{\mathbf{p}}, \hat{\mathbf{v}}]$ and covariance inflation $\sigma_{\max}(\tau)$.
   - Handles track coasting, uncertainty growth, and stale-track pruning.
   - Computes occupancy states (including ray-cast free space vs unobserved voxels).
2. **`GroundTruthWorldModelAdapter` (Phase III/IV Compatibility Adapter):**
   - Implements `WorldModelInterface` wrapping existing `Sequence[ObstacleSphere3D]` and `Sequence[DynamicObstacleSphere3D]`.
   - Uses zero covariance ($\sigma=0$) and exact mathematical equations.
   - Allows all existing Phase III and Phase IV unit tests to run unmodified by passing this adapter to the planners.

### 3. Planner Integration
The signature of `TimeAwareAStarPlanner3D.plan(...)` and `AStarPlanner3D.plan(...)` will be updated to accept `world_model: WorldModelInterface`.
For backward compatibility during the transition:
```python
def plan(
    self,
    start_position: np.ndarray,
    goal_position: np.ndarray,
    world_model: Optional[WorldModelInterface] = None,
    static_obstacles: Optional[Sequence[ObstacleSphere3D]] = None,
    dynamic_obstacles: Optional[Sequence[DynamicObstacleSphere3D]] = None,
    ...
):
    if world_model is None:
        # Wrap legacy static and dynamic obstacles automatically
        world_model = GroundTruthWorldModelAdapter(static_obstacles, dynamic_obstacles)
    ...
```
This guarantees that zero existing tests or scenarios will break when this interface is introduced.

## Alternatives Considered
1. **Directly Modify Planners to Read Tracker Output:**
   *Rejected:* Binds the planner directly to Kalman filter internal structures, preventing alternative mapping representations (e.g. OctoMaps, elevation maps, or neural occupancy fields).
2. **Re-implementing A\* inside the Perception Package:**
   *Rejected:* Duplicates complex 4D graph expansion logic, violating Rule 3 (reuse existing architecture) and Rule 5 (do not rewrite working Phase III or Phase IV subsystems).

## Technical Reasoning
- **Polymorphism via Interface:** Adheres to the Dependency Inversion Principle (High-level planning modules depend on abstractions, not on perception details).
- **Zero Regression Risk:** By providing `GroundTruthWorldModelAdapter`, the entire Phase IV test suite can continue running with zero behavior changes.
- **Unified Swept Clearance:** Both static obstacles and moving tracks are queried through a single, unified method (`is_transition_valid`), simplifying graph search logic.

## Consequences
- **Positive:**
  - Planners remain 100% agnostic to whether obstacles are ground truth, simulated LiDAR, or real sensor tracks.
  - Enables clean A/B comparative testing between ground-truth planning and perception-aware planning under identical scenarios.
- **Negative:**
  - One additional layer of method indirection during collision checks. (Negligible in Python given that collision checks are already dominated by NumPy operations).
- **Neutral:**
  - Requires implementing both `PerceptionWorldModel3D` and `GroundTruthWorldModelAdapter` in Phase V.5.

## Known Limitations
- Continuous swept clearance between moving covariance ellipsoids is approximated using the isotropic bounding sphere model (ADR-005) to maintain real-time performance.

## Future Reconsideration Criteria
- If moving to GPU-accelerated collision checking (e.g. parallel batch ray tracing), the `WorldModelInterface` query methods will be extended to accept batch query tensors.
