# ADR-001: Perception Architecture and Module Boundaries

## Status
ACCEPTED (Architecture Specification for Phase V)

## Context
In ARL Phases I through IV, autonomous 3D navigation and trajectory planning operated using ground-truth geometric representations of obstacles. Specifically:
- `AStarPlanner3D` and `TimeAwareAStarPlanner3D` received lists of `ObstacleSphere3D` and `DynamicObstacleSphere3D` instances directly from the simulation scenario.
- In Phase IV.1, sensor-in-the-loop state estimation was introduced for quadrotor ego-motion (`IMUSensor` and `ComplementaryFilter`), but external obstacle avoidance remained completely reliant on direct simulator state queries (`scenario.obstacles` and `scenario.dynamic_obstacles`).

In real-world robotic flight, true obstacle geometry, positions, and velocities are never accessible. The aircraft perceives its environment exclusively through onboard noisy, range-limited, field-of-view-constrained, and potentially occluded exteroceptive sensors (such as LiDAR or depth cameras). To bridge this critical sim-to-real gap, Phase V must establish an autonomous perception subsystem while preserving the stability, modularity, and testability of existing Phase III and IV navigation planners.

## Decision
We establish a dedicated, decoupled perception subsystem package (`adaptive_rl.perception`) with strictly defined interfaces and unidirectional data flow:

```
[Simulation Environment (Drone / Physics)]
          |
          |  (Hidden True State - Ground Truth Isolation)
          v
  [Sensor Simulation] (LiDAR / Depth Camera)
          |
          |  (Raw Sensor Data: Range Measurements / Point Clouds)
          v
  [Perception Subsystem (adaptive_rl.perception)]
          ├── Detection Layer       (Segmentation & Centroid Estimation)
          ├── Tracking Layer        (Data Association & Kalman Filtering)
          └── World Model Layer     (Dynamic Obstacle & Map Representation)
          |
          |  (Estimated Obstacles + Covariances + Occupancy)
          v
  [Navigation Subsystem (adaptive_rl.navigation)]
          ├── MissionManager
          └── TimeAwareAStarPlanner3D / AStarPlanner3D
```

The module boundaries within `adaptive_rl.perception` are defined as:
1. `adaptive_rl.perception.sensors`: Defines abstract sensor interfaces (`BaseSensor3D`, `SensorConfig3D`) and concrete sensor simulators (`SimulatedLiDAR3D`, `SimulatedDepthCamera3D`).
2. `adaptive_rl.perception.detection`: Consumes raw sensor outputs and generates instantaneous candidate detections (`ObstacleDetection3D`), performing spatial clustering and spatial bounding without temporal persistence.
3. `adaptive_rl.perception.tracking`: Consumes detections over time, associates them with existing tracks, maintains track lifecycles (tentative, confirmed, coasted, deleted), and updates state and velocity estimates via filtering (`DynamicObstacleTracker3D`).
4. `adaptive_rl.perception.world_model`: Maintains the estimated environment state (`PerceptionWorldModel3D`), providing querying APIs for static obstacles, dynamic tracks, uncertainty-inflated collision boundaries, and occupancy status.

Planners in `adaptive_rl.navigation` will interact strictly with `PerceptionWorldModel3D` (or an abstract `WorldModelInterface`), completely isolated from both raw sensor streams and simulation ground-truth structures.

## Alternatives Considered
1. **Monolithic Planner-Perception Coupling:**
   Embedding ray casting, point clustering, and tracking directly inside `TimeAwareAStarPlanner3D` or `MissionManager`.
   *Rejected:* Violates Single Responsibility Principle; makes planners untestable without sensor simulation; prevents benchmarking alternative tracking or planning algorithms independently.
2. **Direct Point-Cloud Path Planning (e.g., OctoMap / Voxel Grid A*):**
   Expanding the 4D graph directly over raw voxels without detecting or tracking distinct dynamic obstacles.
   *Rejected:* 4D time-aware planning requires forward velocity prediction of moving intruders over lookahead horizons ($[t, t + H]$). Pure static occupancy mapping cannot anticipate dynamic obstacle motion, leading to severe corridor freeze or unavoidable collisions.
3. **End-to-End Reinforcement Learning from Raw Sensor Observations:**
   Passing raw range vectors directly into policy networks.
   *Rejected:* Already explored in Phase I/II (`compute_lidar_3d_readings` into SB3 PPO/SAC); lacks formal clearance guarantees, fails to generalize to complex multi-waypoint receding-horizon missions, and conflates perception errors with control failures.

## Technical Reasoning
- **Separation of Concerns:** Modular boundaries allow sensor noise models, detector clustering heuristics, tracker association algorithms, and trajectory planners to be developed, unit-tested, and benchmarked independently.
- **Interchangeability:** Allows swapping a 16-ray LiDAR simulator for a 3D depth camera simulator without altering a single line of planning logic.
- **Graceful Degradation:** A clean boundary allows introducing perception degradation (dropouts, false positives, latency) without destabilizing low-level flight control or mission management.

## Consequences
- **Positive:**
  - Planners are decoupled from sensor implementations.
  - Clear architectural firewall preventing simulator ground-truth leakage.
  - Independent unit and integration testing across perception pipeline stages.
- **Negative:**
  - Introduces data transformation overhead (Sensor $\to$ Detections $\to$ Tracks $\to$ World Model $\to$ Planner).
  - Requires maintaining formal data contracts across 4 internal submodules.
- **Neutral:**
  - Requires existing `MissionManager` to support accepting `PerceptionWorldModel3D` in Phase V.5 while preserving backwards compatibility for ground-truth inputs during regression testing.

## Known Limitations
- High-fidelity point cloud simulation at scale in pure Python can incur performance bottlenecks if ray-casting resolution is set excessively high. Sensor simulation must be vectorized using NumPy.
- Spatial clustering of complex overlapping geometries remains challenging without dense point clouds.

## Future Reconsideration Criteria
- If moving to GPU-accelerated simulation (e.g., Isaac Gym or PyBullet) or hardware-in-the-loop (HITL) flight testing, perception components may be adapted to consume ROS 2 / DDS sensor messages.
