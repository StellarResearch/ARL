# ADR-006: Information-Flow Boundary and Ground-Truth Isolation

## Status
ACCEPTED (Architecture Specification for Phase V)

## Context
In reinforcement learning and robotics research, "ground-truth leakage" is a pervasive failure mode where an autonomous agent silently consumes privileged simulation state (such as exact obstacle coordinates, true velocities, or unobservable ground-truth poses) while claiming to operate purely from onboard sensors.

In Phase IV.1, ARL took an initial step toward state-estimation attribution by tagging telemetry with `state_source: "estimated" | "ground_truth"` and failing closed if unnormalized quaternions or NaNs entered closed-loop control (`test_sensor_in_the_loop_navigation.py`).
However, as audited in Section 3:
- In `MissionManager.start_mission(...)` and `MissionManager.update(...)`, the caller passes `static_obstacles: Sequence[ObstacleSphere3D]` and `dynamic_obstacles: Sequence[DynamicObstacleSphere3D]` directly from the simulator scenario definition (`scenario.obstacles` and `scenario.dynamic_obstacles`).
- The planner currently relies on these ground-truth objects for collision avoidance.

To achieve genuine scientific rigor in Phase V, a strict architectural firewall must prohibit the planner, world model, tracker, and flight controller from accessing simulator ground truth.

## Decision
We establish an **enforced information-flow boundary** separating simulation truth from autonomous avionics estimates.

```
+========================================================================+
|                       SIMULATION TRUTH DOMAIN                          |
|                                                                        |
|  [Drone6DOFEnv / DroneDynamics6DOF]                                    |
|         | True State [p, v, q, omega]                                  |
|         | True Obstacles [ObstacleSphere3D, DynamicObstacleSphere3D]   |
+---------+------------------------------------+-------------------------+
          |                                    |
          | Physical Interaction               | Privileged Ground Truth
          v                                    v
+-----------------------+            +===================================+
|   SENSOR SIMULATION   |            |        EVALUATION SUBSYSTEM       |
|                       |            |                                   |
| - IMUSensor           |            | [FlightScenarioRunner / Metrics]  |
| - SimulatedLiDAR3D    |            |  - Detection Precision & Recall   |
| - Noise & Latency     |            |  - Tracking RMSE (Pos & Vel)      |
+-----------+-----------+            |  - True Minimum Clearance         |
            |                        |  - Ground Truth Collision Count   |
            | Sensor Data Stream     |  - State Estimation Error         |
            | (Points, Ranges, IMU)  |  - Cryptographic Digests          |
            v                        +===================================+
+========================================================================+
|                      AUTONOMOUS AVIONICS DOMAIN                        |
|                     (STRICTLY NO GROUND TRUTH)                         |
|                                                                        |
|  [ComplementaryFilter]     --> Estimated Drone Pose                    |
|  [ObstacleDetector3D]      --> Candidate Detections (Clusters)         |
|  [DynamicObstacleTracker3D]--> Persistent Tracks (p_est, v_est, Cov)   |
|  [PerceptionWorldModel3D]  --> Estimated World & Clearance Queries     |
|  [TimeAwareAStarPlanner3D] --> Trajectory Waypoints                    |
|  [MissionManager]          --> Navigation State Machine                |
|  [FlightController (LQR)]  --> Motor Commands                          |
+========================================================================+
```

### 1. The Architectural Firewall Rules
1. **Rule of Sensor Provenance:**
   The perception layer (`ObstacleDetector3D`, `DynamicObstacleTracker3D`) receives inputs **only** from simulated sensors (`PointCloud3D`, `IMUMeasurement`) and estimated vehicle poses (`EstimatedDroneState`). It has no reference to `scenario.obstacles` or `scenario.dynamic_obstacles`.
2. **Rule of Planner Isolation:**
   The planning layer (`MissionManager`, `TimeAwareAStarPlanner3D`, `AStarPlanner3D`) receives obstacle information **exclusively** through `PerceptionWorldModel3D`. Passing `ObstacleSphere3D` or `DynamicObstacleSphere3D` directly to `MissionManager` in perception-aware mode will raise a runtime `TypeError` or `SecurityViolationError`.
3. **Rule of Evaluation-Only Ground Truth:**
   Ground truth from the simulator is delivered **only** to the offline evaluation harness (`FlightScenarioRunner`, `FlightStepTelemetry`, `EvaluationMetrics`). The harness uses ground truth solely to calculate benchmarking metrics (e.g., localization error $\| \hat{\mathbf{p}} - \mathbf{p}_{\text{true}} \|$, detection precision/recall, and true physical clearance).
4. **Zero Backward Propagation:**
   No evaluation metric or ground-truth comparison may be fed back into the perception, tracking, planning, or control pipelines.

### 2. Runtime Integrity Verification
To guarantee adherence to this firewall in automated tests and benchmarks:
- `PerceptionWorldModel3D` will implement an inspection method `is_ground_truth_isolated() -> bool` verifying that none of its internal obstacle representations contain instances of `ObstacleSphere3D`, `DynamicObstacleSphere3D`, or simulation handles.
- Telemetry records will mandate strict provenance tracking: `perception_source: "estimated_perception" | "ground_truth_bypass"`. Any benchmark run claiming Phase V perception autonomy that has `ground_truth_bypass` will be automatically rejected by the benchmarking suite.

## Alternatives Considered
1. **Passing Ground Truth as an "Optional Fallback":**
   Allowing the planner to read ground truth if the perception tracker fails.
   *Rejected:* Defeats the scientific purpose; introduces silent fallback bugs where tracking failures are hidden by ground-truth substitution.
2. **Global State Singleton:**
   Maintaining an environment object accessible anywhere in memory.
   *Rejected:* Violates encapsulation, causes untraceable state leakage, and prevents running multi-agent or parallel head-to-head simulations.

## Technical Reasoning
- **Scientific Defensibility:** Autonomy claims are valid only when the vehicle makes decisions solely on what it measures.
- **Sim-to-Real Readiness:** A controller or planner that depends on hidden simulation state cannot transition to physical quadrotors, where no simulation object exists.
- **Honest Failure Modes:** When a sensor is blinded or an obstacle is occluded, the system must either detect the uncertainty or experience the failure, allowing safety logic (e.g. fail-safe hover hold) to be properly exercised and validated.

## Consequences
- **Positive:**
  - Absolute guarantee that perception-aware benchmarks measure genuine algorithmic performance.
  - Transparent error attribution: sensor error vs tracking error vs planning error vs control error.
- **Negative:**
  - Requires writing adapter wrappers for existing Phase III/IV scenarios that currently pass ground-truth obstacle sequences into `MissionManager`.
- **Neutral:**
  - Ground-truth inputs will remain supported under an explicit, legacy test flag (`ground_truth_baseline=True`) solely to preserve regression tests for Phase III/IV algorithms.

## Known Limitations
- In a software simulator running in a single Python process, memory isolation requires architectural discipline and test assertions, as Python lacks compile-time memory sandboxing.

## Future Reconsideration Criteria
- If moving to multi-process architecture (e.g. microservices or ROS 2 nodes), process boundaries and message channels will enforce this isolation at the OS transport layer.
