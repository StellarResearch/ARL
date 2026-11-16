# ADR-002: Sensor Observation Data Contract

## Status
ACCEPTED (Architecture Specification for Phase V)

## Context
In ARL, perception begins with exteroceptive sensor observations. Previously, the only exteroceptive range model was `compute_lidar_3d_readings` in `src/adaptive_rl/environments/drone.py`, which produced a 1D normalized range array across 16 fixed ray directions directly from the internal simulation obstacle list.

To support realistic 3D detection, tracking, and mapping, the perception subsystem requires strongly typed, serializable data structures that capture spatial range data, sensor extrinsics, timestamps, noise characteristics, and invalid/dropped measurement flags without relying on heavy external robotics frameworks.

> [!IMPORTANT]
> The data contracts specified in this document are **ARCHITECTURAL SPECIFICATIONS** scheduled for implementation in Phase V.1 (`adaptive_rl.perception.sensors`). They are not present in the current Phase V.0 codebase.

## Decision
We define standard sensor observation contracts for future implementation in `adaptive_rl.perception.sensors`:

```python
@dataclass(frozen=True)
class SensorExtrinsics:
    """Rigid-body pose of the sensor relative to the vehicle body frame (FLU)."""
    translation_b: np.ndarray   # [x, y, z] in meters: sensor origin relative to vehicle CoM in body frame
    rotation_s_to_b: np.ndarray # 3x3 rotation matrix R_S^B (maps sensor vectors to body: v_B = R_S^B * v_S)

@dataclass(frozen=True)
class PointCloud3D:
    """3D point cloud observation produced by an active range sensor."""
    timestamp: float                    # Elapsed simulation time in seconds (measurement epoch)
    frame_id: str                       # Coordinate frame ("sensor", "body", or "world")
    points: np.ndarray                  # (N, 3) float64 array of 3D Cartesian coordinates [x, y, z]
    intensities: Optional[np.ndarray]   # (N,) float32 array in [0, 1] (optional echo strength)
    ranges: Optional[np.ndarray]        # (N,) float32 array of radial distances
    valid_mask: np.ndarray              # (N,) bool array (True = valid echo, False = dropped/out-of-range)
    sensor_id: str
    extrinsics: SensorExtrinsics
    sensor_origin_w: Optional[np.ndarray] = None # Sensor origin in world frame at capture epoch

@dataclass(frozen=True)
class DepthFrame3D:
    """2D depth map observation produced by an optical depth camera."""
    timestamp: float
    frame_id: str
    depth_image: np.ndarray             # (H, W) float32 array in meters
    horizontal_fov_rad: float
    vertical_fov_rad: float
    min_range: float
    max_range: float
    sensor_id: str
    extrinsics: SensorExtrinsics
```

### 1. Data Integrity, Defensive Copying, and Array Immutability
A common Python pitfall is assuming that `@dataclass(frozen=True)` makes NumPy arrays immutable. While `frozen=True` prevents reassigning the field (`point_cloud.points = new_array` raises `FrozenInstanceError`), the underlying NumPy array memory buffer remains mutable (`point_cloud.points[0, 0] = 999.0` will silently alter the data).

To guarantee strict scientific data integrity in Phase V.1:
1. **Read-Only Buffer Enforcement:** Factory constructors will mark array buffers as read-only:
   ```python
   points.flags.writeable = False
   valid_mask.flags.writeable = False
   ```
2. **Defensive Ingestion Copying:** All constructors must ingest arrays via `np.array(arr, copy=True)` or `arr.copy()` to prevent external caller modifications from corrupting stored frames.
3. **Shape and Type Validation:** Preconditions must enforce exact dimensional contracts:
   - `points`: shape `(N, 3)`, dtype `float64`.
   - `ranges`: shape `(N,)`, dtype `float32`.
   - `valid_mask`: shape `(N,)`, dtype `bool`.
   - `rotation_s_to_b`: shape `(3, 3)`, orthonormal $\mathrm{SO}(3)$ check: $\|R R^T - I\| < 10^{-6}, \det(R) = +1$.
4. **Finite Coordinates Enforcement:** All points marked `valid_mask == True` must satisfy `np.all(np.isfinite(points[valid_mask]))`. Invalid returns (NaN/Inf) must be flagged `valid_mask = False` and sanitized.

### 2. Initial Sensor Priority: 3D LiDAR First
We decide to implement the **Simulated 3D LiDAR** first (in Phase V.1), deferring the Depth Camera to Phase V.4:
- **Direct 3D Spatial Geometry:** A 3D LiDAR directly produces sparse Cartesian point clouds $[x, y, z]$ or spherical coordinates $(r, \theta, \phi)$, which can be transformed immediately using linear matrix algebra without requiring camera intrinsic matrix projection, pinhole camera equations, or dense pixel de-projection.
- **Computational Efficiency:** Simulating an array of 64 to 256 rays in pure NumPy/SciPy using vectorized ray-sphere/ray-box intersections is lightweight ($< 5\text{ ms}$ per sweep), maintaining real-time simulation throughput ($> 50\text{ Hz}$ benchmark target against the $20\text{ Hz}$ flight control loop). Simulating a full VGA depth buffer ($640 \times 480 = 307{,}200$ pixels) in Python without GPU rasterization would bottleneck simulation execution.
- **Low Dependency Footprint:** Point-cloud processing can be implemented purely using `numpy` and `scipy.spatial` (KD-trees, spatial distance matrices), adhering strictly to ARL's minimal dependency mandate and avoiding dependencies such as OpenCV or Open3D.

## Alternatives Considered
1. **Depth Camera as Primary Sensor:**
   Simulating structured light or stereo depth frames first.
   *Rejected:* Requires 2D rasterization, image rendering, or ray casting over thousands of pixels. Heavy computational cost in pure Python; requires pinhole projection matrices, lens distortion, and dense depth filtering.
2. **Raw Float Arrays without Metadata:**
   Passing raw `np.ndarray` without timestamp, sensor extrinsics, or validity masks.
   *Rejected:* Causes silent frame-of-reference errors; prevents proper timestamp synchronization; makes dropped beam handling indistinguishable from zero-distance obstacle collisions.
3. **Adopting ROS / ROS 2 Message Types (`sensor_msgs/PointCloud2`):**
   *Rejected:* Introduces heavy external system dependencies (`rclpy`, `ros2`), violating non-negotiable Rule 4 (avoid unnecessary dependencies).

## Technical Reasoning
- **Data Provenance:** Enforcing array read-only flags and defensive copies ensures observation frames cannot be modified downstream by filters or planners, preserving data integrity and avoiding side effects.
- **Validity Masking:** Dropped rays (due to absorption, specular reflection, or out-of-range targets) must report explicit invalidity (`valid_mask = False`) rather than being clamped to `0.0` (which would trigger a false collision) or `max_range` (which masks obstacles).
- **Extrinsic Awareness:** Every frame carries its mounting pose relative to the drone body (`translation_b`, `rotation_s_to_b`), enabling unambiguous transformation to body and world coordinates.

## Consequences
- **Positive:**
  - Standardized interface for all current and future range sensors.
  - Zero external dependencies beyond NumPy and standard library dataclasses.
  - Transparent support for sensor latency and asynchronous update rates.
  - Strict mathematical guarantees against in-place array mutation.
- **Negative:**
  - Defensive array copying incurs minor memory overhead; mitigated by keeping point counts suitable for aerial obstacle avoidance ($N \approx 500 - 2000$).
- **Neutral:**
  - Depth camera support requires a de-projection adapter converting `DepthFrame3D` into `PointCloud3D` for downstream detector consumption.

## Known Limitations
- Current data structures assume global shutter / instantaneous sweep timestamps. True spinning mechanical LiDARs have per-point azimuth timestamps (rolling distortion under vehicle motion); rolling shutter distortion compensation is deferred to Phase V.4.

## Future Reconsideration Criteria
- If moving to camera-based visual navigation or neural radiance fields, dense image frame representations may be prioritized.
