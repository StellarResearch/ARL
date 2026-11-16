"""ARL Perception Subsystem.

Architecture specification and implementation for Phase V:
- sensors: Modular exteroceptive sensors (LiDAR, Depth camera) and observation contracts
- detection: 3D point cloud segmentation, clustering, and candidate detection (Phase V.2)
- tracking: Dynamic obstacle tracking, data association, and state estimation (Phase V.3)
- world_model: Perception world model and covariance-inflated clearance queries (Phase V.5)
"""

from adaptive_rl.perception.sensors import (
    BaseSensor3D,
    PointCloud3D,
    SensorConfig3D,
    SensorExtrinsics,
    SimulatedLiDAR3D,
)

__all__ = [
    "BaseSensor3D",
    "PointCloud3D",
    "SensorConfig3D",
    "SensorExtrinsics",
    "SimulatedLiDAR3D",
]
