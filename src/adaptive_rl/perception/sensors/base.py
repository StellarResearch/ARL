"""Abstract base class for exteroceptive 3D range sensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.perception.sensors.data import PointCloud3D, SensorConfig3D, SensorExtrinsics


class BaseSensor3D(ABC):
    """Abstract base interface for simulated 3D sensors.

    Follows ADR-001 and ADR-002, isolating sensor simulation from downstream
    perception and planning modules.
    """

    def __init__(self, config: Optional[SensorConfig3D] = None) -> None:
        self.config = config if config is not None else SensorConfig3D()

    @property
    def sensor_id(self) -> str:
        """Unique identifier of this sensor instance."""
        return self.config.sensor_id

    @property
    def extrinsics(self) -> SensorExtrinsics:
        """Extrinsic mounting parameters relative to vehicle body frame."""
        assert self.config.extrinsics is not None
        return self.config.extrinsics

    @abstractmethod
    def capture(
        self,
        timestamp: float,
        drone_pos: np.ndarray | Sequence[float],
        drone_orientation: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        obstacles: Optional[Sequence[Any]] = None,
        bounds: Optional[Tuple[float, float, float]] = None,
    ) -> PointCloud3D:
        """Generate a simulated 3D observation point cloud at the given epoch.

        Args:
            timestamp: Elapsed simulation time in seconds.
            drone_pos: Quadrotor center-of-mass position in world frame (3,).
            drone_orientation: Vehicle attitude quaternion [qw, qx, qy, qz] (4,)
                               or rotation matrix R_B^W (3, 3).
            obstacles: Sequence of obstacles (e.g. ObstacleSphere3D, DynamicObstacleSphere3D).
            bounds: Optional arena bounding dimensions (X_max, Y_max, Z_max).

        Returns:
            Immutable PointCloud3D observation in the sensor frame.
        """
        pass
