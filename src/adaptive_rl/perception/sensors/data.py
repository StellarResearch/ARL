"""Data contracts and immutable observation representations for 3D range sensors.

Implements ADR-002 (Sensor Observation Data Contract) and ADR-003 (Coordinate-Frame
and Timestamp Conventions) hardened for Phase V.1.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from adaptive_rl.environments.drone_6dof import quaternion_to_rotation_matrix


def validate_so3_rotation(
    rotation: np.ndarray | Sequence[Sequence[float]],
    tolerance_ortho: float = 1e-5,
    tolerance_det: float = 1e-4,
) -> np.ndarray:
    """Validate that a 3x3 matrix is an orthonormal SO(3) proper rotation matrix.

    Adheres to ADR-003 and Phase V.1.1 hardening standards:
    - Shape must be exactly (3, 3).
    - All elements must be finite (no NaN, no Inf).
    - Orthonormality constraint: ||R @ R^T - I_3||_F <= tolerance_ortho (Frobenius norm).
    - Proper rotation constraint: |det(R) - 1.0| <= tolerance_det (rejects reflections where det=-1,
      scaling, shears, or singular matrices).

    Args:
        rotation: 3x3 array-like rotation matrix.
        tolerance_ortho: Maximum permissible Frobenius matrix norm deviation from orthogonality.
        tolerance_det: Maximum permissible deviation of determinant from +1.0.

    Returns:
        np.ndarray: Defensive float64 copy of the validated (3, 3) rotation matrix.

    Raises:
        ValueError: If tolerances are non-finite, non-positive, or boolean; or if shape is not (3, 3),
                    values are non-finite, matrix Frobenius norm ||R @ R^T - I_3||_F > tolerance_ortho,
                    or determinant is not +1.0 within tolerance_det.
    """
    for name, tol in [("tolerance_ortho", tolerance_ortho), ("tolerance_det", tolerance_det)]:
        if (
            isinstance(tol, bool)
            or not isinstance(tol, (int, float))
            or not math.isfinite(tol)
            or tol <= 0.0
        ):
            raise ValueError(f"{name} must be finite and strictly positive, got {tol!r}")

    rot = np.array(rotation, dtype=np.float64, copy=True)
    if rot.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {rot.shape}")
    if not np.all(np.isfinite(rot)):
        raise ValueError(f"rotation matrix must contain finite values, got {rot}")

    # Enforce SO(3) orthonormality using explicit Frobenius norm: ||R @ R.T - I_3||_F <= tolerance_ortho
    identity_err = float(np.linalg.norm(rot @ rot.T - np.eye(3), ord="fro"))
    if identity_err > tolerance_ortho:
        raise ValueError(
            f"rotation matrix must be orthonormal, ||R*R^T - I||_F = {identity_err:.6e} > {tolerance_ortho:.1e}"
        )

    det = float(np.linalg.det(rot))
    if abs(det - 1.0) > tolerance_det:
        raise ValueError(
            f"rotation matrix must have determinant +1.0 (proper rotation), got det = {det:.6f} "
            f"(error {abs(det - 1.0):.6e} > {tolerance_det:.1e})"
        )

    return rot


@dataclass(frozen=True)
class SensorExtrinsics:
    """Rigid-body pose of the sensor relative to the vehicle body frame (FLU).

    Adheres to ADR-002 and ADR-003:
    - translation_b: 3D vector [x, y, z] in meters of sensor origin relative to vehicle CoM in body frame (FLU).
    - rotation_s_to_b: 3x3 orthonormal rotation matrix R_S^B mapping vectors from sensor frame to body frame:
      v_B = R_S^B * v_S.
    """

    translation_b: np.ndarray
    rotation_s_to_b: np.ndarray

    def __init__(
        self,
        translation_b: np.ndarray | Sequence[float],
        rotation_s_to_b: np.ndarray | Sequence[Sequence[float]],
    ) -> None:
        trans = np.array(translation_b, dtype=np.float64, copy=True)
        if trans.shape != (3,):
            raise ValueError(f"translation_b must have shape (3,), got {trans.shape}")
        if not np.all(np.isfinite(trans)):
            raise ValueError(f"translation_b must contain finite values, got {trans}")

        # Shared SO(3) validation
        rot = validate_so3_rotation(rotation_s_to_b)

        # Enforce read-only memory buffers for data integrity
        trans.flags.writeable = False
        rot.flags.writeable = False

        object.__setattr__(self, "translation_b", trans)
        object.__setattr__(self, "rotation_s_to_b", rot)

    @classmethod
    def default_identity(cls) -> SensorExtrinsics:
        """Create default extrinsics centered at vehicle CoM with aligned axes."""
        return cls(
            translation_b=np.zeros(3, dtype=np.float64),
            rotation_s_to_b=np.eye(3, dtype=np.float64),
        )

    def transform_point_to_body(self, point_s: np.ndarray | Sequence[float]) -> np.ndarray:
        """Transform a 3D point from sensor frame to vehicle body frame: p_B = t_B + R_S^B * p_S."""
        pt = np.asarray(point_s, dtype=np.float64)
        if pt.shape != (3,):
            raise ValueError(f"point_s must have shape (3,), got {pt.shape}")
        if not np.all(np.isfinite(pt)):
            raise ValueError(f"point_s must contain finite values, got {pt}")
        return np.asarray(self.translation_b + self.rotation_s_to_b @ pt, dtype=np.float64)

    def transform_vector_to_body(self, vector_s: np.ndarray | Sequence[float]) -> np.ndarray:
        """Transform a 3D direction vector from sensor frame to vehicle body frame: v_B = R_S^B * v_S."""
        vec = np.asarray(vector_s, dtype=np.float64)
        if vec.shape != (3,):
            raise ValueError(f"vector_s must have shape (3,), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"vector_s must contain finite values, got {vec}")
        return np.asarray(self.rotation_s_to_b @ vec, dtype=np.float64)

    def transform_point_to_sensor(self, point_b: np.ndarray | Sequence[float]) -> np.ndarray:
        """Transform a 3D point from vehicle body frame to sensor frame: p_S = (R_S^B)^T * (p_B - t_B)."""
        pt = np.asarray(point_b, dtype=np.float64)
        if pt.shape != (3,):
            raise ValueError(f"point_b must have shape (3,), got {pt.shape}")
        if not np.all(np.isfinite(pt)):
            raise ValueError(f"point_b must contain finite values, got {pt}")
        return np.asarray(self.rotation_s_to_b.T @ (pt - self.translation_b), dtype=np.float64)

    def transform_vector_to_sensor(self, vector_b: np.ndarray | Sequence[float]) -> np.ndarray:
        """Transform a 3D direction vector from body frame to sensor frame: v_S = (R_S^B)^T * v_B."""
        vec = np.asarray(vector_b, dtype=np.float64)
        if vec.shape != (3,):
            raise ValueError(f"vector_b must have shape (3,), got {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError(f"vector_b must contain finite values, got {vec}")
        return np.asarray(self.rotation_s_to_b.T @ vec, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize extrinsics to dictionary."""
        return {
            "translation_b": [round(float(x), 6) for x in self.translation_b],
            "rotation_s_to_b": [[round(float(c), 6) for c in row] for row in self.rotation_s_to_b],
        }


@dataclass(frozen=True)
class SensorConfig3D:
    """Configuration parameters for 3D range-sensing exteroceptive sensors.

    Follows Section 4.2 of PHASE_V_PERCEPTION_ARCHITECTURE.md with strict P1 hardening.
    """

    sensor_id: str = "lidar_primary"
    update_rate_hz: float = 20.0
    min_range_m: float = 0.2
    max_range_m: float = 25.0
    horizontal_fov_deg: float = 360.0  # Azimuth span in degrees
    vertical_fov_deg: float = 60.0  # Elevation span in degrees
    horizontal_beams: int = 32  # Azimuth rays
    vertical_beams: int = 8  # Elevation layers
    range_noise_std_m: float = 0.03  # Zero-mean Gaussian range standard deviation
    dropout_probability: float = 0.02  # Statistical beam loss probability
    latency_sec: float = 0.02  # Configured transport/processing latency budget (deferred delivery)
    extrinsics: Optional[SensorExtrinsics] = None

    def __post_init__(self) -> None:
        # Validate sensor_id
        if not isinstance(self.sensor_id, str) or not self.sensor_id.strip():
            raise ValueError(f"sensor_id must be a non-empty string, got {self.sensor_id!r}")

        # Validate update_rate_hz
        if (
            isinstance(self.update_rate_hz, bool)
            or not isinstance(self.update_rate_hz, (int, float))
            or not math.isfinite(self.update_rate_hz)
            or self.update_rate_hz <= 0.0
        ):
            raise ValueError(
                f"update_rate_hz must be strictly positive and finite, got {self.update_rate_hz!r}"
            )

        # Validate range bounds
        for name, val in [("min_range_m", self.min_range_m), ("max_range_m", self.max_range_m)]:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite float, got {val!r}")
        if self.min_range_m < 0.0 or self.min_range_m >= self.max_range_m:
            raise ValueError(
                f"Invalid range bounds: min_range_m ({self.min_range_m}) must be >= 0.0 and < max_range_m ({self.max_range_m})"
            )

        # Validate FOV spans
        for name, val, upper in [
            ("horizontal_fov_deg", self.horizontal_fov_deg, 360.0),
            ("vertical_fov_deg", self.vertical_fov_deg, 180.0),
        ]:
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not math.isfinite(val)
                or val <= 0.0
                or val > upper
            ):
                raise ValueError(f"{name} must be finite in (0, {upper}], got {val!r}")

        # Validate beam counts (strictly reject booleans, floats, and counts < 1)
        if (
            isinstance(self.horizontal_beams, bool)
            or not isinstance(self.horizontal_beams, int)
            or self.horizontal_beams < 1
        ):
            raise ValueError(
                f"horizontal_beams must be a positive integer >= 1, got {self.horizontal_beams!r}"
            )
        if (
            isinstance(self.vertical_beams, bool)
            or not isinstance(self.vertical_beams, int)
            or self.vertical_beams < 1
        ):
            raise ValueError(
                f"vertical_beams must be a positive integer >= 1, got {self.vertical_beams!r}"
            )

        # Validate range noise std
        if (
            isinstance(self.range_noise_std_m, bool)
            or not isinstance(self.range_noise_std_m, (int, float))
            or not math.isfinite(self.range_noise_std_m)
            or self.range_noise_std_m < 0.0
        ):
            raise ValueError(
                f"range_noise_std_m must be non-negative and finite, got {self.range_noise_std_m!r}"
            )

        # Validate dropout probability
        if (
            isinstance(self.dropout_probability, bool)
            or not isinstance(self.dropout_probability, (int, float))
            or not math.isfinite(self.dropout_probability)
            or not (0.0 <= self.dropout_probability <= 1.0)
        ):
            raise ValueError(
                f"dropout_probability must be in [0.0, 1.0], got {self.dropout_probability!r}"
            )

        # Validate latency_sec
        if (
            isinstance(self.latency_sec, bool)
            or not isinstance(self.latency_sec, (int, float))
            or not math.isfinite(self.latency_sec)
            or self.latency_sec < 0.0
        ):
            raise ValueError(
                f"latency_sec must be non-negative and finite, got {self.latency_sec!r}"
            )

        # Validate extrinsics
        if self.extrinsics is not None and not isinstance(self.extrinsics, SensorExtrinsics):
            raise TypeError(
                f"extrinsics must be an instance of SensorExtrinsics, got {type(self.extrinsics)}"
            )

        if self.extrinsics is None:
            object.__setattr__(self, "extrinsics", SensorExtrinsics.default_identity())

    @property
    def dt(self) -> float:
        """Nominal sensor period in seconds."""
        return 1.0 / self.update_rate_hz

    @property
    def total_beams(self) -> int:
        """Total number of beams per complete sweep."""
        return self.horizontal_beams * self.vertical_beams

    @property
    def horizontal_fov_rad(self) -> float:
        """Horizontal field of view in radians."""
        return math.radians(self.horizontal_fov_deg)

    @property
    def vertical_fov_rad(self) -> float:
        """Vertical field of view in radians."""
        return math.radians(self.vertical_fov_deg)


@dataclass(frozen=True)
class PointCloud3D:
    """Immutable point cloud observation generated by a 3D range sensor.

    Strictly satisfies ADR-002, ADR-003, and ADR-006:
    - Canonical Invalid Representation: All invalid returns (valid_mask=False)
      MUST report [NaN, NaN, NaN] coordinates, NaN range, and 0.0 intensity.
    - Geometric Point-to-Range Consistency: Every valid measurement must satisfy
      ||p_i - o_frame||_2 ~= r_i within documented tolerance (1 mm).
    - Defensive copying on construction.
    - Read-only array memory flags.
    - Zero ground-truth leakage: Contains NO obstacle IDs, simulation objects,
      or true obstacle velocities.
    """

    timestamp: float
    frame_id: str
    points: np.ndarray
    ranges: np.ndarray
    valid_mask: np.ndarray
    sensor_id: str
    extrinsics: SensorExtrinsics
    sensor_origin_w: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None

    def __init__(
        self,
        timestamp: float,
        frame_id: str,
        points: np.ndarray | Sequence[Sequence[float]],
        ranges: np.ndarray | Sequence[float],
        valid_mask: np.ndarray | Sequence[bool],
        sensor_id: str,
        extrinsics: SensorExtrinsics,
        sensor_origin_w: Optional[np.ndarray | Sequence[float]] = None,
        intensities: Optional[np.ndarray | Sequence[float]] = None,
        tolerance_range_m: float = 1e-3,
    ) -> None:
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0.0
        ):
            raise ValueError(f"timestamp must be non-negative and finite, got {timestamp!r}")
        if (
            isinstance(tolerance_range_m, bool)
            or not isinstance(tolerance_range_m, (int, float))
            or not math.isfinite(tolerance_range_m)
            or tolerance_range_m <= 0.0
        ):
            raise ValueError(
                f"tolerance_range_m must be strictly positive and finite, got {tolerance_range_m!r}"
            )
        if frame_id not in ("sensor", "body", "world"):
            raise ValueError(f"frame_id must be one of 'sensor', 'body', 'world', got '{frame_id}'")
        if not isinstance(sensor_id, str) or not sensor_id.strip():
            raise ValueError(f"sensor_id must be a non-empty string, got {sensor_id!r}")
        if not isinstance(extrinsics, SensorExtrinsics):
            raise TypeError(
                f"extrinsics must be an instance of SensorExtrinsics, got {type(extrinsics)}"
            )

        # Ingest and defensively copy arrays
        pts = np.array(points, dtype=np.float64, copy=True)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
        num_points = pts.shape[0]

        rngs = np.array(ranges, dtype=np.float32, copy=True)
        if rngs.shape != (num_points,):
            raise ValueError(
                f"ranges shape {rngs.shape} does not match points length ({num_points},)"
            )

        mask_raw = np.asarray(valid_mask)
        if mask_raw.dtype != bool:
            raise ValueError(
                f"valid_mask must contain strictly boolean values, got dtype {mask_raw.dtype}"
            )
        mask = np.array(mask_raw, dtype=bool, copy=True)
        if mask.shape != (num_points,):
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match points length ({num_points},)"
            )

        # Ingest optional intensities
        ints: Optional[np.ndarray] = None
        if intensities is not None:
            ints = np.array(intensities, dtype=np.float32, copy=True)
            if ints.shape != (num_points,):
                raise ValueError(
                    f"intensities shape {ints.shape} does not match points length ({num_points},)"
                )

        # Ingest optional sensor_origin_w
        orig_w: Optional[np.ndarray] = None
        if sensor_origin_w is not None:
            orig_w = np.array(sensor_origin_w, dtype=np.float64, copy=True)
            if orig_w.shape != (3,):
                raise ValueError(f"sensor_origin_w must have shape (3,), got {orig_w.shape}")
            if not np.all(np.isfinite(orig_w)):
                raise ValueError(f"sensor_origin_w must have finite coordinates, got {orig_w}")

        # --------------------------------------------------------------------
        # Canonical Invalid Return Validation (valid_mask == False)
        # All invalid returns MUST have:
        # - points: NaN coordinates
        # - ranges: NaN
        # - intensities: 0.0 (if present)
        # --------------------------------------------------------------------
        if np.any(~mask):
            invalid_pts = pts[~mask]
            if not np.all(np.isnan(invalid_pts)):
                raise ValueError(
                    "Invalid returns (valid_mask=False) must have NaN coordinates [NaN, NaN, NaN]"
                )
            invalid_rngs = rngs[~mask]
            if not np.all(np.isnan(invalid_rngs)):
                raise ValueError("Invalid returns (valid_mask=False) must have NaN range values")
            if ints is not None:
                invalid_ints = ints[~mask]
                if not np.all(invalid_ints == 0.0):
                    raise ValueError(
                        "Invalid returns (valid_mask=False) must have 0.0 echo intensity"
                    )

        # --------------------------------------------------------------------
        # Valid Return Validation (valid_mask == True)
        # --------------------------------------------------------------------
        if np.any(mask):
            valid_pts = pts[mask]
            if not np.all(np.isfinite(valid_pts)):
                raise ValueError("All points marked valid_mask=True must have finite coordinates")
            valid_rngs = rngs[mask]
            if not np.all(np.isfinite(valid_rngs)) or np.any(valid_rngs < 0.0):
                raise ValueError(
                    "All ranges marked valid_mask=True must be finite and non-negative"
                )
            if ints is not None:
                valid_ints = ints[mask]
                if not np.all(np.isfinite(valid_ints)) or np.any(
                    (valid_ints < 0.0) | (valid_ints > 1.0)
                ):
                    raise ValueError(
                        "All intensities marked valid_mask=True must be finite in [0.0, 1.0]"
                    )

            # ----------------------------------------------------------------
            # Geometric Point-to-Range Consistency Check
            # ||p_i - o_frame||_2 ~= r_i within tolerance_range_m
            # ----------------------------------------------------------------
            if frame_id == "sensor":
                o_frame = np.zeros(3, dtype=np.float64)
            elif frame_id == "body":
                o_frame = extrinsics.translation_b
            elif frame_id == "world":
                if orig_w is None:
                    raise ValueError(
                        "sensor_origin_w must be provided for world-frame PointCloud3D containing valid points"
                    )
                o_frame = orig_w
            else:
                o_frame = np.zeros(3, dtype=np.float64)

            # Compute Euclidean distances from frame origin to valid points
            dists = np.linalg.norm(valid_pts - o_frame, axis=1)
            discrepancy = np.abs(dists - valid_rngs)
            max_discrepancy = float(np.max(discrepancy))
            if max_discrepancy > tolerance_range_m:
                raise ValueError(
                    f"Geometric point-range inconsistency: max(| ||p - o|| - r |) = {max_discrepancy:.6e} "
                    f"exceeds tolerance {tolerance_range_m:.1e} m in '{frame_id}' frame"
                )

        # Enforce buffer immutability
        pts.flags.writeable = False
        rngs.flags.writeable = False
        mask.flags.writeable = False
        if ints is not None:
            ints.flags.writeable = False
        if orig_w is not None:
            orig_w.flags.writeable = False

        object.__setattr__(self, "timestamp", float(timestamp))
        object.__setattr__(self, "frame_id", str(frame_id))
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "ranges", rngs)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "sensor_id", str(sensor_id))
        object.__setattr__(self, "extrinsics", extrinsics)
        object.__setattr__(self, "sensor_origin_w", orig_w)
        object.__setattr__(self, "intensities", ints)

    @property
    def num_points(self) -> int:
        """Total number of points (beams) in this sweep."""
        return int(len(self.points))

    @property
    def num_valid(self) -> int:
        """Number of valid echo returns in this sweep."""
        return int(np.sum(self.valid_mask))

    @property
    def valid_points(self) -> np.ndarray:
        """Extract (M, 3) array of valid points."""
        return np.asarray(self.points[self.valid_mask], dtype=np.float64)

    @property
    def valid_ranges(self) -> np.ndarray:
        """Extract (M,) array of valid ranges."""
        return np.asarray(self.ranges[self.valid_mask], dtype=np.float32)

    def to_body_frame(self) -> PointCloud3D:
        """Transform point cloud from sensor frame to vehicle body frame (FLU).

        Uses self.extrinsics: p_B = t_B + R_S^B * p_S.
        Preserves canonical invalid return representation and exact radial range.
        """
        if self.frame_id == "body":
            return self
        if self.frame_id == "world":
            raise ValueError(
                "Cannot transform world frame point cloud to body frame without vehicle pose"
            )

        # In sensor frame: transform points
        new_points = np.full_like(self.points, np.nan)
        if self.num_valid > 0:
            # Vectorized transformation: (R @ pts.T).T + t_b
            valid_pts = self.points[self.valid_mask]
            rot_pts = (self.extrinsics.rotation_s_to_b @ valid_pts.T).T
            body_pts = rot_pts + self.extrinsics.translation_b
            new_points[self.valid_mask] = body_pts

        return PointCloud3D(
            timestamp=self.timestamp,
            frame_id="body",
            points=new_points,
            ranges=self.ranges,
            valid_mask=self.valid_mask,
            sensor_id=self.sensor_id,
            extrinsics=self.extrinsics,
            sensor_origin_w=self.sensor_origin_w,
            intensities=self.intensities,
        )

    def to_world_frame(
        self,
        drone_pos: np.ndarray | Sequence[float],
        drone_orientation: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    ) -> PointCloud3D:
        """Transform point cloud to world frame (ENU).

        Args:
            drone_pos: Quadrotor position in world frame [x, y, z] (3,).
            drone_orientation: Quadrotor attitude quaternion [qw, qx, qy, qz] (4,)
                               or 3x3 orthonormal rotation matrix R_B^W.
        """
        if self.frame_id == "world":
            return self

        pos_w = np.asarray(drone_pos, dtype=np.float64)
        if pos_w.shape != (3,):
            raise ValueError(f"drone_pos must have shape (3,), got {pos_w.shape}")
        if not np.all(np.isfinite(pos_w)):
            raise ValueError(f"drone_pos must be finite, got {pos_w}")

        rot_arr = np.asarray(drone_orientation, dtype=np.float64)
        if not np.all(np.isfinite(rot_arr)):
            raise ValueError(f"drone_orientation must contain finite values, got {rot_arr}")

        if rot_arr.shape == (4,):
            q_norm = float(np.linalg.norm(rot_arr))
            if q_norm < 1e-12:
                raise ValueError(
                    f"drone_orientation quaternion must have non-zero norm, got {rot_arr}"
                )
            r_b_to_w = quaternion_to_rotation_matrix(rot_arr)
            r_b_to_w = validate_so3_rotation(r_b_to_w)
        elif rot_arr.shape == (3, 3):
            r_b_to_w = validate_so3_rotation(rot_arr)
        else:
            raise ValueError(
                f"drone_orientation must be quaternion (4,) or rotation matrix (3, 3), got {rot_arr.shape}"
            )

        # First ensure points are in body frame
        body_cloud = self if self.frame_id == "body" else self.to_body_frame()

        sensor_origin_w = pos_w + r_b_to_w @ self.extrinsics.translation_b

        new_points = np.full_like(body_cloud.points, np.nan)
        if body_cloud.num_valid > 0:
            valid_body_pts = body_cloud.points[body_cloud.valid_mask]
            world_pts = (r_b_to_w @ valid_body_pts.T).T + pos_w
            new_points[body_cloud.valid_mask] = world_pts

        return PointCloud3D(
            timestamp=self.timestamp,
            frame_id="world",
            points=new_points,
            ranges=self.ranges,
            valid_mask=self.valid_mask,
            sensor_id=self.sensor_id,
            extrinsics=self.extrinsics,
            sensor_origin_w=sensor_origin_w,
            intensities=self.intensities,
        )
