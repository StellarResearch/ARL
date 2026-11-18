"""Unit tests for Phase V.1 / V.1.1 Simulated 3D LiDAR and PointCloud foundation.

Tests ADR-002, ADR-003, and ADR-006:
- Shared SO(3) validation on rotation matrices and quadrotor orientations.
- SensorExtrinsics validation, orthonormality checks, and array immutability.
- Strict SensorConfig3D validation (rejecting booleans, non-finite values, and malformed bounds).
- PointCloud3D canonical invalid measurement contract (NaN points, NaN ranges, 0.0 intensity).
- Geometric point-to-range consistency checks in sensor, body, and world frames.
- Strict arena bounds validation (shape, finiteness, strictly positive dimensions, sensor origin inside).
- Strict obstacle input validation (rejecting malformed geometry without silent skipping).
- Ray geometry edge cases (grazing hits, origin inside obstacle, obstacle behind, overlapping spheres).
- Synthetic normalized echo-strength intensity semantics.
- Timing and latency semantics (Option A — caller-managed timing).
- SimulatedLiDAR3D ray casting, noise distribution, dropout statistics, boundary intersections,
  extrinsic mounting, dynamic obstacle support, and execution throughput (>= 50 Hz).
- Parameterized scalability benchmark across beam counts and obstacle densities.
"""

from __future__ import annotations

import platform
import time
from typing import List

import numpy as np
import pytest

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.perception.sensors.base import BaseSensor3D
from adaptive_rl.perception.sensors.data import (
    PointCloud3D,
    SensorConfig3D,
    SensorExtrinsics,
    validate_so3_rotation,
)
from adaptive_rl.perception.sensors.lidar3d import SimulatedLiDAR3D

# ============================================================================
# 1. SO(3) Rotation Validation Tests
# ============================================================================


def test_validate_so3_rotation_valid() -> None:
    # Identity
    r_id = validate_so3_rotation(np.eye(3))
    assert np.allclose(r_id, np.eye(3))
    assert r_id.shape == (3, 3)
    assert r_id.dtype == np.float64

    # Pure yaw rotation (+45 deg around Z)
    yaw = np.radians(45.0)
    r_yaw = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    validated_yaw = validate_so3_rotation(r_yaw)
    assert np.allclose(validated_yaw, r_yaw)

    # Pure pitch rotation (+30 deg around Y)
    pitch = np.radians(30.0)
    r_pitch = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    validated_pitch = validate_so3_rotation(r_pitch)
    assert np.allclose(validated_pitch, r_pitch)


def test_validate_so3_rotation_errors() -> None:
    # Wrong shape
    with pytest.raises(ValueError, match="shape \\(3, 3\\)"):
        validate_so3_rotation(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="shape \\(3, 3\\)"):
        validate_so3_rotation(np.zeros(3))

    # Non-finite values
    with pytest.raises(ValueError, match="finite values"):
        bad_nan = np.eye(3)
        bad_nan[0, 0] = np.nan
        validate_so3_rotation(bad_nan)

    with pytest.raises(ValueError, match="finite values"):
        bad_inf = np.eye(3)
        bad_inf[0, 0] = np.inf
        validate_so3_rotation(bad_inf)

    # Non-orthonormal (scaling / shear)
    with pytest.raises(ValueError, match="must be orthonormal"):
        validate_so3_rotation(np.eye(3) * 1.5)

    # Reflection (det = -1)
    with pytest.raises(ValueError, match="determinant \\+1.0"):
        refl = np.diag([1.0, 1.0, -1.0])
        validate_so3_rotation(refl)

    # Singular matrix (det = 0)
    with pytest.raises(ValueError, match="must be orthonormal"):
        validate_so3_rotation(np.zeros((3, 3)))

    # Invalid tolerances: boolean, non-finite, zero, or negative
    for bad_tol in [True, False, 0.0, -1e-5, np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="must be finite and strictly positive"):
            validate_so3_rotation(np.eye(3), tolerance_ortho=bad_tol)
        with pytest.raises(ValueError, match="must be finite and strictly positive"):
            validate_so3_rotation(np.eye(3), tolerance_det=bad_tol)


# ============================================================================
# 2. SensorExtrinsics Tests
# ============================================================================


def test_sensor_extrinsics_default_identity() -> None:
    ext = SensorExtrinsics.default_identity()
    assert np.allclose(ext.translation_b, np.zeros(3))
    assert np.allclose(ext.rotation_s_to_b, np.eye(3))
    assert not ext.translation_b.flags.writeable
    assert not ext.rotation_s_to_b.flags.writeable


def test_sensor_extrinsics_immutability() -> None:
    ext = SensorExtrinsics.default_identity()
    with pytest.raises(ValueError, match="read-only"):
        ext.translation_b[0] = 1.0

    with pytest.raises(ValueError, match="read-only"):
        ext.rotation_s_to_b[0, 0] = 2.0


def test_sensor_extrinsics_defensive_copy() -> None:
    trans = np.array([1.0, 2.0, 3.0])
    rot = np.eye(3)
    ext = SensorExtrinsics(trans, rot)

    # Mutating original inputs must not affect extrinsics
    trans[0] = 999.0
    rot[0, 0] = 999.0
    assert ext.translation_b[0] == 1.0
    assert ext.rotation_s_to_b[0, 0] == 1.0


def test_sensor_extrinsics_validation_errors() -> None:
    # Invalid translation shape
    with pytest.raises(ValueError, match="translation_b must have shape"):
        SensorExtrinsics(np.zeros(4), np.eye(3))

    # Non-finite translation
    with pytest.raises(ValueError, match="finite values"):
        SensorExtrinsics(np.array([1.0, np.nan, 0.0]), np.eye(3))

    # Invalid rotation shape
    with pytest.raises(ValueError, match="rotation matrix must have shape"):
        SensorExtrinsics(np.zeros(3), np.eye(4))

    # Non-orthonormal matrix
    non_ortho = np.eye(3) * 2.0
    with pytest.raises(ValueError, match="must be orthonormal"):
        SensorExtrinsics(np.zeros(3), non_ortho)

    # Transformation input validation
    ext = SensorExtrinsics.default_identity()
    with pytest.raises(ValueError, match="point_s must contain finite values"):
        ext.transform_point_to_body([1.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="vector_s must contain finite values"):
        ext.transform_vector_to_body([np.inf, 0.0, 0.0])
    with pytest.raises(ValueError, match="point_b must contain finite values"):
        ext.transform_point_to_sensor([1.0, 2.0, np.nan])
    with pytest.raises(ValueError, match="vector_b must contain finite values"):
        ext.transform_vector_to_sensor([0.0, np.inf, 1.0])

    # Reflection matrix (det = -1)
    refl = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant \\+1.0"):
        SensorExtrinsics(np.zeros(3), refl)


def test_sensor_extrinsics_point_and_vector_transforms() -> None:
    # Sensor mounted at [0.1, 0.0, 0.05] in body, rotated 90 deg yaw (+Z axis)
    # R_S^B maps sensor X to body Y, sensor Y to body -X, sensor Z to body Z
    yaw = np.radians(90.0)
    rot_s_to_b = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_b = np.array([0.1, 0.0, 0.05])
    ext = SensorExtrinsics(translation_b=t_b, rotation_s_to_b=rot_s_to_b)

    # Sensor boresight point: 2m along sensor +X
    pt_s = np.array([2.0, 0.0, 0.0])
    pt_b = ext.transform_point_to_body(pt_s)
    # Expected in body: t_b + [0, 2, 0] = [0.1, 2.0, 0.05]
    assert np.allclose(pt_b, [0.1, 2.0, 0.05])

    # Round-trip transformation
    pt_s_recovered = ext.transform_point_to_sensor(pt_b)
    assert np.allclose(pt_s_recovered, pt_s)

    # Vector transform ignores translation
    vec_s = np.array([1.0, 0.0, 0.0])
    vec_b = ext.transform_vector_to_body(vec_s)
    assert np.allclose(vec_b, [0.0, 1.0, 0.0])
    assert np.allclose(ext.transform_vector_to_sensor(vec_b), vec_s)


def test_sensor_extrinsics_to_dict() -> None:
    ext = SensorExtrinsics.default_identity()
    d = ext.to_dict()
    assert d["translation_b"] == [0.0, 0.0, 0.0]
    assert d["rotation_s_to_b"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


# ============================================================================
# 3. SensorConfig3D Strict Validation Tests
# ============================================================================


def test_sensor_config_defaults() -> None:
    cfg = SensorConfig3D()
    assert cfg.sensor_id == "lidar_primary"
    assert cfg.update_rate_hz == 20.0
    assert cfg.min_range_m == 0.2
    assert cfg.max_range_m == 25.0
    assert cfg.horizontal_fov_deg == 360.0
    assert cfg.vertical_fov_deg == 60.0
    assert cfg.horizontal_beams == 32
    assert cfg.vertical_beams == 8
    assert cfg.range_noise_std_m == 0.03
    assert cfg.dropout_probability == 0.02
    assert cfg.latency_sec == 0.02
    assert cfg.extrinsics is not None
    assert np.allclose(cfg.extrinsics.translation_b, np.zeros(3))
    assert cfg.dt == pytest.approx(0.05)
    assert cfg.total_beams == 256
    assert cfg.horizontal_fov_rad == pytest.approx(2.0 * np.pi)
    assert cfg.vertical_fov_rad == pytest.approx(np.pi / 3.0)


def test_sensor_config_strict_validation() -> None:
    # Empty or whitespace sensor_id
    with pytest.raises(ValueError, match="sensor_id"):
        SensorConfig3D(sensor_id="")
    with pytest.raises(ValueError, match="sensor_id"):
        SensorConfig3D(sensor_id="   ")

    # Reject booleans and non-ints for beam counts
    with pytest.raises(ValueError, match="horizontal_beams"):
        SensorConfig3D(horizontal_beams=True)  # type: ignore
    with pytest.raises(ValueError, match="horizontal_beams"):
        SensorConfig3D(horizontal_beams=2.5)  # type: ignore
    with pytest.raises(ValueError, match="horizontal_beams"):
        SensorConfig3D(horizontal_beams=0)
    with pytest.raises(ValueError, match="vertical_beams"):
        SensorConfig3D(vertical_beams=False)  # type: ignore
    with pytest.raises(ValueError, match="vertical_beams"):
        SensorConfig3D(vertical_beams=1.2)  # type: ignore
    with pytest.raises(ValueError, match="vertical_beams"):
        SensorConfig3D(vertical_beams=-1)

    # Reject booleans and non-finite for update_rate_hz
    with pytest.raises(ValueError, match="update_rate_hz"):
        SensorConfig3D(update_rate_hz=True)  # type: ignore
    with pytest.raises(ValueError, match="update_rate_hz"):
        SensorConfig3D(update_rate_hz=-1.0)
    with pytest.raises(ValueError, match="update_rate_hz"):
        SensorConfig3D(update_rate_hz=np.nan)
    with pytest.raises(ValueError, match="update_rate_hz"):
        SensorConfig3D(update_rate_hz=np.inf)

    # Range bounds checks
    with pytest.raises(ValueError, match="min_range_m"):
        SensorConfig3D(min_range_m=-0.1)
    with pytest.raises(ValueError, match="min_range_m"):
        SensorConfig3D(min_range_m=True)  # type: ignore
    with pytest.raises(ValueError, match="Invalid range bounds"):
        SensorConfig3D(min_range_m=30.0, max_range_m=20.0)
    with pytest.raises(ValueError, match="Invalid range bounds"):
        SensorConfig3D(min_range_m=10.0, max_range_m=10.0)

    # FOV checks
    with pytest.raises(ValueError, match="horizontal_fov_deg"):
        SensorConfig3D(horizontal_fov_deg=0.0)
    with pytest.raises(ValueError, match="horizontal_fov_deg"):
        SensorConfig3D(horizontal_fov_deg=370.0)
    with pytest.raises(ValueError, match="vertical_fov_deg"):
        SensorConfig3D(vertical_fov_deg=-10.0)
    with pytest.raises(ValueError, match="vertical_fov_deg"):
        SensorConfig3D(vertical_fov_deg=200.0)

    # Noise, dropout, latency checks
    with pytest.raises(ValueError, match="range_noise_std_m"):
        SensorConfig3D(range_noise_std_m=-0.01)
    with pytest.raises(ValueError, match="dropout_probability"):
        SensorConfig3D(dropout_probability=-0.1)
    with pytest.raises(ValueError, match="dropout_probability"):
        SensorConfig3D(dropout_probability=1.5)
    with pytest.raises(ValueError, match="latency_sec"):
        SensorConfig3D(latency_sec=-0.01)

    # Extrinsics type check
    with pytest.raises(TypeError, match="extrinsics must be an instance"):
        SensorConfig3D(extrinsics="invalid")  # type: ignore


# ============================================================================
# 4. PointCloud3D Canonical Contract & Consistency Tests
# ============================================================================


def test_point_cloud_immutability_and_defensive_copies() -> None:
    ext = SensorExtrinsics.default_identity()
    pts = np.array([[3.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    rngs = np.array([3.0, np.nan], dtype=np.float32)
    mask = np.array([True, False])
    ints = np.array([0.8, 0.0], dtype=np.float32)
    orig_w = np.array([1.0, 2.0, 3.0])

    cloud = PointCloud3D(
        timestamp=0.5,
        frame_id="sensor",
        points=pts,
        ranges=rngs,
        valid_mask=mask,
        sensor_id="lidar_01",
        extrinsics=ext,
        sensor_origin_w=orig_w,
        intensities=ints,
    )

    # Ingestion defensive copy test
    pts[0, 0] = 999.0
    rngs[0] = 999.0
    mask[0] = False
    ints[0] = 0.1
    orig_w[0] = 50.0
    assert cloud.points[0, 0] == 3.0
    assert cloud.ranges[0] == pytest.approx(3.0, abs=1e-2)
    assert cloud.valid_mask[0] is np.True_
    assert cloud.intensities is not None and cloud.intensities[0] == pytest.approx(0.8, abs=1e-2)
    assert cloud.sensor_origin_w is not None and cloud.sensor_origin_w[0] == 1.0

    # Read-only buffer tests
    assert not cloud.points.flags.writeable
    assert not cloud.ranges.flags.writeable
    assert not cloud.valid_mask.flags.writeable
    assert cloud.intensities is not None and not cloud.intensities.flags.writeable
    assert cloud.sensor_origin_w is not None and not cloud.sensor_origin_w.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        cloud.points[0, 0] = 50.0
    with pytest.raises(ValueError, match="read-only"):
        cloud.ranges[0] = 50.0
    with pytest.raises(ValueError, match="read-only"):
        cloud.valid_mask[0] = False


def test_point_cloud_canonical_invalid_representation() -> None:
    ext = SensorExtrinsics.default_identity()

    # Case 1: Invalid return (valid_mask=False) has non-NaN point -> must reject
    bad_pts = np.array([[0.0, 0.0, 0.0]])
    bad_rngs = np.array([np.nan], dtype=np.float32)
    mask = np.array([False])
    with pytest.raises(ValueError, match="must have NaN coordinates"):
        PointCloud3D(0.0, "sensor", bad_pts, bad_rngs, mask, "lidar", ext)

    # Case 2: Invalid return has non-NaN range (e.g. inf or 0.0) -> must reject
    bad_pts = np.array([[np.nan, np.nan, np.nan]])
    bad_rngs_inf = np.array([np.inf], dtype=np.float32)
    with pytest.raises(ValueError, match="must have NaN range values"):
        PointCloud3D(0.0, "sensor", bad_pts, bad_rngs_inf, mask, "lidar", ext)

    bad_rngs_zero = np.array([0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="must have NaN range values"):
        PointCloud3D(0.0, "sensor", bad_pts, bad_rngs_zero, mask, "lidar", ext)

    # Case 3: Invalid return has non-zero intensity -> must reject
    bad_rngs = np.array([np.nan], dtype=np.float32)
    bad_ints = np.array([0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="must have 0.0 echo intensity"):
        PointCloud3D(0.0, "sensor", bad_pts, bad_rngs, mask, "lidar", ext, intensities=bad_ints)


def test_point_cloud_geometric_consistency() -> None:
    ext = SensorExtrinsics(translation_b=np.array([0.2, 0.0, 0.1]), rotation_s_to_b=np.eye(3))

    # 1. Sensor frame inconsistency: point norm = 5.0, range = 3.0 -> must reject
    pts_s_bad = np.array([[5.0, 0.0, 0.0]])
    rngs_bad = np.array([3.0], dtype=np.float32)
    mask = np.array([True])
    with pytest.raises(ValueError, match="Geometric point-range inconsistency"):
        PointCloud3D(0.0, "sensor", pts_s_bad, rngs_bad, mask, "lidar", ext)

    # Valid sensor frame point (norm ~= 4.0, range = 4.0) -> must pass
    pts_s_ok = np.array([[4.0, 0.0, 0.0]])
    rngs_ok = np.array([4.0], dtype=np.float32)
    cloud_s = PointCloud3D(0.0, "sensor", pts_s_ok, rngs_ok, mask, "lidar", ext)
    assert cloud_s.num_valid == 1

    # 2. Body frame consistency: o_B = t_B = [0.2, 0, 0.1]
    # Point at [4.2, 0.0, 0.1] has distance to t_B of 4.0m
    pts_b_ok = np.array([[4.2, 0.0, 0.1]])
    cloud_b = PointCloud3D(0.0, "body", pts_b_ok, rngs_ok, mask, "lidar", ext)
    assert cloud_b.num_valid == 1

    # Body frame inconsistency: point distance to t_B != 4.0
    pts_b_bad = np.array([[4.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Geometric point-range inconsistency"):
        PointCloud3D(0.0, "body", pts_b_bad, rngs_ok, mask, "lidar", ext)

    # 3. World frame consistency requires sensor_origin_w
    pts_w = np.array([[10.2, 5.0, 2.1]])
    # Missing sensor_origin_w with valid points -> must reject
    with pytest.raises(ValueError, match="sensor_origin_w must be provided"):
        PointCloud3D(0.0, "world", pts_w, rngs_ok, mask, "lidar", ext, sensor_origin_w=None)

    # Correct sensor_origin_w: [6.2, 5.0, 2.1], dist to point is 4.0m -> passes
    cloud_w = PointCloud3D(
        0.0,
        "world",
        pts_w,
        rngs_ok,
        mask,
        "lidar",
        ext,
        sensor_origin_w=np.array([6.2, 5.0, 2.1]),
    )
    assert cloud_w.num_valid == 1


def test_point_cloud_validation_rules() -> None:
    ext = SensorExtrinsics.default_identity()
    pts = np.array([[1.0, 2.0, 3.0]])
    rngs = np.array([float(np.sqrt(14.0))], dtype=np.float32)
    mask = np.array([True])

    # Timestamp checks
    with pytest.raises(ValueError, match="timestamp must be non-negative"):
        PointCloud3D(-0.1, "sensor", pts, rngs, mask, "lidar", ext)
    with pytest.raises(ValueError, match="timestamp must be non-negative"):
        PointCloud3D(np.nan, "sensor", pts, rngs, mask, "lidar", ext)

    # Frame ID checks
    with pytest.raises(ValueError, match="frame_id must be one of"):
        PointCloud3D(0.0, "camera_optical", pts, rngs, mask, "lidar", ext)

    # Shape mismatch checks
    with pytest.raises(ValueError, match="ranges shape"):
        PointCloud3D(0.0, "sensor", pts, np.array([1.0, 2.0]), mask, "lidar", ext)
    with pytest.raises(ValueError, match="valid_mask shape"):
        PointCloud3D(0.0, "sensor", pts, rngs, np.array([True, False]), "lidar", ext)
    with pytest.raises(ValueError, match="points must have shape \\(N, 3\\)"):
        PointCloud3D(0.0, "sensor", np.zeros((1, 4)), rngs, mask, "lidar", ext)

    # Non-finite coordinates on valid points
    bad_pts = np.array([[np.nan, 2.0, 3.0]])
    with pytest.raises(ValueError, match="finite coordinates"):
        PointCloud3D(0.0, "sensor", bad_pts, rngs, mask, "lidar", ext)

    # Non-finite or negative range on valid points
    bad_rngs = np.array([np.nan], dtype=np.float32)
    with pytest.raises(ValueError, match="finite and non-negative"):
        PointCloud3D(0.0, "sensor", pts, bad_rngs, mask, "lidar", ext)

    # tolerance_range_m checks: reject booleans, zero, negative, nan, inf
    for bad_tol in [True, False, 0.0, -1e-3, np.nan, np.inf]:
        with pytest.raises(
            ValueError, match="tolerance_range_m must be strictly positive and finite"
        ):
            PointCloud3D(0.0, "sensor", pts, rngs, mask, "lidar", ext, tolerance_range_m=bad_tol)

    # valid_mask checks: reject malformed non-boolean masks (e.g. ints, strings, floats)
    for bad_mask in [[1], np.array([1]), ["True"], np.array(["True"]), [1.0]]:
        with pytest.raises(ValueError, match="valid_mask must contain strictly boolean values"):
            PointCloud3D(0.0, "sensor", pts, rngs, bad_mask, "lidar", ext)

    # to_world_frame transformation input checks
    cloud = PointCloud3D(0.0, "sensor", pts, rngs, mask, "lidar", ext)
    with pytest.raises(ValueError, match="drone_pos must be finite"):
        cloud.to_world_frame([np.nan, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="drone_orientation must contain finite values"):
        cloud.to_world_frame([0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="quaternion must have non-zero norm"):
        cloud.to_world_frame([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])


def test_point_cloud_frame_transformations() -> None:
    # Sensor mounted forward +0.2m along body X
    ext = SensorExtrinsics(
        translation_b=np.array([0.2, 0.0, 0.0]),
        rotation_s_to_b=np.eye(3),
    )
    # Point at [3.0, 0.0, 0.0] in sensor frame
    pts = np.array([[3.0, 0.0, 0.0], [np.nan, np.nan, np.nan]])
    rngs = np.array([3.0, np.nan], dtype=np.float32)
    mask = np.array([True, False])

    cloud_sensor = PointCloud3D(
        timestamp=1.0,
        frame_id="sensor",
        points=pts,
        ranges=rngs,
        valid_mask=mask,
        sensor_id="lidar_test",
        extrinsics=ext,
    )
    assert cloud_sensor.num_points == 2
    assert cloud_sensor.num_valid == 1
    assert np.allclose(cloud_sensor.valid_points, [[3.0, 0.0, 0.0]])

    # 1. To body frame: p_B = [0.2, 0, 0] + [3.0, 0, 0] = [3.2, 0.0, 0.0]
    cloud_body = cloud_sensor.to_body_frame()
    assert cloud_body.frame_id == "body"
    assert np.allclose(cloud_body.valid_points, [[3.2, 0.0, 0.0]])
    assert np.isnan(cloud_body.points[1, 0])
    # Idempotent
    assert cloud_body.to_body_frame() is cloud_body

    # 2. To world frame: Drone at [10.0, 5.0, 2.0], rotated yaw 90 deg (+Z)
    yaw = np.radians(90.0)
    q_yaw = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    drone_pos = np.array([10.0, 5.0, 2.0])

    cloud_world = cloud_sensor.to_world_frame(drone_pos=drone_pos, drone_orientation=q_yaw)
    assert cloud_world.frame_id == "world"
    assert np.allclose(cloud_world.valid_points, [[10.0, 8.2, 2.0]])
    assert cloud_world.sensor_origin_w is not None
    assert np.allclose(cloud_world.sensor_origin_w, [10.0, 5.2, 2.0])

    # 3. Reject malformed rotation matrix in to_world_frame
    with pytest.raises(ValueError, match="rotation matrix must be orthonormal"):
        cloud_sensor.to_world_frame(drone_pos=drone_pos, drone_orientation=np.eye(3) * 2.0)


# ============================================================================
# 5. SimulatedLiDAR3D Ray-Casting & Obstacle Tests
# ============================================================================


def test_simulated_lidar_base_interface() -> None:
    lidar = SimulatedLiDAR3D()
    assert isinstance(lidar, BaseSensor3D)
    assert lidar.sensor_id == "lidar_primary"
    assert lidar.extrinsics is not None


def test_simulated_lidar_beam_directions() -> None:
    # 4 horizontal, 3 vertical beams
    cfg = SensorConfig3D(
        horizontal_beams=4,
        vertical_beams=3,
        horizontal_fov_deg=360.0,
        vertical_fov_deg=60.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg)
    rays = lidar.ray_directions_s
    assert rays.shape == (12, 3)

    # Verify unit norm on every ray
    norms = np.linalg.norm(rays, axis=1)
    assert np.allclose(norms, 1.0)

    # Elevation layers check
    z_coords = rays[:, 2]
    unique_z = np.unique(np.round(z_coords, 4))
    assert len(unique_z) == 3
    assert np.isclose(unique_z[0], -0.5)
    assert np.isclose(unique_z[1], 0.0)
    assert np.isclose(unique_z[2], 0.5)


def test_stationary_sphere_boresight_detection() -> None:
    # Obstacle sphere at [5.0, 0.0, 0.0] with radius 1.0m
    # Distance to surface: 5.0 - 1.0 = 4.0m
    obs = ObstacleSphere3D(center=np.array([5.0, 0.0, 0.0]), radius=1.0)

    cfg = SensorConfig3D(
        horizontal_beams=8,
        vertical_beams=3,
        range_noise_std_m=0.0,
        dropout_probability=0.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg)

    drone_pos = np.array([0.0, 0.0, 0.0])
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0])

    cloud = lidar.capture(
        timestamp=0.1,
        drone_pos=drone_pos,
        drone_orientation=drone_quat,
        obstacles=[obs],
    )

    assert cloud.timestamp == 0.1
    assert cloud.frame_id == "sensor"
    assert cloud.num_valid >= 1

    # Boresight beam [1.0, 0.0, 0.0]
    boresight_idx = -1
    for i, r in enumerate(lidar.ray_directions_s):
        if np.allclose(r, [1.0, 0.0, 0.0], atol=1e-3):
            boresight_idx = i
            break
    assert boresight_idx != -1

    assert cloud.valid_mask[boresight_idx] is np.True_
    assert cloud.ranges[boresight_idx] == pytest.approx(4.0, abs=1e-4)
    assert np.allclose(cloud.points[boresight_idx], [4.0, 0.0, 0.0], atol=1e-4)


def test_off_axis_and_yaw_rotation_detection() -> None:
    # Obstacle at [0.0, 5.0, 0.0] (North) with radius 1.0m
    obs = ObstacleSphere3D(center=np.array([0.0, 5.0, 0.0]), radius=1.0)

    cfg = SensorConfig3D(
        horizontal_beams=8,
        vertical_beams=1,
        range_noise_std_m=0.0,
        dropout_probability=0.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg)

    # 1. Drone at origin facing North (yaw = 90 deg)
    yaw = np.radians(90.0)
    q_yaw = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    drone_pos = np.zeros(3)

    cloud = lidar.capture(
        timestamp=0.5,
        drone_pos=drone_pos,
        drone_orientation=q_yaw,
        obstacles=[obs],
    )

    boresight_idx = -1
    for i, r in enumerate(lidar.ray_directions_s):
        if np.allclose(r, [1.0, 0.0, 0.0], atol=1e-3):
            boresight_idx = i
            break
    assert boresight_idx != -1

    assert cloud.valid_mask[boresight_idx] is np.True_
    assert np.allclose(cloud.points[boresight_idx], [4.0, 0.0, 0.0], atol=1e-4)

    cloud_w = cloud.to_world_frame(drone_pos, q_yaw)
    assert np.allclose(cloud_w.points[boresight_idx], [0.0, 4.0, 0.0], atol=1e-4)


def test_arena_boundary_intersections() -> None:
    # Arena box: [20.0, 20.0, 10.0]
    bounds = (20.0, 20.0, 10.0)
    drone_pos = np.array([10.0, 10.0, 5.0])
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0])

    cfg = SensorConfig3D(
        horizontal_beams=4,
        vertical_beams=1,
        range_noise_std_m=0.0,
        dropout_probability=0.0,
        max_range_m=30.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg)

    cloud = lidar.capture(
        timestamp=1.0,
        drone_pos=drone_pos,
        drone_orientation=drone_quat,
        obstacles=[],
        bounds=bounds,
    )

    assert cloud.num_valid == 4
    for i in range(4):
        assert cloud.valid_mask[i] is np.True_
        assert cloud.ranges[i] == pytest.approx(10.0, abs=1e-4)


def test_out_of_range_and_blind_zone_canonical_invalids() -> None:
    # Max range is 10.0m, min range is 0.5m
    cfg = SensorConfig3D(
        min_range_m=0.5,
        max_range_m=10.0,
        range_noise_std_m=0.0,
        dropout_probability=0.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg)

    # 1. Far obstacle at 15.0m (beyond max_range_m)
    far_obs = ObstacleSphere3D(center=np.array([15.0, 0.0, 0.0]), radius=1.0)
    cloud_far = lidar.capture(
        timestamp=0.0,
        drone_pos=np.zeros(3),
        drone_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        obstacles=[far_obs],
    )
    boresight_idx = 0
    for i, r in enumerate(lidar.ray_directions_s):
        if np.allclose(r, [1.0, 0.0, 0.0], atol=1e-3):
            boresight_idx = i
            break
    # Must follow canonical invalid representation (valid_mask=False, range=NaN, points=NaN, intensity=0)
    assert not cloud_far.valid_mask[boresight_idx]
    assert np.isnan(cloud_far.ranges[boresight_idx])
    assert np.all(np.isnan(cloud_far.points[boresight_idx]))
    assert cloud_far.intensities is not None and cloud_far.intensities[boresight_idx] == 0.0

    # 2. Close obstacle at 0.3m (inside blind zone < min_range_m)
    close_obs = ObstacleSphere3D(center=np.array([0.8, 0.0, 0.0]), radius=0.5)
    cloud_close = lidar.capture(
        timestamp=0.0,
        drone_pos=np.zeros(3),
        drone_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        obstacles=[close_obs],
    )
    assert not cloud_close.valid_mask[boresight_idx]
    assert np.isnan(cloud_close.ranges[boresight_idx])
    assert np.all(np.isnan(cloud_close.points[boresight_idx]))


def test_dynamic_obstacle_tracking_over_time() -> None:
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([5.0, 10.0, 5.0]),
        radius=1.0,
        velocity=np.array([0.0, 1.0, 0.0]),
        motion_type="bounce",
    )

    cfg = SensorConfig3D(
        horizontal_beams=16,
        vertical_beams=1,
        range_noise_std_m=0.0,
        dropout_probability=0.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg)

    drone_pos = np.array([0.0, 10.0, 5.0])
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # At t = 0: obstacle center is at [5.0, 10.0, 5.0], boresight distance is 4.0m
    cloud_t0 = lidar.capture(0.0, drone_pos, drone_quat, [dyn_obs])
    cloud_t2 = lidar.capture(2.0, drone_pos, drone_quat, [dyn_obs])

    boresight_idx = -1
    for i, r in enumerate(lidar.ray_directions_s):
        if np.allclose(r, [1.0, 0.0, 0.0], atol=1e-3):
            boresight_idx = i
            break

    assert boresight_idx != -1
    assert cloud_t0.valid_mask[boresight_idx]
    assert cloud_t0.ranges[boresight_idx] == pytest.approx(4.0, abs=1e-3)

    # At t=2, obstacle moved 2m along +Y (now at y=12). Boresight ray (+X) misses it
    assert not cloud_t2.valid_mask[boresight_idx]
    assert cloud_t2.num_valid >= 1


def test_range_noise_statistics() -> None:
    obs = ObstacleSphere3D(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
    noise_sigma = 0.05

    cfg = SensorConfig3D(
        horizontal_beams=1,
        vertical_beams=1,
        horizontal_fov_deg=10.0,
        vertical_fov_deg=10.0,
        range_noise_std_m=noise_sigma,
        dropout_probability=0.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg, seed=42)

    samples = []
    for _ in range(1000):
        c = lidar.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], [obs])
        if c.valid_mask[0]:
            samples.append(float(c.ranges[0]))

    samples_arr = np.array(samples)
    assert len(samples_arr) == 1000

    errors = samples_arr - 4.0
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))

    assert abs(mean_err) < 0.01
    assert std_err == pytest.approx(noise_sigma, rel=0.10)


def test_dropout_statistics() -> None:
    obs = ObstacleSphere3D(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
    target_p = 0.15

    cfg = SensorConfig3D(
        horizontal_beams=1,
        vertical_beams=1,
        range_noise_std_m=0.0,
        dropout_probability=target_p,
    )
    lidar = SimulatedLiDAR3D(config=cfg, seed=123)

    dropped = 0
    total = 2000
    for _ in range(total):
        c = lidar.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], [obs])
        if not c.valid_mask[0]:
            dropped += 1

    empirical_p = dropped / total
    assert empirical_p == pytest.approx(target_p, abs=0.02)


def test_seeded_determinism() -> None:
    obs = ObstacleSphere3D(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
    cfg = SensorConfig3D(range_noise_std_m=0.03, dropout_probability=0.05)

    lidar1 = SimulatedLiDAR3D(config=cfg, seed=999)
    lidar2 = SimulatedLiDAR3D(config=cfg, seed=999)

    c1 = lidar1.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], [obs])
    c2 = lidar2.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], [obs])

    assert np.all(c1.valid_mask == c2.valid_mask)
    v1 = c1.ranges[c1.valid_mask]
    v2 = c2.ranges[c2.valid_mask]
    assert np.all(v1 == v2)


def test_ground_truth_isolation() -> None:
    obs = ObstacleSphere3D(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
    lidar = SimulatedLiDAR3D()

    cloud = lidar.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], [obs])

    assert not hasattr(cloud, "obstacles")
    assert not hasattr(cloud, "true_obstacle_centers")
    assert not hasattr(cloud, "obstacle_velocities")
    assert not hasattr(cloud, "obstacle_ids")


def test_simulated_lidar_non_identity_extrinsics() -> None:
    pitch_down = np.radians(-30.0)
    rot_s_to_b = np.array(
        [
            [np.cos(pitch_down), 0.0, np.sin(pitch_down)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch_down), 0.0, np.cos(pitch_down)],
        ]
    )
    t_b = np.array([0.2, 0.0, 0.0])
    ext = SensorExtrinsics(translation_b=t_b, rotation_s_to_b=rot_s_to_b)

    cfg = SensorConfig3D(
        horizontal_beams=1,
        vertical_beams=1,
        range_noise_std_m=0.0,
        dropout_probability=0.0,
        extrinsics=ext,
    )
    lidar = SimulatedLiDAR3D(config=cfg)

    d_dir = np.array([np.cos(pitch_down), 0.0, -np.sin(pitch_down)])
    obs_center = t_b + 4.0 * d_dir
    obs = ObstacleSphere3D(center=obs_center, radius=1.0)

    cloud = lidar.capture(
        timestamp=0.0,
        drone_pos=np.zeros(3),
        drone_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        obstacles=[obs],
    )

    assert cloud.valid_mask[0]
    assert cloud.ranges[0] == pytest.approx(3.0, abs=1e-4)
    assert np.allclose(cloud.points[0], [3.0, 0.0, 0.0], atol=1e-4)

    cloud_w = cloud.to_world_frame(drone_pos=np.zeros(3), drone_orientation=[1.0, 0, 0, 0])
    expected_w_pt = t_b + 3.0 * d_dir
    assert np.allclose(cloud_w.valid_points[0], expected_w_pt, atol=1e-4)


def test_empty_cloud_handling() -> None:
    ext = SensorExtrinsics.default_identity()
    pts = np.full((5, 3), np.nan)
    rngs = np.full(5, np.nan, dtype=np.float32)
    mask = np.zeros(5, dtype=bool)

    cloud = PointCloud3D(
        timestamp=1.0,
        frame_id="sensor",
        points=pts,
        ranges=rngs,
        valid_mask=mask,
        sensor_id="lidar_empty",
        extrinsics=ext,
    )
    assert cloud.num_points == 5
    assert cloud.num_valid == 0
    assert cloud.valid_points.shape == (0, 3)
    assert cloud.valid_ranges.shape == (0,)

    cloud_b = cloud.to_body_frame()
    assert cloud_b.num_valid == 0
    assert np.all(np.isnan(cloud_b.points))

    cloud_w = cloud.to_world_frame(np.zeros(3), [1.0, 0, 0, 0])
    assert cloud_w.num_valid == 0
    assert np.all(np.isnan(cloud_w.points))


def test_simulated_lidar_invalid_capture_inputs() -> None:
    lidar = SimulatedLiDAR3D()

    # Negative or non-finite timestamp
    with pytest.raises(ValueError, match="timestamp"):
        lidar.capture(-1.0, np.zeros(3), [1.0, 0, 0, 0])
    with pytest.raises(ValueError, match="timestamp"):
        lidar.capture(np.nan, np.zeros(3), [1.0, 0, 0, 0])
    with pytest.raises(ValueError, match="timestamp"):
        lidar.capture(True, np.zeros(3), [1.0, 0, 0, 0])  # type: ignore

    # Invalid drone_pos
    with pytest.raises(ValueError, match="drone_pos"):
        lidar.capture(0.0, np.zeros(4), [1.0, 0, 0, 0])
    with pytest.raises(ValueError, match="drone_pos"):
        lidar.capture(0.0, np.array([np.nan, 0, 0]), [1.0, 0, 0, 0])

    # Invalid orientation
    with pytest.raises(ValueError, match="drone_orientation"):
        lidar.capture(0.0, np.zeros(3), [1.0, 0, 0])
    with pytest.raises(ValueError, match="rotation matrix must be orthonormal"):
        lidar.capture(0.0, np.zeros(3), np.eye(3) * 2.0)


# ============================================================================
# 6. Strict Arena Bounds and Obstacle Input Validation Tests
# ============================================================================


def test_strict_arena_bounds_validation() -> None:
    lidar = SimulatedLiDAR3D()

    # Bounds wrong shape
    with pytest.raises(ValueError, match="bounds must have shape"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], bounds=(20.0, 20.0))  # type: ignore

    # Bounds non-finite
    with pytest.raises(ValueError, match="bounds must be finite"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], bounds=(20.0, np.nan, 10.0))

    # Bounds non-positive
    with pytest.raises(ValueError, match="strictly positive"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], bounds=(20.0, -5.0, 10.0))
    with pytest.raises(ValueError, match="strictly positive"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], bounds=(20.0, 0.0, 10.0))

    # Sensor origin outside bounds
    with pytest.raises(ValueError, match="outside declared arena bounds"):
        lidar.capture(0.0, [25.0, 5.0, 5.0], [1, 0, 0, 0], bounds=(20.0, 20.0, 10.0))
    with pytest.raises(ValueError, match="outside declared arena bounds"):
        lidar.capture(0.0, [-1.0, 5.0, 5.0], [1, 0, 0, 0], bounds=(20.0, 20.0, 10.0))


def test_strict_obstacle_validation() -> None:
    lidar = SimulatedLiDAR3D()

    # Unsupported obstacle type -> TypeError
    with pytest.raises(TypeError, match="Unsupported obstacle type"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], obstacles=["not_an_obstacle"])  # type: ignore

    with pytest.raises(TypeError, match="Unsupported obstacle type"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], obstacles=[{"center": [1, 2, 3]}])  # type: ignore

    # Obstacle with non-finite or wrong-shape center -> ValueError
    class BadCenterObstacle:
        center = np.array([np.nan, 0.0, 0.0])
        radius = 1.0

    with pytest.raises(ValueError, match="Obstacle center must have finite coordinates"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], obstacles=[BadCenterObstacle()])

    class WrongDimCenterObstacle:
        center = np.array([1.0, 2.0])
        radius = 1.0

    with pytest.raises(ValueError, match="Obstacle center must have shape \\(3,\\)"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], obstacles=[WrongDimCenterObstacle()])

    # Obstacle with non-positive or boolean radius -> ValueError
    class NegativeRadiusObstacle:
        center = np.array([1.0, 2.0, 3.0])
        radius = -0.5

    with pytest.raises(ValueError, match="Obstacle radius must be a finite strictly positive"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], obstacles=[NegativeRadiusObstacle()])

    class BoolRadiusObstacle:
        center = np.array([1.0, 2.0, 3.0])
        radius = True

    with pytest.raises(ValueError, match="Obstacle radius must be a finite strictly positive"):
        lidar.capture(0.0, [5, 5, 5], [1, 0, 0, 0], obstacles=[BoolRadiusObstacle()])


# ============================================================================
# 7. Ray Geometry Edge Cases
# ============================================================================


def test_ray_geometry_edge_cases() -> None:
    cfg = SensorConfig3D(
        horizontal_beams=1,
        vertical_beams=1,
        range_noise_std_m=0.0,
        dropout_probability=0.0,
        min_range_m=0.1,
        max_range_m=20.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg)
    drone_quat = [1.0, 0.0, 0.0, 0.0]

    # 1. Sphere behind the ray: ray points +X, obstacle at [-5, 0, 0] -> Miss (NaN)
    obs_behind = ObstacleSphere3D(center=np.array([-5.0, 0.0, 0.0]), radius=1.0)
    c_behind = lidar.capture(0.0, [0, 0, 0], drone_quat, [obs_behind])
    assert not c_behind.valid_mask[0]
    assert np.isnan(c_behind.ranges[0])

    # 2. Tangential grazing hit: obstacle at [5.0, 1.0, 0.0], radius 1.0m
    # Distance from ray line (Y=0, Z=0) to center is exactly 1.0m -> discriminant == 0.0
    obs_tangent = ObstacleSphere3D(center=np.array([5.0, 1.0, 0.0]), radius=1.0)
    c_tangent = lidar.capture(0.0, [0, 0, 0], drone_quat, [obs_tangent])
    assert c_tangent.valid_mask[0]
    assert c_tangent.ranges[0] == pytest.approx(5.0, abs=1e-3)

    # 3. Ray origin inside sphere: center at [0.5, 0.0, 0.0], radius 1.0m
    # Sensor origin is [0, 0, 0] which is inside (dist=0.5 < 1.0)
    # Positive exit intersection: t_exit = -b + sqrt(disc) = -(-0.5) + 1.0 = 1.5m
    obs_inside = ObstacleSphere3D(center=np.array([0.5, 0.0, 0.0]), radius=1.0)
    c_inside = lidar.capture(0.0, [0, 0, 0], drone_quat, [obs_inside])
    assert c_inside.valid_mask[0]
    assert c_inside.ranges[0] == pytest.approx(1.5, abs=1e-3)

    # 3b. Ray origin at exact center of sphere: center at [0, 0, 0], radius 3.0m -> exit distance = 3.0m
    obs_concentric = ObstacleSphere3D(center=np.array([0.0, 0.0, 0.0]), radius=3.0)
    c_concentric = lidar.capture(0.0, [0, 0, 0], drone_quat, [obs_concentric])
    assert c_concentric.valid_mask[0]
    assert c_concentric.ranges[0] == pytest.approx(3.0, abs=1e-3)

    # 3c. Ray origin on boundary pointing inward: center at [1.0, 0, 0], radius 1.0m -> exit distance = 2.0m
    obs_bnd_in = ObstacleSphere3D(center=np.array([1.0, 0.0, 0.0]), radius=1.0)
    c_bnd_in = lidar.capture(0.0, [0, 0, 0], drone_quat, [obs_bnd_in])
    assert c_bnd_in.valid_mask[0]
    assert c_bnd_in.ranges[0] == pytest.approx(2.0, abs=1e-3)

    # 3d. Ray origin on boundary pointing outward: center at [-1.0, 0, 0], radius 1.0m -> Miss (NaN)
    obs_bnd_out = ObstacleSphere3D(center=np.array([-1.0, 0.0, 0.0]), radius=1.0)
    c_bnd_out = lidar.capture(0.0, [0, 0, 0], drone_quat, [obs_bnd_out])
    assert not c_bnd_out.valid_mask[0]
    assert np.isnan(c_bnd_out.ranges[0])

    # 4. Overlapping spheres: Near obstacle at [4.0, 0, 0] (r=1, dist=3m)
    # Far obstacle at [6.0, 0, 0] (r=1.5, dist=4.5m)
    obs_near = ObstacleSphere3D(center=np.array([4.0, 0.0, 0.0]), radius=1.0)
    obs_far = ObstacleSphere3D(center=np.array([6.0, 0.0, 0.0]), radius=1.5)
    c_overlap = lidar.capture(0.0, [0, 0, 0], drone_quat, [obs_far, obs_near])
    assert c_overlap.valid_mask[0]
    # Must report the nearest obstacle surface (3.0m), not the far one
    assert c_overlap.ranges[0] == pytest.approx(3.0, abs=1e-3)

    # 5. Ray parallel to boundary wall: ray points +X (axis 0)
    # When testing against bounds=(10, 10, 10) with drone at [5, 5, 5]:
    # Ray along +X hits East boundary wall at x=10 (dist = 5.0m)
    c_wall = lidar.capture(0.0, [5, 5, 5], drone_quat, [], bounds=(10, 10, 10))
    assert c_wall.valid_mask[0]
    assert c_wall.ranges[0] == pytest.approx(5.0, abs=1e-3)


# ============================================================================
# 8. Intensity Proxy and Timing Semantics Tests
# ============================================================================


def test_intensity_proxy_semantics() -> None:
    cfg = SensorConfig3D(
        horizontal_beams=1,
        vertical_beams=1,
        max_range_m=20.0,
        range_noise_std_m=0.0,
        dropout_probability=0.0,
    )
    lidar = SimulatedLiDAR3D(config=cfg)

    # Closer obstacle at 4m (surface at 3m)
    obs_close = ObstacleSphere3D(center=np.array([4.0, 0.0, 0.0]), radius=1.0)
    c_close = lidar.capture(0.0, [0, 0, 0], [1, 0, 0, 0], [obs_close])

    # Farther obstacle at 11m (surface at 10m)
    obs_far = ObstacleSphere3D(center=np.array([11.0, 0.0, 0.0]), radius=1.0)
    c_far = lidar.capture(0.0, [0, 0, 0], [1, 0, 0, 0], [obs_far])

    assert c_close.intensities is not None
    assert c_far.intensities is not None

    # Echo strength falls off with range: I_close > I_far
    i_close = float(c_close.intensities[0])
    i_far = float(c_far.intensities[0])
    assert 0.0 < i_far < i_close <= 1.0

    # Theoretical values: 1 / (1 + (r / max_r)^2)
    expected_close = 1.0 / (1.0 + (3.0 / 20.0) ** 2)
    expected_far = 1.0 / (1.0 + (10.0 / 20.0) ** 2)
    assert i_close == pytest.approx(expected_close, abs=1e-4)
    assert i_far == pytest.approx(expected_far, abs=1e-4)


def test_sensor_timing_and_latency_semantics() -> None:
    cfg = SensorConfig3D(update_rate_hz=25.0, latency_sec=0.04)
    lidar = SimulatedLiDAR3D(config=cfg)

    assert cfg.dt == pytest.approx(0.04)
    assert cfg.latency_sec == pytest.approx(0.04)

    # capture() records exact simulation acquisition epoch in cloud.timestamp
    t_meas = 3.14159
    cloud = lidar.capture(t_meas, [0, 0, 0], [1, 0, 0, 0])
    assert cloud.timestamp == pytest.approx(t_meas)


# ============================================================================
# 9. Performance Throughput & Scalability Benchmark
# ============================================================================


def test_sensor_throughput_benchmark() -> None:
    """Verify execution throughput >= 50 Hz on the standard 256-ray workload."""
    obstacles = [
        ObstacleSphere3D(center=np.array([float(x), float(y), 2.0]), radius=0.8)
        for x in [3.0, 6.0, 9.0, 12.0, 15.0]
        for y in [-4.0, 4.0]
    ]
    cfg = SensorConfig3D(horizontal_beams=32, vertical_beams=8)  # 256 rays
    lidar = SimulatedLiDAR3D(config=cfg, seed=42)

    # Warmup
    for _ in range(5):
        lidar.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], obstacles)

    # Timed benchmark: 100 sweeps
    num_sweeps = 100
    t0 = time.perf_counter()
    for i in range(num_sweeps):
        lidar.capture(i * 0.05, np.zeros(3), [1.0, 0, 0, 0], obstacles)
    elapsed = time.perf_counter() - t0

    sweep_time_sec = elapsed / num_sweeps
    rate_hz = num_sweeps / elapsed

    # Phase V.1 / V.1.1 Acceptance Criteria: ray casting throughput >= 50 Hz (< 20 ms per sweep)
    assert sweep_time_sec < 0.020, f"Sweep time too slow: {sweep_time_sec * 1000:.2f} ms"
    assert rate_hz >= 50.0, f"Rate below 50 Hz target: {rate_hz:.1f} Hz"


@pytest.mark.parametrize("num_rays", [64, 128, 256])
@pytest.mark.parametrize("num_obstacles", [5, 10])
def test_lidar_sweep_scalability_benchmark(num_rays: int, num_obstacles: int) -> None:
    """Benchmark LiDAR sweep latency and throughput across beam and obstacle counts."""
    # Configure beams
    if num_rays == 64:
        h_beams, v_beams = 16, 4
    elif num_rays == 128:
        h_beams, v_beams = 32, 4
    else:
        h_beams, v_beams = 32, 8

    cfg = SensorConfig3D(horizontal_beams=h_beams, vertical_beams=v_beams)
    lidar = SimulatedLiDAR3D(config=cfg, seed=101)

    # Build obstacles
    all_obs = [
        ObstacleSphere3D(center=np.array([float(x), float(y), 2.0]), radius=0.7)
        for x in [3.0, 6.0, 9.0, 12.0, 15.0]
        for y in [-3.0, 3.0]
    ]
    obstacles = all_obs[:num_obstacles]

    # Warmup sweeps
    for _ in range(10):
        lidar.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], obstacles)

    # Timed measurement sweeps
    num_measured_sweeps = 50
    latencies_ms: List[float] = []
    for i in range(num_measured_sweeps):
        t_start = time.perf_counter()
        lidar.capture(i * 0.05, np.zeros(3), [1.0, 0, 0, 0], obstacles)
        latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    median_ms = float(np.median(latencies_arr))
    p95_ms = float(np.percentile(latencies_arr, 95))
    p99_ms = float(np.percentile(latencies_arr, 99))
    throughput_hz = 1000.0 / median_ms if median_ms > 0 else float("inf")

    # Record platform and performance metadata
    print(
        f"\n[BENCHMARK] Python: {platform.python_version()} | NumPy: {np.__version__} | "
        f"Rays: {num_rays:3d} | Obstacles: {num_obstacles:2d} | "
        f"Median: {median_ms:.3f} ms | p95: {p95_ms:.3f} ms | p99: {p99_ms:.3f} ms | "
        f"Throughput: {throughput_hz:.1f} Hz"
    )

    # Verify that even on largest workload, median latency is comfortably under 20 ms (50 Hz)
    assert median_ms < 20.0
