"""Unit tests for Phase V.1 Simulated 3D LiDAR and PointCloud foundation.

Tests ADR-002, ADR-003, and ADR-006:
- SensorExtrinsics validation, orthonormality checks, and array immutability.
- SensorConfig3D validation and defaults.
- PointCloud3D defensive copying, buffer immutability, shape validation, and frame transformations.
- SimulatedLiDAR3D ray casting, noise distribution, dropout statistics, boundary intersections,
  extrinsic mounting, dynamic obstacle support, and execution throughput (>= 50 Hz).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.perception.sensors.base import BaseSensor3D
from adaptive_rl.perception.sensors.data import (
    PointCloud3D,
    SensorConfig3D,
    SensorExtrinsics,
)
from adaptive_rl.perception.sensors.lidar3d import SimulatedLiDAR3D

# ============================================================================
# 1. SensorExtrinsics Tests
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
    with pytest.raises(ValueError, match="rotation_s_to_b must have shape"):
        SensorExtrinsics(np.zeros(3), np.eye(4))

    # Non-finite rotation
    with pytest.raises(ValueError, match="finite values"):
        bad_rot = np.eye(3)
        bad_rot[0, 0] = np.inf
        SensorExtrinsics(np.zeros(3), bad_rot)

    # Non-orthonormal matrix
    non_ortho = np.eye(3) * 2.0
    with pytest.raises(ValueError, match="must be orthonormal"):
        SensorExtrinsics(np.zeros(3), non_ortho)

    # Reflection matrix (det = -1)
    refl = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant \\+1"):
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
# 2. SensorConfig3D Tests
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


def test_sensor_config_validation() -> None:
    # Invalid rate
    with pytest.raises(ValueError, match="update_rate_hz"):
        SensorConfig3D(update_rate_hz=-1.0)

    # Invalid range bounds
    with pytest.raises(ValueError, match="Invalid range bounds"):
        SensorConfig3D(min_range_m=-0.1)
    with pytest.raises(ValueError, match="Invalid range bounds"):
        SensorConfig3D(min_range_m=30.0, max_range_m=20.0)

    # Invalid FOV
    with pytest.raises(ValueError, match="horizontal_fov_deg"):
        SensorConfig3D(horizontal_fov_deg=0.0)
    with pytest.raises(ValueError, match="horizontal_fov_deg"):
        SensorConfig3D(horizontal_fov_deg=370.0)
    with pytest.raises(ValueError, match="vertical_fov_deg"):
        SensorConfig3D(vertical_fov_deg=-10.0)
    with pytest.raises(ValueError, match="vertical_fov_deg"):
        SensorConfig3D(vertical_fov_deg=200.0)

    # Invalid beam counts
    with pytest.raises(ValueError, match="Beams must be >= 1"):
        SensorConfig3D(horizontal_beams=0)
    with pytest.raises(ValueError, match="Beams must be >= 1"):
        SensorConfig3D(vertical_beams=0)

    # Invalid noise & dropout
    with pytest.raises(ValueError, match="range_noise_std_m"):
        SensorConfig3D(range_noise_std_m=-0.01)
    with pytest.raises(ValueError, match="dropout_probability"):
        SensorConfig3D(dropout_probability=-0.1)
    with pytest.raises(ValueError, match="dropout_probability"):
        SensorConfig3D(dropout_probability=1.5)
    with pytest.raises(ValueError, match="latency_sec"):
        SensorConfig3D(latency_sec=-0.01)


# ============================================================================
# 3. PointCloud3D Tests
# ============================================================================


def test_point_cloud_immutability_and_defensive_copies() -> None:
    ext = SensorExtrinsics.default_identity()
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rngs = np.array([3.74, 8.77], dtype=np.float32)
    mask = np.array([True, True])

    cloud = PointCloud3D(
        timestamp=0.5,
        frame_id="sensor",
        points=pts,
        ranges=rngs,
        valid_mask=mask,
        sensor_id="lidar_01",
        extrinsics=ext,
    )

    # Ingestion defensive copy test
    pts[0, 0] = 999.0
    rngs[0] = 999.0
    mask[0] = False
    assert cloud.points[0, 0] == 1.0
    assert cloud.ranges[0] == pytest.approx(3.74, abs=1e-2)
    assert cloud.valid_mask[0] is np.True_

    # Read-only flag tests
    assert not cloud.points.flags.writeable
    assert not cloud.ranges.flags.writeable
    assert not cloud.valid_mask.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        cloud.points[0, 0] = 50.0
    with pytest.raises(ValueError, match="read-only"):
        cloud.ranges[0] = 50.0
    with pytest.raises(ValueError, match="read-only"):
        cloud.valid_mask[0] = False


def test_point_cloud_validation_rules() -> None:
    ext = SensorExtrinsics.default_identity()
    pts = np.array([[1.0, 2.0, 3.0]])
    rngs = np.array([3.74], dtype=np.float32)
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

    # Non-finite range on valid points
    bad_rngs = np.array([np.nan], dtype=np.float32)
    with pytest.raises(ValueError, match="finite and non-negative"):
        PointCloud3D(0.0, "sensor", pts, bad_rngs, mask, "lidar", ext)


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
    # R_B^W rotates [3.2, 0, 0]_B to [0, 3.2, 0]_W
    # p_W = [10.0, 5.0, 2.0] + [0, 3.2, 0] = [10.0, 8.2, 2.0]
    yaw = np.radians(90.0)
    q_yaw = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    drone_pos = np.array([10.0, 5.0, 2.0])

    cloud_world = cloud_sensor.to_world_frame(drone_pos=drone_pos, drone_orientation=q_yaw)
    assert cloud_world.frame_id == "world"
    assert np.allclose(cloud_world.valid_points, [[10.0, 8.2, 2.0]])
    assert cloud_world.sensor_origin_w is not None
    # Sensor origin in world: [10, 5, 2] + R_B^W * [0.2, 0, 0] = [10.0, 5.2, 2.0]
    assert np.allclose(cloud_world.sensor_origin_w, [10.0, 5.2, 2.0])


# ============================================================================
# 4. SimulatedLiDAR3D Ray-Casting & Obstacle Tests
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
    # Top layer is sin(+30 deg) = 0.5, bottom is sin(-30 deg) = -0.5
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
        range_noise_std_m=0.0,  # Zero noise
        dropout_probability=0.0,  # Zero dropout
    )
    lidar = SimulatedLiDAR3D(config=cfg)

    drone_pos = np.array([0.0, 0.0, 0.0])
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity orientation

    cloud = lidar.capture(
        timestamp=0.1,
        drone_pos=drone_pos,
        drone_orientation=drone_quat,
        obstacles=[obs],
    )

    assert cloud.timestamp == 0.1
    assert cloud.frame_id == "sensor"
    assert cloud.num_valid >= 1

    # Find the boresight beam [1.0, 0.0, 0.0]
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

    # Since drone faces North, boresight beam (+X_S) points towards [0, 5, 0] in world!
    boresight_idx = -1
    for i, r in enumerate(lidar.ray_directions_s):
        if np.allclose(r, [1.0, 0.0, 0.0], atol=1e-3):
            boresight_idx = i
            break
    assert boresight_idx != -1

    assert cloud.valid_mask[boresight_idx] is np.True_
    # In sensor frame, it is at +X_S = 4.0m
    assert np.allclose(cloud.points[boresight_idx], [4.0, 0.0, 0.0], atol=1e-4)

    # In world frame, it must project to [0.0, 4.0, 0.0]
    cloud_w = cloud.to_world_frame(drone_pos, q_yaw)
    assert np.allclose(cloud_w.points[boresight_idx], [0.0, 4.0, 0.0], atol=1e-4)


def test_arena_boundary_intersections() -> None:
    # Arena box: [20.0, 20.0, 10.0]
    # Drone at center [10.0, 10.0, 5.0] facing East (+X_W)
    bounds = (20.0, 20.0, 10.0)
    drone_pos = np.array([10.0, 10.0, 5.0])
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0])

    cfg = SensorConfig3D(
        horizontal_beams=4,
        vertical_beams=1,  # 4 cardinal beams: East, North, West, South
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
    # All cardinal directions from center [10, 10, 5] should hit walls at distance 10.0m
    for i in range(4):
        assert cloud.valid_mask[i] is np.True_
        assert cloud.ranges[i] == pytest.approx(10.0, abs=1e-4)


def test_out_of_range_and_blind_zone_rejection() -> None:
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
    # Boresight ray must be invalid and set to inf
    boresight_idx = 0  # will verify ray
    for i, r in enumerate(lidar.ray_directions_s):
        if np.allclose(r, [1.0, 0.0, 0.0], atol=1e-3):
            boresight_idx = i
            break
    assert not cloud_far.valid_mask[boresight_idx]
    assert np.isinf(cloud_far.ranges[boresight_idx])
    assert np.all(np.isnan(cloud_far.points[boresight_idx]))

    # 2. Close obstacle at 0.3m (inside blind zone < min_range_m)
    # Distance to surface: 0.8 - 0.5 = 0.3m
    close_obs = ObstacleSphere3D(center=np.array([0.8, 0.0, 0.0]), radius=0.5)
    cloud_close = lidar.capture(
        timestamp=0.0,
        drone_pos=np.zeros(3),
        drone_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        obstacles=[close_obs],
    )
    assert not cloud_close.valid_mask[boresight_idx]


def test_dynamic_obstacle_tracking_over_time() -> None:
    # Dynamic obstacle translating at 1.0 m/s along +Y inside arena bounds
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

    # Drone located at [0.0, 10.0, 5.0] facing East (+X_W)
    drone_pos = np.array([0.0, 10.0, 5.0])
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # At t = 0: obstacle center is at [5.0, 10.0, 5.0], boresight distance is 4.0m
    cloud_t0 = lidar.capture(0.0, drone_pos, drone_quat, [dyn_obs])
    # At t = 2: obstacle center is at [5.0, 12.0, 5.0]
    cloud_t2 = lidar.capture(2.0, drone_pos, drone_quat, [dyn_obs])

    # Find boresight beam
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

    # But rays pointing towards +Y detect it
    assert cloud_t2.num_valid >= 1


def test_range_noise_statistics() -> None:
    # Target at exactly 5.0m - 1.0m = 4.0m
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

    # Phase V.1 Acceptance Criteria: range noise matches target sigma within +-10%
    assert abs(mean_err) < 0.01  # zero-mean
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
    # Check within 2% margin
    assert empirical_p == pytest.approx(target_p, abs=0.02)


def test_seeded_determinism() -> None:
    obs = ObstacleSphere3D(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
    cfg = SensorConfig3D(range_noise_std_m=0.03, dropout_probability=0.05)

    lidar1 = SimulatedLiDAR3D(config=cfg, seed=999)
    lidar2 = SimulatedLiDAR3D(config=cfg, seed=999)

    c1 = lidar1.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], [obs])
    c2 = lidar2.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], [obs])

    assert np.all(c1.valid_mask == c2.valid_mask)
    # Valid ranges must be bit-exact
    v1 = c1.ranges[c1.valid_mask]
    v2 = c2.ranges[c2.valid_mask]
    assert np.all(v1 == v2)


def test_ground_truth_isolation() -> None:
    obs = ObstacleSphere3D(center=np.array([5.0, 0.0, 0.0]), radius=1.0)
    lidar = SimulatedLiDAR3D()

    cloud = lidar.capture(0.0, np.zeros(3), [1.0, 0, 0, 0], [obs])

    # Assert that PointCloud3D does NOT hold obstacle objects or true velocities
    assert not hasattr(cloud, "obstacles")
    assert not hasattr(cloud, "true_obstacle_centers")
    assert not hasattr(cloud, "obstacle_velocities")
    assert not hasattr(cloud, "obstacle_ids")


def test_simulated_lidar_non_identity_extrinsics() -> None:
    # Sensor mounted forward +0.2m along body X and pitched down 30 deg
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

    # Obstacle placed along pitched ray direction in front of drone
    # Sensor boresight ray in sensor frame is [1, 0, 0]
    # In body frame (and world frame for drone at origin with identity attitude):
    # d_W = rot_s_to_b @ [1, 0, 0] = [cos(-30 deg), 0, -sin(-30 deg)] = [sqrt(3)/2, 0, -0.5]
    # Origin in world: [0.2, 0, 0]
    # If obstacle center is placed at origin + 4.0 * d_W with radius 1.0m:
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
    # Measured distance must be 4.0 - 1.0 = 3.0m
    assert cloud.ranges[0] == pytest.approx(3.0, abs=1e-4)
    # Point in sensor frame: 3.0 * [1, 0, 0] = [3.0, 0.0, 0.0]
    assert np.allclose(cloud.points[0], [3.0, 0.0, 0.0], atol=1e-4)

    # Transform to world frame
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

    # Invalid drone_pos
    with pytest.raises(ValueError, match="drone_pos"):
        lidar.capture(0.0, np.zeros(4), [1.0, 0, 0, 0])
    with pytest.raises(ValueError, match="drone_pos"):
        lidar.capture(0.0, np.array([np.nan, 0, 0]), [1.0, 0, 0, 0])

    # Invalid orientation
    with pytest.raises(ValueError, match="drone_orientation"):
        lidar.capture(0.0, np.zeros(3), [1.0, 0, 0])


def test_sensor_throughput_benchmark() -> None:
    # 256 rays against 10 obstacles
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

    # Phase V.1 Acceptance Criteria: ray casting throughput >= 50 Hz (< 20 ms per sweep)
    assert sweep_time_sec < 0.020, f"Sweep time too slow: {sweep_time_sec * 1000:.2f} ms"
    assert rate_hz >= 50.0, f"Rate below 50 Hz target: {rate_hz:.1f} Hz"
    # 256 rays against 10 obstacles
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

    # Phase V.1 Acceptance Criteria: ray casting throughput >= 50 Hz (< 20 ms per sweep)
    assert sweep_time_sec < 0.020, f"Sweep time too slow: {sweep_time_sec * 1000:.2f} ms"
    assert rate_hz >= 50.0, f"Rate below 50 Hz target: {rate_hz:.1f} Hz"
