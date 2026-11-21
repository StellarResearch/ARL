import numpy as np
import pytest

from adaptive_rl.perception.detection.obstacle_detection import (
    ObstacleDetector3D,
    ObstacleDetectorConfig,
)
from adaptive_rl.perception.sensors.data import SensorExtrinsics
from adaptive_rl.perception.sensors.lidar3d import PointCloud3D


def make_pointcloud(points, valid_mask=None, sensor_id="lidar_primary", timestamp=0.0):
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    if valid_mask is None:
        valid_mask = np.ones(N, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
    ranges = np.linalg.norm(points, axis=1).astype(np.float32)
    intensities = np.full(N, 0.5, dtype=np.float32)
    return PointCloud3D(
        timestamp=timestamp,
        frame_id="sensor",
        points=points,
        ranges=ranges,
        valid_mask=valid_mask,
        sensor_id=sensor_id,
        extrinsics=SensorExtrinsics.default_identity(),
        sensor_origin_w=None,
        intensities=intensities,
    )


def test_empty_pointcloud_returns_no_detections():
    pc = make_pointcloud(points=np.empty((0, 3)), valid_mask=np.empty((0,), dtype=bool))
    detector = ObstacleDetector3D()
    detections = detector.detect(pc)
    assert detections == []


def test_single_obstacle_detection():
    center = np.array([5.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    points = center + rng.normal(scale=0.05, size=(100, 3))
    pc = make_pointcloud(points)
    detector = ObstacleDetector3D(ObstacleDetectorConfig(clustering_tolerance=0.2))
    detections = detector.detect(pc)
    assert len(detections) == 1
    det = detections[0]
    assert np.allclose(det.centroid, center, atol=0.1)
    assert abs(det.radius - 1.0) < 0.2
    assert isinstance(det.detection_id, str)
    assert det.is_valid
    with pytest.raises(AttributeError):
        det.centroid = np.zeros(3)


def test_multiple_clusters_and_determinism():
    rng = np.random.default_rng(1)
    points_a = rng.normal(scale=0.05, size=(50, 3))
    points_b = np.array([3.0, 3.0, 3.0]) + rng.normal(scale=0.05, size=(60, 3))
    points = np.vstack([points_a, points_b])
    pc = make_pointcloud(points)
    config = ObstacleDetectorConfig(clustering_tolerance=0.3)
    detector = ObstacleDetector3D(config)
    detections1 = detector.detect(pc)
    detections2 = detector.detect(pc)
    assert len(detections1) == 2
    ids1 = [d.detection_id for d in detections1]
    ids2 = [d.detection_id for d in detections2]
    assert ids1 == ids2
    centroids = np.stack([d.centroid for d in detections1])
    assert any(np.linalg.norm(centroids - np.array([0.0, 0.0, 0.0]), axis=1) < 0.2)
    assert any(np.linalg.norm(centroids - np.array([3.0, 3.0, 3.0]), axis=1) < 0.2)


def test_invalid_input_raises():
    points = np.array([[np.nan, 0.0, 0.0]])
    detector = ObstacleDetector3D()
    with pytest.raises(ValueError):
        pc = make_pointcloud(points, valid_mask=[True])
        detector.detect(pc)
    points = np.zeros((2, 3))
    pc = make_pointcloud(points, valid_mask=[True])
    with pytest.raises(ValueError):
        detector.detect(pc)


def test_ground_truth_isolation():
    import inspect
    import re

    src = inspect.getsource(ObstacleDetector3D)
    forbidden = ["ObstacleSphere3D", "DynamicObstacleSphere3D"]
    for name in forbidden:
        assert not re.search(r"\b" + name + r"\b", src)
