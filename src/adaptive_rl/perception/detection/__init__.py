# Detection package for Phase V.2

"""Phase V.2: 3D Obstacle Detection

Provides immutable data structures and a detector that operates solely on
`PointCloud3D` observations.

The implementation follows the architecture described in
`docs/PHASE_V_PERCEPTION_ARCHITECTURE.md`.
"""

from .obstacle_detection import (
    ObstacleDetection3D,
    ObstacleDetectorConfig,
    ObstacleDetector3D,
)
