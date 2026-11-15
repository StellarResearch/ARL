"""Deterministic Dynamic (Moving) Obstacles for 3D Quadrotor Navigation.

Provides moving spherical obstacles with deterministic linear reflection,
sinusoidal oscillation, and continuous time-aware clearance checking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D


@dataclass
class DynamicObstacleSphere3D:
    """A spherical obstacle that translates deterministically through the 3D flight arena."""

    center_initial: np.ndarray
    radius: float
    velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0)
    motion_type: str = "bounce"  # "bounce" or "oscillate"
    oscillation_period: float = 6.0
    oscillation_amplitude: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 4.0, 0.0], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        self.center_initial = np.asarray(self.center_initial, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.oscillation_amplitude = np.asarray(self.oscillation_amplitude, dtype=np.float64)

    def center_at(self, time_sec: float) -> np.ndarray:
        """Compute the deterministic 3D position of the obstacle at a given simulation time."""
        t = float(time_sec)
        if self.motion_type == "oscillate":
            omega = 2.0 * math.pi / max(0.1, self.oscillation_period)
            offset = self.oscillation_amplitude * math.sin(omega * t)
            return self.center_initial + offset

        # Standard bounce motion: linear motion reflected off boundaries
        pos = self.center_initial + self.velocity * t
        reflected = np.zeros(3, dtype=np.float64)
        for i in range(3):
            b_min = self.radius
            b_max = self.bounds[i] - self.radius
            span = max(0.1, b_max - b_min)
            # Normalized cycle in [0, 2*span]
            val = (pos[i] - b_min) % (2.0 * span)
            if val < 0.0:
                val += 2.0 * span
            if val <= span:
                reflected[i] = b_min + val
            else:
                reflected[i] = b_max - (val - span)
        return reflected

    def distance_to(self, point: np.ndarray, time_sec: float) -> float:
        """Compute the Euclidean distance from a 3D point to the obstacle center at time t."""
        c = self.center_at(time_sec)
        p = np.asarray(point, dtype=np.float64)
        return float(np.linalg.norm(p - c))

    def as_static_obstacle_at(self, time_sec: float) -> ObstacleSphere3D:
        """Export an instantaneous ObstacleSphere3D at a specific timestamp."""
        return ObstacleSphere3D(center=self.center_at(time_sec), radius=self.radius)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_initial": [round(float(x), 4) for x in self.center_initial],
            "radius": round(float(self.radius), 4),
            "velocity": [round(float(x), 4) for x in self.velocity],
            "motion_type": self.motion_type,
            "bounds": list(self.bounds),
        }
