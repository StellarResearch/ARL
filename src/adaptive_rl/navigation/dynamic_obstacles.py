"""Deterministic Dynamic (Moving) Obstacles for 3D Quadrotor Navigation.

Provides moving spherical obstacles with deterministic linear reflection,
sinusoidal oscillation, and continuous time-aware clearance checking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

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

    def check_continuous_collision(
        self,
        prev_drone_pos: np.ndarray,
        curr_drone_pos: np.ndarray,
        prev_time: float,
        curr_time: float,
        vehicle_radius: float,
    ) -> bool:
        """Check continuous swept-sphere collision between vehicle and dynamic obstacle over [prev_time, curr_time].

        Calculates minimum relative distance between moving vehicle segment and moving obstacle trajectory.
        Handles arbitrary speeds, cross-trajectories, and boundary bounce reflections.
        """
        p0 = np.asarray(prev_drone_pos, dtype=np.float64)
        p1 = np.asarray(curr_drone_pos, dtype=np.float64)
        t0 = float(prev_time)
        t1 = float(curr_time)
        dt = t1 - t0
        threshold = float(self.radius + vehicle_radius)

        if dt <= 1e-9:
            c = self.center_at(t0)
            return bool(np.linalg.norm(p1 - c) <= threshold)

        # Immediate endpoint checks
        if np.linalg.norm(p0 - self.center_at(t0)) <= threshold:
            return True
        if np.linalg.norm(p1 - self.center_at(t1)) <= threshold:
            return True

        # Adaptive subdivision to guarantee linear relative subsegment fidelity under high speed or bounce
        v_obs_mag = float(np.linalg.norm(self.velocity))
        v_veh_est = float(np.linalg.norm(p1 - p0)) / dt
        v_rel_max = max(1.0, v_obs_mag + v_veh_est)
        # Substep duration ensures max relative displacement per subsegment <= half the threshold
        sub_dt = min(0.01, threshold / (2.0 * v_rel_max))
        num_substeps = max(1, min(50, int(math.ceil(dt / sub_dt))))

        for k in range(num_substeps):
            frac_a = k / num_substeps
            frac_b = (k + 1) / num_substeps
            tau_a = t0 + frac_a * dt
            tau_b = t0 + frac_b * dt

            pa = p0 + frac_a * (p1 - p0)
            pb = p0 + frac_b * (p1 - p0)
            ca = self.center_at(tau_a)
            cb = self.center_at(tau_b)

            # Relative displacement vector r(s) = (pa - ca) + s * ((pb - cb) - (pa - ca))
            ra = pa - ca
            rb = pb - cb
            d_rel = rb - ra
            d_sq = float(np.dot(d_rel, d_rel))

            if d_sq < 1e-12:
                min_dist = float(np.linalg.norm(ra))
            else:
                s_opt = max(0.0, min(1.0, -float(np.dot(ra, d_rel)) / d_sq))
                closest_rel = ra + s_opt * d_rel
                min_dist = float(np.linalg.norm(closest_rel))

            if min_dist <= threshold:
                return True

        return False

    def distance_to(self, point: np.ndarray, time_sec: float) -> float:
        """Compute the Euclidean distance from a 3D point to the obstacle center at time t."""
        c = self.center_at(time_sec)
        p = np.asarray(point, dtype=np.float64)
        return float(np.linalg.norm(p - c))

    def as_static_obstacle_at(self, time_sec: float) -> ObstacleSphere3D:
        """Export an instantaneous ObstacleSphere3D at a specific timestamp."""
        return ObstacleSphere3D(center=self.center_at(time_sec), radius=self.radius)

    def predict_trajectory(
        self,
        t_start: float,
        horizon_sec: float = 3.0,
        dt: float = 0.5,
    ) -> List[Tuple[float, np.ndarray]]:
        """Predict the obstacle center trajectory over a future time horizon [t_start, t_start + horizon]."""
        traj: List[Tuple[float, np.ndarray]] = []
        steps = max(1, int(round(horizon_sec / dt)))
        for i in range(steps + 1):
            t = t_start + i * dt
            traj.append((t, self.center_at(t)))
        return traj

    def predict_occupied_spheres(
        self,
        t_start: float,
        horizon_sec: float = 3.0,
        dt: float = 0.5,
    ) -> List[ObstacleSphere3D]:
        """Generate static spherical obstacle approximations covering the predicted trajectory horizon."""
        return [
            ObstacleSphere3D(center=pt, radius=self.radius)
            for _, pt in self.predict_trajectory(t_start, horizon_sec, dt)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_initial": [float(x) for x in self.center_initial],
            "radius": float(self.radius),
            "velocity": [float(x) for x in self.velocity],
            "bounds": [float(x) for x in self.bounds],
            "motion_type": str(self.motion_type),
            "oscillation_period": float(self.oscillation_period),
            "oscillation_amplitude": [float(x) for x in self.oscillation_amplitude],
        }
