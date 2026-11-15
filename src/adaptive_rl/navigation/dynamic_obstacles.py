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
        self.radius = float(self.radius)
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError(
                f"Obstacle radius must be strictly positive and finite, got {self.radius}"
            )

        if self.motion_type not in ("bounce", "oscillate"):
            raise ValueError(
                f"Unsupported motion_type: '{self.motion_type}'. Must be 'bounce' or 'oscillate'"
            )

        self.oscillation_period = float(self.oscillation_period)
        if not math.isfinite(self.oscillation_period) or self.oscillation_period <= 0.0:
            raise ValueError(
                f"Oscillation period must be strictly positive and finite, got {self.oscillation_period}"
            )

        self.center_initial = np.asarray(self.center_initial, dtype=np.float64)
        if self.center_initial.shape != (3,) or not np.all(np.isfinite(self.center_initial)):
            raise ValueError(
                f"center_initial must be a finite 3D vector, got {self.center_initial}"
            )

        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.velocity.shape != (3,) or not np.all(np.isfinite(self.velocity)):
            raise ValueError(f"velocity must be a finite 3D vector, got {self.velocity}")

        self.oscillation_amplitude = np.asarray(self.oscillation_amplitude, dtype=np.float64)
        if self.oscillation_amplitude.shape != (3,) or not np.all(
            np.isfinite(self.oscillation_amplitude)
        ):
            raise ValueError(
                f"oscillation_amplitude must be a finite 3D vector, got {self.oscillation_amplitude}"
            )

        if len(self.bounds) != 3 or any(
            not math.isfinite(b) or b <= 2.0 * self.radius for b in self.bounds
        ):
            raise ValueError(
                f"Arena bounds must all exceed 2 * radius ({2.0 * self.radius}), got {self.bounds}"
            )

    def max_speed(self) -> float:
        """Calculate the theoretical maximum translation speed of the obstacle."""
        if self.motion_type == "oscillate":
            omega = 2.0 * math.pi / max(0.1, self.oscillation_period)
            amp_mag = float(np.linalg.norm(self.oscillation_amplitude))
            return omega * amp_mag
        return float(np.linalg.norm(self.velocity))

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
        Conservative collision evaluation accounts for secant chord approximation errors and high speeds.
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

        # Adaptive subdivision based on valid maximum speed
        v_obs_mag = self.max_speed()
        v_veh_est = float(np.linalg.norm(p1 - p0)) / dt
        v_rel_max = max(1.0, v_obs_mag + v_veh_est)

        # Substep duration ensures relative displacement per subsegment <= half the threshold
        sub_dt = min(0.01, threshold / (2.0 * v_rel_max))
        max_substeps = 500
        desired_substeps = max(1, int(math.ceil(dt / sub_dt)))
        num_substeps = min(max_substeps, desired_substeps)

        # Theoretical secant approximation error bound eps_sec over subsegment tau_step
        tau_step = dt / num_substeps
        if self.motion_type == "oscillate":
            omega = 2.0 * math.pi / max(0.1, self.oscillation_period)
            amp_mag = float(np.linalg.norm(self.oscillation_amplitude))
            eps_sec = (1.0 / 8.0) * (tau_step**2) * (omega**2) * amp_mag
        else:
            eps_sec = v_obs_mag * tau_step

        effective_threshold = threshold + eps_sec

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

            if min_dist <= effective_threshold:
                return True

        # When computational limits are reached (desired_substeps > max_substeps),
        # fail conservatively if swept bounding volumes intersect.
        if desired_substeps > max_substeps:
            mid_drone = 0.5 * (p0 + p1)
            mid_obs = 0.5 * (self.center_at(t0) + self.center_at(t1))
            envelope_dist = float(np.linalg.norm(mid_drone - mid_obs))
            swept_bound = threshold + 0.5 * v_rel_max * dt
            if envelope_dist <= swept_bound:
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
        max_spheres: int = 2000,
    ) -> List[ObstacleSphere3D]:
        """Generate static spherical obstacle approximations covering the predicted continuous swept volume.

        Guarantees that consecutive spheres overlap with spacing <= 1.2 * radius,
        leaving zero gaps in the predicted flight corridor.
        Uses a mathematically derived upper bound on the distance travelled across the horizon
        (v_max * horizon_sec) rather than discrete endpoint displacement.
        """
        t_start = float(t_start)
        horizon_sec = float(horizon_sec)
        if horizon_sec <= 0.0:
            return [ObstacleSphere3D(center=self.center_at(t_start), radius=self.radius)]

        v_max = self.max_speed()
        max_spacing = 1.2 * self.radius

        # If obstacle is essentially stationary, a single sphere at initial position suffices
        if v_max <= 1e-9:
            return [ObstacleSphere3D(center=self.center_at(t_start), radius=self.radius)]

        # Total distance upper bound across the entire prediction horizon
        # For any trajectory c(t), arc length s <= integral of ||c'(t)|| dt <= v_max * horizon_sec
        total_dist_bound = v_max * horizon_sec

        # Required number of temporal steps to guarantee center spacing <= 1.2 * radius everywhere
        needed_steps = max(1, int(math.ceil(total_dist_bound / max_spacing)))

        # Handle computational limits conservatively:
        # If needed_steps exceeds max_spheres, we clamp the number of evaluations, but we
        # CONSERVATIVELY inflate the sphere radius so that adjacent spheres still overlap
        # without leaving any gaps in the predicted flight corridor.
        if needed_steps > max_spheres:
            actual_steps = max_spheres
            step_dt = horizon_sec / actual_steps
            step_disp_bound = v_max * step_dt
            eff_radius = max(self.radius, step_disp_bound / 1.2)
        else:
            actual_steps = needed_steps
            eff_radius = self.radius

        spheres: List[ObstacleSphere3D] = []
        for i in range(actual_steps + 1):
            t = t_start + (i / actual_steps) * horizon_sec
            spheres.append(ObstacleSphere3D(center=self.center_at(t), radius=eff_radius))

        return spheres

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
