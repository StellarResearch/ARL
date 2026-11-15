"""Deterministic Dynamic (Moving) Obstacles for 3D Quadrotor Navigation.

Provides moving spherical obstacles with deterministic linear reflection,
sinusoidal oscillation, and continuous time-aware clearance checking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, (int, float))
            or not math.isfinite(self.radius)
            or self.radius <= 0.0
        ):
            raise ValueError(
                f"Obstacle radius must be strictly positive and finite, got {self.radius!r}"
            )
        self.radius = float(self.radius)

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

        if self.motion_type == "oscillate":
            for i in range(3):
                b_min = self.radius
                b_max = self.bounds[i] - self.radius
                min_c = self.center_initial[i] - abs(self.oscillation_amplitude[i])
                max_c = self.center_initial[i] + abs(self.oscillation_amplitude[i])
                if min_c < b_min or max_c > b_max:
                    raise ValueError(
                        f"Oscillation envelope along axis {i} [{min_c}, {max_c}] exceeds "
                        f"permitted obstacle center region [{b_min}, {b_max}] for arena bounds {self.bounds} "
                        f"and radius {self.radius}"
                    )

    def max_speed(self) -> float:
        """Calculate the theoretical maximum translation speed of the obstacle."""
        if self.motion_type == "oscillate":
            omega = 2.0 * math.pi / self.oscillation_period
            amp_mag = float(np.linalg.norm(self.oscillation_amplitude))
            return omega * amp_mag
        return float(np.linalg.norm(self.velocity))

    def center_at(self, time_sec: float) -> np.ndarray:
        """Compute the deterministic 3D position of the obstacle at a given simulation time."""
        if (
            isinstance(time_sec, bool)
            or not isinstance(time_sec, (int, float))
            or not math.isfinite(time_sec)
        ):
            raise ValueError(f"time_sec must be a finite float, got {time_sec!r}")

        t = float(time_sec)
        if self.motion_type == "oscillate":
            omega = 2.0 * math.pi / self.oscillation_period
            offset = self.oscillation_amplitude * math.sin(omega * t)
            return self.center_initial + offset

        # Standard bounce motion: linear motion reflected off boundaries
        pos = self.center_initial + self.velocity * t
        reflected = np.zeros(3, dtype=np.float64)
        for i in range(3):
            b_min = self.radius
            b_max = self.bounds[i] - self.radius
            span = b_max - b_min
            # Normalized cycle in [0, 2*span]
            val = (pos[i] - b_min) % (2.0 * span)
            if val < 0.0:
                val += 2.0 * span
            if val <= span:
                reflected[i] = b_min + val
            else:
                reflected[i] = b_max - (val - span)
        return reflected

    def get_bounce_times(self, t0: float, t1: float) -> List[float]:
        """Identify all exact boundary reflection timestamps in (t0, t1) for bounce motion."""
        if self.motion_type != "bounce":
            return []
        t0_f = float(t0)
        t1_f = float(t1)
        if t1_f <= t0_f:
            return []

        bounce_times: List[float] = []
        for i in range(3):
            vi = float(self.velocity[i])
            if abs(vi) <= 1e-9:
                continue
            b_min = float(self.radius)
            b_max = float(self.bounds[i] - self.radius)
            span = b_max - b_min
            if span <= 1e-9:
                continue

            x0 = float(self.center_initial[i]) + vi * t0_f
            x1 = float(self.center_initial[i]) + vi * t1_f
            x_min = min(x0, x1)
            x_max = max(x0, x1)

            k_start = int(math.ceil((x_min - b_min) / span))
            k_end = int(math.floor((x_max - b_min) / span))

            for k in range(k_start, k_end + 1):
                wall_pos = b_min + k * span
                t_bounce = (wall_pos - float(self.center_initial[i])) / vi
                if t0_f + 1e-7 < t_bounce < t1_f - 1e-7:
                    bounce_times.append(float(t_bounce))

        if not bounce_times:
            return []

        bounce_times.sort()
        unique_times: List[float] = [bounce_times[0]]
        for t in bounce_times[1:]:
            if t - unique_times[-1] > 1e-7:
                unique_times.append(t)

        return unique_times

    def _check_subsegment_collision_linear(
        self,
        p0_arr: np.ndarray,
        p1_arr: np.ndarray,
        t0: float,
        t1: float,
        threshold: float,
        veh_acc_max: float,
        veh_interval_dt: Optional[float] = None,
    ) -> bool:
        """Evaluate continuous collision across a single interval [t0, t1] where obstacle motion is non-reflecting."""
        dt = t1 - t0
        if dt <= 1e-9:
            c = self.center_at(t0)
            return bool(
                np.linalg.norm(p0_arr - c) <= threshold or np.linalg.norm(p1_arr - c) <= threshold
            )

        # Immediate endpoint checks
        if np.linalg.norm(p0_arr - self.center_at(t0)) <= threshold:
            return True
        if np.linalg.norm(p1_arr - self.center_at(t1)) <= threshold:
            return True

        # Adaptive subdivision based on valid maximum speed
        v_obs_mag = self.max_speed()
        v_veh_est = float(np.linalg.norm(p1_arr - p0_arr)) / dt
        v_rel_max = max(1.0, v_obs_mag + v_veh_est)

        # Substep duration ensures relative displacement per subsegment <= half the threshold
        sub_dt = min(0.01, threshold / (2.0 * v_rel_max))
        max_substeps = 500
        desired_substeps = max(1, int(math.ceil(dt / sub_dt)))
        num_substeps = min(max_substeps, desired_substeps)

        # Theoretical curvature error bounds:
        # 1. Obstacle curvature bound (over subsegment tau_step):
        tau_step = dt / num_substeps
        if self.motion_type == "oscillate":
            omega = 2.0 * math.pi / self.oscillation_period
            amp_mag = float(np.linalg.norm(self.oscillation_amplitude))
            eps_obs = (1.0 / 8.0) * (tau_step**2) * (omega**2) * amp_mag
        else:
            eps_obs = 0.0

        # 2. Vehicle curvature bound:
        # If genuine intermediate trajectory was provided, veh_interval_dt is the substep dt;
        # if only endpoints were provided, veh_interval_dt is the full interval duration dt.
        effective_veh_dt = dt if veh_interval_dt is None else float(veh_interval_dt)
        eps_veh = (1.0 / 8.0) * (effective_veh_dt**2) * veh_acc_max

        effective_threshold = threshold + eps_obs + eps_veh

        for k in range(num_substeps):
            frac_a = k / num_substeps
            frac_b = (k + 1) / num_substeps
            tau_a = t0 + frac_a * dt
            tau_b = t0 + frac_b * dt

            pa = p0_arr + frac_a * (p1_arr - p0_arr)
            pb = p0_arr + frac_b * (p1_arr - p0_arr)
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

        if desired_substeps > max_substeps:
            mid_drone = 0.5 * (p0_arr + p1_arr)
            mid_obs = 0.5 * (self.center_at(t0) + self.center_at(t1))
            envelope_dist = float(np.linalg.norm(mid_drone - mid_obs))
            swept_bound = effective_threshold + 0.5 * v_rel_max * dt
            if envelope_dist <= swept_bound:
                return True

        return False

    def _check_subsegment_collision(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        t0: float,
        t1: float,
        threshold: float,
        veh_acc_max: float = 35.0,
        veh_interval_dt: Optional[float] = None,
    ) -> bool:
        """Evaluate continuous swept-sphere collision across [t0, t1], splitting at bounce events."""
        t0_f = float(t0)
        t1_f = float(t1)
        if t1_f < t0_f:
            raise ValueError(f"t1 ({t1_f}) must be >= t0 ({t0_f})")

        p0_arr = np.asarray(p0, dtype=np.float64)
        p1_arr = np.asarray(p1, dtype=np.float64)
        dt = t1_f - t0_f

        if dt <= 1e-9:
            c = self.center_at(t0_f)
            return bool(
                np.linalg.norm(p0_arr - c) <= threshold or np.linalg.norm(p1_arr - c) <= threshold
            )

        # Identify boundary reflections and split interval at reflection events
        bounce_times = self.get_bounce_times(t0_f, t1_f)
        if not bounce_times:
            return self._check_subsegment_collision_linear(
                p0_arr, p1_arr, t0_f, t1_f, threshold, veh_acc_max, veh_interval_dt
            )

        # Multiple subintervals split at exact bounce events
        split_times = [t0_f] + bounce_times + [t1_f]
        for m in range(len(split_times) - 1):
            ta = split_times[m]
            tb = split_times[m + 1]
            fa = (ta - t0_f) / dt
            fb = (tb - t0_f) / dt
            pa = p0_arr + fa * (p1_arr - p0_arr)
            pb = p0_arr + fb * (p1_arr - p0_arr)
            if self._check_subsegment_collision_linear(
                pa, pb, ta, tb, threshold, veh_acc_max, veh_interval_dt
            ):
                return True

        return False

    def check_continuous_collision(
        self,
        prev_drone_pos: np.ndarray,
        curr_drone_pos: np.ndarray,
        prev_time: float,
        curr_time: float,
        vehicle_radius: float,
        intermediate_drone_positions: Optional[List[np.ndarray]] = None,
        veh_acc_max: float = 35.0,
    ) -> bool:
        """Check continuous swept-sphere collision between vehicle and dynamic obstacle over [prev_time, curr_time].

        Calculates relative clearance between vehicle trajectory and moving obstacle trajectory.
        Supports actual 6-DOF intermediate RK4 substep positions when available, and bounds
        trajectory curvature using physical acceleration limits (eps_veh <= 1/8 * a_max * dt^2).
        Fails closed on malformed or inconsistent intermediate trajectory data.
        """
        if (
            isinstance(prev_time, bool)
            or not isinstance(prev_time, (int, float))
            or not math.isfinite(prev_time)
        ):
            raise ValueError(f"prev_time must be a finite float, got {prev_time!r}")
        if (
            isinstance(curr_time, bool)
            or not isinstance(curr_time, (int, float))
            or not math.isfinite(curr_time)
        ):
            raise ValueError(f"curr_time must be a finite float, got {curr_time!r}")

        t0 = float(prev_time)
        t1 = float(curr_time)
        if t1 < t0:
            raise ValueError(f"curr_time ({t1}) must be >= prev_time ({t0})")

        if (
            isinstance(vehicle_radius, bool)
            or not isinstance(vehicle_radius, (int, float))
            or not math.isfinite(vehicle_radius)
            or vehicle_radius < 0.0
        ):
            raise ValueError(
                f"vehicle_radius must be a non-negative finite float, got {vehicle_radius!r}"
            )

        if (
            isinstance(veh_acc_max, bool)
            or not isinstance(veh_acc_max, (int, float))
            or not math.isfinite(veh_acc_max)
            or veh_acc_max < 0.0
        ):
            raise ValueError(
                f"veh_acc_max must be a non-negative finite float, got {veh_acc_max!r}"
            )

        threshold = float(self.radius + vehicle_radius)
        dt = t1 - t0

        p_start_check = np.asarray(prev_drone_pos, dtype=np.float64)
        p_end_check = np.asarray(curr_drone_pos, dtype=np.float64)

        if p_start_check.shape != (3,) or not np.all(np.isfinite(p_start_check)):
            raise ValueError(f"prev_drone_pos must be a finite 3D vector, got {prev_drone_pos}")
        if p_end_check.shape != (3,) or not np.all(np.isfinite(p_end_check)):
            raise ValueError(f"curr_drone_pos must be a finite 3D vector, got {curr_drone_pos}")

        if dt <= 1e-9:
            c = self.center_at(t0)
            return bool(
                np.linalg.norm(p_start_check - c) <= threshold
                or np.linalg.norm(p_end_check - c) <= threshold
            )

        if intermediate_drone_positions is not None:
            if not isinstance(intermediate_drone_positions, (list, tuple)):
                raise ValueError(
                    "intermediate_drone_positions must be a sequence of 3D coordinates"
                )
            if len(intermediate_drone_positions) < 2:
                raise ValueError("intermediate_drone_positions must contain at least 2 positions")

            validated_pts: List[np.ndarray] = []
            for idx, pt in enumerate(intermediate_drone_positions):
                arr = np.asarray(pt, dtype=np.float64)
                if arr.shape != (3,):
                    raise ValueError(
                        f"intermediate_drone_positions[{idx}] must have shape (3,), got {arr.shape}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"intermediate_drone_positions[{idx}] contains non-finite values: {arr}"
                    )
                validated_pts.append(arr)

            # Agreement with endpoints
            if np.linalg.norm(validated_pts[0] - p_start_check) > 1e-3:
                raise ValueError(
                    f"intermediate_drone_positions[0] {validated_pts[0]} does not match prev_drone_pos {p_start_check}"
                )
            if np.linalg.norm(validated_pts[-1] - p_end_check) > 1e-3:
                raise ValueError(
                    f"intermediate_drone_positions[-1] {validated_pts[-1]} does not match curr_drone_pos {p_end_check}"
                )

            num_segs = len(validated_pts) - 1
            for j in range(num_segs):
                p_a = validated_pts[j]
                p_b = validated_pts[j + 1]
                t_a = t0 + (j / num_segs) * dt
                t_b = t0 + ((j + 1) / num_segs) * dt
                sub_dt = t_b - t_a
                if self._check_subsegment_collision(
                    p_a,
                    p_b,
                    t_a,
                    t_b,
                    threshold,
                    veh_acc_max=veh_acc_max,
                    veh_interval_dt=sub_dt,
                ):
                    return True
            return False

        # If intermediate positions are not provided, evaluate chord with full-interval vehicle curvature bound
        return self._check_subsegment_collision(
            p_start_check,
            p_end_check,
            t0,
            t1,
            threshold,
            veh_acc_max=veh_acc_max,
            veh_interval_dt=dt,
        )

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

        Parameters
        ----------
        t_start : float
            Start simulation time for the prediction horizon (finite float).
        horizon_sec : float
            Prediction horizon duration in seconds (non-negative finite float).
        dt : float
            Nominal discrete sampling interval (strictly positive finite float).
            Note: The modern algorithm computes optimal continuous swept subdivisions
            based on v_max and 1.2 * R overlap; dt is validated and preserved for
            backwards compatibility.
        max_spheres : int
            Strict upper bound on the number of returned ObstacleSphere3D instances
            (strictly positive integer >= 1). If computational limits clamp the
            sphere count, the sphere radii are conservatively inflated to guarantee
            zero gaps in the predicted flight corridor.
        """
        if isinstance(max_spheres, bool) or not isinstance(max_spheres, int) or max_spheres <= 0:
            raise ValueError(
                f"max_spheres must be a strictly positive integer (>= 1), got {max_spheres!r}"
            )

        if (
            isinstance(t_start, bool)
            or not isinstance(t_start, (int, float))
            or not math.isfinite(t_start)
        ):
            raise ValueError(f"t_start must be a finite float, got {t_start!r}")

        if (
            isinstance(horizon_sec, bool)
            or not isinstance(horizon_sec, (int, float))
            or not math.isfinite(horizon_sec)
        ):
            raise ValueError(f"horizon_sec must be a finite float, got {horizon_sec!r}")

        if horizon_sec < 0.0:
            raise ValueError(f"horizon_sec must be non-negative, got {horizon_sec!r}")

        if (
            dt is None
            or isinstance(dt, bool)
            or not isinstance(dt, (int, float))
            or not math.isfinite(dt)
            or dt <= 0.0
        ):
            raise ValueError(f"dt must be a strictly positive finite float, got {dt!r}")

        t_start_f = float(t_start)
        horizon_f = float(horizon_sec)

        if horizon_f <= 0.0:
            return [ObstacleSphere3D(center=self.center_at(t_start_f), radius=self.radius)]

        v_max = self.max_speed()
        max_spacing = 1.2 * self.radius

        # If obstacle is essentially stationary, a single sphere at initial position suffices
        if v_max <= 1e-9:
            return [ObstacleSphere3D(center=self.center_at(t_start_f), radius=self.radius)]

        # Total distance upper bound across the entire prediction horizon
        # For any trajectory c(t), arc length s <= integral of ||c'(t)|| dt <= v_max * horizon_sec
        total_dist_bound = v_max * horizon_f

        # Required number of temporal steps to guarantee center spacing <= 1.2 * radius everywhere
        needed_steps = max(1, int(math.ceil(total_dist_bound / max_spacing)))

        if max_spheres == 1:
            t_mid = t_start_f + 0.5 * horizon_f
            eff_radius = self.radius + 0.5 * total_dist_bound
            return [ObstacleSphere3D(center=self.center_at(t_mid), radius=eff_radius)]

        actual_steps = min(needed_steps, max_spheres - 1)
        actual_steps = max(1, actual_steps)
        step_dt = horizon_f / actual_steps
        step_disp_bound = v_max * step_dt
        eff_radius = max(self.radius + 0.5 * step_disp_bound, step_disp_bound / 1.2)

        spheres: List[ObstacleSphere3D] = []
        for i in range(actual_steps + 1):
            t = t_start_f + (i / actual_steps) * horizon_f
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


@dataclass
class DivergentDynamicObstacleSphere3D(DynamicObstacleSphere3D):
    """Dynamic obstacle with deterministic trajectory divergence after a defined timestamp.

    Enables experimental evaluation of autonomous navigation under model prediction mismatch.
    Prior to divergence_time, the actual trajectory exactly matches the unperturbed prediction.
    At or after divergence_time, the obstacle transitions deterministically to a new velocity.
    """

    divergence_time: float = 0.0
    divergence_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.divergence_time = float(self.divergence_time)
        if not math.isfinite(self.divergence_time) or self.divergence_time < 0.0:
            raise ValueError(
                f"divergence_time must be finite and >= 0.0, got {self.divergence_time}"
            )
        self.divergence_velocity = np.asarray(self.divergence_velocity, dtype=np.float64)
        if self.divergence_velocity.shape != (3,) or not np.all(
            np.isfinite(self.divergence_velocity)
        ):
            raise ValueError(
                f"divergence_velocity must be a finite 3D vector, got {self.divergence_velocity}"
            )

    def max_speed(self) -> float:
        """Calculate the maximum translation speed across pre- and post-divergence regimes."""
        v_pre = super().max_speed()
        v_post = float(np.linalg.norm(self.divergence_velocity))
        return max(v_pre, v_post)

    def predicted_center_at(self, time_sec: float) -> np.ndarray:
        """Nominal center position as predicted before divergence occurred."""
        return super().center_at(time_sec)

    def center_at(self, time_sec: float) -> np.ndarray:
        """Actual center position including deterministic trajectory divergence."""
        t = float(time_sec)
        if t <= self.divergence_time:
            return super().center_at(t)

        # Baseline position at the moment divergence initiates
        p_div = super().center_at(self.divergence_time)
        dt_post = t - self.divergence_time
        pos = p_div + self.divergence_velocity * dt_post

        # Reflect cleanly within arena bounds
        reflected = np.zeros(3, dtype=np.float64)
        for i in range(3):
            b_min = self.radius
            b_max = self.bounds[i] - self.radius
            span = b_max - b_min
            if span <= 1e-9:
                reflected[i] = b_min
                continue
            norm_c = (pos[i] - b_min) % (2.0 * span)
            if norm_c < 0.0:
                norm_c += 2.0 * span
            if norm_c <= span:
                reflected[i] = b_min + norm_c
            else:
                reflected[i] = b_max - (norm_c - span)
        return reflected

    def prediction_error(self, time_sec: float) -> float:
        """Euclidean distance between actual obstacle position and unperturbed prediction."""
        p_actual = self.center_at(time_sec)
        p_pred = self.predicted_center_at(time_sec)
        return float(np.linalg.norm(p_actual - p_pred))

    def as_predicted_obstacle(self) -> DynamicObstacleSphere3D:
        """Return an unperturbed DynamicObstacleSphere3D matching initial prediction."""
        return DynamicObstacleSphere3D(
            center_initial=self.center_initial.copy(),
            radius=self.radius,
            velocity=self.velocity.copy(),
            bounds=self.bounds,
            motion_type=self.motion_type,
            oscillation_period=self.oscillation_period,
            oscillation_amplitude=self.oscillation_amplitude.copy(),
        )

    def as_updated_obstacle(self, current_time: float) -> DynamicObstacleSphere3D:
        """Return an updated DynamicObstacleSphere3D fitted to post-divergence observation."""
        curr_pos = self.center_at(current_time)
        curr_vel = (
            self.divergence_velocity if current_time >= self.divergence_time else self.velocity
        )
        return DynamicObstacleSphere3D(
            center_initial=curr_pos - curr_vel * current_time,
            radius=self.radius,
            velocity=curr_vel.copy(),
            bounds=self.bounds,
            motion_type="bounce",
        )
