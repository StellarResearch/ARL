"""Simulated 3D LiDAR sensor model.

Implements ADR-002, ADR-003, and ADR-006:
Vectorized ray-sphere and ray-box intersection against simulation geometry,
adding Gaussian range error, statistical dropout, and extrinsic transformation,
outputting strongly typed, immutable PointCloud3D observations with zero ground-truth leakage.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.environments.drone_6dof import quaternion_to_rotation_matrix
from adaptive_rl.perception.sensors.base import BaseSensor3D
from adaptive_rl.perception.sensors.data import PointCloud3D, SensorConfig3D


class SimulatedLiDAR3D(BaseSensor3D):
    """Vectorized simulated 3D LiDAR sensor generating timestamped point clouds."""

    def __init__(
        self,
        config: Optional[SensorConfig3D] = None,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(config=config)

        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Precompute sensor-frame ray unit vectors (FLU frame)
        self._ray_directions_s = self._generate_sensor_ray_directions()
        self._ray_directions_s.flags.writeable = False

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the internal random number generator for deterministic replay."""
        self.rng = np.random.default_rng(seed)

    @property
    def ray_directions_s(self) -> np.ndarray:
        """Read-only array of (N, 3) unit beam directions in sensor frame (FLU)."""
        return self._ray_directions_s

    def _generate_sensor_ray_directions(self) -> np.ndarray:
        """Generate (N, 3) normalized ray unit vectors in sensor FLU frame.

        Azimuth (theta): 0 rad is boresight (+X_S), +pi/2 is left (+Y_S), -pi/2 is right (-Y_S).
        Elevation (phi): 0 rad is horizontal, +pi/2 is up (+Z_S), -pi/2 is down (-Z_S).
        """
        h_beams = self.config.horizontal_beams
        v_beams = self.config.vertical_beams

        if self.config.horizontal_fov_deg >= 360.0:
            # Omnidirectional 360-deg azimuth starting at boresight (0 rad = +X_S)
            azimuths = np.linspace(0.0, 2.0 * np.pi, h_beams, endpoint=False)
        else:
            half_h = math.radians(self.config.horizontal_fov_deg) / 2.0
            if h_beams == 1:
                azimuths = np.array([0.0], dtype=np.float64)
            else:
                azimuths = np.linspace(-half_h, half_h, h_beams, endpoint=True)

        half_v = math.radians(self.config.vertical_fov_deg) / 2.0
        if v_beams == 1:
            elevations = np.array([0.0], dtype=np.float64)
        else:
            elevations = np.linspace(-half_v, half_v, v_beams, endpoint=True)

        # Build lattice (v_beams, h_beams)
        rays: List[List[float]] = []
        for phi in elevations:
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            for theta in azimuths:
                x = cos_phi * math.cos(theta)
                y = cos_phi * math.sin(theta)
                z = sin_phi
                rays.append([x, y, z])

        arr = np.array(rays, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # Numerical guard against division by zero
        norms = np.where(norms < 1e-12, 1.0, norms)
        return arr / norms

    def capture(
        self,
        timestamp: float,
        drone_pos: np.ndarray | Sequence[float],
        drone_orientation: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        obstacles: Optional[Sequence[Any]] = None,
        bounds: Optional[Tuple[float, float, float]] = None,
    ) -> PointCloud3D:
        """Render a 3D LiDAR point cloud observation at the given epoch.

        Internally simulates ray-casting against spherical obstacles and arena
        bounding walls using vectorized geometry. Outputs an immutable PointCloud3D.
        """
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"timestamp must be non-negative and finite, got {timestamp}")

        pos_w = np.asarray(drone_pos, dtype=np.float64)
        if pos_w.shape != (3,) or not np.all(np.isfinite(pos_w)):
            raise ValueError(f"drone_pos must be a finite 3D vector, got {drone_pos}")

        rot_arr = np.asarray(drone_orientation, dtype=np.float64)
        if rot_arr.shape == (4,):
            r_b_to_w = quaternion_to_rotation_matrix(rot_arr)
        elif rot_arr.shape == (3, 3):
            r_b_to_w = rot_arr
        else:
            raise ValueError(
                f"drone_orientation must be quaternion (4,) or rotation matrix (3, 3), got {rot_arr.shape}"
            )

        # 1. Compute sensor origin in world frame ENU: p_S^W = p_drone^W + R_B^W * t_B
        t_b = self.extrinsics.translation_b
        r_s_to_b = self.extrinsics.rotation_s_to_b
        sensor_origin_w = pos_w + r_b_to_w @ t_b

        # 2. Transform sensor ray directions to world frame: d_W = R_B^W * R_S^B * d_S
        r_total = r_b_to_w @ r_s_to_b  # (3, 3)
        rays_w = self._ray_directions_s @ r_total.T  # (N, 3)

        num_rays = self._ray_directions_s.shape[0]

        # 3. Vectorized Ray-Sphere Intersections
        t_spheres = np.full(num_rays, np.inf, dtype=np.float64)
        if obstacles:
            centers: List[np.ndarray] = []
            radii: List[float] = []
            for obs in obstacles:
                # Support static ObstacleSphere3D, DynamicObstacleSphere3D, or duck-typed obstacles
                if hasattr(obs, "center_at") and callable(obs.center_at):
                    c = obs.center_at(timestamp)
                elif hasattr(obs, "center"):
                    c = obs.center
                else:
                    continue

                r = float(obs.radius) if hasattr(obs, "radius") else 0.0
                if r > 0.0:
                    centers.append(np.asarray(c, dtype=np.float64))
                    radii.append(r)

            if centers:
                c_arr = np.array(centers, dtype=np.float64)  # (M, 3)
                r_arr = np.array(radii, dtype=np.float64)  # (M,)
                num_obs = c_arr.shape[0]

                # Vector from obstacle center to sensor origin: m_j = o - c_j
                m_vecs = sensor_origin_w[np.newaxis, :] - c_arr  # (M, 3)
                # b_ij = d_i . m_j = (rays_w @ m_vecs.T)
                b_mat = rays_w @ m_vecs.T  # (N, M)
                # c_const_j = ||m_j||^2 - r_j^2
                c_const = np.sum(m_vecs * m_vecs, axis=1) - r_arr * r_arr  # (M,)

                # Discriminant: Delta_ij = b_ij^2 - c_const_j
                disc = b_mat * b_mat - c_const[np.newaxis, :]  # (N, M)

                # Initialize distances matrix
                dist_mat = np.full((num_rays, num_obs), np.inf, dtype=np.float64)

                # Condition 1: Origin inside sphere (c_const <= 0)
                inside_mask = c_const <= 0.0
                if np.any(inside_mask):
                    dist_mat[:, inside_mask] = 0.0

                # Condition 2: Origin outside, ray points toward sphere, discriminant >= 0
                outside_valid = (~inside_mask)[np.newaxis, :] & (b_mat <= 0.0) & (disc >= 0.0)
                if np.any(outside_valid):
                    # Distance to nearest entry intersection: t = -b - sqrt(disc)
                    sqrt_disc = np.sqrt(np.maximum(0.0, disc))
                    t_entry = -b_mat - sqrt_disc
                    # Filter positive distances
                    valid_t = outside_valid & (t_entry >= 0.0)
                    dist_mat[valid_t] = t_entry[valid_t]

                t_spheres = np.min(dist_mat, axis=1)

        # 4. Vectorized Ray-Box Intersections (Arena boundaries)
        t_box = np.full(num_rays, np.inf, dtype=np.float64)
        if bounds is not None:
            b_arr = np.asarray(bounds, dtype=np.float64)
            if b_arr.shape == (3,) and np.all(b_arr > 0.0):
                t_exit_axes = np.full((num_rays, 3), np.inf, dtype=np.float64)
                for axis in range(3):
                    d_axis = rays_w[:, axis]
                    pos_axis = sensor_origin_w[axis]
                    max_axis = b_arr[axis]

                    pos_dir = d_axis > 1e-7
                    neg_dir = d_axis < -1e-7

                    t_exit_axes[pos_dir, axis] = (max_axis - pos_axis) / d_axis[pos_dir]
                    t_exit_axes[neg_dir, axis] = (0.0 - pos_axis) / d_axis[neg_dir]

                # Filter non-negative exit times
                t_exit_axes[t_exit_axes < 0.0] = np.inf
                t_box = np.min(t_exit_axes, axis=1)

        # 5. Combine nearest true distances
        r_true = np.minimum(t_spheres, t_box)

        # 6. Apply physical detection limits, noise, and stochastic dropout
        min_r = self.config.min_range_m
        max_r = self.config.max_range_m
        noise_std = self.config.range_noise_std_m
        dropout_p = self.config.dropout_probability

        # Determine in-range candidates
        in_range_mask = (r_true >= min_r) & (r_true <= max_r)

        # Evaluate dropout probability
        dropout_roll = self.rng.uniform(0.0, 1.0, size=num_rays)
        not_dropped = dropout_roll >= dropout_p

        valid_mask = in_range_mask & not_dropped

        # Allocate output arrays
        pts_s = np.full((num_rays, 3), np.nan, dtype=np.float64)
        ranges = np.full(num_rays, np.nan, dtype=np.float32)
        intensities = np.zeros(num_rays, dtype=np.float32)

        # For rays that missed everything within max_range, set range to inf
        out_of_range_mask = r_true > max_r
        ranges[out_of_range_mask] = np.inf

        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            true_valid_ranges = r_true[valid_indices]

            # Add zero-mean Gaussian noise
            if noise_std > 0.0:
                noise = self.rng.normal(0.0, noise_std, size=len(valid_indices))
                measured_ranges = true_valid_ranges + noise
                # Physical range guard: do not allow measured range below min_range_m
                measured_ranges = np.maximum(min_r, measured_ranges)
            else:
                measured_ranges = true_valid_ranges

            # Check if noise pushed range beyond max_range
            beyond_max = measured_ranges > max_r
            if np.any(beyond_max):
                invalid_from_noise = valid_indices[beyond_max]
                valid_mask[invalid_from_noise] = False
                ranges[invalid_from_noise] = np.inf

            # Remaining valid rays
            remaining_valid = valid_mask[valid_indices]
            final_valid_indices = valid_indices[remaining_valid]
            final_ranges = measured_ranges[remaining_valid].astype(np.float32)

            ranges[final_valid_indices] = final_ranges
            # Compute 3D Cartesian coordinates in sensor frame: p_S = r * d_S
            pts_s[final_valid_indices] = (
                final_ranges[:, np.newaxis] * self._ray_directions_s[final_valid_indices]
            )

            # Synthetic echo intensity: inverse square falloff normalized to [0.5, 1.0]
            norm_r = final_ranges / max_r
            intensities[final_valid_indices] = (1.0 / (1.0 + norm_r * norm_r)).astype(
                np.float32
            )

        return PointCloud3D(
            timestamp=timestamp,
            frame_id="sensor",
            points=pts_s,
            ranges=ranges,
            valid_mask=valid_mask,
            sensor_id=self.config.sensor_id,
            extrinsics=self.extrinsics,
            sensor_origin_w=sensor_origin_w,
            intensities=intensities,
        )
