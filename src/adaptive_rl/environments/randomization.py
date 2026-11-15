"""Domain Randomization and Environmental Robustness Framework for Aerial Simulation.

Provides configurable parameter randomization for quadrotor mass, moments of inertia,
motor coefficients, damping, aerodynamic wind, and IMU sensor uncertainty.
Ensures reproducible seeding, strict train/evaluation split isolation, and verifiable configs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from adaptive_rl.environments.drone_6dof import DroneDynamics6DOF


@dataclass
class RandomizationBounds:
    """Configurable range bounds for a single randomized parameter."""

    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class DomainRandomizationConfig:
    """Specification of domain randomization bounds for Sim-to-Real robustness testing."""

    mass_scale: Tuple[float, float] = (0.85, 1.15)
    inertia_scale: Tuple[float, float] = (0.80, 1.20)
    max_thrust_scale: Tuple[float, float] = (0.90, 1.10)
    torque_ratio_scale: Tuple[float, float] = (0.85, 1.15)
    linear_damping_scale: Tuple[float, float] = (0.70, 1.30)
    angular_damping_scale: Tuple[float, float] = (0.70, 1.30)
    motor_time_constant_range: Tuple[float, float] = (0.02, 0.05)
    wind_speed_max: float = 4.0  # Maximum wind magnitude (m/s) in arbitrary 3D direction
    imu_accel_noise_range: Tuple[float, float] = (0.02, 0.10)
    imu_gyro_noise_range: Tuple[float, float] = (0.002, 0.010)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainRandomizationConfig:
        return cls(**data)


class DomainRandomizer:
    """Samples and applies physical parameter shifts to 6-DOF simulation models."""

    def __init__(
        self,
        config: Optional[DomainRandomizationConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or DomainRandomizationConfig()
        self.rng = np.random.default_rng(seed)

    def sample_parameters(
        self,
        nominal_dynamics: Optional[DroneDynamics6DOF] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Sample a reproducible dictionary of randomized physical parameters."""
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        cfg = self.config

        base_mass = nominal_dynamics.mass if nominal_dynamics else 1.0
        base_inertia = (
            nominal_dynamics.inertia_diag if nominal_dynamics else np.array([0.01, 0.01, 0.02])
        )
        base_thrust = nominal_dynamics.max_thrust_per_motor if nominal_dynamics else 5.0
        base_c_tau = nominal_dynamics.c_tau if nominal_dynamics else 0.01
        base_lin_damp = nominal_dynamics.linear_damping if nominal_dynamics else 0.1
        base_ang_damp = nominal_dynamics.angular_damping if nominal_dynamics else 0.02

        # Sample multipliers
        mass_factor = float(rng.uniform(cfg.mass_scale[0], cfg.mass_scale[1]))
        inertia_factors = rng.uniform(cfg.inertia_scale[0], cfg.inertia_scale[1], size=3)
        thrust_factor = float(rng.uniform(cfg.max_thrust_scale[0], cfg.max_thrust_scale[1]))
        c_tau_factor = float(rng.uniform(cfg.torque_ratio_scale[0], cfg.torque_ratio_scale[1]))
        lin_damp_factor = float(
            rng.uniform(cfg.linear_damping_scale[0], cfg.linear_damping_scale[1])
        )
        ang_damp_factor = float(
            rng.uniform(cfg.angular_damping_scale[0], cfg.angular_damping_scale[1])
        )

        tau_motor = float(
            rng.uniform(cfg.motor_time_constant_range[0], cfg.motor_time_constant_range[1])
        )

        # Wind vector with random spherical direction and magnitude
        wind_mag = float(rng.uniform(0.0, cfg.wind_speed_max))
        wind_dir = rng.normal(0.0, 1.0, size=3)
        wind_dir_norm = float(np.linalg.norm(wind_dir))
        if wind_dir_norm > 1e-12:
            wind_vector = (wind_dir / wind_dir_norm) * wind_mag
        else:
            wind_vector = np.zeros(3, dtype=np.float64)

        return {
            "mass": round(base_mass * mass_factor, 4),
            "inertia": [round(float(x), 5) for x in (base_inertia * inertia_factors)],
            "max_thrust_per_motor": round(base_thrust * thrust_factor, 4),
            "torque_to_thrust_ratio": round(base_c_tau * c_tau_factor, 5),
            "linear_damping": round(base_lin_damp * lin_damp_factor, 4),
            "angular_damping": round(base_ang_damp * ang_damp_factor, 5),
            "motor_time_constant": round(tau_motor, 4),
            "wind_vector": [round(float(x), 4) for x in wind_vector],
            "imu_accel_noise": round(
                float(rng.uniform(cfg.imu_accel_noise_range[0], cfg.imu_accel_noise_range[1])), 4
            ),
            "imu_gyro_noise": round(
                float(rng.uniform(cfg.imu_gyro_noise_range[0], cfg.imu_gyro_noise_range[1])), 5
            ),
        }

    def apply_to_dynamics(
        self,
        dynamics: DroneDynamics6DOF,
        params: Dict[str, Any],
    ) -> None:
        """Apply sampled parameters directly to an active DroneDynamics6DOF instance."""
        dynamics.mass = float(params.get("mass", dynamics.mass))
        if "inertia" in params:
            dynamics.inertia_diag = np.array(params["inertia"], dtype=np.float64)
            dynamics.inv_inertia = 1.0 / dynamics.inertia_diag
        dynamics.max_thrust_per_motor = float(
            params.get("max_thrust_per_motor", dynamics.max_thrust_per_motor)
        )
        dynamics.c_tau = float(params.get("torque_to_thrust_ratio", dynamics.c_tau))
        dynamics.linear_damping = float(params.get("linear_damping", dynamics.linear_damping))
        dynamics.angular_damping = float(params.get("angular_damping", dynamics.angular_damping))
        if "wind_vector" in params:
            dynamics.wind_vector = np.array(params["wind_vector"], dtype=np.float64)
        if "motor_time_constant" in params:
            dynamics.motor_time_constant = float(params["motor_time_constant"])
        dynamics.hover_thrust_per_motor = (dynamics.mass * dynamics.gravity) / 4.0

    def export_config(self, filepath: Path | str) -> Path:
        """Export domain randomization configuration to JSON."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        return target
