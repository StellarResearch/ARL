"""Physically Consistent 6-Axis Inertial Measurement Unit (IMU) Sensor Model.

Models body-frame specific force (accelerometer) and angular velocity (gyroscope)
with configurable Gaussian noise, deterministic bias, random-walk bias drift,
saturation clipping, and Gymnasium-compatible seeded random number generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from adaptive_rl.environments.drone_6dof import DroneState6DOF, quaternion_to_rotation_matrix


@dataclass
class IMUMeasurement:
    """Raw sensor readings produced by a 6-axis IMU."""

    timestamp: float
    accelerometer: np.ndarray  # Specific force in body frame (m/s^2) [ax, ay, az]
    gyroscope: np.ndarray  # Angular velocity in body frame (rad/s) [p, q, r]
    accel_bias: np.ndarray
    gyro_bias: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "accelerometer": [round(float(x), 4) for x in self.accelerometer],
            "gyroscope": [round(float(x), 5) for x in self.gyroscope],
            "accel_bias": [round(float(x), 5) for x in self.accel_bias],
            "gyro_bias": [round(float(x), 6) for x in self.gyro_bias],
        }


class IMUSensor:
    """6-Axis IMU (Accelerometer + Gyroscope) with physical specific force projection."""

    def __init__(
        self,
        gravity: float = 9.81,
        accel_noise_std: float = 0.05,  # m/s^2 noise std
        gyro_noise_std: float = 0.005,  # rad/s noise std
        accel_bias_init: Optional[np.ndarray] = None,
        gyro_bias_init: Optional[np.ndarray] = None,
        bias_drift_std: float = 1e-4,  # Random walk drift rate
        accel_saturation: float = 50.0,  # Max measurable specific force (m/s^2)
        gyro_saturation: float = 35.0,  # Max measurable angular rate (rad/s)
        seed: Optional[int] = None,
    ) -> None:
        self.gravity = float(gravity)
        self.accel_noise_std = float(accel_noise_std)
        self.gyro_noise_std = float(gyro_noise_std)
        self.bias_drift_std = float(bias_drift_std)
        self.accel_saturation = float(accel_saturation)
        self.gyro_saturation = float(gyro_saturation)

        self.rng = np.random.default_rng(seed)

        self.accel_bias = (
            np.asarray(accel_bias_init, dtype=np.float64).copy()
            if accel_bias_init is not None
            else np.zeros(3, dtype=np.float64)
        )
        self.gyro_bias = (
            np.asarray(gyro_bias_init, dtype=np.float64).copy()
            if gyro_bias_init is not None
            else np.zeros(3, dtype=np.float64)
        )

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset sensor RNG and biases."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.accel_bias.fill(0.0)
        self.gyro_bias.fill(0.0)

    def measure(
        self,
        drone_state: DroneState6DOF,
        inertial_acceleration: np.ndarray,
        dt: float = 0.05,
        timestamp: float = 0.0,
    ) -> IMUMeasurement:
        """Generate physically consistent IMU measurements from drone state.

        Physics:
          An accelerometer measures specific force in the body frame:
            a_spec_W = a_inertial - g_vector (where g_vector = [0, 0, -g])
            a_spec_B = R(q)^T * a_spec_W
          A gyroscope measures angular velocity in the body frame:
            omega_B = [p, q, r]

        Returns:
            IMUMeasurement containing noisy body-frame specific force and angular velocity.
        """
        # 1. Update bias random walk drift: b(t + dt) = b(t) + N(0, drift_std * sqrt(dt))
        drift_scale = math.sqrt(max(1e-6, dt)) * self.bias_drift_std
        self.accel_bias += self.rng.normal(0.0, drift_scale, size=3)
        self.gyro_bias += self.rng.normal(0.0, drift_scale, size=3)

        # 2. Specific force in world frame
        # Gravity vector in world frame points down: g = [0, 0, -9.81]
        g_world = np.array([0.0, 0.0, -self.gravity], dtype=np.float64)
        a_inertial = np.asarray(inertial_acceleration, dtype=np.float64)
        spec_force_world = a_inertial - g_world  # upward when hovering!

        # Transform to body frame
        r_mat = quaternion_to_rotation_matrix(drone_state.quaternion)
        spec_force_body_true = r_mat.T @ spec_force_world

        # Add noise and bias
        accel_noise = self.rng.normal(0.0, self.accel_noise_std, size=3)
        accel_measured = spec_force_body_true + self.accel_bias + accel_noise
        accel_clamped = np.clip(accel_measured, -self.accel_saturation, self.accel_saturation)

        # 3. Gyroscope body angular velocity
        omega_body_true = drone_state.angular_velocity.copy()
        gyro_noise = self.rng.normal(0.0, self.gyro_noise_std, size=3)
        gyro_measured = omega_body_true + self.gyro_bias + gyro_noise
        gyro_clamped = np.clip(gyro_measured, -self.gyro_saturation, self.gyro_saturation)

        return IMUMeasurement(
            timestamp=float(timestamp),
            accelerometer=accel_clamped,
            gyroscope=gyro_clamped,
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
        )
