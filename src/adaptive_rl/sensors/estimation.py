"""State Estimation and Attitude Filter for Quadrotor Avionics.

Implements a non-linear quaternion complementary filter fusing IMU gyroscope
and accelerometer measurements, providing real-time attitude, velocity, and
position estimation with documented observability boundaries.

OBSERVABILITY AND ASSUMPTION NOTE:
----------------------------------
1. Attitude: Accelerometer specific force provides long-term gravimetric leveling
   for roll and pitch angles. Heading (yaw) is unobservable from gravity alone and
   must be tracked via gyro integration (or external magnetometer/visual heading).
2. Translation: Open-loop double integration of accelerometer readings drifts quadratically
   with time. The position filter incorporates an altitude reference (e.g. range/baro)
   for vertical stability, while horizontal position requires external reference updates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from adaptive_rl.environments.drone_6dof import (
    DroneState6DOF,
    quaternion_derivative,
    quaternion_to_euler_angles,
    quaternion_to_rotation_matrix,
)
from adaptive_rl.sensors.imu import IMUMeasurement


@dataclass
class EstimatedDroneState:
    """Estimated state produced by the sensor fusion avionics filter."""

    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    quaternion: np.ndarray  # [qw, qx, qy, qz] attitude
    euler_angles_deg: np.ndarray  # [roll, pitch, yaw] in degrees
    angular_velocity: np.ndarray  # [p, q, r] in rad/s
    position_error: float = 0.0  # Euclidean distance to ground truth
    attitude_error_deg: float = 0.0  # Angular tilt error to ground truth

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": [round(float(x), 4) for x in self.position],
            "velocity": [round(float(x), 4) for x in self.velocity],
            "quaternion": [round(float(x), 5) for x in self.quaternion],
            "euler_angles_deg": [round(float(x), 2) for x in self.euler_angles_deg],
            "angular_velocity": [round(float(x), 4) for x in self.angular_velocity],
            "position_error": round(self.position_error, 4),
            "attitude_error_deg": round(self.attitude_error_deg, 3),
        }


class ComplementaryFilter:
    """Non-linear quaternion complementary filter for quadrotor state estimation."""

    def __init__(
        self,
        gravity: float = 9.81,
        kp_accel: float = 0.8,  # Accelerometer tilt correction gain
        ki_accel: float = 0.02,  # Gyro bias estimation gain
        kp_pos: float = 1.2,  # Position reference correction gain
        kp_vel: float = 0.8,  # Velocity damping gain
    ) -> None:
        self.gravity = float(gravity)
        self.kp_accel = float(kp_accel)
        self.ki_accel = float(ki_accel)
        self.kp_pos = float(kp_pos)
        self.kp_vel = float(kp_vel)

        self.estimated_position = np.zeros(3, dtype=np.float64)
        self.estimated_velocity = np.zeros(3, dtype=np.float64)
        self.estimated_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.gyro_bias_est = np.zeros(3, dtype=np.float64)

    def reset(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        quaternion: Optional[np.ndarray] = None,
    ) -> None:
        """Reset estimator state."""
        self.estimated_position = (
            np.asarray(position, dtype=np.float64).copy()
            if position is not None
            else np.zeros(3, dtype=np.float64)
        )
        self.estimated_velocity = (
            np.asarray(velocity, dtype=np.float64).copy()
            if velocity is not None
            else np.zeros(3, dtype=np.float64)
        )
        self.estimated_quaternion = (
            np.asarray(quaternion, dtype=np.float64).copy()
            if quaternion is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        self.gyro_bias_est.fill(0.0)

    def update(
        self,
        imu: IMUMeasurement,
        dt: float = 0.05,
        altitude_ref: Optional[float] = None,
        position_ref: Optional[np.ndarray] = None,
        true_state: Optional[DroneState6DOF] = None,
    ) -> EstimatedDroneState:
        """Fuse IMU measurements with reference observations to update estimated state."""
        dt = max(1e-4, float(dt))
        q = self.estimated_quaternion

        # 1. Accelerometer gravity direction correction
        # Normalize accelerometer reading
        a_meas = imu.accelerometer
        a_norm = float(np.linalg.norm(a_meas))

        correction = np.zeros(3, dtype=np.float64)
        if 5.0 < a_norm < 15.0:  # Only trust gravity when dynamic acceleration is modest
            a_dir = a_meas / a_norm  # measured body-frame gravity direction (upward specific force)

            # Estimated body +Z axis from current quaternion: R(q)^T * [0, 0, 1]
            r_mat = quaternion_to_rotation_matrix(q)
            z_body_est = r_mat.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # Error between measured specific force direction and estimated upright
            error_tilt = np.cross(a_dir, z_body_est)

            # Proportional and integral feedback
            correction = self.kp_accel * error_tilt
            self.gyro_bias_est += -self.ki_accel * error_tilt * dt

        # Corrected angular velocity
        omega_corrected = (imu.gyroscope - self.gyro_bias_est) + correction

        # Integrate quaternion kinematics: dq/dt = 0.5 * q (x) omega
        dq = quaternion_derivative(q, omega_corrected)
        q_new = q + dq * dt
        norm_q = float(np.linalg.norm(q_new))
        if norm_q > 1e-12:
            q_new /= norm_q
        self.estimated_quaternion = q_new

        # 2. Translational kinematics: project specific force to world frame
        r_new = quaternion_to_rotation_matrix(q_new)
        a_world = r_new @ imu.accelerometer
        # Linear acceleration in world: a_lin = a_world + g (since a_world = a_inertial - g)
        g_vec = np.array([0.0, 0.0, -self.gravity], dtype=np.float64)
        a_inertial_est = a_world + g_vec

        # Integrate velocity and position
        self.estimated_velocity += a_inertial_est * dt
        self.estimated_position += self.estimated_velocity * dt

        # Apply altitude/position reference correction if available
        if altitude_ref is not None:
            z_err = float(altitude_ref) - self.estimated_position[2]
            self.estimated_position[2] += self.kp_pos * z_err * dt
            self.estimated_velocity[2] += self.kp_vel * z_err * dt

        if position_ref is not None:
            pos_ref_arr = np.asarray(position_ref, dtype=np.float64)
            pos_err = pos_ref_arr - self.estimated_position
            self.estimated_position += self.kp_pos * pos_err * dt
            self.estimated_velocity += self.kp_vel * pos_err * dt

        # Compute estimation errors relative to ground truth if provided
        pos_err_mag = 0.0
        att_err_deg = 0.0
        if true_state is not None:
            pos_err_mag = float(np.linalg.norm(self.estimated_position - true_state.position))
            r_true = quaternion_to_rotation_matrix(true_state.quaternion)
            r_diff = r_new.T @ r_true
            tr = float(np.trace(r_diff))
            angle_rad = math.acos(max(-1.0, min(1.0, 0.5 * (tr - 1.0))))
            att_err_deg = math.degrees(angle_rad)

        roll, pitch, yaw = quaternion_to_euler_angles(self.estimated_quaternion)
        euler_deg = np.array(
            [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)], dtype=np.float64
        )

        return EstimatedDroneState(
            position=self.estimated_position.copy(),
            velocity=self.estimated_velocity.copy(),
            quaternion=self.estimated_quaternion.copy(),
            euler_angles_deg=euler_deg,
            angular_velocity=omega_corrected.copy(),
            position_error=pos_err_mag,
            attitude_error_deg=att_err_deg,
        )
