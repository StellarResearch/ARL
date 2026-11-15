"""Aerospace reference cascaded PID controller for 6-DOF quadrotor flight dynamics.

Provides a classical cascaded PID control architecture for independent physics
validation, benchmark baselines, and reproducible autonomous flight scenario testing.

Architecture:
-------------
1. Outer Loop (Position & Velocity Control):
   - Computes desired 3D acceleration from position and velocity tracking errors.
   - Converts desired horizontal acceleration into target roll (phi) and pitch (theta) angles.
   - Computes total required collective thrust along the quadrotor body Z-axis.

2. Inner Loop (Attitude & Angular Rate Control):
   - Computes attitude orientation error using quaternion algebra.
   - Computes desired roll, pitch, and yaw body torques [tau_x, tau_y, tau_z].

3. Control Allocation (Motor Thrust Mixer):
   - Inverts the physical X-configuration thrust-torque geometry matrix analytically:
       [T_total, tau_x, tau_y, tau_z]^T = M * [f1, f2, f3, f4]^T
   - Clamps individual motor thrusts to actuator limits [0, max_thrust_per_motor].
   - Maps physical thrusts to normalized environment actions in [-1.0, 1.0]^4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.environments.drone_6dof import (
    DroneState6DOF,
    euler_to_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
)


@dataclass
class PIDGains:
    """Proportional-Integral-Derivative gain set with output clamping."""

    kp: float
    ki: float
    kd: float
    max_integral: float = 5.0
    max_output: float = 10.0


class QuadrotorPIDController:
    """Cascaded PID reference flight controller for 6-DOF quadrotors.

    Designed for deterministic physics validation, axis-isolation verification,
    and reference scenario benchmarking.
    """

    def __init__(
        self,
        mass: float = 1.0,
        gravity: float = 9.81,
        arm_length: float = 0.2,
        inertia: Optional[Tuple[float, float, float]] = None,
        max_thrust_per_motor: float = 5.0,
        torque_to_thrust_ratio: float = 0.01,
        dt: float = 0.05,
        pos_gains_xy: Optional[PIDGains] = None,
        pos_gains_z: Optional[PIDGains] = None,
        att_gains_rp: Optional[PIDGains] = None,
        att_gains_yaw: Optional[PIDGains] = None,
        max_tilt_angle_rad: float = math.radians(35.0),
    ) -> None:
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.arm_length = float(arm_length)
        self.d = self.arm_length / math.sqrt(2.0)
        self.max_thrust_per_motor = float(max_thrust_per_motor)
        self.max_total_thrust = 4.0 * self.max_thrust_per_motor
        self.c_tau = float(torque_to_thrust_ratio)
        self.dt = float(dt)
        self.max_tilt_angle = float(max_tilt_angle_rad)

        if inertia is None:
            self.inertia = np.array([0.01, 0.01, 0.02], dtype=np.float64)
        else:
            self.inertia = np.array(
                [float(inertia[0]), float(inertia[1]), float(inertia[2])], dtype=np.float64
            )

        # Default tuned gains for 1.0 kg quadrotor
        self.pos_gains_xy = pos_gains_xy or PIDGains(
            kp=2.2, ki=0.15, kd=2.8, max_integral=2.0, max_output=6.0
        )
        self.pos_gains_z = pos_gains_z or PIDGains(
            kp=5.0, ki=0.8, kd=4.0, max_integral=3.0, max_output=15.0
        )
        self.att_gains_rp = att_gains_rp or PIDGains(
            kp=8.0, ki=0.05, kd=2.5, max_integral=1.0, max_output=8.0
        )
        self.att_gains_yaw = att_gains_yaw or PIDGains(
            kp=4.0, ki=0.02, kd=1.5, max_integral=1.0, max_output=4.0
        )

        # Integral error accumulators
        self.pos_integral = np.zeros(3, dtype=np.float64)
        self.att_integral = np.zeros(3, dtype=np.float64)

        # Previous errors for derivative calculation if velocity is unavailable
        self.prev_pos_error = np.zeros(3, dtype=np.float64)
        self.prev_att_error = np.zeros(3, dtype=np.float64)

        # Target setpoints
        self.target_position = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        self.target_velocity = np.zeros(3, dtype=np.float64)
        self.target_yaw = 0.0

        # Hover thrust baseline
        self.hover_thrust_total = self.mass * self.gravity
        self.hover_thrust_per_motor = self.hover_thrust_total / 4.0

    def reset(self) -> None:
        """Reset internal integrators and tracking memory."""
        self.pos_integral.fill(0.0)
        self.att_integral.fill(0.0)
        self.prev_pos_error.fill(0.0)
        self.prev_att_error.fill(0.0)

    def set_target(
        self,
        position: np.ndarray | Sequence[float],
        velocity: Optional[np.ndarray | Sequence[float]] = None,
        yaw: float = 0.0,
    ) -> None:
        """Update target flight setpoints."""
        self.target_position = np.asarray(position, dtype=np.float64).copy()
        if velocity is not None:
            self.target_velocity = np.asarray(velocity, dtype=np.float64).copy()
        else:
            self.target_velocity.fill(0.0)
        self.target_yaw = float(yaw)

    def compute_action(
        self,
        state_or_obs: DroneState6DOF | np.ndarray | Sequence[float],
        target_position: Optional[np.ndarray | Sequence[float]] = None,
        target_velocity: Optional[np.ndarray | Sequence[float]] = None,
        target_yaw: Optional[float] = None,
        bounds: Optional[Tuple[float, float, float]] = None,
        dt: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute 4 normalized actuator commands [-1, 1] for current state and target.

        Args:
            state_or_obs: DroneState6DOF object or raw observation vector (36-D).
            target_position: Optional target coordinate override [x, y, z].
            target_velocity: Optional target velocity override [vx, vy, vz].
            target_yaw: Optional target yaw angle in radians.
            bounds: Optional arena bounds for denormalizing 36-D observation.
            dt: Optional time step override.

        Returns:
            Tuple of (action_array_4, telemetry_dict)
        """
        effective_dt = dt or self.dt
        if effective_dt <= 0.0:
            effective_dt = 0.05

        if target_position is not None:
            self.target_position = np.asarray(target_position, dtype=np.float64).copy()
        if target_velocity is not None:
            self.target_velocity = np.asarray(target_velocity, dtype=np.float64).copy()
        if target_yaw is not None:
            self.target_yaw = float(target_yaw)

        # 1. Extract physical state from input
        if isinstance(state_or_obs, DroneState6DOF):
            pos = state_or_obs.position.copy()
            vel = state_or_obs.velocity.copy()
            quat = state_or_obs.quaternion.copy()
            omega = state_or_obs.angular_velocity.copy()
        else:
            arr = np.asarray(state_or_obs, dtype=np.float64)
            if len(arr) == 36:
                bx, by, bz = bounds or (30.0, 30.0, 15.0)
                pos = np.array([arr[0] * bx, arr[1] * by, arr[2] * bz], dtype=np.float64)
                vel = np.array([arr[3] * 10.0, arr[4] * 10.0, arr[5] * 10.0], dtype=np.float64)
                quat = np.array([arr[6], arr[7], arr[8], arr[9]], dtype=np.float64)
                omega = np.array([arr[10] * 10.0, arr[11] * 10.0, arr[12] * 10.0], dtype=np.float64)
            elif len(arr) == 13:
                pos = arr[0:3].copy()
                vel = arr[3:6].copy()
                quat = arr[6:10].copy()
                omega = arr[10:13].copy()
            else:
                raise ValueError(
                    f"Unsupported state/obs representation with shape {arr.shape}. "
                    "Expected DroneState6DOF, 36-D obs, or 13-D state vector."
                )

        q_norm = np.linalg.norm(quat)
        if q_norm > 1e-12:
            quat /= q_norm
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # -------------------------------------------------------------
        # Outer Loop: Position & Velocity Tracking -> Desired World Acc
        # -------------------------------------------------------------
        pos_error = self.target_position - pos
        vel_error = self.target_velocity - vel

        # Integrate XY position error with anti-windup clamping
        self.pos_integral[:2] += pos_error[:2] * effective_dt
        self.pos_integral[:2] = np.clip(
            self.pos_integral[:2],
            -self.pos_gains_xy.max_integral,
            self.pos_gains_xy.max_integral,
        )

        # Integrate Z position error with anti-windup clamping
        self.pos_integral[2] += pos_error[2] * effective_dt
        self.pos_integral[2] = np.clip(
            self.pos_integral[2],
            -self.pos_gains_z.max_integral,
            self.pos_gains_z.max_integral,
        )

        # Desired horizontal accelerations (world frame)
        a_x_des = (
            self.pos_gains_xy.kp * pos_error[0]
            + self.pos_gains_xy.kd * vel_error[0]
            + self.pos_gains_xy.ki * self.pos_integral[0]
        )
        a_y_des = (
            self.pos_gains_xy.kp * pos_error[1]
            + self.pos_gains_xy.kd * vel_error[1]
            + self.pos_gains_xy.ki * self.pos_integral[1]
        )
        a_x_des = float(
            np.clip(a_x_des, -self.pos_gains_xy.max_output, self.pos_gains_xy.max_output)
        )
        a_y_des = float(
            np.clip(a_y_des, -self.pos_gains_xy.max_output, self.pos_gains_xy.max_output)
        )

        # Desired vertical acceleration (world frame)
        a_z_des = (
            self.pos_gains_z.kp * pos_error[2]
            + self.pos_gains_z.kd * vel_error[2]
            + self.pos_gains_z.ki * self.pos_integral[2]
        )
        a_z_des = float(np.clip(a_z_des, -self.pos_gains_z.max_output, self.pos_gains_z.max_output))

        # -------------------------------------------------------------
        # Attitude Setpoint Generation from Desired Accelerations
        # -------------------------------------------------------------
        # In ENU:
        # Thrust along world Z opposes gravity: F_z_net = m * (g + a_z_des)
        total_acc_z = self.gravity + a_z_des
        if total_acc_z < 0.1:
            total_acc_z = 0.1  # prevent negative or zero thrust demand

        # Compute total required thrust magnitude
        thrust_magnitude = self.mass * math.sqrt(
            a_x_des * a_x_des + a_y_des * a_y_des + total_acc_z * total_acc_z
        )
        total_thrust_des = float(np.clip(thrust_magnitude, 0.0, self.max_total_thrust))

        # Invert attitude mapping to align body +Z_B with desired thrust vector:
        # Desired tilt angles in world ENU:
        # Accelerating towards +X requires pitching down (-theta) or pitching up (+theta)
        # Note: in body frame (+X forward, +Y left, +Z up):
        # R(q) * [0, 0, 1]^T gives world thrust direction.
        # R31 = 2(qx*qz - qw*qy), R32 = 2(qy*qz + qw*qx), R33 = 1 - 2(qx^2 + qy^2)
        # For small angles: R * [0,0,1]^T ~ [theta, -phi, 1]^T.
        # Thus, positive a_x requires pitch angle theta = a_x / g.
        # Positive a_y requires roll angle phi = -a_y / g.
        cos_yaw = math.cos(self.target_yaw)
        sin_yaw = math.sin(self.target_yaw)

        # Rotate horizontal acceleration into quadrotor heading frame:
        a_forward = a_x_des * cos_yaw + a_y_des * sin_yaw
        a_left = -a_x_des * sin_yaw + a_y_des * cos_yaw

        pitch_des = math.atan2(a_forward, total_acc_z)
        roll_des = math.atan2(-a_left, total_acc_z)

        # Clamp attitude setpoints to maximum safe tilt
        pitch_des = float(np.clip(pitch_des, -self.max_tilt_angle, self.max_tilt_angle))
        roll_des = float(np.clip(roll_des, -self.max_tilt_angle, self.max_tilt_angle))
        yaw_des = float(self.target_yaw)

        # -------------------------------------------------------------
        # Inner Loop: Attitude & Angular Rate Control -> Desired Torques
        # -------------------------------------------------------------
        q_target = euler_to_quaternion(roll_des, pitch_des, yaw_des)

        # Quaternion orientation error: q_err = q_conj (x) q_target
        q_conj = quaternion_conjugate(quat)
        q_err = quaternion_multiply(q_conj, q_target)
        if q_err[0] < 0.0:
            q_err = -q_err  # shortest path on S^3

        # Small angle error vector: e_att = 2 * [qx, qy, qz]
        att_error = 2.0 * q_err[1:4]

        # Integrate attitude errors
        self.att_integral[:2] += att_error[:2] * effective_dt
        self.att_integral[:2] = np.clip(
            self.att_integral[:2],
            -self.att_gains_rp.max_integral,
            self.att_gains_rp.max_integral,
        )
        self.att_integral[2] += att_error[2] * effective_dt
        self.att_integral[2] = np.clip(
            self.att_integral[2],
            -self.att_gains_yaw.max_integral,
            self.att_gains_yaw.max_integral,
        )

        # Body torques PID: tau = I * (Kp*e - Kd*omega + Ki*int_e)
        tau_x = self.inertia[0] * (
            self.att_gains_rp.kp * att_error[0]
            - self.att_gains_rp.kd * omega[0]
            + self.att_gains_rp.ki * self.att_integral[0]
        )
        tau_y = self.inertia[1] * (
            self.att_gains_rp.kp * att_error[1]
            - self.att_gains_rp.kd * omega[1]
            + self.att_gains_rp.ki * self.att_integral[1]
        )
        tau_z = self.inertia[2] * (
            self.att_gains_yaw.kp * att_error[2]
            - self.att_gains_yaw.kd * omega[2]
            + self.att_gains_yaw.ki * self.att_integral[2]
        )

        tau_x = float(np.clip(tau_x, -self.att_gains_rp.max_output, self.att_gains_rp.max_output))
        tau_y = float(np.clip(tau_y, -self.att_gains_rp.max_output, self.att_gains_rp.max_output))
        tau_z = float(np.clip(tau_z, -self.att_gains_yaw.max_output, self.att_gains_yaw.max_output))

        # -------------------------------------------------------------
        # Control Allocation / Analytical Mixer Inversion
        # -------------------------------------------------------------
        # Physical rotor thrusts:
        #   f1 = T/4 - tau_x/(4d) - tau_y/(4d) + tau_z/(4*c_tau)
        #   f2 = T/4 + tau_x/(4d) - tau_y/(4d) - tau_z/(4*c_tau)
        #   f3 = T/4 + tau_x/(4d) + tau_y/(4d) + tau_z/(4*c_tau)
        #   f4 = T/4 - tau_x/(4d) + tau_y/(4d) - tau_z/(4*c_tau)
        motor_thrusts = self.allocate_thrusts(
            total_thrust=total_thrust_des,
            tau_x=tau_x,
            tau_y=tau_y,
            tau_z=tau_z,
        )

        # Map to normalized actions in [-1, 1]
        # action = 2 * (thrust / max_thrust) - 1
        actions = 2.0 * (motor_thrusts / self.max_thrust_per_motor) - 1.0
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

        telemetry: Dict[str, Any] = {
            "target_position": self.target_position.copy(),
            "target_velocity": self.target_velocity.copy(),
            "target_attitude_euler": np.array([roll_des, pitch_des, yaw_des], dtype=np.float64),
            "position_error": pos_error,
            "velocity_error": vel_error,
            "attitude_error": att_error,
            "desired_acceleration": np.array([a_x_des, a_y_des, a_z_des], dtype=np.float64),
            "desired_total_thrust": total_thrust_des,
            "desired_torques": np.array([tau_x, tau_y, tau_z], dtype=np.float64),
            "allocated_motor_thrusts": motor_thrusts,
            "normalized_actions": actions,
        }

        return actions, telemetry

    def allocate_thrusts(
        self,
        total_thrust: float,
        tau_x: float,
        tau_y: float,
        tau_z: float,
    ) -> np.ndarray:
        """Invert the physical X-quadrotor allocation matrix analytically."""
        d = self.d
        c = self.c_tau

        t_term = total_thrust * 0.25
        tx_term = tau_x / (4.0 * d)
        ty_term = tau_y / (4.0 * d)
        tz_term = tau_z / (4.0 * c) if c > 1e-9 else 0.0

        f1 = t_term - tx_term - ty_term + tz_term
        f2 = t_term + tx_term - ty_term - tz_term
        f3 = t_term + tx_term + ty_term + tz_term
        f4 = t_term - tx_term + ty_term - tz_term

        thrusts = np.array([f1, f2, f3, f4], dtype=np.float64)
        return np.clip(thrusts, 0.0, self.max_thrust_per_motor)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Any] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Evaluation interface conforming to BaseAlgorithm/Evaluator.predict."""
        actions, _ = self.compute_action(observation)
        return actions, state


# Canonical alias for reference controller
ReferenceFlightController = QuadrotorPIDController
