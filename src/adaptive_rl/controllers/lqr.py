"""Linear Quadratic Regulator (LQR) Optimal Flight Controller for 6-DOF Quadrotors.

Formulates continuous-time linearized quadrotor state-space dynamics around hover equilibrium,
solves the Algebraic Riccati Equation (ARE) for optimal gain matrix K, and applies
exact analytical mixer inversion to map virtual control inputs to four motor thrusts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from adaptive_rl.environments.drone_6dof import DroneState6DOF, quaternion_to_euler_angles


class QuadrotorLQRController:
    """Optimal Linear Quadratic Regulator (LQR) for 6-DOF quadrotors."""

    def __init__(
        self,
        mass: float = 1.0,
        gravity: float = 9.81,
        arm_length: float = 0.2,
        inertia: Tuple[float, float, float] = (0.01, 0.01, 0.02),
        max_thrust_per_motor: float = 5.0,
        torque_to_thrust_ratio: float = 0.01,
        dt: float = 0.05,
        q_pos: float = 8.0,
        q_vel: float = 4.0,
        q_att: float = 15.0,
        q_rate: float = 2.0,
        r_thrust: float = 1.0,
        r_torque: float = 5.0,
    ) -> None:
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.arm_length = float(arm_length)
        self.d = self.arm_length / math.sqrt(2.0)
        self.inertia = np.array(inertia, dtype=np.float64)
        self.max_thrust_per_motor = float(max_thrust_per_motor)
        self.c_tau = float(torque_to_thrust_ratio)
        self.dt = float(dt)

        self.hover_thrust_total = self.mass * self.gravity
        self.hover_thrust_per_motor = self.hover_thrust_total / 4.0

        # State weighting matrix Q (12x12)
        # States: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.Q = np.diag(
            [
                q_pos,
                q_pos,
                q_pos * 1.5,
                q_vel,
                q_vel,
                q_vel * 1.2,
                q_att,
                q_att,
                q_att * 0.5,
                q_rate,
                q_rate,
                q_rate * 0.5,
            ]
        )

        # Control weighting matrix R (4x4)
        # Controls: [Delta_T, tau_x, tau_y, tau_z]
        self.R = np.diag([r_thrust, r_torque, r_torque, r_torque * 1.5])

        # Compute optimal gain matrix K
        self.K = self._compute_lqr_gain()
        self.target_position = np.array([0.0, 0.0, 2.0], dtype=np.float64)

    def _compute_lqr_gain(self) -> np.ndarray:
        """Formulate linearized state space (A, B) and solve the continuous ARE."""
        A = np.zeros((12, 12), dtype=np.float64)
        B = np.zeros((12, 4), dtype=np.float64)

        # dx/dt = v
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0

        # dv/dt = g * tilt
        A[3, 7] = self.gravity  # dvx/dt = g * theta
        A[4, 6] = -self.gravity  # dvy/dt = -g * phi

        # d(att)/dt = omega
        A[6, 9] = 1.0
        A[7, 10] = 1.0
        A[8, 11] = 1.0

        # Control mapping B
        B[5, 0] = 1.0 / self.mass  # dvz/dt = Delta_T / m
        B[9, 1] = 1.0 / self.inertia[0]  # dp/dt = tau_x / Ixx
        B[10, 2] = 1.0 / self.inertia[1]  # dq/dt = tau_y / Iyy
        B[11, 3] = 1.0 / self.inertia[2]  # dr/dt = tau_z / Izz

        try:
            from scipy.linalg import solve_continuous_are

            P = solve_continuous_are(A, B, self.Q, self.R)
            R_inv = np.linalg.inv(self.R)
            K = R_inv @ (B.T @ P)
            return np.asarray(K, dtype=np.float64)
        except Exception:
            # Analytical decoupled LQR gain approximation if scipy is unavailable
            K_approx = np.zeros((4, 12), dtype=np.float64)
            # z altitude channel: [z, vz] -> Delta_T
            K_approx[0, 2] = math.sqrt(self.Q[2, 2] / self.R[0, 0])
            K_approx[0, 5] = math.sqrt(self.Q[5, 5] / self.R[0, 0])
            # roll channel: [y, vy, phi, p] -> tau_x
            K_approx[1, 1] = -math.sqrt(self.Q[1, 1] / self.R[1, 1]) * 0.5
            K_approx[1, 4] = -math.sqrt(self.Q[4, 4] / self.R[1, 1]) * 0.5
            K_approx[1, 6] = math.sqrt(self.Q[6, 6] / self.R[1, 1])
            K_approx[1, 9] = math.sqrt(self.Q[9, 9] / self.R[1, 1])
            # pitch channel: [x, vx, theta, q] -> tau_y
            K_approx[2, 0] = math.sqrt(self.Q[0, 0] / self.R[2, 2]) * 0.5
            K_approx[2, 3] = math.sqrt(self.Q[3, 3] / self.R[2, 2]) * 0.5
            K_approx[2, 7] = math.sqrt(self.Q[7, 7] / self.R[2, 2])
            K_approx[2, 10] = math.sqrt(self.Q[10, 10] / self.R[2, 2])
            # yaw channel: [psi, r] -> tau_z
            K_approx[3, 8] = math.sqrt(self.Q[8, 8] / self.R[3, 3])
            K_approx[3, 11] = math.sqrt(self.Q[11, 11] / self.R[3, 3])
            return K_approx

    def set_target(self, target_position: np.ndarray) -> None:
        """Update target position waypoint."""
        self.target_position = np.asarray(target_position, dtype=np.float64).copy()

    def reset(self) -> None:
        """Reset internal controller state."""
        pass

    def compute_action(
        self,
        state: DroneState6DOF,
        target_position: Optional[np.ndarray] = None,
        target_velocity: Optional[np.ndarray] = None,
        target_yaw: float = 0.0,
        dt: float = 0.05,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal control action for the 6-DOF quadrotor.

        Returns:
            (action_in_minus1_plus1, debug_info_dict)
        """
        target_pos = (
            np.asarray(target_position, dtype=np.float64)
            if target_position is not None
            else self.target_position
        )
        target_vel = (
            np.asarray(target_velocity, dtype=np.float64)
            if target_velocity is not None
            else np.zeros(3, dtype=np.float64)
        )

        pos = state.position
        vel = state.velocity
        roll, pitch, yaw = quaternion_to_euler_angles(state.quaternion)
        omega = state.angular_velocity

        # State error vector (12,)
        # delta_x = x - x_ref
        delta_x = np.array(
            [
                pos[0] - target_pos[0],
                pos[1] - target_pos[1],
                pos[2] - target_pos[2],
                vel[0] - target_vel[0],
                vel[1] - target_vel[1],
                vel[2] - target_vel[2],
                roll,
                pitch,
                yaw - target_yaw,
                omega[0],
                omega[1],
                omega[2],
            ],
            dtype=np.float64,
        )

        # Optimal virtual control: u = -K * delta_x
        # u = [Delta_T, tau_x, tau_y, tau_z]
        u = -self.K @ delta_x

        delta_T = float(u[0])
        tau_x = float(u[1])
        tau_y = float(u[2])
        tau_z = float(u[3])

        total_thrust = self.hover_thrust_total + delta_T
        total_thrust = max(0.0, min(4.0 * self.max_thrust_per_motor, total_thrust))

        # Exact analytical mixer inversion for X-quadrotor
        d = self.d
        c_tau = self.c_tau
        t1 = 0.25 * total_thrust - 0.25 * tau_x / d - 0.25 * tau_y / d - 0.25 * tau_z / c_tau
        t2 = 0.25 * total_thrust - 0.25 * tau_x / d + 0.25 * tau_y / d + 0.25 * tau_z / c_tau
        t3 = 0.25 * total_thrust + 0.25 * tau_x / d + 0.25 * tau_y / d - 0.25 * tau_z / c_tau
        t4 = 0.25 * total_thrust + 0.25 * tau_x / d - 0.25 * tau_y / d + 0.25 * tau_z / c_tau

        thrusts = np.array([t1, t2, t3, t4], dtype=np.float64)
        thrusts_clamped = np.clip(thrusts, 0.0, self.max_thrust_per_motor)

        # Normalize to Gym action space [-1, 1]
        action = np.clip(
            (2.0 * thrusts_clamped / self.max_thrust_per_motor) - 1.0, -1.0, 1.0
        ).astype(np.float32)

        info = {
            "total_thrust": total_thrust,
            "delta_T": delta_T,
            "tau_x": tau_x,
            "tau_y": tau_y,
            "tau_z": tau_z,
            "motor_thrusts": thrusts_clamped,
            "position_error": float(np.linalg.norm(target_pos - pos)),
        }
        return action, info

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Gym-compatible interface matching RL policy predict signature."""
        from adaptive_rl.environments.drone_6dof import DroneState6DOF

        pos = obs[0:3]
        vel = obs[3:6]
        quat = obs[6:10]
        omega = obs[10:13]
        target = obs[13:16]

        state = DroneState6DOF(
            position=np.array(pos, dtype=np.float64),
            velocity=np.array(vel, dtype=np.float64),
            quaternion=np.array(quat, dtype=np.float64),
            angular_velocity=np.array(omega, dtype=np.float64),
        )
        action, info = self.compute_action(state, target_position=target)
        return action, info
