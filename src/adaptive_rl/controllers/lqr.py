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
        bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0),
        max_velocity: float = 10.0,
        max_angular_velocity: float = 10.0,
        allow_fallback: bool = False,
    ) -> None:
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.arm_length = float(arm_length)
        self.d = self.arm_length / math.sqrt(2.0)
        self.inertia = np.array(inertia, dtype=np.float64)
        self.max_thrust_per_motor = float(max_thrust_per_motor)
        self.c_tau = float(torque_to_thrust_ratio)
        self.dt = float(dt)
        self.bounds = tuple(float(b) for b in bounds)
        self.max_velocity = float(max_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        self.allow_fallback = bool(allow_fallback)

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

        self.solver_status: str = "INITIALIZING"
        self.is_are_solution: bool = False

        # Compute optimal gain matrix K
        self.K = self._compute_lqr_gain()
        self.target_position = np.array([0.0, 0.0, 2.0], dtype=np.float64)

    @staticmethod
    def get_allocation_matrix(d: float, c_tau: float) -> np.ndarray:
        """Construct the physical forward allocation matrix M (4x4) for an X-quadrotor.

        Maps individual rotor thrusts [f1, f2, f3, f4]^T to [T, tau_x, tau_y, tau_z]^T:
            T     = f1 + f2 + f3 + f4
            tau_x = d * (-f1 + f2 + f3 - f4)
            tau_y = d * (-f1 - f2 + f3 + f4)
            tau_z = c_tau * (+f1 - f2 + f3 - f4)
        """
        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [-d, d, d, -d],
                [-d, -d, d, d],
                [c_tau, -c_tau, c_tau, -c_tau],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def get_allocation_matrix_inverse(d: float, c_tau: float) -> np.ndarray:
        """Construct the exact analytical inverse allocation matrix M^{-1} (4x4).

        Maps virtual control wrench [T, tau_x, tau_y, tau_z]^T to [f1, f2, f3, f4]^T:
            f1 = T/4 - tau_x/(4d) - tau_y/(4d) + tau_z/(4*c_tau)
            f2 = T/4 + tau_x/(4d) - tau_y/(4d) - tau_z/(4*c_tau)
            f3 = T/4 + tau_x/(4d) + tau_y/(4d) + tau_z/(4*c_tau)
            f4 = T/4 - tau_x/(4d) + tau_y/(4d) - tau_z/(4*c_tau)
        """
        if d <= 1e-9:
            raise ValueError(f"Arm diagonal offset d must be positive, got {d}")
        if c_tau <= 1e-12:
            raise ValueError(f"Torque constant c_tau must be positive, got {c_tau}")

        return np.array(
            [
                [0.25, -0.25 / d, -0.25 / d, 0.25 / c_tau],
                [0.25, 0.25 / d, -0.25 / d, -0.25 / c_tau],
                [0.25, 0.25 / d, 0.25 / d, 0.25 / c_tau],
                [0.25, -0.25 / d, 0.25 / d, -0.25 / c_tau],
            ],
            dtype=np.float64,
        )

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

        return np.array([f1, f2, f3, f4], dtype=np.float64)

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
            self.solver_status = "ARE_SOLVED"
            self.is_are_solution = True
            return np.asarray(K, dtype=np.float64)
        except Exception as err:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Continuous Algebraic Riccati Equation solver failed: {err}. "
                    "Exact LQR control requires scipy.linalg.solve_continuous_are. "
                    "Do not silently call an approximate gain matrix an ARE-derived LQR solution; "
                    "set allow_fallback=True only if an explicit heuristic approximation is acceptable."
                ) from err

            self.solver_status = "UNRESOLVED_APPROXIMATION"
            self.is_are_solution = False
            # Analytical decoupled gain approximation
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
        state_or_obs: DroneState6DOF | np.ndarray,
        target_position: Optional[np.ndarray] = None,
        target_velocity: Optional[np.ndarray] = None,
        target_yaw: float = 0.0,
        dt: float = 0.05,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal control action for the 6-DOF quadrotor.

        Accepts either a physical DroneState6DOF object, a 36-D normalized Gymnasium
        observation from Drone6DOFEnv, or an unnormalized 13-D state vector.

        Returns:
            (action_in_minus1_plus1, debug_info_dict)
        """
        if isinstance(state_or_obs, DroneState6DOF):
            pos = state_or_obs.position.copy()
            vel = state_or_obs.velocity.copy()
            roll, pitch, yaw = quaternion_to_euler_angles(state_or_obs.quaternion)
            omega = state_or_obs.angular_velocity.copy()
            target_pos = (
                np.asarray(target_position, dtype=np.float64).copy()
                if target_position is not None
                else self.target_position.copy()
            )
        else:
            arr = np.asarray(state_or_obs, dtype=np.float64)
            if len(arr) == 36:
                bx, by, bz = self.bounds
                pos = np.array([arr[0] * bx, arr[1] * by, arr[2] * bz], dtype=np.float64)
                vel = np.array(
                    [
                        arr[3] * self.max_velocity,
                        arr[4] * self.max_velocity,
                        arr[5] * self.max_velocity,
                    ],
                    dtype=np.float64,
                )
                quat = np.array([arr[6], arr[7], arr[8], arr[9]], dtype=np.float64)
                roll, pitch, yaw = quaternion_to_euler_angles(quat)
                omega = np.array(
                    [
                        arr[10] * self.max_angular_velocity,
                        arr[11] * self.max_angular_velocity,
                        arr[12] * self.max_angular_velocity,
                    ],
                    dtype=np.float64,
                )
                if target_position is not None:
                    target_pos = np.asarray(target_position, dtype=np.float64).copy()
                else:
                    # De-normalize target from observation norm_goal indices 13:16
                    target_pos = np.array(
                        [arr[13] * bx, arr[14] * by, arr[15] * bz], dtype=np.float64
                    )
            elif len(arr) == 13:
                pos = arr[0:3].copy()
                vel = arr[3:6].copy()
                quat = arr[6:10].copy()
                roll, pitch, yaw = quaternion_to_euler_angles(quat)
                omega = arr[10:13].copy()
                target_pos = (
                    np.asarray(target_position, dtype=np.float64).copy()
                    if target_position is not None
                    else self.target_position.copy()
                )
            else:
                raise ValueError(
                    f"Unsupported state/obs representation with shape {arr.shape}. "
                    "Expected DroneState6DOF, 36-D normalized Drone6DOFEnv observation, "
                    "or 13-D unnormalized physical state vector."
                )

        target_vel = (
            np.asarray(target_velocity, dtype=np.float64)
            if target_velocity is not None
            else np.zeros(3, dtype=np.float64)
        )

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
        raw_thrusts = self.allocate_thrusts(total_thrust, tau_x, tau_y, tau_z)
        thrusts_clamped = np.clip(raw_thrusts, 0.0, self.max_thrust_per_motor)

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
            "raw_motor_thrusts": raw_thrusts,
            "position_error": float(np.linalg.norm(target_pos - pos)),
            "solver_status": self.solver_status,
            "is_are_solution": self.is_are_solution,
        }
        return action, info

    def predict(
        self,
        observation: np.ndarray | DroneState6DOF,
        state: Optional[Any] = None,
        episode_start: Optional[Any] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Gym-compatible interface matching RL policy predict signature.

        Correctly decodes normalized 36-D observation from Drone6DOFEnv or accepts DroneState6DOF.
        Rejects unsupported formats with an explicit ValueError.
        """
        action, info = self.compute_action(observation)
        return action, info
