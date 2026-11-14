"""Unit tests for rigid-body 6-DOF quadrotor dynamics environment (Issue #253)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from adaptive_rl.environments import get, make_env
from adaptive_rl.environments.drone_6dof import (
    Drone6DOFEnv,
    DroneDynamics6DOF,
    quaternion_to_rotation_matrix,
)


def test_drone_6dof_registration() -> None:
    """Verify that drone-6dof is registered and instantiable via registry."""
    factory = get("drone-6dof")
    env = factory()
    assert isinstance(env, Drone6DOFEnv)
    env.close()

    env2 = make_env("drone-6dof")
    assert isinstance(env2, Drone6DOFEnv)
    env2.close()


def test_drone_6dof_gymnasium_check_env() -> None:
    """Verify standard Gymnasium API compliance via check_env."""
    env = Drone6DOFEnv()
    check_env(env.unwrapped)
    env.close()


def test_hover_equilibrium() -> None:
    """Hover thrust T = mg produces zero vertical acceleration in level attitude."""
    dynamics = DroneDynamics6DOF(mass=1.0, gravity=9.81, dt=0.01, substeps=1)
    hover_f = dynamics.hover_thrust_per_motor
    assert math.isclose(hover_f * 4.0, 9.81, rel_tol=1e-5)

    # Initial state at rest, level attitude
    dynamics.reset(position=np.array([10.0, 10.0, 5.0]), velocity=np.zeros(3))
    motor_thrusts = np.array([hover_f, hover_f, hover_f, hover_f])

    total_t, torques = dynamics.compute_forces_and_torques(motor_thrusts)
    assert math.isclose(total_t, 9.81, rel_tol=1e-5)
    assert np.allclose(torques, 0.0, atol=1e-6)

    d_pos, d_vel, d_quat, d_omega = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        total_t,
        torques,
    )
    # At hover with v=0 and q=[1,0,0,0], d_vel should be exactly zero
    assert np.allclose(d_vel, 0.0, atol=1e-6)
    assert np.allclose(d_pos, 0.0, atol=1e-6)
    assert np.allclose(d_omega, 0.0, atol=1e-6)

    # Step forward 50 steps
    for _ in range(50):
        state = dynamics.step_rk4(motor_thrusts)

    assert math.isclose(state.position[2], 5.0, abs_tol=1e-4)
    assert math.isclose(state.velocity[2], 0.0, abs_tol=1e-4)


def test_free_fall_gravity() -> None:
    """Zero thrust causes vertical acceleration of -g."""
    dynamics = DroneDynamics6DOF(mass=1.0, gravity=9.81, linear_damping=0.0, dt=0.01, substeps=1)
    dynamics.reset(position=np.array([0.0, 0.0, 10.0]), velocity=np.zeros(3))

    zero_thrusts = np.zeros(4)
    total_t, torques = dynamics.compute_forces_and_torques(zero_thrusts)

    _, d_vel, _, _ = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        total_t,
        torques,
    )
    assert np.allclose(d_vel, [0.0, 0.0, -9.81], atol=1e-5)


def test_pure_roll_torque() -> None:
    """Differential left/right thrust produces angular acceleration about body x-axis."""
    dynamics = DroneDynamics6DOF(arm_length=0.2, inertia=(0.01, 0.01, 0.02))
    # Motors: F1 (FR), F2 (FL), F3 (RL), F4 (RR)
    # Roll torque = d * (-F1 + F2 + F3 - F4)
    # To get positive roll: F2 and F3 > F1 and F4
    f_hover = dynamics.hover_thrust_per_motor
    delta = 0.5
    thrusts = np.array([f_hover - delta, f_hover + delta, f_hover + delta, f_hover - delta])

    total_t, torques = dynamics.compute_forces_and_torques(thrusts)
    assert math.isclose(total_t, 4.0 * f_hover, rel_tol=1e-5)
    assert torques[0] > 0.0  # Positive roll torque
    assert math.isclose(torques[1], 0.0, abs_tol=1e-6)  # Pitch is zero
    assert math.isclose(torques[2], 0.0, abs_tol=1e-6)  # Yaw is zero

    _, _, _, d_omega = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        total_t,
        torques,
    )
    assert d_omega[0] > 0.0
    assert math.isclose(d_omega[1], 0.0, abs_tol=1e-6)
    assert math.isclose(d_omega[2], 0.0, abs_tol=1e-6)


def test_pure_pitch_torque() -> None:
    """Differential front/rear thrust produces angular acceleration about body y-axis."""
    dynamics = DroneDynamics6DOF(arm_length=0.2, inertia=(0.01, 0.01, 0.02))
    # Pitch torque = d * (-F1 - F2 + F3 + F4)
    # Positive pitch (nose up): rear motors (F3, F4) > front motors (F1, F2)
    f_hover = dynamics.hover_thrust_per_motor
    delta = 0.5
    thrusts = np.array([f_hover - delta, f_hover - delta, f_hover + delta, f_hover + delta])

    total_t, torques = dynamics.compute_forces_and_torques(thrusts)
    assert math.isclose(total_t, 4.0 * f_hover, rel_tol=1e-5)
    assert math.isclose(torques[0], 0.0, abs_tol=1e-6)  # Roll is zero
    assert torques[1] > 0.0  # Positive pitch torque
    assert math.isclose(torques[2], 0.0, abs_tol=1e-6)  # Yaw is zero

    _, _, _, d_omega = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        total_t,
        torques,
    )
    assert math.isclose(d_omega[0], 0.0, abs_tol=1e-6)
    assert d_omega[1] > 0.0
    assert math.isclose(d_omega[2], 0.0, abs_tol=1e-6)


def test_pure_yaw_torque() -> None:
    """Differential CCW/CW rotor thrust produces angular acceleration about body z-axis."""
    dynamics = DroneDynamics6DOF(torque_to_thrust_ratio=0.01, inertia=(0.01, 0.01, 0.02))
    # Yaw torque = c_tau * (+F1 - F2 + F3 - F4)
    # Positive yaw: F1, F3 > F2, F4
    f_hover = dynamics.hover_thrust_per_motor
    delta = 0.5
    thrusts = np.array([f_hover + delta, f_hover - delta, f_hover + delta, f_hover - delta])

    total_t, torques = dynamics.compute_forces_and_torques(thrusts)
    assert math.isclose(total_t, 4.0 * f_hover, rel_tol=1e-5)
    assert math.isclose(torques[0], 0.0, abs_tol=1e-6)  # Roll is zero
    assert math.isclose(torques[1], 0.0, abs_tol=1e-6)  # Pitch is zero
    assert torques[2] > 0.0  # Positive yaw torque

    _, _, _, d_omega = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        total_t,
        torques,
    )
    assert math.isclose(d_omega[0], 0.0, abs_tol=1e-6)
    assert math.isclose(d_omega[1], 0.0, abs_tol=1e-6)
    assert d_omega[2] > 0.0


def test_quaternion_normalization_and_orthogonality() -> None:
    """Quaternion attitude remains strictly normalized and rotation matrix orthogonal."""
    dynamics = DroneDynamics6DOF(dt=0.05, substeps=5)
    rng = np.random.default_rng(12345)

    for _ in range(200):
        random_thrusts = rng.uniform(0.0, 5.0, size=4)
        state = dynamics.step_rk4(random_thrusts)
        norm = np.linalg.norm(state.quaternion)
        assert math.isclose(norm, 1.0, abs_tol=1e-6)

        r_mat = quaternion_to_rotation_matrix(state.quaternion)
        # Check R * R^T = I
        ident = r_mat @ r_mat.T
        assert np.allclose(ident, np.eye(3), atol=1e-5)
        # Check det(R) = 1
        assert math.isclose(np.linalg.det(r_mat), 1.0, abs_tol=1e-5)


def test_motor_saturation() -> None:
    """Motor thrust commands beyond [0, max_thrust] are safely clamped."""
    dynamics = DroneDynamics6DOF(max_thrust_per_motor=5.0)
    over_thrusts = np.array([-10.0, 100.0, 2.5, -0.1])
    total_t, _ = dynamics.compute_forces_and_torques(over_thrusts)
    # Expected clamped: [0.0, 5.0, 2.5, 0.0] -> sum = 7.5
    assert math.isclose(total_t, 7.5, abs_tol=1e-5)


def test_numerical_stability_1000_random_steps() -> None:
    """Environment remains strictly finite (no NaN, no Inf) over 1000 random actions."""
    env = Drone6DOFEnv(max_steps=1500)
    obs, info = env.reset(seed=42)

    assert np.all(np.isfinite(obs))
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)), f"Non-finite observation at step {step}"
        assert math.isfinite(reward), f"Non-finite reward at step {step}"
        assert np.all(np.isfinite(info["position"]))
        assert np.all(np.isfinite(info["velocity"]))
        assert np.all(np.isfinite(info["quaternion"]))
        assert np.all(np.isfinite(info["angular_velocity"]))
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


def test_drone_6dof_deterministic_reset() -> None:
    """Same seed produces identical initial observation and obstacle placement."""
    env1 = Drone6DOFEnv()
    obs1, info1 = env1.reset(seed=999)

    env2 = Drone6DOFEnv()
    obs2, info2 = env2.reset(seed=999)

    assert np.allclose(obs1, obs2)
    assert (
        len(info1["num_obstacles"])
        if isinstance(info1["num_obstacles"], list)
        else info1["num_obstacles"] == info2["num_obstacles"]
    )
    env1.close()
    env2.close()


@pytest.mark.parametrize(
    "invalid_kwargs,match_str",
    [
        ({"bounds": (30.0, 30.0)}, "Bounds"),
        ({"bounds": (30.0, 30.0, -5.0)}, "Bounds"),
        ({"bounds": (float("nan"), 30.0, 15.0)}, "Bounds"),
        ({"bounds": (float("inf"), 30.0, 15.0)}, "Bounds"),
        ({"start_pos": (5.0, 5.0)}, "start_pos"),
        ({"start_pos": (float("nan"), 5.0, 5.0)}, "start_pos"),
        ({"start_pos": (-1.0, 5.0, 5.0)}, "outside valid arena bounds"),
        ({"start_pos": (50.0, 5.0, 5.0)}, "outside valid arena bounds"),
        ({"goal_pos": (5.0, 5.0)}, "goal_pos"),
        ({"goal_pos": (5.0, float("inf"), 5.0)}, "goal_pos"),
        ({"goal_pos": (5.0, -2.0, 5.0)}, "outside valid arena bounds"),
        ({"num_obstacles": -1}, "Obstacle count"),
        ({"num_obstacles": True}, "Obstacle count"),
        ({"num_obstacles": 2.5}, "Obstacle count"),
        ({"obstacle_radius": 0.0}, "Obstacle radius"),
        ({"obstacle_radius": float("nan")}, "Obstacle radius"),
        ({"obstacle_radius": -1.0}, "Obstacle radius"),
        ({"target_radius": 0.0}, "target_radius"),
        ({"collision_radius": -1.0}, "collision_radius"),
        ({"num_lidar_rays": 0}, "LiDAR ray count"),
        ({"num_lidar_rays": True}, "LiDAR ray count"),
        ({"num_lidar_rays": -5}, "LiDAR ray count"),
        ({"lidar_range": 0.0}, "LiDAR range"),
        ({"lidar_range": float("inf")}, "LiDAR range"),
        ({"lidar_noise_std": -0.1}, "lidar_noise_std"),
        ({"lidar_noise_std": float("nan")}, "lidar_noise_std"),
        ({"lidar_dropout_prob": 1.5}, "lidar_dropout_prob"),
        ({"lidar_dropout_prob": float("nan")}, "lidar_dropout_prob"),
        ({"lidar_min_range": -0.1}, "lidar_min_range"),
        ({"lidar_min_range": 25.0}, "lidar_min_range"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
        ({"max_steps": -10}, "max_steps"),
        ({"step_penalty": float("nan")}, "step_penalty"),
        ({"goal_reward": float("inf")}, "goal_reward"),
        ({"collision_reward": float("nan")}, "collision_reward"),
        ({"progress_weight": float("nan")}, "progress_weight"),
        ({"action_penalty_weight": -0.01}, "action_penalty_weight"),
        ({"angular_rate_penalty_weight": float("nan")}, "angular_rate_penalty_weight"),
        ({"tilt_penalty_weight": -0.5}, "tilt_penalty_weight"),
        ({"inertia": (0.01, 0.01)}, "Inertia"),
        ({"inertia": (0.01, 0.01, 0.02, 0.03)}, "Inertia"),
        ({"inertia": (0.01, -0.01, 0.02)}, "Inertia"),
        ({"inertia": (0.01, float("nan"), 0.02)}, "Inertia"),
    ],
)
def test_drone_6dof_invalid_input_validation(invalid_kwargs: dict, match_str: str) -> None:
    """Verify that invalid shapes, non-finite values, and out-of-range inputs raise ValueError."""
    with pytest.raises(ValueError, match=match_str):
        Drone6DOFEnv(**invalid_kwargs)
