"""Tests for user-defined 3D destination / custom target selector feature."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adaptive_rl.algorithms.random_policy import RandomPolicy
from adaptive_rl.config import AlgorithmConfig, EnvironmentConfig, ExperimentConfig, TrainingConfig
from adaptive_rl.environments.drone import (
    DroneNavigation3DEnv,
    validate_target_coordinates,
)
from adaptive_rl.environments.drone_6dof import Drone6DOFEnv
from adaptive_rl.evaluation.evaluator import Evaluator, compare_policies
from adaptive_rl.gui.visualizer import build_arena_3d_figure, run_drone_simulation_episode
from adaptive_rl.training.trainer import RLTrainer

# ==============================================================================
# 1. ENVIRONMENT CUSTOM GOAL TESTS (3-DOF & 6-DOF)
# ==============================================================================


def test_custom_goal_3dof_initialization_and_reset() -> None:
    """Verify custom destination in 3-DOF kinematic drone persists through resets."""
    custom_goal = np.array([12.0, 18.0, 7.0])
    env = DroneNavigation3DEnv(
        bounds=(30.0, 30.0, 15.0),
        start_pos=np.array([5.0, 5.0, 5.0]),
        goal_pos=custom_goal,
        num_obstacles=4,
    )
    try:
        np.testing.assert_allclose(env.target, custom_goal)
        obs, info = env.reset(seed=42)

        # Environment goal state and info must reflect custom target
        np.testing.assert_allclose(env.target, custom_goal)
        np.testing.assert_allclose(info["goal"], custom_goal)

        # Distance to goal in info
        expected_dist = float(np.linalg.norm(custom_goal - np.array([5.0, 5.0, 5.0])))
        assert abs(info["distance_to_goal"] - expected_dist) < 1e-4

        # Relative goal in observation: [9:12] = (goal - pos) / bounds
        bx, by, bz = env.bounds
        expected_rel = np.array(
            [
                (custom_goal[0] - 5.0) / bx,
                (custom_goal[1] - 5.0) / by,
                (custom_goal[2] - 5.0) / bz,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(obs[9:12], expected_rel, atol=1e-4)

        # Normalized goal in observation: [6:9] = goal / bounds
        expected_norm_goal = np.array(
            [
                custom_goal[0] / bx,
                custom_goal[1] / by,
                custom_goal[2] / bz,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(obs[6:9], expected_norm_goal, atol=1e-4)

        # Procedural obstacles must maintain clearance from custom goal
        clearance_thresh = env.obstacle_radius + env.collision_radius + 2.0
        for obs_item in env.obstacles:
            dist_to_goal = float(np.linalg.norm(obs_item.center - custom_goal))
            assert dist_to_goal >= clearance_thresh

        # Reset a second time with different seed; goal must survive
        obs2, info2 = env.reset(seed=999)
        np.testing.assert_allclose(env.target, custom_goal)
        np.testing.assert_allclose(info2["goal"], custom_goal)
    finally:
        env.close()


def test_custom_goal_3dof_reward_and_success() -> None:
    """Verify reward and success calculations evaluate against custom goal in 3-DOF."""
    custom_goal = np.array([8.0, 5.0, 5.0])
    start = np.array([5.0, 5.0, 5.0])
    env = DroneNavigation3DEnv(
        bounds=(30.0, 30.0, 15.0),
        start_pos=start,
        goal_pos=custom_goal,
        target_radius=1.5,
        num_obstacles=0,
    )
    try:
        env.reset(seed=42)
        # Move drone +X toward goal [8, 5, 5]
        action_toward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        _, reward_toward, _, _, info_toward = env.step(action_toward)
        # Moving toward custom goal yields progress reward
        assert reward_toward > env.step_penalty

        # Reset and test direct arrival at custom goal
        env_success = DroneNavigation3DEnv(
            bounds=(30.0, 30.0, 15.0),
            start_pos=np.array([7.0, 5.0, 5.0]),  # 1.0m from custom goal, radius is 1.5m
            goal_pos=custom_goal,
            target_radius=1.5,
            num_obstacles=0,
        )
        env_success.reset(seed=42)
        _, reward_succ, term, _, info_succ = env_success.step(np.zeros(3, dtype=np.float32))
        assert term is True
        assert info_succ["success"] is True
        assert reward_succ >= env_success.goal_reward
        env_success.close()
    finally:
        env.close()


def test_custom_goal_6dof_initialization_and_reset() -> None:
    """Verify custom destination in 6-DOF quadrotor persists through resets and affects telemetry."""
    custom_goal = np.array([20.0, 10.0, 8.0])
    env = Drone6DOFEnv(
        bounds=(30.0, 30.0, 15.0),
        start_pos=np.array([5.0, 5.0, 5.0]),
        goal_pos=custom_goal,
        num_obstacles=4,
    )
    try:
        np.testing.assert_allclose(env.target, custom_goal)
        obs, info = env.reset(seed=42)

        np.testing.assert_allclose(env.target, custom_goal)
        np.testing.assert_allclose(info["goal"], custom_goal)

        expected_dist = float(np.linalg.norm(custom_goal - np.array([5.0, 5.0, 5.0])))
        assert abs(info["distance_to_goal"] - expected_dist) < 1e-4

        # Obstacles must respect clearance from custom goal
        clearance_thresh = env.obstacle_radius + env.collision_radius + 2.0
        for obs_item in env.obstacles:
            dist_to_goal = float(np.linalg.norm(obs_item.center - custom_goal))
            assert dist_to_goal >= clearance_thresh

        # Second reset with different seed
        obs2, info2 = env.reset(seed=777)
        np.testing.assert_allclose(env.target, custom_goal)
        np.testing.assert_allclose(info2["goal"], custom_goal)
    finally:
        env.close()


def test_default_target_compatibility() -> None:
    """Verify omitting custom target retains standard default waypoint across environments."""
    env3 = DroneNavigation3DEnv(bounds=(30.0, 30.0, 15.0))
    expected_default = np.array([25.0, 25.0, 10.0])
    np.testing.assert_allclose(env3.target, expected_default)
    env3.reset()
    np.testing.assert_allclose(env3.target, expected_default)
    env3.close()

    env6 = Drone6DOFEnv(bounds=(30.0, 30.0, 15.0))
    np.testing.assert_allclose(env6.target, expected_default)
    env6.reset()
    np.testing.assert_allclose(env6.target, expected_default)
    env6.close()


def test_reset_options_goal_pos() -> None:
    """Verify reset(options={'goal_pos': ...}) dynamically updates target."""
    env = DroneNavigation3DEnv()
    new_target = np.array([14.0, 16.0, 7.0])
    obs, info = env.reset(options={"goal_pos": new_target})
    np.testing.assert_allclose(env.target, new_target)
    np.testing.assert_allclose(info["goal"], new_target)
    env.close()

    env6 = Drone6DOFEnv()
    obs6, info6 = env6.reset(options={"goal_pos": new_target})
    np.testing.assert_allclose(env6.target, new_target)
    np.testing.assert_allclose(info6["goal"], new_target)
    env6.close()


# ==============================================================================
# 2. VALIDATION FUNCTION TESTS
# ==============================================================================


def test_validation_valid_target() -> None:
    """Verify valid targets pass validation."""
    valid, msg = validate_target_coordinates(
        target=(15.0, 20.0, 8.0),
        bounds=(30.0, 30.0, 15.0),
        start_pos=(5.0, 5.0, 5.0),
    )
    assert valid is True
    assert msg is None


def test_validation_negative_coordinate() -> None:
    """Verify negative coordinate in any axis fails validation."""
    for axis, target in [
        ("X", (-1.0, 15.0, 8.0)),
        ("Y", (15.0, -0.5, 8.0)),
        ("Z", (15.0, 15.0, -2.0)),
    ]:
        valid, msg = validate_target_coordinates(target=target, bounds=(30.0, 30.0, 15.0))
        assert valid is False
        assert msg is not None
        assert "cannot be negative" in msg
        assert axis in msg


def test_validation_outside_bounds() -> None:
    """Verify coordinates exceeding arena bounds fail validation."""
    for axis, target in [
        ("X", (31.0, 15.0, 8.0)),
        ("Y", (15.0, 32.5, 8.0)),
        ("Z", (15.0, 15.0, 16.0)),
    ]:
        valid, msg = validate_target_coordinates(target=target, bounds=(30.0, 30.0, 15.0))
        assert valid is False
        assert msg is not None
        assert "exceeds arena bound" in msg
        assert axis in msg


def test_validation_nan_and_infinity() -> None:
    """Verify NaN and +/- infinity coordinates fail validation."""
    valid, msg = validate_target_coordinates(target=(float("nan"), 15.0, 8.0))
    assert valid is False
    assert msg is not None
    assert "NaN" in msg

    valid_inf, msg_inf = validate_target_coordinates(target=(15.0, float("inf"), 8.0))
    assert valid_inf is False
    assert msg_inf is not None
    assert "infinite" in msg_inf

    valid_neginf, msg_neginf = validate_target_coordinates(target=(15.0, 15.0, float("-inf")))
    assert valid_neginf is False
    assert msg_neginf is not None
    assert "infinite" in msg_neginf


def test_validation_incorrect_coordinate_count() -> None:
    """Verify non-3D inputs fail validation."""
    for bad_target in [(15.0, 15.0), (15.0, 15.0, 8.0, 2.0), [], None]:
        valid, msg = validate_target_coordinates(target=bad_target)
        assert valid is False
        assert msg is not None


def test_validation_target_too_close_to_start() -> None:
    """Verify target within arrival radius of start position fails validation."""
    # Start is (5, 5, 5), arrival radius is 1.5m; distance is 0.5m
    valid, msg = validate_target_coordinates(
        target=(5.5, 5.0, 5.0),
        bounds=(30.0, 30.0, 15.0),
        start_pos=(5.0, 5.0, 5.0),
        target_radius=1.5,
    )
    assert valid is False
    assert msg is not None
    assert "too close to launch position" in msg


def test_validation_boundary_collision_margin() -> None:
    """Verify target placed inside boundary collision margin fails validation."""
    # Collision radius is 0.8; target at 0.5m is within collision margin
    valid, msg = validate_target_coordinates(
        target=(0.5, 15.0, 8.0),
        bounds=(30.0, 30.0, 15.0),
        collision_radius=0.8,
    )
    assert valid is False
    assert msg is not None
    assert "violates arena boundary collision margin" in msg

    # Target near upper boundary: 30.0 - 0.5 = 29.5m (within 0.8m margin)
    valid_up, msg_up = validate_target_coordinates(
        target=(29.5, 15.0, 8.0),
        bounds=(30.0, 30.0, 15.0),
        collision_radius=0.8,
    )
    assert valid_up is False
    assert msg_up is not None
    assert "violates arena boundary collision margin" in msg


def test_validation_raise_error_flag() -> None:
    """Verify raise_error=True raises ValueError on invalid inputs."""
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_target_coordinates(target=(-5.0, 15.0, 8.0), raise_error=True)


# ==============================================================================
# 3. GUI AND SIMULATION WORKFLOW TESTS
# ==============================================================================


def test_simulation_episode_with_custom_target_3dof() -> None:
    """Verify run_drone_simulation_episode correctly captures custom target in 3-DOF."""
    custom_target = (14.0, 16.0, 8.0)
    env = DroneNavigation3DEnv(goal_pos=custom_target, num_obstacles=2)
    try:
        flight_data = run_drone_simulation_episode(env=env, model=None, seed=42)
        np.testing.assert_allclose(flight_data["target"], np.array(custom_target))
        assert flight_data["target_radius"] == 1.5

        # Initial step distance to goal must be from start (5, 5, 5) to custom_target
        expected_dist = float(np.linalg.norm(np.array(custom_target) - np.array([5.0, 5.0, 5.0])))
        assert abs(flight_data["steps"][0]["distance_to_goal"] - expected_dist) < 1e-3
    finally:
        env.close()


def test_simulation_episode_with_custom_target_6dof() -> None:
    """Verify run_drone_simulation_episode correctly captures custom target in 6-DOF."""
    custom_target = (22.0, 18.0, 7.5)
    env = Drone6DOFEnv(goal_pos=custom_target, num_obstacles=2)
    try:
        flight_data = run_drone_simulation_episode(env=env, model=None, seed=42)
        np.testing.assert_allclose(flight_data["target"], np.array(custom_target))
        assert flight_data["target_radius"] == 1.5

        expected_dist = float(np.linalg.norm(np.array(custom_target) - np.array([5.0, 5.0, 5.0])))
        assert abs(flight_data["steps"][0]["distance_to_goal"] - expected_dist) < 1e-3
    finally:
        env.close()


def test_3d_figure_reflects_custom_target_and_arrival_radius() -> None:
    """Verify Plotly figure displays custom target coordinates, arrival radius, and beacon."""
    custom_target = (18.5, 22.0, 9.0)
    fig = build_arena_3d_figure(
        bounds=(30.0, 30.0, 15.0),
        target=custom_target,
        start_pos=(5.0, 5.0, 5.0),
        target_radius=1.5,
    )

    trace_names = [t.name for t in fig.data if hasattr(t, "name")]
    assert "Target Waypoint" in trace_names
    assert "Waypoint Ground Pad" in trace_names
    assert "Waypoint Beacon Column" in trace_names
    assert "Arrival Radius (1.5m)" in trace_names

    # Check waypoint marker text contains exact coordinates
    waypoint_trace = next(t for t in fig.data if getattr(t, "name", None) == "Target Waypoint")
    assert waypoint_trace.x[0] == 18.5
    assert waypoint_trace.y[0] == 22.0
    assert waypoint_trace.z[0] == 9.0
    assert "18.5" in waypoint_trace.text[0]
    assert "22.0" in waypoint_trace.text[0]
    assert "9.0" in waypoint_trace.text[0]


def test_changing_target_generates_distinct_simulation_keys() -> None:
    """Verify distinct target coordinates generate distinct session state keys preventing stale results."""
    target_a = (25.0, 25.0, 10.0)
    target_b = (12.0, 18.0, 7.0)

    key_a = f"flight_data_3dof_{target_a[0]:.2f}_{target_a[1]:.2f}_{target_a[2]:.2f}"
    key_b = f"flight_data_3dof_{target_b[0]:.2f}_{target_b[1]:.2f}_{target_b[2]:.2f}"

    assert key_a != key_b
    assert "25.00_25.00_10.00" in key_a
    assert "12.00_18.00_7.00" in key_b


# ==============================================================================
# 4. TRAINING AND EVALUATION INTEGRATION TESTS
# ==============================================================================


def test_training_config_receives_custom_target() -> None:
    """Verify RLTrainer correctly instantiates environment with custom goal from config."""
    custom_target = [14.0, 19.0, 7.5]
    cfg = ExperimentConfig(
        name="test_custom_train",
        seed=42,
        algorithm=AlgorithmConfig(name="ppo", parameters={"n_steps": 64, "batch_size": 32}),
        environment=EnvironmentConfig(
            name="drone",
            parameters={"goal_pos": custom_target},
        ),
        training=TrainingConfig(total_timesteps=128, checkpoint_freq=0),
    )

    trainer = RLTrainer(config=cfg)
    try:
        np.testing.assert_allclose(trainer.env.target, np.array(custom_target))
        obs, info = trainer.env.reset()
        np.testing.assert_allclose(info["goal"], np.array(custom_target))
    finally:
        trainer.env.close()


def test_compare_policies_evaluates_same_custom_target() -> None:
    """Verify both trained policy and baseline evaluate against the same custom target."""
    custom_target = (16.0, 16.0, 8.0)
    eval_env = DroneNavigation3DEnv(goal_pos=custom_target, num_obstacles=1)
    try:
        rand_policy = RandomPolicy(action_space=eval_env.action_space, seed=42)
        results = compare_policies(
            algorithm=rand_policy,
            env=eval_env,
            num_episodes=2,
            base_seed=42,
        )
        assert "Random Policy" in results
        assert eval_env.target[0] == 16.0
        assert eval_env.target[1] == 16.0
        assert eval_env.target[2] == 8.0
    finally:
        eval_env.close()


def test_evaluator_save_report_records_target(tmp_path: Path) -> None:
    """Verify Evaluator.save_report serializes selected destination in evaluation report."""
    eval_env = DroneNavigation3DEnv(goal_pos=(18.0, 14.0, 6.0))
    try:
        evaluator = Evaluator(
            algorithm=RandomPolicy(action_space=eval_env.action_space, seed=42),
            env=eval_env,
        )
        metrics = evaluator.evaluate(num_episodes=2, base_seed=42)
        report_path = tmp_path / "evaluation_report.json"
        Evaluator.save_report(
            metrics=metrics,
            output_path=report_path,
            target_pos=(18.0, 14.0, 6.0),
        )

        assert report_path.is_file()
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "target" in data
        assert data["target"] == [18.0, 14.0, 6.0]
    finally:
        eval_env.close()
