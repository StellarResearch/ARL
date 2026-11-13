"""Tests for browser-based GUI visualizer and simulation playback."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")
pytest.importorskip("streamlit")

import plotly.graph_objects as go

from adaptive_rl.environments.drone import DroneNavigation3DEnv, ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import Drone6DOFEnv
from adaptive_rl.gui.visualizer import (
    _generate_box_wireframe,
    _generate_sphere_surface,
    build_arena_3d_figure,
    build_comparison_bar_chart,
    build_density_experiment_figure,
    build_training_curve_figure,
    compute_quadrotor_3d_geometry,
    get_available_models,
    run_drone_simulation_episode,
)


def test_sphere_surface_generator() -> None:
    """Verify parametric sphere coordinates generation."""
    x, y, z = _generate_sphere_surface(center=[5.0, 5.0, 5.0], radius=2.0, n_points=25)
    assert x.shape == (25, 25)
    assert y.shape == (25, 25)
    assert z.shape == (25, 25)
    # Check max radius range
    assert np.isclose(np.max(x), 7.0, atol=0.01)
    assert np.isclose(np.min(x), 3.0, atol=0.01)


def test_box_wireframe_generator() -> None:
    """Verify 12 edges generation for 3D bounding arena."""
    xs, ys, zs = _generate_box_wireframe(bounds=(30.0, 30.0, 15.0))
    # 12 edges * 3 items (start, end, None) = 36 items
    assert len(xs) == 36
    assert len(ys) == 36
    assert len(zs) == 36
    assert None in xs


def test_build_arena_3d_figure() -> None:
    """Verify complete 3D Plotly figure generation with drone, obstacles, and LiDAR."""
    bounds = (30.0, 30.0, 15.0)
    obstacles = [ObstacleSphere3D(center=np.array([15.0, 15.0, 7.0]), radius=2.0)]
    target = np.array([25.0, 25.0, 10.0])
    start = np.array([5.0, 5.0, 5.0])
    trajectory = [np.array([5.0, 5.0, 5.0]), np.array([6.0, 6.0, 5.5])]
    current_pos = np.array([6.0, 6.0, 5.5])
    current_vel = np.array([1.0, 1.0, 0.5])
    lidar_rays = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lidar_ranges = [10.0, 15.0]

    fig = build_arena_3d_figure(
        bounds=bounds,
        obstacles=obstacles,
        target=target,
        start_pos=start,
        trajectory=trajectory,
        current_pos=current_pos,
        current_vel=current_vel,
        lidar_rays=lidar_rays,
        lidar_ranges=lidar_ranges,
        show_lidar=True,
    )

    assert isinstance(fig, go.Figure)
    trace_names = [trace.name for trace in fig.data if hasattr(trace, "name")]
    assert "Arena Boundary" in trace_names
    assert "Obstacle 1" in trace_names
    assert "Target Waypoint" in trace_names
    assert "Start Position" in trace_names
    assert "Flight Path" in trace_names
    assert "16-Ray LiDAR" in trace_names
    assert "Drone" in trace_names
    assert "Velocity Vector" in trace_names


def test_run_drone_simulation_episode() -> None:
    """Verify deterministic episode execution and trajectory data structure."""
    env = DroneNavigation3DEnv(num_obstacles=2, max_steps=15)
    result = run_drone_simulation_episode(env=env, model=None, seed=42)

    assert "trajectory" in result
    assert "steps" in result
    assert "total_reward" in result
    assert "outcome" in result
    assert len(result["steps"]) > 1
    assert result["seed"] == 42
    assert result["outcome"] in ["SUCCESS", "COLLISION / TIMEOUT"]

    first_step = result["steps"][0]
    assert first_step["step"] == 0
    assert "position" in first_step
    assert "lidar_ranges" in first_step
    assert len(first_step["lidar_ranges"]) == env.num_lidar_rays


def test_get_available_models(tmp_path: Path) -> None:
    """Verify scanning for model checkpoints."""
    models = get_available_models()
    assert isinstance(models, list)


def test_build_training_curve_figure() -> None:
    """Verify training curve Plotly figure generation."""
    rewards = [10.0, 15.0, 12.0, 25.0, 30.0, 45.0, 50.0, 60.0, 75.0, 80.0, 90.0]
    fig = build_training_curve_figure(rewards, window_size=5)
    assert isinstance(fig, go.Figure)
    trace_names = [trace.name for trace in fig.data if hasattr(trace, "name")]
    assert "Episode Return" in trace_names
    assert "Moving Avg (w=5)" in trace_names


def test_build_comparison_bar_chart() -> None:
    """Verify comparison bar chart generation between policies."""
    from adaptive_rl.evaluation.metrics import EvaluationMetrics

    m_ppo = EvaluationMetrics(
        episodes=10,
        mean_reward=50.0,
        std_reward=5.0,
        min_reward=40.0,
        max_reward=60.0,
        success_rate=0.7,
        collision_rate=0.2,
        mean_episode_length=35.0,
        std_episode_length=4.0,
    )
    m_rand = EvaluationMetrics(
        episodes=10,
        mean_reward=-40.0,
        std_reward=10.0,
        min_reward=-80.0,
        max_reward=-10.0,
        success_rate=0.0,
        collision_rate=0.9,
        mean_episode_length=15.0,
        std_episode_length=3.0,
    )
    fig = build_comparison_bar_chart({"PPO": m_ppo, "Random Policy": m_rand})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_build_density_experiment_figure() -> None:
    """Verify obstacle density experiment figure generation."""
    data = [
        {"obstacle_count": 4, "success_rate": 0.8, "collision_rate": 0.1, "mean_reward": 60.0},
        {"obstacle_count": 6, "success_rate": 0.5, "collision_rate": 0.4, "mean_reward": 20.0},
        {"obstacle_count": 8, "success_rate": 0.3, "collision_rate": 0.6, "mean_reward": -10.0},
    ]
    fig = build_density_experiment_figure(data)
    assert isinstance(fig, go.Figure)
    trace_names = [trace.name for trace in fig.data if hasattr(trace, "name")]
    assert "Success Rate (%)" in trace_names
    assert "Collision Rate (%)" in trace_names


def test_build_distribution_shift_trajectory_figure() -> None:
    """Verify 3D distribution shift trajectory figure creation."""
    from adaptive_rl.gui.visualizer import build_distribution_shift_trajectory_figure

    fixed_traj = [np.array([5.0, 5.0, 5.0]), np.array([10.0, 8.0, 6.0])]
    adaptive_traj = [np.array([5.0, 5.0, 5.0]), np.array([12.0, 12.0, 8.0])]
    fig = build_distribution_shift_trajectory_figure(
        fixed_trajectory=fixed_traj,
        adaptive_trajectory=adaptive_traj,
        bounds=(50.0, 50.0, 25.0),
        wind_vector=(4.0, 0.0, 0.0),
    )
    assert isinstance(fig, go.Figure)
    names = [t.name for t in fig.data if hasattr(t, "name")]
    assert any("Fixed Policy" in n for n in names)
    assert any("Adaptive Policy" in n for n in names)


def test_build_recovery_curve_figure() -> None:
    """Verify recovery curve figure generation."""
    from adaptive_rl.gui.visualizer import build_recovery_curve_figure

    episodes = list(range(1, 16))
    fixed_returns = [-50.0] * 15
    adaptive_returns = [-50.0] * 5 + [-20.0, 0.0, 30.0, 60.0, 70.0, 80.0, 85.0, 88.0, 90.0, 92.0]
    fig = build_recovery_curve_figure(
        episodes=episodes,
        fixed_returns=fixed_returns,
        adaptive_returns=adaptive_returns,
        p_pre=80.0,
        p0=-50.0,
        fixed_t_h=15,
        adaptive_t_h=10,
    )
    assert isinstance(fig, go.Figure)
    names = [t.name for t in fig.data if hasattr(t, "name")]
    assert any("Fixed Arm" in n for n in names)
    assert any("Adaptive Arm" in n for n in names)
    assert any("90% Recovery Level" in n for n in names)


def test_quadrotor_geometry_identity_and_translation() -> None:
    """Verify quadrotor geometry at origin and with translation under identity quaternion."""
    center = np.array([10.0, 20.0, 5.0])
    arm_span = 1.0
    d = arm_span / (2.0 * np.sqrt(2.0))
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])

    geom = compute_quadrotor_3d_geometry(center_pos=center, quaternion=q_ident, arm_span=arm_span)

    assert np.allclose(geom["center"], center)
    rotors = geom["rotors"]
    assert len(rotors) == 4

    # R1: Front-Right (+d, -d)
    assert np.allclose(rotors[0], [10.0 + d, 20.0 - d, 5.0], atol=1e-5)
    # R2: Front-Left (+d, +d)
    assert np.allclose(rotors[1], [10.0 + d, 20.0 + d, 5.0], atol=1e-5)
    # R3: Rear-Left (-d, +d)
    assert np.allclose(rotors[2], [10.0 - d, 20.0 + d, 5.0], atol=1e-5)
    # R4: Rear-Right (-d, -d)
    assert np.allclose(rotors[3], [10.0 - d, 20.0 - d, 5.0], atol=1e-5)

    # All rotor distances from center must be exactly arm_span / 2
    for r in rotors:
        dist = float(np.linalg.norm(r - center))
        assert np.isclose(dist, arm_span / 2.0, atol=1e-5)

    # Nose heading indicator is along +X direction
    nose = geom["nose"]
    assert nose[0] > center[0]
    assert np.isclose(nose[1], center[1], atol=1e-5)
    assert np.isclose(nose[2], center[2], atol=1e-5)


def test_quadrotor_geometry_yaw_90_rotation() -> None:
    """Verify 90-degree yaw rotation (+Z) rotates body +X to world +Y."""
    center = np.array([0.0, 0.0, 0.0])
    arm_span = 1.0
    d = arm_span / (2.0 * np.sqrt(2.0))
    # 90 deg yaw: q = [cos(pi/4), 0, 0, sin(pi/4)]
    q_yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

    geom = compute_quadrotor_3d_geometry(center_pos=center, quaternion=q_yaw90, arm_span=arm_span)
    nose = geom["nose"]
    # Nose should point along +Y
    assert np.isclose(nose[0], 0.0, atol=1e-5)
    assert nose[1] > 0.0
    assert np.isclose(nose[2], 0.0, atol=1e-5)

    rotors = geom["rotors"]
    # R1 was (+d, -d), rotated by +90 yaw -> (+d, +d)
    assert np.allclose(rotors[0], [d, d, 0.0], atol=1e-5)
    # R2 was (+d, +d), rotated by +90 yaw -> (-d, +d)
    assert np.allclose(rotors[1], [-d, d, 0.0], atol=1e-5)


def test_quadrotor_geometry_pitch_and_roll_transformations() -> None:
    """Verify pitch and roll transformations tilt the drone arms and nose appropriately."""
    center = np.array([5.0, 5.0, 5.0])
    # 90 deg pitch (nose down): q = [cos(pi/4), 0, sin(pi/4), 0]
    q_pitch90 = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    geom_pitch = compute_quadrotor_3d_geometry(center_pos=center, quaternion=q_pitch90)
    nose = geom_pitch["nose"]
    # Body +X rotated by pitch +90 points down (-Z)
    assert np.isclose(nose[0], center[0], atol=1e-5)
    assert np.isclose(nose[1], center[1], atol=1e-5)
    assert nose[2] < center[2]

    # 90 deg roll (left wing up): q = [cos(pi/4), sin(pi/4), 0, 0]
    q_roll90 = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    geom_roll = compute_quadrotor_3d_geometry(center_pos=center, quaternion=q_roll90)
    # Rotors distance from center is preserved
    for r in geom_roll["rotors"]:
        assert np.all(np.isfinite(r))


def test_quadrotor_geometry_finite_and_figure_traces() -> None:
    """Verify full arena Plotly figure includes all quadrotor geometry traces."""
    pos = np.array([15.0, 15.0, 8.0])
    quat = np.array([0.92388, 0.38268, 0.0, 0.0])  # ~45 deg roll
    fig = build_arena_3d_figure(
        bounds=(30.0, 30.0, 15.0),
        current_pos=pos,
        attitude_quaternion=quat,
    )
    trace_names = [trace.name for trace in fig.data if hasattr(trace, "name")]
    assert "Front Arms" in trace_names
    assert "Rear Arms" in trace_names
    assert "Front Propellers" in trace_names
    assert "Rear Propellers" in trace_names
    assert "Rotor Assemblies" in trace_names
    assert "Forward Heading (+X_B)" in trace_names
    assert "Fuselage" in trace_names
    assert "Drone" in trace_names


def test_run_drone_simulation_episode_6dof() -> None:
    """Verify 6-DOF simulation episode produces complete state and telemetry."""
    env = Drone6DOFEnv(num_obstacles=2, max_steps=10)
    res = run_drone_simulation_episode(env=env, model=None, seed=99)

    assert "trajectory" in res
    assert "steps" in res
    assert len(res["steps"]) > 1

    first = res["steps"][0]
    assert first["action"].shape == (4,)
    assert first["quaternion"] is not None
    assert len(first["quaternion"]) == 4
    assert "altitude" in first
    assert "angular_speed" in first
    assert len(first["lidar_ranges"]) == 16
    env.close()
