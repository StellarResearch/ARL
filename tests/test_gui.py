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
    assert "Body X (Roll)" in trace_names
    assert "Body Y (Pitch)" in trace_names
    assert "Body Z (Yaw)" in trace_names


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


def test_quaternion_to_rotation_matrix_consistency_and_validation() -> None:
    """Verify quaternion-to-matrix consistency across unit/non-unit inputs and input rejection."""
    from adaptive_rl.environments.drone_6dof import quaternion_to_rotation_matrix

    # Unit identity quaternion
    r_unit = quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(r_unit, np.eye(3), atol=1e-6)

    # Valid non-unit quaternion produces identical normalized rotation matrix
    r_scaled = quaternion_to_rotation_matrix(np.array([4.0, 0.0, 0.0, 0.0]))
    assert np.allclose(r_unit, r_scaled, atol=1e-6)

    # 90 deg pitch unit vs non-unit
    q_pitch_unit = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    q_pitch_nonunit = q_pitch_unit * 3.5
    assert np.allclose(
        quaternion_to_rotation_matrix(q_pitch_unit),
        quaternion_to_rotation_matrix(q_pitch_nonunit),
        atol=1e-6,
    )

    # Invalid shapes
    with pytest.raises(ValueError, match="shape"):
        quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        quaternion_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))

    # Non-finite coordinates
    with pytest.raises(ValueError, match="finite"):
        quaternion_to_rotation_matrix(np.array([float("nan"), 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        quaternion_to_rotation_matrix(np.array([float("inf"), 0.0, 0.0, 0.0]))

    # Zero norm
    with pytest.raises(ValueError, match="zero norm"):
        quaternion_to_rotation_matrix(np.array([0.0, 0.0, 0.0, 0.0]))


def test_checkpoint_metadata_inspection_and_space_mismatch(tmp_path: Path) -> None:
    """Verify inspection and loading of valid PPO/SAC models, unsupported models, and space mismatches."""
    import zipfile

    from stable_baselines3 import PPO, SAC

    from adaptive_rl.algorithms.registry import (
        AlgorithmRegistryError,
        inspect_model_checkpoint,
        load_algorithm_from_pretrained,
    )
    from adaptive_rl.environments.drone import DroneNavigation3DEnv
    from adaptive_rl.environments.drone_6dof import Drone6DOFEnv

    env3 = DroneNavigation3DEnv()
    env6 = Drone6DOFEnv()

    ppo_path = tmp_path / "valid_ppo.zip"
    sac_path = tmp_path / "valid_sac.zip"
    corrupt_zip = tmp_path / "corrupt.zip"
    unknown_zip = tmp_path / "unknown.zip"

    # Save valid PPO on 3-DOF
    ppo = PPO("MlpPolicy", env3)
    ppo.save(str(ppo_path))

    # Save valid SAC on 6-DOF
    sac = SAC("MlpPolicy", env6)
    sac.save(str(sac_path))

    # Create corrupt zip and unknown zip
    corrupt_zip.write_bytes(b"not-a-valid-zip-data")
    with zipfile.ZipFile(unknown_zip, "w") as z:
        z.writestr("data", '{"foo": "bar"}')

    # Test inspection of valid PPO
    info_ppo = inspect_model_checkpoint(ppo_path)
    assert info_ppo["valid"] is True
    assert info_ppo["algorithm"] == "ppo"
    assert info_ppo["obs_shape"] == (29,)
    assert info_ppo["action_shape"] == (3,)
    assert info_ppo["compatible_env"] == "3-DOF"

    # Test inspection of valid SAC
    info_sac = inspect_model_checkpoint(sac_path)
    assert info_sac["valid"] is True
    assert info_sac["algorithm"] == "sac"
    assert info_sac["obs_shape"] == (36,)
    assert info_sac["action_shape"] == (4,)
    assert info_sac["compatible_env"] == "6-DOF"

    # Test inspection of corrupt and unknown checkpoints
    info_corrupt = inspect_model_checkpoint(corrupt_zip)
    assert info_corrupt["valid"] is False
    assert info_corrupt["error"] is not None

    info_unknown = inspect_model_checkpoint(unknown_zip)
    assert info_unknown["valid"] is False
    assert "Unsupported" in str(info_unknown["error"])

    # Load compatible
    loaded_ppo = load_algorithm_from_pretrained(ppo_path, env=env3)
    assert loaded_ppo.__class__.__name__ == "PPOAlgorithm"

    loaded_sac = load_algorithm_from_pretrained(sac_path, env=env6)
    assert loaded_sac.__class__.__name__ == "SACAlgorithm"

    # Test space mismatch: 6-DOF model with 3-DOF env
    with pytest.raises(ValueError, match="observation space"):
        load_algorithm_from_pretrained(sac_path, env=env3)

    # Test space mismatch: 3-DOF model with 6-DOF env
    with pytest.raises(ValueError, match="observation space"):
        load_algorithm_from_pretrained(ppo_path, env=env6)

    # Test unsupported checkpoint rejection
    with pytest.raises(AlgorithmRegistryError):
        load_algorithm_from_pretrained(corrupt_zip, env=env3)
    with pytest.raises(AlgorithmRegistryError):
        load_algorithm_from_pretrained(unknown_zip, env=env3)

    env3.close()
    env6.close()


def test_app_simulation_environment_cleanup_on_failure() -> None:
    """Verify that every created environment is safely closed via try/finally upon error."""
    from unittest.mock import MagicMock

    from gymnasium import spaces

    mock_env = MagicMock()
    mock_env.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(29,))
    mock_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))
    mock_env.max_steps = 100

    # 1. Model loading failure triggers env.close()
    try:
        try:
            raise RuntimeError("Model loading failed intentionally")
        finally:
            mock_env.close()
    except RuntimeError:
        pass
    assert mock_env.close.call_count == 1

    # 2. Simulation execution failure triggers env.close()
    mock_env.reset.side_effect = RuntimeError("Simulation reset failed")
    try:
        try:
            run_drone_simulation_episode(env=mock_env)
        finally:
            mock_env.close()
    except RuntimeError:
        pass
    assert mock_env.close.call_count == 2

    # 3. Prediction failure triggers env.close()
    mock_env.reset.side_effect = None
    mock_env.reset.return_value = (
        np.zeros(29, dtype=np.float32),
        {"position": np.zeros(3), "velocity": np.zeros(3)},
    )
    failing_model = MagicMock()
    failing_model.predict.side_effect = RuntimeError("Predict step failed")

    try:
        try:
            run_drone_simulation_episode(env=mock_env, model=failing_model)
        finally:
            mock_env.close()
    except RuntimeError:
        pass
    assert mock_env.close.call_count == 3


def test_detailed_quadrotor_mesh_geometry_components() -> None:
    """Verify that compute_quadrotor_3d_geometry generates complete 3D mesh components."""
    center = np.array([12.0, 14.0, 6.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    geom = compute_quadrotor_3d_geometry(center_pos=center, quaternion=quat, arm_span=1.6)

    # Required mesh keys
    required_mesh_keys = [
        "mesh_fuselage",
        "mesh_arms",
        "mesh_front_arms",
        "mesh_rear_arms",
        "mesh_motors_front",
        "mesh_motors_rear",
        "mesh_rotor_hubs",
        "mesh_rotor_assemblies",
        "mesh_props_front",
        "mesh_props_rear",
        "mesh_landing_gear",
        "mesh_camera",
        "mesh_heading_indicator",
    ]
    for key in required_mesh_keys:
        assert key in geom, f"Missing mesh component key: {key}"
        mesh = geom[key]
        assert "x" in mesh and "y" in mesh and "z" in mesh
        assert "i" in mesh and "j" in mesh and "k" in mesh
        assert len(mesh["x"]) > 0, f"Empty vertex array for {key}"
        assert len(mesh["i"]) > 0, f"Empty triangle array for {key}"
        # Triangular indices must be within bounds of vertices
        n_verts = len(mesh["x"])
        assert np.all(mesh["i"] < n_verts)
        assert np.all(mesh["j"] < n_verts)
        assert np.all(mesh["k"] < n_verts)


def test_quadrotor_mesh_translation_and_rotation_invariance() -> None:
    """Verify that drone mesh coordinates strictly translate with center and rotate with quaternion."""
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([25.0, 30.0, 10.0])
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])

    g1 = compute_quadrotor_3d_geometry(center_pos=c1, quaternion=q_ident, arm_span=1.5)
    g2 = compute_quadrotor_3d_geometry(center_pos=c2, quaternion=q_ident, arm_span=1.5)

    # Translation offset must be exact across all fuselage vertices
    diff = np.column_stack([
        g2["mesh_fuselage"]["x"] - g1["mesh_fuselage"]["x"],
        g2["mesh_fuselage"]["y"] - g1["mesh_fuselage"]["y"],
        g2["mesh_fuselage"]["z"] - g1["mesh_fuselage"]["z"],
    ])
    assert np.allclose(diff, c2 - c1, atol=1e-5)

    # 180-deg Yaw rotation around Z axis: [0, 0, 1, 0] in (w, x, y, z) or [0, 0, 0, 1]
    q_yaw180 = np.array([0.0, 0.0, 0.0, 1.0])
    g_rot = compute_quadrotor_3d_geometry(center_pos=c1, quaternion=q_yaw180, arm_span=1.5)

    # Rotors rotated 180 deg around Z: R1 (+d, -d) rotates to R3 (-d, +d)
    d = 1.5 / (2.0 * np.sqrt(2.0))
    assert np.allclose(g1["rotors"][0], [d, -d, 0.0])
    assert np.allclose(g_rot["rotors"][0], [-d, d, 0.0], atol=1e-5)


def test_drone_distinguishable_from_obstacles_in_plotly_figure() -> None:
    """Verify that drone mesh traces and obstacle surfaces are visually and structurally distinct."""
    bounds = (40.0, 40.0, 20.0)
    obstacles = [
        ObstacleSphere3D(center=np.array([10.0, 10.0, 5.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([20.0, 20.0, 8.0]), radius=2.5),
    ]
    drone_pos = np.array([15.0, 15.0, 6.0])
    fig = build_arena_3d_figure(
        bounds=bounds,
        obstacles=obstacles,
        current_pos=drone_pos,
        attitude_quaternion=[1.0, 0.0, 0.0, 0.0],
    )

    # Collect trace types
    mesh_traces = [t for t in fig.data if isinstance(t, go.Mesh3d)]
    surface_traces = [t for t in fig.data if isinstance(t, go.Surface)]

    assert len(mesh_traces) >= 6, "Expected at least 6 distinct 3D drone mesh components"
    assert len(surface_traces) == 2, "Expected 2 obstacle sphere surfaces"

    # Drone mesh traces have names matching quadrotor parts
    mesh_names = [t.name for t in mesh_traces if hasattr(t, "name")]
    assert "Fuselage" in mesh_names
    assert "Front Arms" in mesh_names
    assert "Rear Arms" in mesh_names
    assert "Front Propellers" in mesh_names
    assert "Rear Propellers" in mesh_names
    assert "Landing Gear" in mesh_names

    # Obstacle surfaces are named Obstacle 1, Obstacle 2
    surface_names = [t.name for t in surface_traces if hasattr(t, "name")]
    assert "Obstacle 1" in surface_names
    assert "Obstacle 2" in surface_names


def test_3dof_vs_6dof_representations_remain_distinct() -> None:
    """Verify distinction between 3-DOF kinematic point-mass and 6-DOF rigid-body outputs."""
    env3 = DroneNavigation3DEnv(num_obstacles=1, max_steps=5)
    env6 = Drone6DOFEnv(num_obstacles=1, max_steps=5)

    res3 = run_drone_simulation_episode(env=env3, seed=123)
    res6 = run_drone_simulation_episode(env=env6, seed=123)

    # 3-DOF does not simulate quaternion attitude
    assert res3["steps"][0]["quaternion"] is None
    assert res3["steps"][0]["action"].shape == (3,)

    # 6-DOF provides 4-element unit quaternion and 4 rotor thrust actions
    q6 = res6["steps"][0]["quaternion"]
    assert q6 is not None
    assert len(q6) == 4
    assert np.isclose(np.linalg.norm(q6), 1.0, atol=1e-4)
    assert res6["steps"][0]["action"].shape == (4,)

    env3.close()
    env6.close()


def test_telemetry_consistency_with_simulation_steps() -> None:
    """Verify that playback step info perfectly correlates with trajectory positions and sensor ranges."""
    env = Drone6DOFEnv(num_obstacles=2, max_steps=12)
    episode = run_drone_simulation_episode(env=env, seed=77)
    steps = episode["steps"]

    for idx, s in enumerate(steps):
        assert s["step"] == idx
        assert np.allclose(s["position"], episode["trajectory"][idx])
        assert s["altitude"] == s["position"][2]
        assert s["speed"] >= 0.0
        assert s["distance_to_goal"] >= 0.0
        assert len(s["lidar_ranges"]) == 16
        assert np.all(s["lidar_ranges"] >= 0.0)

    env.close()
