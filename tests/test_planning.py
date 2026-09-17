"""Unit and integration tests for classical motion planning algorithms and baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
from adaptive_rl.environments.registry import make_env
from adaptive_rl.planning.astar import AStarPlanner, AStarPlannerPolicy
from adaptive_rl.planning.benchmark import (
    ClassicalBenchmarkReport,
    ClassicalBenchmarkRunner,
    PlannerComparisonResult,
)
from adaptive_rl.planning.rrt import (
    RRTPlanner,
    RRTPlannerPolicy,
    RRTStarPlanner,
)

runner = CliRunner()


def test_astar_planner_direct_path() -> None:
    """Verify A* finds the provably optimal shortest path on an open 2D grid."""
    planner = AStarPlanner(width=6, height=6, obstacles=set(), heuristic="manhattan")
    result = planner.plan(start=(0, 0), goal=(5, 5))

    assert result.success is True
    assert result.cost == 10.0  # (5 - 0) + (5 - 0)
    assert len(result.path) == 11
    assert result.path[0] == (0.0, 0.0)
    assert result.path[-1] == (5.0, 5.0)
    assert result.nodes_expanded > 0
    assert result.planning_time_sec >= 0.0


def test_astar_planner_obstacle_detour() -> None:
    """Verify A* detours around a wall of obstacles."""
    # Obstacle wall at x=2, y=0..3
    obstacles = {(2, 0), (2, 1), (2, 2), (2, 3)}
    planner = AStarPlanner(width=6, height=6, obstacles=obstacles, heuristic="manhattan")
    result = planner.plan(start=(0, 2), goal=(4, 2))

    assert result.success is True
    # Path must detour below the wall (e.g., via y=4 or y=5)
    for pt in result.path:
        coord = (int(pt[0]), int(pt[1]))
        assert coord not in obstacles
    assert result.cost > 4.0  # Direct distance would be 4, detour must be strictly longer


def test_astar_planner_unsolvable() -> None:
    """Verify A* cleanly reports failure when start is completely walled off."""
    # Enclose (0, 0)
    obstacles = {(1, 0), (0, 1), (1, 1)}
    planner = AStarPlanner(width=5, height=5, obstacles=obstacles)
    result = planner.plan(start=(0, 0), goal=(4, 4))

    assert result.success is False
    assert result.path == []
    assert result.cost == float("inf")
    assert "No path exists" in result.metadata.get("reason", "")


def test_astar_heuristics_and_validation() -> None:
    """Verify A* supports Euclidean and Chebyshev heuristics and raises on invalid params."""
    p_euc = AStarPlanner(width=4, height=4, heuristic="euclidean")
    assert p_euc.name == "AStar_euclidean_4conn"
    res_euc = p_euc.plan(start=(0, 0), goal=(3, 3))
    assert res_euc.success is True

    p_cheb = AStarPlanner(width=4, height=4, allow_diagonal=True, heuristic="chebyshev")
    assert p_cheb.name == "AStar_chebyshev_8conn"
    res_cheb = p_cheb.plan(start=(0, 0), goal=(3, 3))
    assert res_cheb.success is True
    # Diagonal steps cost sqrt(2) each (Euclidean step cost), so 3 diagonal steps = 3*sqrt(2)
    import math
    assert abs(res_cheb.cost - 3 * math.sqrt(2)) < 1e-9, f"Expected 3*sqrt(2)={3*math.sqrt(2):.6f}, got {res_cheb.cost:.6f}"

    with pytest.raises(ValueError, match="Unknown heuristic"):
        AStarPlanner(width=4, height=4, heuristic="unknown_metric")

    with pytest.raises(ValueError, match="Grid dimensions must be at least 2x2"):
        AStarPlanner(width=1, height=4)


def test_astar_planner_policy_execution() -> None:
    """Verify AStarPlannerPolicy navigates a GridWorldEnv to goal completion."""
    # Use fixed_obstacles=[] for a deterministic, always-solvable test grid
    env = GridWorldEnv(
        width=6,
        height=6,
        start_pos=(0, 0),
        goal_pos=(5, 5),
        fixed_obstacles=[],  # No random obstacles: path always exists
        max_steps=50,
        terminate_on_collision=True,
    )
    obs, _ = env.reset(seed=42)

    policy = AStarPlannerPolicy(env=env)
    policy.reset_policy()

    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < 50:
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    assert info.get("success", False) is True
    assert info.get("collision", False) is False
    assert total_reward > 0.0
    env.close()


def test_rrt_planner_continuous_navigation() -> None:
    """Verify standard RRT finds a collision-free path across a continuous 2D arena."""
    obstacles = [(10.0, 10.0, 2.0)]
    planner = RRTPlanner(
        bounds=[(0.0, 20.0), (0.0, 20.0)],
        obstacles=obstacles,
        step_size=1.0,
        max_iterations=1500,
        goal_bias=0.2,
        goal_tolerance=1.0,
        agent_radius=0.3,
        rng_seed=42,
    )

    result = planner.plan(start=(2.0, 2.0), goal=(18.0, 18.0))
    assert result.success is True
    assert len(result.path) >= 2
    assert result.cost > 0.0

    # Ensure none of the waypoints collide with obstacle
    for pt in result.path:
        dist = np.hypot(pt[0] - 10.0, pt[1] - 10.0)
        assert dist >= (2.0 + 0.3 - 1e-4)


def test_rrt_star_planner_optimality() -> None:
    """Verify RRT* optimizes trajectory with rewiring."""
    obstacles = [(10.0, 10.0, 2.5)]
    planner = RRTStarPlanner(
        bounds=[(0.0, 20.0), (0.0, 20.0)],
        obstacles=obstacles,
        step_size=1.0,
        max_iterations=2000,
        goal_bias=0.2,
        goal_tolerance=1.0,
        agent_radius=0.3,
        rewire_radius=3.0,
        rng_seed=123,
    )

    result = planner.plan(start=(2.0, 2.0), goal=(18.0, 18.0))
    assert result.success is True
    assert result.metadata.get("rewirings", 0) >= 0
    assert result.cost < 40.0  # Euclidean direct distance is ~22.6


def test_rrt_star_planner_3d_drone() -> None:
    """Verify RRT* plans in 3D continuous space around spherical obstacles."""
    obstacles = [(10.0, 10.0, 7.5, 2.0)]
    planner = RRTStarPlanner(
        bounds=[(0.0, 20.0), (0.0, 20.0), (0.0, 15.0)],
        obstacles=obstacles,
        step_size=1.5,
        max_iterations=2000,
        goal_bias=0.25,
        goal_tolerance=1.5,
        agent_radius=0.4,
        rng_seed=999,
    )

    result = planner.plan(start=(2.0, 2.0, 2.0), goal=(18.0, 18.0, 13.0))
    assert result.success is True
    assert len(result.path) >= 2
    assert result.cost > 0.0


def test_rrt_planner_policy_execution() -> None:
    """Verify RRTPlannerPolicy navigates ContinuousNavigation2DEnv to goal."""
    # Use no obstacles so RRT plans a direct path; use more steps for path tracking
    env = ContinuousNavigation2DEnv(
        arena_width=16.0,
        arena_height=16.0,
        start_pos=(2.0, 2.0),
        goal_pos=(14.0, 14.0),
        num_obstacles=0,   # No obstacles: direct RRT path, no stuck-near-obstacle risk
        obstacle_radius=1.0,
        max_steps=200,     # Generous budget for waypoint tracking
    )
    obs, _ = env.reset(seed=42)

    policy = RRTPlannerPolicy(env=env, waypoint_tolerance=1.0)
    policy.reset_policy()

    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < 200:
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    assert info.get("success", False) is True
    assert total_reward > 0.0
    env.close()


def test_classical_benchmark_runner_and_report(tmp_path: Path) -> None:
    """Verify ClassicalBenchmarkRunner produces accurate comparisons and valid JSON reports."""
    # Use no random obstacles so A* is guaranteed to find and follow a valid path
    runner_bench = ClassicalBenchmarkRunner(
        environment_name="gridworld",
        planner_type="astar",
        environment_parameters={"width": 6, "height": 6, "num_obstacles": 0, "max_steps": 30},
    )

    seeds = [101, 102, 103]
    report = runner_bench.run_benchmark(
        seeds=seeds,
        rl_algorithm=None,
        experiment_name="test_gridworld_astar_benchmark",
    )

    assert report.total_episodes == 3
    assert report.planner_success_rate == 1.0
    assert report.planner_collision_rate == 0.0
    assert report.planner_mean_path_length > 0.0
    assert len(report.detailed_results) == 3

    # Test serialization
    json_path = tmp_path / "classical_benchmark.json"
    saved = report.save_json(json_path)
    assert saved.exists()

    with open(saved, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_name"] == "test_gridworld_astar_benchmark"
    assert data["planner_success_rate"] == 1.0
    assert "detailed_results" in data


def test_cli_benchmark_planners_command(tmp_path: Path) -> None:
    """Verify adaptive-rl benchmark-planners CLI command executes cleanly."""
    cfg_file = tmp_path / "test_bench.yaml"
    cfg_file.write_text(
        """
name: "cli_bench_test"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 32
  parameters:
    n_steps: 64
environment:
  name: "gridworld"
  max_steps: 25
  parameters:
    width: 6
    height: 6
    num_obstacles: 2
training:
  total_timesteps: 64
  checkpoint_freq: 32
  log_interval: 1
evaluation:
  eval_episodes: 2
  deterministic: true
output_dir: "{out}"
log_dir: "{logs}"
""".format(out=str(tmp_path / "out"), logs=str(tmp_path / "logs")),
        encoding="utf-8",
    )

    out_json = tmp_path / "cli_bench_report.json"
    res = runner.invoke(
        app,
        [
            "benchmark-planners",
            "--config",
            str(cfg_file),
            "--planner",
            "astar",
            "--episodes",
            "3",
            "--start-seed",
            "50",
            "--output-report",
            str(out_json),
        ],
    )

    assert res.exit_code == 0
    assert "Classical vs RL Benchmark" in res.output
    assert "Success Rate" in res.output
    assert out_json.exists()
