"""Tests for Phase 12 — A* planner and planner evaluation adapter."""

from __future__ import annotations

from typing import Set

import pytest

from adaptive_rl.planners.astar import AStarPlanner
from adaptive_rl.planners.base import GridCoordinate, PlannerResult

# ---------------------------------------------------------------------------
# A* planner tests
# ---------------------------------------------------------------------------


class TestAStarPlanner:
    """Tests for the AStarPlanner implementation."""

    def setup_method(self) -> None:
        self.planner = AStarPlanner()

    def test_name(self) -> None:
        assert self.planner.name == "astar"

    def test_simple_path_found(self) -> None:
        """A* finds a path on an open 6x5 grid with no obstacles."""
        result = self.planner.plan(
            start=(0, 0),
            goal=(5, 4),
            obstacles=set(),
            width=6,
            height=5,
        )
        assert result.success
        assert len(result.path) > 0
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (5, 4)
        assert result.path_length > 0
        assert result.planning_time_seconds >= 0.0

    def test_path_length_is_optimal(self) -> None:
        """Manhattan-optimal path from (0,0) to (5,4) is 9 steps (no obstacles)."""
        result = self.planner.plan(
            start=(0, 0),
            goal=(5, 4),
            obstacles=set(),
            width=6,
            height=5,
        )
        assert result.success
        # Manhattan optimal = |5-0| + |4-0| = 9 steps (edges)
        assert result.path_length == 9

    def test_obstacle_avoidance(self) -> None:
        """A* navigates around obstacles and returns a longer but valid path."""
        # Block the direct horizontal path
        obstacles: Set[GridCoordinate] = {(1, 0), (2, 0), (3, 0), (4, 0)}
        result = self.planner.plan(
            start=(0, 0),
            goal=(5, 0),
            obstacles=obstacles,
            width=6,
            height=5,
        )
        assert result.success
        # Path must avoid all obstacles
        for cell in result.path:
            assert cell not in obstacles, f"Path passes through obstacle {cell}"
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (5, 0)

    def test_impossible_path_returns_failure(self) -> None:
        """A* returns failure when goal is completely surrounded by obstacles."""
        # Fully blocked: small grid where goal is surrounded
        result = self.planner.plan(
            start=(0, 0),
            goal=(1, 1),
            obstacles={(1, 0), (0, 1)},
            width=2,
            height=2,
        )
        assert not result.success
        assert result.path == []
        assert result.failure_reason is not None
        assert "No path found" in result.failure_reason

    def test_start_equals_goal(self) -> None:
        """When start equals goal, path is trivially [start] with length 0."""
        result = self.planner.plan(
            start=(2, 2),
            goal=(2, 2),
            obstacles=set(),
            width=6,
            height=5,
        )
        assert result.success
        assert result.path == [(2, 2)]
        assert result.path_length == 0

    def test_deterministic_output(self) -> None:
        """Same inputs always produce exactly the same path."""
        obstacles: Set[GridCoordinate] = {(2, 2), (3, 2), (1, 3)}
        r1 = self.planner.plan((0, 0), (5, 4), obstacles, 6, 5)
        r2 = self.planner.plan((0, 0), (5, 4), obstacles, 6, 5)
        assert r1.success == r2.success
        assert r1.path == r2.path
        assert r1.path_length == r2.path_length

    def test_path_is_connected(self) -> None:
        """Adjacent path cells differ by exactly 1 in one coordinate."""
        result = self.planner.plan(
            start=(0, 0),
            goal=(5, 4),
            obstacles={(2, 1), (3, 3)},
            width=6,
            height=5,
        )
        assert result.success
        for i in range(len(result.path) - 1):
            a = result.path[i]
            b = result.path[i + 1]
            manhattan = abs(a[0] - b[0]) + abs(a[1] - b[1])
            assert manhattan == 1, f"Non-adjacent step: {a} -> {b}"

    def test_out_of_bounds_start_raises(self) -> None:
        """Starting position outside grid raises ValueError."""
        with pytest.raises(ValueError, match="outside grid bounds"):
            self.planner.plan(
                start=(10, 0),
                goal=(5, 4),
                obstacles=set(),
                width=6,
                height=5,
            )

    def test_out_of_bounds_goal_raises(self) -> None:
        """Goal position outside grid raises ValueError."""
        with pytest.raises(ValueError, match="outside grid bounds"):
            self.planner.plan(
                start=(0, 0),
                goal=(6, 4),  # col=6 out of bounds for width=6
                obstacles=set(),
                width=6,
                height=5,
            )

    def test_start_on_obstacle_raises(self) -> None:
        """Start position on obstacle raises ValueError."""
        with pytest.raises(ValueError, match="obstacle"):
            self.planner.plan(
                start=(2, 2),
                goal=(5, 4),
                obstacles={(2, 2)},
                width=6,
                height=5,
            )

    def test_goal_on_obstacle_raises(self) -> None:
        """Goal position on obstacle raises ValueError."""
        with pytest.raises(ValueError, match="obstacle"):
            self.planner.plan(
                start=(0, 0),
                goal=(5, 4),
                obstacles={(5, 4)},
                width=6,
                height=5,
            )

    def test_invalid_heuristic_raises(self) -> None:
        """Unsupported heuristic name raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unsupported heuristic"):
            AStarPlanner(heuristic="invalid_heuristic")

    def test_euclidean_and_chebyshev_heuristics(self) -> None:
        """Supported heuristics (euclidean, chebyshev) initialize and find valid paths."""
        for h in ("euclidean", "chebyshev"):
            planner = AStarPlanner(heuristic=h)
            assert planner.heuristic_name == h
            res = planner.plan(
                start=(0, 0),
                goal=(5, 4),
                obstacles=set(),
                width=6,
                height=5,
            )
            assert res.success
            assert planner.validate_path(res.path, obstacles=set(), width=6, height=5)

    def test_planning_time_recorded(self) -> None:
        """Planning time is positive after a search."""
        result = self.planner.plan(
            start=(0, 0),
            goal=(5, 4),
            obstacles=set(),
            width=6,
            height=5,
        )
        assert result.planning_time_seconds > 0.0

    def test_nodes_explored_positive(self) -> None:
        """Nodes explored counter is positive after a non-trivial search."""
        result = self.planner.plan(
            start=(0, 0),
            goal=(5, 4),
            obstacles=set(),
            width=6,
            height=5,
        )
        assert result.nodes_explored > 0


class TestPlannerResult:
    """Tests for PlannerResult validation helpers."""

    def test_is_collision_free_clean_path(self) -> None:
        path = [(0, 0), (1, 0), (2, 0)]
        obstacles = {(3, 0), (4, 0)}
        result = PlannerResult(success=True, path=path)
        assert result.is_collision_free(obstacles)

    def test_is_collision_free_with_collision(self) -> None:
        path = [(0, 0), (1, 0), (2, 0)]
        obstacles = {(1, 0)}
        result = PlannerResult(success=True, path=path)
        assert not result.is_collision_free(obstacles)

    def test_is_valid_success(self) -> None:
        path = [(0, 0), (1, 0), (2, 0)]
        result = PlannerResult(success=True, path=path, path_length=2)
        assert result.is_valid(start=(0, 0), goal=(2, 0), obstacles=set())

    def test_is_valid_wrong_start(self) -> None:
        path = [(0, 0), (1, 0), (2, 0)]
        result = PlannerResult(success=True, path=path, path_length=2)
        assert not result.is_valid(start=(1, 0), goal=(2, 0), obstacles=set())

    def test_is_valid_failure_result(self) -> None:
        result = PlannerResult(success=False, path=[])
        assert not result.is_valid(start=(0, 0), goal=(2, 0), obstacles=set())


class TestPlannerAdapter:
    """Tests for the PlannerAdapter GridWorldEnv integration."""

    def test_basic_evaluation(self) -> None:
        """Adapter evaluates A* on GridWorld and returns planner metrics."""
        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.planners.adapter import PlannerAdapter

        env = GridWorldEnv(width=6, height=5, num_obstacles=2)
        planner = AStarPlanner()
        adapter = PlannerAdapter(planner=planner, env=env)

        metrics = adapter.evaluate(num_episodes=5, base_seed=42)

        assert metrics.episodes == 5
        assert 0.0 <= metrics.success_rate <= 1.0
        assert metrics.collision_rate == 0.0  # validated paths
        assert len(metrics.all_path_lengths) == 5
        assert len(metrics.all_planning_times) == 5

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces same metrics."""
        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.planners.adapter import PlannerAdapter

        env1 = GridWorldEnv(width=6, height=5, num_obstacles=2)
        env2 = GridWorldEnv(width=6, height=5, num_obstacles=2)
        planner = AStarPlanner()

        m1 = PlannerAdapter(planner=planner, env=env1).evaluate(num_episodes=3, base_seed=42)
        m2 = PlannerAdapter(planner=planner, env=env2).evaluate(num_episodes=3, base_seed=42)

        assert m1.success_rate == m2.success_rate
        assert m1.all_path_lengths == m2.all_path_lengths

    def test_rejects_non_gridworld(self) -> None:
        """PlannerAdapter raises TypeError for non-GridWorldEnv input."""
        from adaptive_rl.environments.testing import DummyTestEnv
        from adaptive_rl.planners.adapter import PlannerAdapter

        env = DummyTestEnv()
        planner = AStarPlanner()

        with pytest.raises(TypeError, match="GridWorldEnv"):
            PlannerAdapter(planner=planner, env=env)  # type: ignore

    def test_invalid_num_episodes(self) -> None:
        """Adapter raises ValueError for num_episodes < 1."""
        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.planners.adapter import PlannerAdapter

        env = GridWorldEnv()
        planner = AStarPlanner()
        adapter = PlannerAdapter(planner=planner, env=env)

        with pytest.raises(ValueError, match="num_episodes"):
            adapter.evaluate(num_episodes=0)

    def test_metrics_to_dict(self) -> None:
        """PlannerEvaluationMetrics serializes to a JSON-compatible dict."""
        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.planners.adapter import PlannerAdapter

        env = GridWorldEnv()
        planner = AStarPlanner()
        adapter = PlannerAdapter(planner=planner, env=env)
        metrics = adapter.evaluate(num_episodes=3, base_seed=42)

        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "success_rate" in d
        assert "episodes" in d
        assert "mean_path_length" in d
        assert "mean_planning_time" in d


# ---------------------------------------------------------------------------
# RRT* planner tests
# ---------------------------------------------------------------------------


class TestRRTStarPlanner:
    """Tests for the continuous-space RRT* planner implementation."""

    def test_name(self) -> None:
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner()
        assert planner.name == "rrt_star"

    def test_open_space_path(self) -> None:
        """RRT* finds a continuous path across an obstacle-free arena."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner(max_iterations=1000, step_size=1.0, seed=42)
        result = planner.plan(
            start=(2.0, 2.0),
            goal=(8.0, 8.0),
            arena_width=10.0,
            arena_height=10.0,
            obstacles=[],
            agent_radius=0.3,
            goal_radius=0.8,
        )
        assert result.success
        assert len(result.path) >= 2
        assert result.path[0] == (2.0, 2.0)
        assert result.path[-1] == (8.0, 8.0)
        assert result.path_length > 0.0
        assert result.planning_time_seconds >= 0.0

    def test_obstacle_avoidance(self) -> None:
        """RRT* routes around a central circular obstacle."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner(max_iterations=2000, step_size=0.8, search_radius=2.0, seed=42)
        # Big circular obstacle right in the center between (2,5) and (8,5)
        obstacles = [(5.0, 5.0, 1.5)]
        result = planner.plan(
            start=(2.0, 5.0),
            goal=(8.0, 5.0),
            arena_width=10.0,
            arena_height=10.0,
            obstacles=obstacles,
            agent_radius=0.3,
            goal_radius=0.8,
        )
        assert result.success
        # Verify no segment collides with the obstacle
        for i in range(len(result.path) - 1):
            assert planner.is_segment_valid(
                result.path[i],
                result.path[i + 1],
                arena_width=10.0,
                arena_height=10.0,
                obstacles=obstacles,
                agent_radius=0.3,
            )

    def test_deterministic_seeded_behavior(self) -> None:
        """Identical seeds produce identical trees and identical path lengths."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        p1 = RRTStarPlanner(max_iterations=500, step_size=0.5, seed=123)
        p2 = RRTStarPlanner(max_iterations=500, step_size=0.5, seed=123)

        r1 = p1.plan((1.0, 1.0), (5.0, 5.0), 6.0, 6.0, obstacles=[(3.0, 3.0, 0.5)])
        r2 = p2.plan((1.0, 1.0), (5.0, 5.0), 6.0, 6.0, obstacles=[(3.0, 3.0, 0.5)])

        assert r1.success == r2.success
        assert len(r1.path) == len(r2.path)
        assert pytest.approx(r1.path_length, rel=1e-5) == r2.path_length

    def test_invalid_start_collision(self) -> None:
        """Start position inside obstacle returns failure without raising."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner()
        result = planner.plan(
            start=(5.0, 5.0),
            goal=(8.0, 8.0),
            arena_width=10.0,
            arena_height=10.0,
            obstacles=[(5.0, 5.0, 1.0)],
            agent_radius=0.3,
        )
        assert not result.success
        assert "Start position" in (result.failure_reason or "")

    def test_invalid_goal_collision(self) -> None:
        """Goal position inside obstacle returns failure without raising."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner()
        result = planner.plan(
            start=(1.0, 1.0),
            goal=(5.0, 5.0),
            arena_width=10.0,
            arena_height=10.0,
            obstacles=[(5.0, 5.0, 1.0)],
            agent_radius=0.3,
        )
        assert not result.success
        assert "Goal position" in (result.failure_reason or "")

    def test_adapter_evaluation_continuous_navigation(self) -> None:
        """PlannerAdapter successfully evaluates RRT* on ContinuousNavigation2DEnv."""
        from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        env = ContinuousNavigation2DEnv(
            arena_width=10.0,
            arena_height=10.0,
            num_obstacles=2,
            obstacle_radius=0.8,
            start_pos=(1.5, 1.5),
            goal_pos=(8.5, 8.5),
        )
        planner = RRTStarPlanner(max_iterations=800, step_size=0.8, seed=42)
        adapter = PlannerAdapter(planner=planner, env=env)

        metrics = adapter.evaluate(num_episodes=3, base_seed=42)
        assert metrics.episodes == 3
        assert metrics.collision_rate == 0.0
        assert len(metrics.all_planning_times) == 3
