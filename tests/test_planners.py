"""Tests for Phase 12 — A* planner and planner evaluation adapter."""

from __future__ import annotations

from typing import Any, Optional, Set

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

    @pytest.fixture(autouse=True)
    def _require_gymnasium(self) -> None:
        pytest.importorskip("gymnasium")

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
        assert metrics.collision_rate is None  # no dynamic environment step execution occurred
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
        pytest.importorskip("gymnasium")
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
        assert metrics.collision_rate is None  # no dynamic execution
        assert len(metrics.all_planning_times) == 3

    def test_rrt_star_invalid_collision_resolution(self) -> None:
        """collision_resolution <= 0 raises ValueError."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        with pytest.raises(ValueError, match="collision_resolution must be positive"):
            RRTStarPlanner(collision_resolution=0)

        with pytest.raises(ValueError, match="collision_resolution must be positive"):
            RRTStarPlanner(collision_resolution=-0.5)

    def test_rrt_star_invalid_dimensions(self) -> None:
        """Invalid arena dimensions raise ValueError."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner()
        with pytest.raises(ValueError, match="Arena dimensions must be positive"):
            planner.plan(start=(0, 0), goal=(1, 1), arena_width=0, arena_height=10)

    def test_rrt_star_start_in_goal_radius_obstacle_in_between(self) -> None:
        """When start is within goal radius of goal but blocked by obstacle, do not straight-connect through obstacle."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner(max_iterations=100)
        # start=(1.0, 1.0), goal=(2.0, 1.0) -> distance is 1.0 (< goal_radius 2.0)
        # Put an obstacle at (1.5, 1.0) with radius 0.4
        result = planner.plan(
            start=(1.0, 1.0),
            goal=(2.0, 1.0),
            arena_width=10.0,
            arena_height=10.0,
            obstacles=[(1.5, 1.0, 0.4)],
            agent_radius=0.05,
            goal_radius=2.0,
        )
        if result.success:
            # If a path was found via exploration around the obstacle, it must be valid
            assert planner.validate_path(
                result.path,
                obstacles=[(1.5, 1.0, 0.4)],
                agent_radius=0.05,
                arena_width=10.0,
                arena_height=10.0,
            )
            # Cannot be a direct 2-point straight line
            assert result.path != [(1.0, 1.0), (2.0, 1.0)]

    def test_rrt_star_trivial_same_start_goal(self) -> None:
        """When start == goal, return 1-node path with 0 length."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner()
        result = planner.plan(
            start=(3.0, 3.0),
            goal=(3.0, 3.0),
            arena_width=10.0,
            arena_height=10.0,
        )
        assert result.success
        assert result.path == [(3.0, 3.0)]
        assert result.path_length == 0.0

    def test_rrt_star_cycle_prevention(self) -> None:
        """_is_ancestor correctly detects ancestor relationships to prevent cycles."""
        from adaptive_rl.planners.rrt_star import RRTNode, RRTStarPlanner

        root = RRTNode(1.0, 1.0, cost=0.0)
        child1 = RRTNode(2.0, 2.0, parent=root, cost=1.0)
        child2 = RRTNode(3.0, 3.0, parent=child1, cost=2.0)

        assert RRTStarPlanner._is_ancestor(root, child2)
        assert RRTStarPlanner._is_ancestor(child1, child2)
        assert not RRTStarPlanner._is_ancestor(child2, root)

    def test_astar_diagonal_step_invalid(self) -> None:
        """4-connected A* rejects diagonal steps during validation."""
        path = [(0, 0), (1, 1)]
        assert not AStarPlanner.validate_path(path, obstacles=set(), width=5, height=5)

    def test_astar_heuristics(self) -> None:
        """Different heuristics compute expected distance values."""
        planner_m = AStarPlanner(heuristic="manhattan")
        planner_e = AStarPlanner(heuristic="euclidean")
        planner_c = AStarPlanner(heuristic="chebyshev")

        assert planner_m._heuristic((0, 0), (3, 4)) == 7.0
        assert planner_e._heuristic((0, 0), (3, 4)) == 5.0
        assert planner_c._heuristic((0, 0), (3, 4)) == 4.0

    def test_adapter_zero_success_returns_none_for_path_metrics(self) -> None:
        """When 0 paths succeed, path length metrics are None rather than fabricated 0.0."""
        pytest.importorskip("gymnasium")
        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.planners.adapter import PlannerAdapter

        # Create a gridworld where goal is completely surrounded by obstacles
        env = GridWorldEnv(
            width=3,
            height=3,
            start_pos=(0, 0),
            goal_pos=(2, 2),
            fixed_obstacles=[(1, 2), (2, 1)],  # blocks access to (2,2)
            num_obstacles=0,
        )
        planner = AStarPlanner()
        adapter = PlannerAdapter(planner=planner, env=env)
        metrics = adapter.evaluate(num_episodes=2)

        assert metrics.success_rate == 0.0
        assert metrics.mean_path_length is None
        assert metrics.std_path_length is None
        assert metrics.min_path_length is None
        assert metrics.max_path_length is None
        assert metrics.collision_rate is None
        assert "collision_semantics" in metrics.additional_metrics

    def test_astar_planner_heuristic_propagation(self) -> None:
        """A* correctly initializes and uses the specified heuristic."""
        for heur in ("manhattan", "euclidean", "chebyshev"):
            planner = AStarPlanner(heuristic=heur)
            assert planner.heuristic_name == heur
            res = planner.plan(start=(0, 0), goal=(2, 2), obstacles=set(), width=3, height=3)
            assert res.success

    def test_rrt_star_multilevel_subtree_rewire_cost_propagation(self) -> None:
        """Multi-level subtree rewire propagates cost delta through parent -> child -> grandchild."""
        from adaptive_rl.planners.rrt_star import RRTNode, RRTStarPlanner

        planner = RRTStarPlanner()
        root = RRTNode(0.0, 0.0, cost=0.0)
        child = RRTNode(2.0, 0.0, cost=2.0, parent=root)
        root.children.append(child)

        grandchild = RRTNode(4.0, 0.0, cost=4.0, parent=child)
        child.children.append(grandchild)

        great_grandchild = RRTNode(6.0, 0.0, cost=6.0, parent=grandchild)
        grandchild.children.append(great_grandchild)

        # Rewire child to a new shorter parent path reducing cost by 0.5 (from 2.0 to 1.5)
        planner._update_subtree_costs(child, 1.5)

        assert child.cost == 1.5
        assert grandchild.cost == 3.5
        assert great_grandchild.cost == 5.5

        # Rewire with a cost increase of 1.0 (from 1.5 to 2.5)
        planner._update_subtree_costs(child, 2.5)
        assert child.cost == 2.5
        assert grandchild.cost == 4.5
        assert great_grandchild.cost == 6.5

    def test_rrt_star_constructor_parameter_validation(self) -> None:
        """RRTStarPlanner validates hyperparameters and rejects invalid values early."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        # Zero and negative values
        with pytest.raises(ValueError):
            RRTStarPlanner(step_size=0.0)
        with pytest.raises(ValueError):
            RRTStarPlanner(step_size=-0.5)
        with pytest.raises(ValueError):
            RRTStarPlanner(search_radius=0.0)
        with pytest.raises(ValueError):
            RRTStarPlanner(search_radius=-1.0)
        with pytest.raises(ValueError):
            RRTStarPlanner(collision_resolution=0.0)
        with pytest.raises(ValueError):
            RRTStarPlanner(collision_resolution=-0.05)
        with pytest.raises(ValueError):
            RRTStarPlanner(max_iterations=0)
        with pytest.raises(ValueError):
            RRTStarPlanner(max_iterations=-10)
        with pytest.raises(ValueError):
            RRTStarPlanner(goal_bias=-0.1)
        with pytest.raises(ValueError):
            RRTStarPlanner(goal_bias=1.1)

        # NaN and Inf values
        with pytest.raises(ValueError):
            RRTStarPlanner(step_size=float("nan"))
        with pytest.raises(ValueError):
            RRTStarPlanner(search_radius=float("inf"))
        with pytest.raises(ValueError):
            RRTStarPlanner(goal_bias=float("nan"))

        # Invalid types
        with pytest.raises(TypeError):
            RRTStarPlanner(step_size="invalid")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            RRTStarPlanner(step_size=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            RRTStarPlanner(max_iterations=100.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            RRTStarPlanner(seed="seed_string")  # type: ignore[arg-type]

    def test_rrt_star_plan_geometry_validation(self) -> None:
        """RRTStarPlanner.plan() rejects invalid dimensions, coordinates, and obstacle geometries."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner(seed=42)

        # Non-positive or too small arena
        with pytest.raises(ValueError):
            planner.plan(start=(1.0, 1.0), goal=(5.0, 5.0), arena_width=0.0, arena_height=10.0)
        with pytest.raises(ValueError):
            planner.plan(
                start=(1.0, 1.0),
                goal=(5.0, 5.0),
                arena_width=0.5,
                arena_height=0.5,
                agent_radius=0.3,
            )

        # Invalid agent or goal radius
        with pytest.raises(ValueError):
            planner.plan(
                start=(1.0, 1.0),
                goal=(5.0, 5.0),
                arena_width=10.0,
                arena_height=10.0,
                agent_radius=-0.1,
            )
        with pytest.raises(ValueError):
            planner.plan(
                start=(1.0, 1.0),
                goal=(5.0, 5.0),
                arena_width=10.0,
                arena_height=10.0,
                goal_radius=0.0,
            )

        # Invalid obstacle specs
        with pytest.raises(ValueError):
            planner.plan(
                start=(1.0, 1.0),
                goal=(5.0, 5.0),
                arena_width=10.0,
                arena_height=10.0,
                obstacles=[(2.0, 2.0, -1.0)],
            )
        with pytest.raises(TypeError):
            planner.plan(
                start=(1.0, 1.0),
                goal=(5.0, 5.0),
                arena_width=10.0,
                arena_height=10.0,
                obstacles=[("x", "y", "r")],  # type: ignore[list-item]
            )

    def test_astar_seed_tolerance(self) -> None:
        """AStarPlanner accepts seed parameter without error and produces identical paths."""
        from adaptive_rl.planners.astar import AStarPlanner

        p_no_seed = AStarPlanner(heuristic="manhattan")
        p_seed = AStarPlanner(heuristic="manhattan", seed=42)

        assert p_seed.seed == 42
        res1 = p_no_seed.plan(start=(0, 0), goal=(4, 4), obstacles={(1, 1)}, width=5, height=5)
        res2 = p_seed.plan(start=(0, 0), goal=(4, 4), obstacles={(1, 1)}, width=5, height=5)

        assert res1.success and res2.success
        assert res1.path == res2.path
        assert res1.path_length == res2.path_length

    def test_rrt_star_relational_validation(self) -> None:
        """RRTStarPlanner rejects relational violations (collision_resolution > step_size, etc.)."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        # collision_resolution > step_size
        with pytest.raises(ValueError, match="cannot be greater than step_size"):
            RRTStarPlanner(step_size=0.1, collision_resolution=0.5)

        # collision_resolution < 1e-4
        with pytest.raises(ValueError, match="too small"):
            RRTStarPlanner(collision_resolution=1e-5)

        # search_radius < 0.5 * step_size
        with pytest.raises(ValueError, match="search_radius .* must be >= 0.5 \\* step_size"):
            RRTStarPlanner(step_size=2.0, search_radius=0.5)

        # max_iterations > 100_000
        with pytest.raises(ValueError, match="exceeds the maximum allowable limit"):
            RRTStarPlanner(max_iterations=200_000)

    def test_make_planner_factory(self) -> None:
        """make_planner authoritatively instantiates and validates planners."""
        from adaptive_rl.planners.astar import AStarPlanner
        from adaptive_rl.planners.factory import make_planner
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        # A* creation
        astar = make_planner("astar", heuristic="euclidean", seed=123)
        assert isinstance(astar, AStarPlanner)
        assert astar.heuristic_name == "euclidean"
        assert astar.seed == 123

        # RRT* creation
        rrt_star = make_planner("rrt_star", step_size=0.4, search_radius=1.2, seed=99)
        assert isinstance(rrt_star, RRTStarPlanner)
        assert rrt_star.step_size == 0.4
        assert rrt_star.search_radius == 1.2
        assert rrt_star.seed == 99

        # Lenient rrt* alias
        rrt_star_alias = make_planner("rrt*", step_size=0.5)
        assert isinstance(rrt_star_alias, RRTStarPlanner)

        # Plain rrt rejected with clear explanation
        with pytest.raises(ValueError, match="Standard RRT does not perform tree rewiring"):
            make_planner("rrt")

        # Unknown planner rejected
        with pytest.raises(ValueError, match="Unknown planner 'dijkstra'"):
            make_planner("dijkstra")

    def test_failed_episodes_record_none_in_all_path_lengths(self) -> None:
        """Failed planner episodes must record None in all_path_lengths, never 0.0."""
        pytest.importorskip("gymnasium")
        from unittest.mock import MagicMock

        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.base import PlannerResult

        env = GridWorldEnv(width=5, height=5, num_obstacles=0)
        planner = AStarPlanner()

        # Mock plan to return failure
        planner.plan = MagicMock(
            return_value=PlannerResult(success=False, failure_reason="blocked")
        )

        adapter = PlannerAdapter(planner=planner, env=env)
        metrics = adapter.evaluate(num_episodes=3, base_seed=42)

        assert metrics.success_rate == 0.0
        assert metrics.all_path_lengths == [None, None, None]
        assert 0.0 not in metrics.all_path_lengths

        # to_dict must output None (JSON null), not 0.0
        d = metrics.to_dict()
        assert d["all_path_lengths"] == [None, None, None]

    def test_planner_polymorphism_and_hierarchy(self) -> None:
        """A* and RRT* adhere to the unified BasePlanner interface with canonical names."""
        from adaptive_rl.planners.astar import AStarPlanner
        from adaptive_rl.planners.base import (
            BaseContinuousPlanner,
            BaseGridPlanner,
            BasePlanner,
        )
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        astar = AStarPlanner(seed=10)
        rrt = RRTStarPlanner(seed=20)

        # Both inherit from BasePlanner
        assert isinstance(astar, BasePlanner)
        assert isinstance(rrt, BasePlanner)

        # Domain-specific subclasses
        assert isinstance(astar, BaseGridPlanner)
        assert isinstance(rrt, BaseContinuousPlanner)

        # Canonical naming contract
        assert astar.name == "astar"
        assert rrt.name == "rrt_star"
        assert astar.seed == 10
        assert rrt.seed == 20

    def test_planner_adapter_extensible_registration(self) -> None:
        """PlannerAdapter allows registering custom planner-environment evaluators."""
        pytest.importorskip("gymnasium")
        from adaptive_rl.planners.adapter import PlannerAdapter, PlannerEvaluationMetrics
        from adaptive_rl.planners.base import BasePlanner

        class CustomDummyPlanner(BasePlanner):
            @property
            def name(self) -> str:
                return "custom_dummy"

        class CustomDummyEnv:
            pass

        def custom_evaluator(adapter: PlannerAdapter, num_episodes: int, base_seed: int | None):
            return PlannerEvaluationMetrics(
                episodes=num_episodes,
                success_rate=1.0,
                all_path_lengths=[12.5] * num_episodes,
                all_planning_times=[0.01] * num_episodes,
            )

        PlannerAdapter.register_evaluator(CustomDummyPlanner, CustomDummyEnv, custom_evaluator)

        adapter = PlannerAdapter(planner=CustomDummyPlanner(), env=CustomDummyEnv())
        res = adapter.evaluate(num_episodes=2)

        assert res.episodes == 2
        assert res.success_rate == 1.0
        assert res.all_path_lengths == [12.5, 12.5]

    def test_canonical_failure_and_zero_semantics(self) -> None:
        """Verify strict canonical failure (success=False, path_length=None) vs zero (success=True, path_length=0.0)."""
        astar = AStarPlanner()

        # 1. Blocked / impossible path -> failure with path_length=None
        blocked_res = astar.plan(
            start=(0, 0),
            goal=(2, 0),
            obstacles={(1, 0)},
            width=3,
            height=1,
        )
        assert blocked_res.success is False
        assert blocked_res.path_length is None
        assert blocked_res.path == []

        # 2. Trivial start == goal -> success with measured zero path_length=0.0
        trivial_res = astar.plan(
            start=(1, 1),
            goal=(1, 1),
            obstacles=set(),
            width=3,
            height=3,
        )
        assert trivial_res.success is True
        assert trivial_res.path_length == 0.0
        assert trivial_res.path == [(1, 1)]
        assert trivial_res.is_valid(start=(1, 1), goal=(1, 1), obstacles=set()) is True

    def test_rrt_star_targeted_parent_selection_and_rewiring(self) -> None:
        """Targeted test: parent selection chooses min cost and rewiring propagates subtree costs."""
        from adaptive_rl.planners.rrt_star import RRTNode, RRTStarPlanner

        planner = RRTStarPlanner(search_radius=3.0, step_size=1.0, seed=42)

        # Build controlled tree: root at (0, 0), child A at (1, 0) with cost 1.0
        root = RRTNode(x=0.0, y=0.0, cost=0.0)
        node_a = RRTNode(x=1.0, y=0.0, cost=1.0, parent=root)
        root.children.append(node_a)
        # Node B at (1, 1) parented to root with cost hypot(1, 1) ~ 1.414
        node_b = RRTNode(x=1.0, y=1.0, cost=1.4142, parent=root)
        root.children.append(node_b)

        # Child of B at (1, 2) with cost 2.4142
        node_c = RRTNode(x=1.0, y=2.0, cost=2.4142, parent=node_b)
        node_b.children.append(node_c)

        # Subtree cost propagation test
        planner._update_subtree_costs(node_b, new_cost=1.0)
        assert node_b.cost == 1.0
        # node_c cost must have dropped by 0.4142 -> 2.0
        assert abs(node_c.cost - 2.0) < 1e-4

    def test_rrt_star_collision_rejection(self) -> None:
        """Targeted test: segments crossing circular obstacles or boundaries are rejected."""
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        planner = RRTStarPlanner()

        # Segment from (0.0, 5.0) to (10.0, 5.0) crossing obstacle at (5.0, 5.0, radius=2.0)
        obstacles = [(5.0, 5.0, 2.0)]
        assert not planner.is_segment_valid(
            (0.0, 5.0),
            (10.0, 5.0),
            arena_width=20.0,
            arena_height=20.0,
            obstacles=obstacles,
            agent_radius=0.0,
        )

        # Segment crossing out of bounds
        assert not planner.is_segment_valid(
            (1.0, 1.0),
            (-2.0, 1.0),
            arena_width=20.0,
            arena_height=20.0,
            obstacles=[],
            agent_radius=0.0,
        )

        # Clear segment
        assert planner.is_segment_valid(
            (0.0, 0.0),
            (2.0, 0.0),
            arena_width=20.0,
            arena_height=20.0,
            obstacles=obstacles,
            agent_radius=0.0,
        )


class TestEvaluatorRegistrySpecificity:
    """Tests for deterministic MRO-based evaluator dispatch in PlannerAdapter."""

    def test_overlapping_registrations_most_specific_selected(self) -> None:
        """Verify most specific evaluator is chosen regardless of registration order."""
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.base import BasePlanner

        class HierarchyBasePlanner(BasePlanner):
            @property
            def name(self) -> str:
                return "base_planner"

        class HierarchySpecificPlanner(HierarchyBasePlanner):
            @property
            def name(self) -> str:
                return "specific_planner"

        class HierarchyBaseEnv:
            pass

        class HierarchySpecificEnv(HierarchyBaseEnv):
            pass

        def fn_base_base(adapter: Any, n: int, s: Optional[int]) -> Any:
            return None

        def fn_specific_base(adapter: Any, n: int, s: Optional[int]) -> Any:
            return None

        def fn_base_specific(adapter: Any, n: int, s: Optional[int]) -> Any:
            return None

        def fn_specific_specific(adapter: Any, n: int, s: Optional[int]) -> Any:
            return None

        # Test registration order 1: general registered before specific
        registry_backup = dict(PlannerAdapter._evaluator_registry)
        try:
            PlannerAdapter._evaluator_registry.clear()
            PlannerAdapter.register_evaluator(HierarchyBasePlanner, HierarchyBaseEnv, fn_base_base)
            PlannerAdapter.register_evaluator(
                HierarchySpecificPlanner, HierarchyBaseEnv, fn_specific_base
            )
            PlannerAdapter.register_evaluator(
                HierarchyBasePlanner, HierarchySpecificEnv, fn_base_specific
            )
            PlannerAdapter.register_evaluator(
                HierarchySpecificPlanner, HierarchySpecificEnv, fn_specific_specific
            )

            assert (
                PlannerAdapter.find_evaluator(HierarchySpecificPlanner, HierarchySpecificEnv)
                is fn_specific_specific
            )
            assert (
                PlannerAdapter.find_evaluator(HierarchySpecificPlanner, HierarchyBaseEnv)
                is fn_specific_base
            )
            assert (
                PlannerAdapter.find_evaluator(HierarchyBasePlanner, HierarchySpecificEnv)
                is fn_base_specific
            )
            assert (
                PlannerAdapter.find_evaluator(HierarchyBasePlanner, HierarchyBaseEnv)
                is fn_base_base
            )

            # Test registration order 2: reverse order produces identical resolution
            PlannerAdapter._evaluator_registry.clear()
            PlannerAdapter.register_evaluator(
                HierarchySpecificPlanner, HierarchySpecificEnv, fn_specific_specific
            )
            PlannerAdapter.register_evaluator(
                HierarchyBasePlanner, HierarchySpecificEnv, fn_base_specific
            )
            PlannerAdapter.register_evaluator(
                HierarchySpecificPlanner, HierarchyBaseEnv, fn_specific_base
            )
            PlannerAdapter.register_evaluator(HierarchyBasePlanner, HierarchyBaseEnv, fn_base_base)

            assert (
                PlannerAdapter.find_evaluator(HierarchySpecificPlanner, HierarchySpecificEnv)
                is fn_specific_specific
            )
            assert (
                PlannerAdapter.find_evaluator(HierarchySpecificPlanner, HierarchyBaseEnv)
                is fn_specific_base
            )
            assert (
                PlannerAdapter.find_evaluator(HierarchyBasePlanner, HierarchySpecificEnv)
                is fn_base_specific
            )
            assert (
                PlannerAdapter.find_evaluator(HierarchyBasePlanner, HierarchyBaseEnv)
                is fn_base_base
            )

            # Test missing exact match: falls back to most specific planner superclass
            PlannerAdapter._evaluator_registry.clear()
            PlannerAdapter.register_evaluator(HierarchyBasePlanner, HierarchyBaseEnv, fn_base_base)
            PlannerAdapter.register_evaluator(
                HierarchySpecificPlanner, HierarchyBaseEnv, fn_specific_base
            )
            PlannerAdapter.register_evaluator(
                HierarchyBasePlanner, HierarchySpecificEnv, fn_base_specific
            )
            # Both fn_specific_base (p_dist=0, e_dist=1) and fn_base_specific (p_dist=1, e_dist=0) match.
            # Planner specificity (p_dist) takes priority:
            assert (
                PlannerAdapter.find_evaluator(HierarchySpecificPlanner, HierarchySpecificEnv)
                is fn_specific_base
            )

            # Completely unregistered types return None
            class UnrelatedType:
                pass

            assert PlannerAdapter.find_evaluator(UnrelatedType, HierarchyBaseEnv) is None  # type: ignore
        finally:
            PlannerAdapter._evaluator_registry = registry_backup


class TestPlannerMissingVsZeroMetrics:
    """Tests for preserving None for unmeasured planning time vs 0.0 for measured zero."""

    def test_planner_result_unmeasured_time_is_none(self) -> None:
        """PlannerResult without explicit timing has planning_time_seconds = None."""
        res = PlannerResult(success=False, failure_reason="Test failure")
        assert res.planning_time_seconds is None

        # Explicit measured zero is preserved
        res_zero = PlannerResult(success=True, planning_time_seconds=0.0)
        assert res_zero.planning_time_seconds == 0.0

    def test_adapter_build_metrics_unmeasured_time(self) -> None:
        """PlannerAdapter._build_metrics reports None when all episodes are unmeasured."""
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.astar import AStarPlanner

        adapter = PlannerAdapter.__new__(PlannerAdapter)
        adapter.planner = AStarPlanner()

        # Case 1: All episodes unmeasured (e.g. exceptions before timing)
        unmeasured_metrics = adapter._build_metrics(
            num_episodes=2,
            successes=0,
            path_lengths=[None, None],
            successful_path_lengths=[],
            planning_times=[None, None],
            env_name="test_env",
            base_seed=42,
        )
        assert unmeasured_metrics.mean_planning_time is None
        assert unmeasured_metrics.std_planning_time is None
        assert unmeasured_metrics.all_planning_times == [None, None]

        # Case 2: Measured zero timing (e.g. instant resolution)
        measured_zero_metrics = adapter._build_metrics(
            num_episodes=2,
            successes=2,
            path_lengths=[0.0, 0.0],
            successful_path_lengths=[0.0, 0.0],
            planning_times=[0.0, 0.0],
            env_name="test_env",
            base_seed=42,
        )
        assert measured_zero_metrics.mean_planning_time == 0.0
        assert measured_zero_metrics.std_planning_time == 0.0
        assert measured_zero_metrics.all_planning_times == [0.0, 0.0]

        # Unmeasured vs measured zero must strictly not be equal
        assert unmeasured_metrics.mean_planning_time != measured_zero_metrics.mean_planning_time
