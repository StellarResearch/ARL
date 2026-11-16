"""Unit and integration tests for 3D A* path planning, obstacle inflation, and navigation mission management."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import DroneState6DOF
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.mission import MissionManager, MissionState
from adaptive_rl.navigation.planner import AStarPlanner3D


def test_planner_direct_path_when_unobstructed() -> None:
    """When no obstacles intersect the direct segment, planner returns 2-point direct path."""
    planner = AStarPlanner3D(bounds=(30.0, 30.0, 15.0), resolution=1.0)
    start = np.array([5.0, 5.0, 5.0])
    goal = np.array([25.0, 25.0, 10.0])
    obstacles: List[ObstacleSphere3D] = []

    res = planner.plan(start, goal, obstacles)
    assert res.status == "SUCCESS"
    assert len(res.waypoints) == 2
    np.testing.assert_allclose(res.waypoints[0], start)
    np.testing.assert_allclose(res.waypoints[1], goal)
    expected_dist = float(np.linalg.norm(goal - start))
    assert math.isclose(res.path_length, expected_dist, rel_tol=1e-5)


def test_planner_navigates_around_obstacle_in_direct_line() -> None:
    """When an obstacle lies directly on the straight-line path, A* finds a detour."""
    planner = AStarPlanner3D(
        bounds=(30.0, 30.0, 15.0), resolution=1.0, vehicle_radius=0.8, safety_margin=0.4
    )
    start = np.array([5.0, 15.0, 7.0])
    goal = np.array([25.0, 15.0, 7.0])

    # Place obstacle directly in the center of the straight line
    obs = ObstacleSphere3D(center=np.array([15.0, 15.0, 7.0]), radius=2.5)
    obstacles = [obs]

    # Verify straight line is rejected
    direct_clear, _ = planner.is_segment_valid(start, goal, obstacles)
    assert direct_clear is False

    # Run A*
    res = planner.plan(start, goal, obstacles)
    assert res.status == "SUCCESS"
    assert len(res.waypoints) >= 3  # Detour waypoints inserted
    assert res.min_clearance >= 0.0  # Respects inflation margin

    # Verify every segment in planned path is collision-free
    for i in range(len(res.waypoints) - 1):
        valid, seg_clear = planner.is_segment_valid(
            res.waypoints[i], res.waypoints[i + 1], obstacles
        )
        assert valid is True
        assert seg_clear >= 0.0


def test_planner_blocked_start_and_blocked_goal() -> None:
    """Start or goal inside an inflated obstacle is immediately detected and reported."""
    planner = AStarPlanner3D(
        bounds=(30.0, 30.0, 15.0), resolution=1.0, vehicle_radius=0.8, safety_margin=0.4
    )
    obs = ObstacleSphere3D(center=np.array([10.0, 10.0, 5.0]), radius=2.0)
    obstacles = [obs]

    # Point inside obstacle
    inside_pt = np.array([10.5, 10.0, 5.0])
    free_pt = np.array([25.0, 25.0, 10.0])

    # Blocked start
    res_start = planner.plan(inside_pt, free_pt, obstacles)
    assert res_start.status == "BLOCKED_START"
    assert len(res_start.waypoints) == 0

    # Blocked goal
    res_goal = planner.plan(free_pt, inside_pt, obstacles)
    assert res_goal.status == "BLOCKED_GOAL"
    assert len(res_goal.waypoints) == 0


def test_planner_no_path_scenario() -> None:
    """When a wall of obstacles completely blocks access to goal, returns NO_PATH."""
    planner = AStarPlanner3D(
        bounds=(10.0, 10.0, 10.0), resolution=1.0, vehicle_radius=0.5, safety_margin=0.2
    )
    start = np.array([2.0, 5.0, 5.0])
    goal = np.array([8.0, 5.0, 5.0])

    # Create an impassable wall of large obstacles along x=5.0
    wall: List[ObstacleSphere3D] = []
    for y in range(0, 11, 2):
        for z in range(0, 11, 2):
            wall.append(ObstacleSphere3D(center=np.array([5.0, float(y), float(z)]), radius=2.0))

    res = planner.plan(start, goal, wall, max_expansions=1000)
    assert res.status == "NO_PATH"
    assert len(res.waypoints) == 0


def test_planner_narrow_passage_navigation() -> None:
    """Planner can thread through a narrow gap between two obstacles if clearance permits."""
    # Gap between obstacles: centers at y=12 and y=18, radius=2.0 -> gap from 14.0 to 16.0 (width 2.0m)
    # Required width: 2 * (0.5 vehicle + 0.2 margin) = 1.4m < 2.0m -> passable!
    planner = AStarPlanner3D(
        bounds=(30.0, 30.0, 15.0), resolution=0.5, vehicle_radius=0.5, safety_margin=0.2
    )
    start = np.array([5.0, 15.0, 7.0])
    goal = np.array([25.0, 15.0, 7.0])

    obs_left = ObstacleSphere3D(center=np.array([15.0, 12.0, 7.0]), radius=2.0)
    obs_right = ObstacleSphere3D(center=np.array([15.0, 18.0, 7.0]), radius=2.0)
    obstacles = [obs_left, obs_right]

    res = planner.plan(start, goal, obstacles)
    assert res.status == "SUCCESS"
    assert res.min_clearance >= 0.0


def test_path_simplification_never_violates_obstacles() -> None:
    """Line-of-sight shortcutting must never smooth across an obstacle."""
    planner = AStarPlanner3D(
        bounds=(30.0, 30.0, 15.0), resolution=1.0, vehicle_radius=0.8, safety_margin=0.2
    )
    obs = ObstacleSphere3D(center=np.array([15.0, 15.0, 7.0]), radius=2.5)
    obstacles = [obs]

    # Create dummy waypoint detour: [5, 15, 7] -> [15, 8, 7] -> [25, 15, 7]
    raw_wps = [
        np.array([5.0, 15.0, 7.0]),
        np.array([10.0, 10.0, 7.0]),
        np.array([15.0, 8.0, 7.0]),
        np.array([20.0, 10.0, 7.0]),
        np.array([25.0, 15.0, 7.0]),
    ]
    simplified = planner.simplify_path(raw_wps, obstacles)
    assert len(simplified) <= len(raw_wps)

    # Verify all simplified segments are strictly collision-free
    for i in range(len(simplified) - 1):
        valid, clear = planner.is_segment_valid(simplified[i], simplified[i + 1], obstacles)
        assert valid is True
        assert clear >= 0.0


def test_dynamic_obstacle_deterministic_motion() -> None:
    """Dynamic obstacle deterministic motion produces exact expected coordinates over time."""
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([10.0, 10.0, 5.0]),
        radius=1.5,
        velocity=np.array([0.0, 2.0, 0.0]),
        bounds=(30.0, 30.0, 15.0),
        motion_type="bounce",
    )

    # At t=0, position matches initial
    np.testing.assert_allclose(dyn_obs.center_at(0.0), [10.0, 10.0, 5.0])
    # At t=2.0, y advanced by 2.0 * 2.0 = 4.0m -> [10.0, 14.0, 5.0]
    np.testing.assert_allclose(dyn_obs.center_at(2.0), [10.0, 14.0, 5.0])

    # Distance computation
    test_pt = np.array([10.0, 14.0, 8.0])
    assert math.isclose(dyn_obs.distance_to(test_pt, time_sec=2.0), 3.0, rel_tol=1e-5)


def test_mission_manager_lifecycle_and_replanning() -> None:
    """MissionManager starts mission, sequences waypoints, and executes Return-to-Home."""
    planner = AStarPlanner3D(bounds=(30.0, 30.0, 15.0), resolution=1.0)
    manager = MissionManager(planner=planner, waypoint_acceptance_radius=1.0)

    start_pos = np.array([5.0, 5.0, 3.0])
    goal1 = np.array([15.0, 15.0, 6.0])
    goal2 = np.array([25.0, 25.0, 8.0])
    obstacles = [ObstacleSphere3D(center=np.array([10.0, 10.0, 4.5]), radius=1.5)]

    success = manager.start_mission(start_pos, [goal1, goal2], obstacles)
    assert success is True
    assert manager.state == MissionState.EXECUTING
    assert len(manager.active_path) >= 3

    # Simulate drone reaching first waypoint
    drone_state = DroneState6DOF(
        position=manager.active_path[1].copy(),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    target, telem = manager.update(drone_state, time_sec=5.0, static_obstacles=obstacles)
    assert telem.state in (MissionState.EXECUTING.value, MissionState.ARRIVED.value)

    # Test Return to Home
    rth_ok = manager.trigger_return_to_home(drone_state.position, obstacles)
    assert rth_ok is True
    np.testing.assert_allclose(manager.final_goal, start_pos)


def test_planner_diagonal_corner_cutting_prevention() -> None:
    """Verify that A* neighbor expansion never cuts diagonally through obstacles."""
    planner = AStarPlanner3D(
        bounds=(10.0, 10.0, 10.0), resolution=1.0, vehicle_radius=0.4, safety_margin=0.1
    )
    # Start at (2, 2, 5), Goal at (4, 4, 5)
    start = np.array([2.0, 2.0, 5.0])
    goal = np.array([4.0, 4.0, 5.0])

    # Place an obstacle centered at (3.0, 3.0, 5.0) with radius 0.6
    # The direct diagonal line passes straight through (3.0, 3.0, 5.0), penetrating the obstacle.
    # The grid points (2, 2, 5) and (4, 4, 5) have distance sqrt(2) ~ 1.414 from (3,3,5), so points are free,
    # but the diagonal connecting segment is strictly in collision!
    obs = ObstacleSphere3D(center=np.array([3.0, 3.0, 5.0]), radius=0.6)
    obstacles = [obs]

    res = planner.plan(start, goal, obstacles)
    assert res.status == "SUCCESS"

    # Invariant: every returned segment must pass is_segment_valid
    for i in range(len(res.waypoints) - 1):
        valid, clear = planner.is_segment_valid(res.waypoints[i], res.waypoints[i + 1], obstacles)
        assert valid is True
        assert clear >= 0.0


def test_planner_simplification_never_appends_unvalidated_segment() -> None:
    """Verify simplify_path returns None rather than appending an invalid segment."""
    planner = AStarPlanner3D(
        bounds=(20.0, 20.0, 10.0), resolution=1.0, vehicle_radius=0.5, safety_margin=0.2
    )
    # Construct a path where the intermediate step to the next waypoint is blocked
    p1 = np.array([2.0, 5.0, 5.0])
    p2 = np.array([6.0, 5.0, 5.0])
    p3 = np.array([10.0, 5.0, 5.0])

    # Place a large impassable obstacle completely covering p2 and the segments to/from p2
    obs = ObstacleSphere3D(center=np.array([6.0, 5.0, 5.0]), radius=3.0)
    obstacles = [obs]

    # Attempting to simplify an unsafe list of waypoints must return None
    res_simplified = planner.simplify_path([p1, p2, p3], obstacles)
    assert res_simplified is None


def test_planner_tangential_contact_and_boundary_grazing() -> None:
    """Waypoints grazing the boundary or obstacle perimeter are properly validated."""
    planner = AStarPlanner3D(
        bounds=(10.0, 10.0, 10.0), resolution=1.0, vehicle_radius=0.5, safety_margin=0.2
    )
    # Point grazing floor boundary (z = 0.5 is exactly vehicle_radius, z < 0.5 is invalid)
    pt_safe = np.array([5.0, 5.0, 0.7])
    valid_safe, _ = planner.is_point_valid(pt_safe, [])
    assert valid_safe is True

    pt_boundary_breach = np.array([5.0, 5.0, 0.4])
    valid_breach, _ = planner.is_point_valid(pt_boundary_breach, [])
    assert valid_breach is False


def test_planner_overlapping_obstacles_and_determinism() -> None:
    """Complex multi-obstacle field produces deterministic identical paths upon repeated planning."""
    planner = AStarPlanner3D(
        bounds=(20.0, 20.0, 10.0), resolution=1.0, vehicle_radius=0.5, safety_margin=0.2
    )
    start = np.array([2.0, 2.0, 5.0])
    goal = np.array([18.0, 18.0, 5.0])

    obs1 = ObstacleSphere3D(center=np.array([8.0, 8.0, 5.0]), radius=2.0)
    obs2 = ObstacleSphere3D(center=np.array([10.0, 9.0, 5.0]), radius=2.0)  # Overlaps obs1
    obs3 = ObstacleSphere3D(center=np.array([14.0, 14.0, 5.0]), radius=1.8)
    obstacles = [obs1, obs2, obs3]

    res1 = planner.plan(start, goal, obstacles)
    res2 = planner.plan(start, goal, obstacles)

    assert res1.status == "SUCCESS"
    assert res2.status == "SUCCESS"
    assert len(res1.waypoints) == len(res2.waypoints)
    for w1, w2 in zip(res1.waypoints, res2.waypoints):
        np.testing.assert_allclose(w1, w2, atol=1e-10)

    # Invariant: Every consecutive segment is strictly collision-free
    for i in range(len(res1.waypoints) - 1):
        valid, clear = planner.is_segment_valid(res1.waypoints[i], res1.waypoints[i + 1], obstacles)
        assert valid is True
        assert clear >= 0.0


def test_planner_multiple_resolutions() -> None:
    """Planner functions correctly at coarse (1.0m) and fine (0.5m) grid resolutions."""
    for res_val in [1.0, 0.5]:
        planner = AStarPlanner3D(
            bounds=(15.0, 15.0, 10.0), resolution=res_val, vehicle_radius=0.5, safety_margin=0.2
        )
        start = np.array([2.0, 7.5, 5.0])
        goal = np.array([13.0, 7.5, 5.0])
        obs = ObstacleSphere3D(center=np.array([7.5, 7.5, 5.0]), radius=2.0)
        res = planner.plan(start, goal, [obs])
        assert res.status == "SUCCESS"
        assert len(res.waypoints) >= 3
        for i in range(len(res.waypoints) - 1):
            valid, _ = planner.is_segment_valid(res.waypoints[i], res.waypoints[i + 1], [obs])
            assert valid is True


def test_planner_arbitrary_non_grid_aligned_coordinates() -> None:
    """Planner handles arbitrary non-integer, non-grid-aligned coordinates across resolutions."""
    for res_val in [0.5, 1.0, 2.0]:
        planner = AStarPlanner3D(
            bounds=(20.0, 20.0, 10.0), resolution=res_val, vehicle_radius=0.5, safety_margin=0.2
        )
        start = np.array([2.37, 7.82, 4.19])
        goal = np.array([17.63, 8.44, 5.71])
        obs = ObstacleSphere3D(center=np.array([10.0, 8.0, 5.0]), radius=2.2)

        res = planner.plan(start, goal, [obs])
        assert res.status == "SUCCESS"
        assert len(res.waypoints) >= 2
        np.testing.assert_allclose(res.waypoints[0], start, atol=1e-5)
        np.testing.assert_allclose(res.waypoints[-1], goal, atol=1e-5)

        # Invariant: Every consecutive segment is strictly collision-free
        for i in range(len(res.waypoints) - 1):
            valid, clear = planner.is_segment_valid(res.waypoints[i], res.waypoints[i + 1], [obs])
            assert valid is True
            assert clear >= 0.0


def test_planner_non_grid_start_nearest_node_blocked_finds_neighbor() -> None:
    """When the nearest rounded grid node is blocked, planner connects through a collision-free neighbor."""
    planner = AStarPlanner3D(
        bounds=(20.0, 20.0, 10.0), resolution=2.0, vehicle_radius=0.4, safety_margin=0.1
    )
    # Grid resolution is 2.0. Base node for (4.8, 4.8, 4.0) is (4.0, 4.0, 4.0).
    # Place obstacle at (4.0, 4.0, 4.0) with radius 0.5 (inflated radius 1.0).
    # Node (4.0, 4.0, 4.0) is blocked.
    # Start (4.8, 4.8, 4.0) is at distance sqrt(0.8^2 + 0.8^2) = 1.131 > 1.0 (valid).
    obs = ObstacleSphere3D(center=np.array([4.0, 4.0, 4.0]), radius=0.5)
    start = np.array([4.8, 4.8, 4.0])
    goal = np.array([12.0, 12.0, 4.0])

    res = planner.plan(start, goal, [obs])
    assert res.status == "SUCCESS"
    np.testing.assert_allclose(res.waypoints[0], start, atol=1e-5)
    np.testing.assert_allclose(res.waypoints[-1], goal, atol=1e-5)
    for i in range(len(res.waypoints) - 1):
        valid, _ = planner.is_segment_valid(res.waypoints[i], res.waypoints[i + 1], [obs])
        assert valid is True
