"""Tests for Phase IV Time-Aware 3D A* Path Planner and Route Evaluation."""

from __future__ import annotations

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.time_aware_planner import (
    RouteEvaluator,
    TimeAwareAStarPlanner3D,
    TimedWaypoint,
)


def test_time_aware_planner_basic_clear_path() -> None:
    """Clear environment with no obstacles finds direct timed 3D trajectory."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 10.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
    )
    start = np.array([2.0, 2.0, 2.0])
    goal = np.array([10.0, 2.0, 2.0])

    res = planner.plan(start, goal, static_obstacles=[], dynamic_obstacles=[], t_start=0.0)

    assert res.status == "SUCCESS"
    assert len(res.spatial_waypoints) >= 2
    assert np.allclose(res.spatial_waypoints[0], start)
    assert np.allclose(res.spatial_waypoints[-1], goal)
    assert res.timed_waypoints[0].time_sec == 0.0
    assert res.timed_waypoints[-1].time_sec > 0.0
    assert res.metrics is not None
    assert res.metrics.is_collision_free is True
    assert res.metrics.path_length >= 8.0


def test_time_aware_planner_crossing_moving_obstacle_anticipation() -> None:
    """Time-aware planner anticipates a crossing dynamic obstacle and routes around it."""
    # Flight path is straight from [2.0, 5.0, 2.0] to [10.0, 5.0, 2.0]
    # A moving obstacle starts at [6.0, 1.0, 2.0] and moves at vy = +2.0 m/s
    # At t = 2.0s, the obstacle crosses y = 5.0 at x = 6.0!
    # A stationary planner would see y=5 clear at t=0 and plan straight through, causing collision.
    # The time-aware planner must route above, below, or wait.
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 10.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
        vehicle_radius=0.5,
        safety_margin=0.3,
    )
    start = np.array([2.0, 5.0, 2.0])
    goal = np.array([10.0, 5.0, 2.0])

    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([6.0, 1.0, 2.0]),
        radius=1.2,
        velocity=np.array([0.0, 2.0, 0.0]),
        bounds=(20.0, 20.0, 10.0),
        motion_type="bounce",
    )

    res = planner.plan(start, goal, static_obstacles=[], dynamic_obstacles=[dyn_obs], t_start=0.0)

    assert res.status == "SUCCESS"
    assert res.metrics is not None
    assert res.metrics.is_collision_free is True

    # Verify that along the trajectory, minimum clearance to moving obstacle is strictly positive
    for tw in res.timed_waypoints:
        obs_c = dyn_obs.center_at(tw.time_sec)
        dist = float(np.linalg.norm(tw.position - obs_c))
        assert dist >= dyn_obs.radius + planner.vehicle_radius, f"Collision at t={tw.time_sec}!"


def test_time_aware_planner_waiting_transition_in_narrow_passage() -> None:
    """When a narrow passage is temporarily blocked, the planner can wait in place until it clears."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(15.0, 15.0, 8.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
        vehicle_radius=0.4,
        safety_margin=0.2,
        max_wait_steps=10,
    )
    start = np.array([3.0, 5.0, 2.0])
    goal = np.array([7.0, 5.0, 2.0])

    # Static obstacles restricting flight to a slot at y=5, z=2
    stat_obs = [
        ObstacleSphere3D(center=np.array([5.0, 3.0, 2.0]), radius=1.2),
        ObstacleSphere3D(center=np.array([5.0, 7.0, 2.0]), radius=1.2),
        ObstacleSphere3D(center=np.array([5.0, 5.0, 4.0]), radius=1.2),
    ]

    # Dynamic obstacle crossing through slot at t=1.0s
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([5.0, 2.0, 2.0]),
        radius=1.0,
        velocity=np.array([0.0, 3.0, 0.0]),
        bounds=(15.0, 15.0, 8.0),
        motion_type="bounce",
    )

    res = planner.plan(
        start, goal, static_obstacles=stat_obs, dynamic_obstacles=[dyn_obs], t_start=0.0
    )

    assert res.status == "SUCCESS"
    assert res.metrics is not None
    assert res.metrics.is_collision_free is True
    assert res.metrics.waiting_time > 0.0


def test_time_aware_planner_alternative_routes_generation() -> None:
    """Planner generates distinct alternative corridors ranked by multi-objective evaluation."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 10.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
        vehicle_radius=0.5,
        safety_margin=0.3,
    )
    start = np.array([2.0, 5.0, 5.0])
    goal = np.array([12.0, 5.0, 5.0])

    # Static obstacle directly in center of corridor
    stat_obs = [ObstacleSphere3D(center=np.array([7.0, 5.0, 5.0]), radius=1.5)]

    alts = planner.plan_alternatives(
        start,
        goal,
        static_obstacles=stat_obs,
        dynamic_obstacles=[],
        t_start=0.0,
        num_alternatives=3,
    )

    assert len(alts) >= 2
    # All alternative routes must be collision-free
    for r in alts:
        assert r.status == "SUCCESS"
        assert r.metrics is not None
        assert r.metrics.is_collision_free is True

    # The alternatives must be spatially diverse
    dist_diff = sum(
        float(np.linalg.norm(p1 - p2))
        for p1, p2 in zip(alts[0].spatial_waypoints[:3], alts[1].spatial_waypoints[:3])
    )
    assert dist_diff >= 0.5


def test_time_aware_planner_bounded_complexity_and_timeouts() -> None:
    """Planner respects max_expansions and time_limit_sec bounds under impossible/dense constraints."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(30.0, 30.0, 15.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
    )
    start = np.array([2.0, 2.0, 2.0])
    goal = np.array([28.0, 28.0, 12.0])

    # Strict search expansion limit
    res_limited = planner.plan(
        start, goal, static_obstacles=[], dynamic_obstacles=[], max_expansions=10
    )
    assert res_limited.status == "SEARCH_BUDGET_EXCEEDED"
    assert res_limited.nodes_expanded <= 10

    # Strict time limit
    res_timeout = planner.plan(
        start, goal, static_obstacles=[], dynamic_obstacles=[], time_limit_sec=0.0001
    )
    assert res_timeout.status in ("TIMEOUT", "SEARCH_BUDGET_EXCEEDED")


def test_route_evaluator_multi_objective_properties() -> None:
    """RouteEvaluator computes strictly non-negative, deterministic normalized costs."""
    evaluator = RouteEvaluator(
        weight_distance=1.0,
        weight_time=1.0,
        weight_waiting=0.8,
        weight_risk=0.5,
        weight_curvature=0.2,
    )

    wps = [
        TimedWaypoint(position=np.array([0.0, 0.0, 1.0]), time_sec=0.0),
        TimedWaypoint(position=np.array([2.0, 0.0, 1.0]), time_sec=1.0),
        TimedWaypoint(position=np.array([4.0, 2.0, 1.0]), time_sec=2.5),
    ]

    metrics = evaluator.evaluate(
        timed_waypoints=wps,
        static_obstacles=[],
        dynamic_obstacles=[],
        vehicle_radius=0.5,
        reference_distance=4.0,
    )

    assert metrics.path_length > 4.0
    assert metrics.travel_time == 2.5
    assert metrics.waiting_time == 0.0
    assert metrics.curvature_cost > 0.0
    assert metrics.total_objective_cost > 0.0
    assert metrics.is_collision_free is True


def test_time_aware_planner_head_on_moving_obstacle_avoidance() -> None:
    """Planner detours around a head-on approaching obstacle moving in the opposite direction."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 10.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
        vehicle_radius=0.5,
        safety_margin=0.3,
    )
    start = np.array([2.0, 5.0, 5.0])
    goal = np.array([12.0, 5.0, 5.0])

    # Obstacle starts at goal x=12.0 and flies directly head-on towards start x=2.0 at vx = -2.0 m/s
    head_on_obs = DynamicObstacleSphere3D(
        center_initial=np.array([12.0, 5.0, 5.0]),
        radius=1.0,
        velocity=np.array([-2.0, 0.0, 0.0]),
        bounds=(20.0, 20.0, 10.0),
        motion_type="bounce",
    )

    res = planner.plan(
        start, goal, static_obstacles=[], dynamic_obstacles=[head_on_obs], t_start=0.0
    )

    assert res.status == "SUCCESS"
    assert res.metrics is not None
    assert res.metrics.is_collision_free is True
    # The drone must deviate in y or z to avoid the head-on collision
    max_y_dev = max(abs(tw.position[1] - 5.0) for tw in res.timed_waypoints)
    max_z_dev = max(abs(tw.position[2] - 5.0) for tw in res.timed_waypoints)
    assert max_y_dev >= 1.0 or max_z_dev >= 1.0


def test_time_aware_planner_oscillating_obstacle_traversal() -> None:
    """Planner times crossing through an oscillating obstacle path during a clear window."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 10.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
        vehicle_radius=0.4,
        safety_margin=0.2,
    )
    start = np.array([2.0, 5.0, 5.0])
    goal = np.array([10.0, 5.0, 5.0])

    # Obstacle oscillates vertically along z around z=5.0 with amplitude 3.0m and period 4.0s
    osc_obs = DynamicObstacleSphere3D(
        center_initial=np.array([6.0, 5.0, 5.0]),
        radius=1.2,
        velocity=np.array([0.0, 0.0, 0.0]),
        bounds=(20.0, 20.0, 10.0),
        motion_type="oscillate",
        oscillation_period=4.0,
        oscillation_amplitude=np.array([0.0, 0.0, 3.0]),
    )

    res = planner.plan(start, goal, static_obstacles=[], dynamic_obstacles=[osc_obs], t_start=0.0)

    assert res.status == "SUCCESS"
    assert res.metrics is not None
    assert res.metrics.is_collision_free is True


def test_time_aware_planner_blocked_start_and_goal_validation() -> None:
    """Planner rejects planning requests when start or goal is blocked or out of bounds."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 10.0), vehicle_radius=0.5, safety_margin=0.3
    )

    stat_obs = [ObstacleSphere3D(center=np.array([2.0, 2.0, 2.0]), radius=1.0)]

    # Start blocked by static obstacle
    res_start = planner.plan(np.array([2.0, 2.0, 2.0]), np.array([10.0, 10.0, 5.0]), stat_obs, [])
    assert res_start.status == "BLOCKED_START"

    # Goal blocked by static obstacle
    res_goal = planner.plan(np.array([10.0, 10.0, 5.0]), np.array([2.0, 2.0, 2.0]), stat_obs, [])
    assert res_goal.status == "BLOCKED_GOAL"

    # Out of bounds
    res_oob = planner.plan(np.array([-1.0, 5.0, 5.0]), np.array([10.0, 10.0, 5.0]), [], [])
    assert res_oob.status == "OUT_OF_BOUNDS"
