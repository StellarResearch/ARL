"""Tests for continuous trajectory risk evaluation, swept-sphere clearance, and static chord clearance."""

from __future__ import annotations

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.time_aware_planner import (
    RouteEvaluator,
    TimeAwareAStarPlanner3D,
    TimedWaypoint,
)


def test_static_chord_analytical_clearance() -> None:
    """Validate point-to-segment analytical distance and clearance against static spheres."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 20.0),
        vehicle_radius=0.5,
        safety_margin=0.2,
    )

    # Static obstacle centered at (5, 5, 5) with radius 1.0
    obs = ObstacleSphere3D(center=np.array([5.0, 5.0, 5.0]), radius=1.0)

    # 1. Segment passing straight through the obstacle center: (2, 5, 5) to (8, 5, 5)
    valid, clr = planner.is_static_segment_valid(
        np.array([2.0, 5.0, 5.0]),
        np.array([8.0, 5.0, 5.0]),
        [obs],
    )
    assert valid is False
    # At center (5, 5, 5), distance is 0.0, clearance = 0.0 - (1.0 + 0.5) = -1.5
    assert clr <= -0.5

    # 2. Segment passing alongside at distance 2.0 m: (2, 7, 5) to (8, 7, 5)
    # Distance to center is 2.0 m. Clearance = 2.0 - (1.0 + 0.5) = +0.5 m
    # Inflation margin is vehicle_radius + safety_margin = 0.5 + 0.2 = 0.7 m.
    # Since d = 2.0 >= 1.0 + 0.7 = 1.7 m, valid is True.
    valid_side, clr_side = planner.is_static_segment_valid(
        np.array([2.0, 7.0, 5.0]),
        np.array([8.0, 7.0, 5.0]),
        [obs],
    )
    assert valid_side is True
    assert np.isclose(clr_side, 0.5, atol=1e-3)

    # 3. Segment passing just within safety margin: distance = 1.6 m
    # 1.6 < 1.0 + 0.7 = 1.7, so valid is False, but clearance = 1.6 - 1.5 = +0.1 m
    valid_margin, clr_margin = planner.is_static_segment_valid(
        np.array([2.0, 6.6, 5.0]),
        np.array([8.0, 6.6, 5.0]),
        [obs],
    )
    assert valid_margin is False
    assert np.isclose(clr_margin, 0.1, atol=1e-3)


def test_dynamic_transition_swept_collision() -> None:
    """Validate continuous swept-sphere collision detection between moving drone and moving obstacle."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 20.0),
        vehicle_radius=0.5,
        safety_margin=0.2,
    )

    # Moving obstacle starts at (5, 0, 5) moving at +2.0 m/s in y: center(t) = (5, 2*t, 5)
    # Radius = 1.0 m. Total effective radius = 1.0 + 0.5 + 0.2 = 1.7 m.
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([5.0, 0.0, 5.0]),
        radius=1.0,
        velocity=np.array([0.0, 2.0, 0.0]),
        bounds=(20.0, 20.0, 20.0),
        motion_type="bounce",
    )

    # Drone moves along x from (2, 4, 5) to (8, 4, 5) over [t=1.5, t=2.5]
    # At t=2.0s:
    # Drone position is (5, 4, 5)
    # Obstacle center is (5, 4, 5) -> Exact impact!
    collides = planner.check_dynamic_transition_collision(
        p_start=np.array([2.0, 4.0, 5.0]),
        p_end=np.array([8.0, 4.0, 5.0]),
        t_start=1.5,
        t_end=2.5,
        dynamic_obstacles=[dyn_obs],
    )
    assert collides is True

    # Same spatial segment, but earlier in time: [t=0.0, t=0.5]
    # Obstacle y is in [0.0, 1.0], while drone y is 4.0 (clearance >= 3.0 m - 1.7 m > 1.0 m)
    safe_earlier = planner.check_dynamic_transition_collision(
        p_start=np.array([2.0, 4.0, 5.0]),
        p_end=np.array([8.0, 4.0, 5.0]),
        t_start=0.0,
        t_end=0.5,
        dynamic_obstacles=[dyn_obs],
    )
    assert safe_earlier is False


def test_waiting_interval_clearance() -> None:
    """Validate continuous clearance when vehicle holds position while obstacle passes."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 20.0),
        vehicle_radius=0.5,
        safety_margin=0.2,
    )

    # Obstacle passes through (5, 5, 5) at t=2.0s
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([5.0, 1.0, 5.0]),
        radius=0.8,
        velocity=np.array([0.0, 2.0, 0.0]),
        bounds=(20.0, 20.0, 20.0),
        motion_type="bounce",
    )

    # Vehicle holds stationary at (5, 5, 5) from t=1.8 to t=2.2 -> Collision during wait
    collides_wait = planner.check_dynamic_transition_collision(
        p_start=np.array([5.0, 5.0, 5.0]),
        p_end=np.array([5.0, 5.0, 5.0]),
        t_start=1.8,
        t_end=2.2,
        dynamic_obstacles=[dyn_obs],
    )
    assert collides_wait is True

    # Vehicle holds stationary safely at (5, 9, 5) from t=1.8 to t=2.2 -> Safe
    safe_wait = planner.check_dynamic_transition_collision(
        p_start=np.array([5.0, 9.0, 5.0]),
        p_end=np.array([5.0, 9.0, 5.0]),
        t_start=1.8,
        t_end=2.2,
        dynamic_obstacles=[dyn_obs],
    )
    assert safe_wait is False


def test_route_evaluator_metrics_and_scoring() -> None:
    """Validate RouteEvaluator multi-objective metrics on complete timed trajectory."""
    evaluator = RouteEvaluator(
        nominal_speed=2.0,
        weight_time=1.0,
        weight_distance=0.5,
        weight_curvature=0.2,
        weight_risk=1.5,
        weight_waiting=0.8,
    )

    waypoints = [
        TimedWaypoint(position=np.array([0.0, 0.0, 2.0]), time_sec=0.0, is_waiting=False),
        TimedWaypoint(position=np.array([2.0, 0.0, 2.0]), time_sec=1.0, is_waiting=True),
        TimedWaypoint(position=np.array([2.0, 0.0, 2.0]), time_sec=2.0, is_waiting=False),
        TimedWaypoint(position=np.array([4.0, 0.0, 2.0]), time_sec=3.0, is_waiting=False),
    ]

    static_obs = [ObstacleSphere3D(center=np.array([2.0, 3.0, 2.0]), radius=1.0)]

    metrics = evaluator.evaluate(
        timed_waypoints=waypoints,
        static_obstacles=static_obs,
        dynamic_obstacles=[],
    )

    assert metrics.is_collision_free is True
    assert np.isclose(metrics.path_length, 4.0, atol=1e-3)
    assert np.isclose(metrics.travel_time, 3.0, atol=1e-3)
    assert np.isclose(metrics.waiting_time, 1.0, atol=1e-3)
    # Static clearance from (2, 0, 2) to center (2, 3, 2) is 3.0 - (1.0 + 0.8) = 1.2
    assert metrics.min_clearance > 0.0
    assert metrics.risk_score >= 0.0
    assert metrics.total_objective_cost > 0.0
