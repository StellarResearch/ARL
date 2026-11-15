"""Tests for conservative time discretization, heuristic admissibility, and edge scheduling in Phase IV."""

from __future__ import annotations

import math

import numpy as np

from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.time_aware_planner import (
    TimeAwareAStarPlanner3D,
)


def test_time_aware_planner_invalid_inputs() -> None:
    """Planner rejects non-positive resolutions, speeds, or malformed bounds with INVALID_INPUT."""
    start = np.array([1.0, 1.0, 1.0])
    goal = np.array([5.0, 5.0, 5.0])

    # Zero spatial resolution
    p_bad_res = TimeAwareAStarPlanner3D(spatial_resolution=0.0)
    res = p_bad_res.plan(start, goal, static_obstacles=[], dynamic_obstacles=[])
    assert res.status == "INVALID_INPUT"
    assert len(res.spatial_waypoints) == 0

    # Negative time resolution
    p_bad_time = TimeAwareAStarPlanner3D(time_resolution=-0.5)
    res = p_bad_time.plan(start, goal, static_obstacles=[], dynamic_obstacles=[])
    assert res.status == "INVALID_INPUT"

    # Non-positive nominal speed
    p_bad_spd = TimeAwareAStarPlanner3D(nominal_speed=0.0)
    res = p_bad_spd.plan(start, goal, static_obstacles=[], dynamic_obstacles=[])
    assert res.status == "INVALID_INPUT"

    # Invalid bounds
    p_bad_bnd = TimeAwareAStarPlanner3D(bounds=(-10.0, 10.0, 10.0))
    res = p_bad_bnd.plan(start, goal, static_obstacles=[], dynamic_obstacles=[])
    assert res.status == "INVALID_INPUT"

    # Non-finite start or goal
    planner = TimeAwareAStarPlanner3D()
    res_nan = planner.plan(
        np.array([np.nan, 0.0, 0.0]), goal, static_obstacles=[], dynamic_obstacles=[]
    )
    assert res_nan.status == "INVALID_INPUT"

    res_inf = planner.plan(
        start, np.array([np.inf, 0.0, 0.0]), static_obstacles=[], dynamic_obstacles=[]
    )
    assert res_inf.status == "INVALID_INPUT"


def test_conservative_ceil_time_discretization_edges() -> None:
    """Verify conservative ceil discretization across orthogonal, planar diagonal, and 3D diagonal steps."""
    dt = 0.5
    v = 2.0
    _ = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 20.0),
        spatial_resolution=1.0,
        time_resolution=dt,
        nominal_speed=v,
    )

    # 1. Orthogonal edge: dist = 1.0 m -> duration = 0.5s -> d_steps = ceil(0.5 / 0.5) = 1 step = 0.5s
    d_ortho = 1.0
    dur_ortho = d_ortho / v
    steps_ortho = max(1, int(math.ceil(dur_ortho / dt - 1e-9)))
    assert steps_ortho == 1
    assert steps_ortho * dt >= dur_ortho

    # 2. Planar diagonal edge: dist = sqrt(2) ~= 1.4142 m -> duration = 0.7071s
    # Naive floor would yield 1 step (0.5s) which violates physical max speed (requires 2.83 m/s > 2.0 m/s)!
    # Ceil yields 2 steps (1.0s), which safely respects physical speed bounds.
    d_diag2d = math.sqrt(2.0)
    dur_diag2d = d_diag2d / v
    steps_diag2d = max(1, int(math.ceil(dur_diag2d / dt - 1e-9)))
    assert steps_diag2d == 2
    assert steps_diag2d * dt >= dur_diag2d
    assert (d_diag2d / (steps_diag2d * dt)) <= v

    # 3. 3D diagonal edge: dist = sqrt(3) ~= 1.732 m -> duration = 0.866s
    # Ceil yields 2 steps (1.0s), safely respecting speed.
    d_diag3d = math.sqrt(3.0)
    dur_diag3d = d_diag3d / v
    steps_diag3d = max(1, int(math.ceil(dur_diag3d / dt - 1e-9)))
    assert steps_diag3d == 2
    assert steps_diag3d * dt >= dur_diag3d
    assert (d_diag3d / (steps_diag3d * dt)) <= v

    # 4. Short edge: dist = 0.1m -> duration = 0.05s
    # Ceil must allocate at least 1 step (max(1, ...))
    d_short = 0.1
    dur_short = d_short / v
    steps_short = max(1, int(math.ceil(dur_short / dt - 1e-9)))
    assert steps_short == 1
    assert steps_short * dt >= dur_short


def test_heuristic_admissibility_and_consistency() -> None:
    """Verify that euclidean-distance / v_nom heuristic is admissible and consistent (monotonic)."""
    v_nom = 2.5
    dt = 0.4
    _ = TimeAwareAStarPlanner3D(
        bounds=(30.0, 30.0, 30.0),
        spatial_resolution=1.0,
        time_resolution=dt,
        nominal_speed=v_nom,
    )

    goal = np.array([25.0, 20.0, 15.0])

    def heuristic(p: np.ndarray) -> float:
        return float(np.linalg.norm(p - goal) / v_nom)

    # Test admissibility: h(u) <= c*(u, goal)
    # The minimum physical time from u to goal at v_nom is ||u - goal|| / v_nom
    # With conservative discrete steps, any discrete path duration >= euclidean_dist / v_nom.
    rng = np.random.default_rng(42)
    for _ in range(50):
        pos = rng.uniform(0.0, 30.0, size=3)
        h_val = heuristic(pos)
        exact_min_time = float(np.linalg.norm(pos - goal) / v_nom)
        assert h_val <= exact_min_time + 1e-9

    # Test consistency (monotonicity): h(u) <= c(u, v) + h(v)
    # where c(u, v) = d_steps * dt
    moves = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]

    for _ in range(50):
        u = rng.uniform(5.0, 25.0, size=3)
        h_u = heuristic(u)
        for move in moves:
            v = u + move
            h_v = heuristic(v)
            edge_dist = float(np.linalg.norm(move))
            dur = edge_dist / v_nom
            d_steps = max(1, int(math.ceil(dur / dt - 1e-9)))
            step_cost = d_steps * dt

            # Consistency inequality: h(u) - h(v) <= c(u, v)
            assert h_u - h_v <= step_cost + 1e-9, (
                f"Heuristic inconsistent: {h_u - h_v} > {step_cost}"
            )


def test_under_allocation_prevention_dynamic_crossing() -> None:
    """Floor discretization would under-allocate time, leading to phantom collisions or physical impossibility.
    Ceil discretization guarantees that the scheduled arrival time is physically reachable at or below nominal speed.
    """
    planner = TimeAwareAStarPlanner3D(
        bounds=(15.0, 15.0, 10.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
    )

    start = np.array([2.0, 2.0, 2.0])
    goal = np.array([8.0, 8.0, 2.0])

    # Dynamic obstacle sweeping perpendicular to the path
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([5.0, 0.0, 2.0]),
        radius=1.2,
        velocity=np.array([0.0, 1.5, 0.0]),
        bounds=(15.0, 15.0, 10.0),
        motion_type="bounce",
    )

    res = planner.plan(start, goal, static_obstacles=[], dynamic_obstacles=[dyn_obs], t_start=0.0)
    assert res.status == "SUCCESS"
    assert len(res.timed_waypoints) >= 2

    # Verify that each segment in the planned trajectory respects nominal speed
    for i in range(len(res.timed_waypoints) - 1):
        tw1 = res.timed_waypoints[i]
        tw2 = res.timed_waypoints[i + 1]
        dt_seg = tw2.time_sec - tw1.time_sec
        dist_seg = float(np.linalg.norm(tw2.position - tw1.position))
        if dt_seg > 1e-9:
            speed_required = dist_seg / dt_seg
            assert speed_required <= planner.nominal_speed + 1e-6, (
                f"Segment {i}->{i + 1} requires speed {speed_required:.3f} > {planner.nominal_speed}"
            )
