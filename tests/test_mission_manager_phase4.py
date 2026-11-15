"""Tests for Phase IV Enhanced MissionManager: State Machine, Hysteresis, Waiting & Replanning."""

from __future__ import annotations

from typing import Optional

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import DroneState6DOF
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.mission import MissionManager, MissionState
from adaptive_rl.navigation.planner import AStarPlanner3D
from adaptive_rl.navigation.time_aware_planner import TimeAwareAStarPlanner3D


def _make_state(pos: np.ndarray, vel: Optional[np.ndarray] = None) -> DroneState6DOF:
    return DroneState6DOF(
        position=pos.copy(),
        velocity=vel.copy() if vel is not None else np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )


def test_mission_state_machine_deterministic_transitions() -> None:
    """MissionManager traverses IDLE -> PLANNING -> EXECUTING -> ARRIVED deterministically."""
    mgr = MissionManager(waypoint_acceptance_radius=1.0)
    assert mgr.state == MissionState.IDLE

    start = np.array([2.0, 2.0, 2.0])
    goal = np.array([10.0, 2.0, 2.0])

    ok = mgr.start_mission(start, [goal], static_obstacles=[])
    assert ok is True
    assert mgr.state == MissionState.EXECUTING

    # Midway point
    mid_state = _make_state(np.array([5.0, 2.0, 2.0]))
    target, telem = mgr.update(mid_state, time_sec=0.5, static_obstacles=[])
    assert mgr.state == MissionState.EXECUTING
    assert telem.state == MissionState.EXECUTING

    # Reaching final goal
    goal_state = _make_state(np.array([10.0, 2.0, 2.0]), vel=np.array([0.1, 0.0, 0.0]))
    _, telem_final = mgr.update(goal_state, time_sec=5.0, static_obstacles=[])
    assert mgr.state == MissionState.ARRIVED
    assert telem_final.state == MissionState.ARRIVED


def test_mission_replan_cooldown_and_critical_safety_override() -> None:
    """Non-critical replanning observes cooldown, but critical clearance bypasses cooldown immediately."""
    planner = AStarPlanner3D(bounds=(20.0, 20.0, 10.0), vehicle_radius=0.5, safety_margin=0.3)
    mgr = MissionManager(
        planner=planner,
        waypoint_acceptance_radius=1.0,
        replan_cooldown_sec=1.0,  # 1.0s cooldown
    )

    start = np.array([2.0, 5.0, 5.0])
    goal = np.array([10.0, 5.0, 5.0])
    mgr.start_mission(start, [goal], static_obstacles=[])
    assert mgr.state == MissionState.EXECUTING

    # 1. First trigger at t=0.5s: non-critical dynamic obstacle intersects corridor
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([5.0, 5.0, 5.0]),
        radius=1.0,
        velocity=np.zeros(3),
        bounds=(20.0, 20.0, 10.0),
        motion_type="bounce",
    )
    state = _make_state(start)
    mgr.update(state, time_sec=0.5, static_obstacles=[], dynamic_obstacles=[dyn_obs])
    assert mgr.replan_count == 1
    assert mgr.last_replan_time_sec == 0.5

    # 2. At t=0.7s (within 1.0s cooldown): non-critical trigger is suppressed by cooldown
    mgr.update(state, time_sec=0.7, static_obstacles=[], dynamic_obstacles=[dyn_obs])
    assert mgr.replan_count == 1  # Not incremented due to cooldown

    # 3. At t=0.8s: CRITICAL obstacle proximity (< vehicle_radius + 0.1) immediately bypasses cooldown!
    crit_dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([2.5, 5.0, 5.0]),  # Distance is 0.5 <= 0.5 + 0.1
        radius=0.2,
        velocity=np.zeros(3),
        bounds=(20.0, 20.0, 10.0),
        motion_type="bounce",
    )
    mgr.update(state, time_sec=0.8, static_obstacles=[], dynamic_obstacles=[crit_dyn_obs])
    assert mgr.replan_count == 2  # Bypassed cooldown for immediate safety response!
    assert mgr.last_replan_time_sec == 0.8


def test_mission_off_track_path_deviation_trigger() -> None:
    """When drone position drifts excessively from planned corridor, replanning is triggered."""
    planner = AStarPlanner3D(bounds=(20.0, 20.0, 10.0), vehicle_radius=0.5, safety_margin=0.3)
    mgr = MissionManager(
        planner=planner,
        waypoint_acceptance_radius=1.0,
        max_path_deviation_threshold=1.5,
    )
    start = np.array([2.0, 5.0, 5.0])
    goal = np.array([10.0, 5.0, 5.0])
    mgr.start_mission(start, [goal], static_obstacles=[])

    # Drone drifts laterally to y=8.0 (deviation = 3.0m > 1.5m threshold)
    drifted_state = _make_state(np.array([4.0, 8.0, 5.0]))
    mgr.update(drifted_state, time_sec=1.0, static_obstacles=[])
    assert mgr.replan_count == 1
    assert mgr.replan_reason == "excessive_path_deviation"
    assert mgr.planned_vs_actual_deviation >= 2.5


def test_mission_waiting_state_and_timeout_transition() -> None:
    """When corridor is temporarily blocked, time-aware manager enters WAITING and times out to FAILED_BLOCKED."""
    ta_planner = TimeAwareAStarPlanner3D(
        bounds=(15.0, 15.0, 8.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
        vehicle_radius=0.4,
        safety_margin=0.2,
    )
    mgr = MissionManager(
        time_aware_planner=ta_planner,
        max_waiting_duration_sec=2.0,  # 2.0s maximum waiting
    )
    start = np.array([3.0, 5.0, 2.0])
    goal = np.array([7.0, 5.0, 2.0])
    mgr.start_mission(start, [goal], static_obstacles=[])
    assert mgr.state == MissionState.EXECUTING

    # Impassable barrier completely surrounding goal and drone
    barrier_obs = [
        ObstacleSphere3D(center=np.array([5.0, 5.0, 2.0]), radius=2.5),
    ]

    state = _make_state(start)
    # At t=0.5s: trigger replanning which fails to find any route -> enters WAITING
    target, telem = mgr.update(state, time_sec=0.5, static_obstacles=barrier_obs)
    assert mgr.state == MissionState.WAITING
    assert telem.state == MissionState.WAITING
    assert np.allclose(target, start)

    # At t=1.5s: wait elapsed = 1.0s < 2.0s -> still WAITING
    _, telem_wait = mgr.update(state, time_sec=1.5, static_obstacles=barrier_obs)
    assert mgr.state == MissionState.WAITING
    assert telem_wait.waiting_duration >= 1.0

    # At t=3.0s: wait elapsed = 2.5s >= 2.0s -> transitions to FAILED_BLOCKED
    _, telem_blocked = mgr.update(state, time_sec=3.0, static_obstacles=barrier_obs)
    assert mgr.state == MissionState.FAILED_BLOCKED
    assert telem_blocked.state == MissionState.FAILED_BLOCKED
    assert "waiting_duration_exceeded" in mgr.replan_reason


def test_mission_non_finite_sensor_state_aborts_immediately() -> None:
    """Non-finite sensor observation fails closed immediately without silent fallback."""
    mgr = MissionManager()
    mgr.start_mission(np.array([2.0, 2.0, 2.0]), [np.array([5.0, 5.0, 5.0])], static_obstacles=[])

    bad_state = _make_state(np.array([np.nan, 2.0, 2.0]))
    _, telem = mgr.update(bad_state, time_sec=1.0, static_obstacles=[])

    assert mgr.state == MissionState.ABORTED
    assert telem.state == MissionState.ABORTED
    assert "non_finite" in telem.replan_reason
