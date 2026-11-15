"""Tests for Phase IV strict fail-closed sensor state validation and sensor initialization."""

from __future__ import annotations

import numpy as np

from adaptive_rl.environments.drone_6dof import DroneState6DOF
from adaptive_rl.flight_testing.scenarios import FlightScenarioRunner, get_scenario
from adaptive_rl.navigation.mission import MissionManager, MissionState
from adaptive_rl.navigation.time_aware_planner import TimeAwareAStarPlanner3D


def _create_test_mission_manager() -> MissionManager:
    planner = TimeAwareAStarPlanner3D(
        bounds=(20.0, 20.0, 10.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
    )
    manager = MissionManager(
        time_aware_planner=planner,
        state_source="estimated",
        waypoint_acceptance_radius=1.0,
    )
    start = np.array([2.0, 2.0, 2.0])
    goal = np.array([10.0, 2.0, 2.0])
    manager.start_mission(start, [goal], static_obstacles=[])
    return manager


def test_fail_closed_nan_inf_sensor_state() -> None:
    """MissionManager immediately transitions to ABORTED when estimated state has non-finite values."""
    # 1. NaN position
    mgr = _create_test_mission_manager()
    bad_state = DroneState6DOF(
        position=np.array([np.nan, 2.0, 2.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    _, telem = mgr.update(bad_state, time_sec=0.1, static_obstacles=[])
    assert mgr.state == MissionState.ABORTED
    assert "fail_closed" in mgr.replan_reason
    assert telem.state == MissionState.ABORTED

    # 2. Inf velocity
    mgr = _create_test_mission_manager()
    bad_state = DroneState6DOF(
        position=np.array([2.0, 2.0, 2.0]),
        velocity=np.array([np.inf, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    _, telem = mgr.update(bad_state, time_sec=0.1, static_obstacles=[])
    assert mgr.state == MissionState.ABORTED
    assert "fail_closed" in mgr.replan_reason

    # 3. NaN angular velocity
    mgr = _create_test_mission_manager()
    bad_state = DroneState6DOF(
        position=np.array([2.0, 2.0, 2.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, np.nan, 0.0]),
    )
    _, telem = mgr.update(bad_state, time_sec=0.1, static_obstacles=[])
    assert mgr.state == MissionState.ABORTED
    assert "fail_closed" in mgr.replan_reason


def test_fail_closed_unnormalized_quaternion() -> None:
    """MissionManager immediately aborts if quaternion norm deviates significantly from 1.0."""
    # 1. Zero quaternion (norm = 0.0)
    mgr = _create_test_mission_manager()
    bad_state = DroneState6DOF(
        position=np.array([2.0, 2.0, 2.0]),
        velocity=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    _, telem = mgr.update(bad_state, time_sec=0.1, static_obstacles=[])
    assert mgr.state == MissionState.ABORTED
    assert "fail_closed" in mgr.replan_reason

    # 2. Under-normalized quaternion (norm = 0.5)
    mgr = _create_test_mission_manager()
    bad_state = DroneState6DOF(
        position=np.array([2.0, 2.0, 2.0]),
        velocity=np.zeros(3),
        quaternion=np.array([0.5, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    _, telem = mgr.update(bad_state, time_sec=0.1, static_obstacles=[])
    assert mgr.state == MissionState.ABORTED
    assert "fail_closed" in mgr.replan_reason

    # 3. Over-normalized quaternion (norm = 2.0)
    mgr = _create_test_mission_manager()
    bad_state = DroneState6DOF(
        position=np.array([2.0, 2.0, 2.0]),
        velocity=np.zeros(3),
        quaternion=np.array([2.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    _, telem = mgr.update(bad_state, time_sec=0.1, static_obstacles=[])
    assert mgr.state == MissionState.ABORTED
    assert "fail_closed" in mgr.replan_reason


def test_sensor_in_the_loop_pre_step1_initialization() -> None:
    """Verify Scenario 25 initializes sensor estimator before step 1 and runs safely."""
    runner = FlightScenarioRunner()
    sc = get_scenario("sensor_in_the_loop_navigation")
    sc.max_steps = 20
    result = runner.run_scenario(sc)
    assert result.metrics.collision_count == 0
    assert result.metrics.outcome != "sensor_estimation_failure"
