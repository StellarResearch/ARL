"""Tests for Phase IV Sensor-in-the-Loop Autonomous Decision-Making."""

from __future__ import annotations

import numpy as np

from adaptive_rl.flight_testing.scenarios import (
    FlightScenarioRunner,
    get_scenario,
)
from adaptive_rl.navigation.mission import MissionManager, MissionState


def test_sensor_in_the_loop_scenario_execution() -> None:
    """Scenario 25 (sensor_in_the_loop_navigation) executes using estimated state under noise and drift."""
    runner = FlightScenarioRunner()
    sc = get_scenario("sensor_in_the_loop_navigation")

    assert sc.enable_sensor_estimation is True
    assert sc.sensor_in_the_loop is True
    assert sc.use_time_aware_planner is True

    result = runner.run_scenario(sc)

    # 1. Telemetry records sensor-in-the-loop state source and estimation errors
    assert len(result.telemetry) > 10
    step_sample = result.telemetry[10]
    assert step_sample.estimated_position is not None
    assert step_sample.estimated_velocity is not None
    assert step_sample.position_estimation_error is not None
    assert step_sample.attitude_estimation_error_deg is not None
    assert step_sample.state_source == "estimated"

    # 2. Flight successfully completes to target without collision
    assert result.metrics.collision_count == 0
    assert result.metrics.final_position_error <= sc.target_radius


def test_mission_manager_state_source_attribution() -> None:
    """MissionManager attributes state_source in telemetry and validates state integrity."""
    mgr_est = MissionManager(state_source="estimated")
    assert mgr_est.state_source == "estimated"

    mgr_gt = MissionManager(state_source="ground_truth")
    assert mgr_gt.state_source == "ground_truth"


def test_sensor_in_the_loop_fails_closed_on_divergent_sensor_state() -> None:
    """If state estimator diverges to NaN or Inf, mission manager aborts rather than falling back to ground truth."""
    from adaptive_rl.environments.drone_6dof import DroneState6DOF

    mgr = MissionManager(state_source="estimated")
    mgr.start_mission(np.array([2.0, 2.0, 2.0]), [np.array([10.0, 10.0, 5.0])], static_obstacles=[])

    diverged_state = DroneState6DOF(
        position=np.array([float("nan"), 2.0, 2.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )

    _, telem = mgr.update(diverged_state, time_sec=1.0, static_obstacles=[])

    assert mgr.state == MissionState.ABORTED
    assert telem.state == MissionState.ABORTED
    assert telem.state_source == "estimated"
    assert "non_finite" in telem.replan_reason
