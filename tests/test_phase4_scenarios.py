"""Unit tests for Phase IV Predictive Autonomous Mission Planning and Flight Navigation Scenarios.

Verifies end-to-end simulation execution across Scenarios 20 through 27 using FlightScenarioRunner.
"""

from __future__ import annotations

import numpy as np

from adaptive_rl.flight_testing.scenarios import (
    FlightScenarioRunner,
    get_scenario,
)


def test_scenario_20_dynamic_obstacle_crossing() -> None:
    """Scenario 20: Time-aware 4D planning anticipates and evades a perpendicular crossing obstacle."""
    runner = FlightScenarioRunner()
    sc = get_scenario("dynamic_obstacle_crossing_time_aware")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0, f"Collision occurred in scenario 20: {res.metrics}"
    assert res.metrics.success is True, f"Scenario 20 did not succeed: {res.metrics}"
    assert res.metrics.outcome == "success"
    assert res.metrics.final_position_error <= sc.target_radius
    assert res.metrics.min_obstacle_clearance > 0.0


def test_scenario_21_multi_dynamic_obstacle_corridor() -> None:
    """Scenario 21: Autonomous corridor navigation through multiple moving obstacles."""
    runner = FlightScenarioRunner()
    sc = get_scenario("multi_dynamic_obstacle_corridor")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0, f"Collision occurred in scenario 21: {res.metrics}"
    assert res.metrics.success is True, f"Scenario 21 did not succeed: {res.metrics}"
    assert res.metrics.outcome == "success"
    assert res.metrics.min_obstacle_clearance > 0.0


def test_scenario_22_temporary_corridor_blockage_wait() -> None:
    """Scenario 22: Corridor blockage leads to safe in-place hover waiting until obstacle clears."""
    runner = FlightScenarioRunner()
    sc = get_scenario("temporary_corridor_blockage_wait")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0, f"Collision occurred in scenario 22: {res.metrics}"
    assert res.metrics.success is True, f"Scenario 22 did not succeed: {res.metrics}"
    assert res.metrics.outcome == "success"

    # Telemetry must verify non-zero waiting occurred or WAITING state was engaged
    waiting_steps = [
        t
        for t in res.telemetry
        if t.mission_state == "WAITING" or (t.waiting_duration or 0.0) > 0.0
    ]
    assert len(waiting_steps) > 0, "Mission never entered WAITING state during corridor blockage"


def test_scenario_23_sequential_multi_waypoint_adaptive() -> None:
    """Scenario 23: Multi-waypoint mission achieves all goals in strictly sequential order."""
    runner = FlightScenarioRunner()
    sc = get_scenario("sequential_multi_waypoint_adaptive")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0, f"Collision occurred in scenario 23: {res.metrics}"

    # Verify chronological sequence of arrivals: t_wp0 < t_wp1 < t_wp2 < t_wp3
    arrival_times = []
    for wp in sc.waypoints:
        reached_t = None
        for telem in res.telemetry:
            if np.linalg.norm(telem.position - wp) <= sc.target_radius:
                reached_t = telem.time_sec
                break
        arrival_times.append(reached_t)

    for idx, t_arr in enumerate(arrival_times):
        assert t_arr is not None, f"Waypoint {idx} ({sc.waypoints[idx]}) was never reached"

    for i in range(len(arrival_times) - 1):
        assert arrival_times[i] < arrival_times[i + 1], (
            f"Waypoints reached out of order: {arrival_times}"
        )


def test_scenario_24_repeated_dynamic_replanning() -> None:
    """Scenario 24: Active oscillating dynamic obstacles trigger continual receding-horizon replanning."""
    runner = FlightScenarioRunner()
    sc = get_scenario("repeated_dynamic_replanning")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0, f"Collision occurred in scenario 24: {res.metrics}"
    assert res.metrics.replan_count > 0, (
        "No dynamic replans were triggered under oscillating obstacles"
    )
    assert len(res.telemetry) > 50


def test_scenario_25_sensor_in_the_loop_navigation() -> None:
    """Scenario 25: Autonomous navigation under avionics sensor fusion state estimation."""
    runner = FlightScenarioRunner()
    sc = get_scenario("sensor_in_the_loop_navigation")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0, f"Collision occurred in scenario 25: {res.metrics}"
    assert res.metrics.success is True, f"Scenario 25 failed: {res.metrics}"
    assert res.metrics.final_position_error <= sc.target_radius

    # Telemetry checks for estimated state
    assert len(res.telemetry) > 20
    sample = res.telemetry[20]
    assert sample.state_source == "estimated"
    assert sample.estimated_position is not None
    assert sample.position_estimation_error is not None


def test_scenario_26_unreachable_mission_safe_blocked() -> None:
    """Scenario 26: Fully enclosed unreachable destination causes fail-closed safe blocked abort."""
    runner = FlightScenarioRunner()
    sc = get_scenario("unreachable_mission_safe_blocked")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0
    assert res.metrics.success is False
    assert res.metrics.outcome == "blocked_no_path"


def test_scenario_27_prediction_divergence_recovery() -> None:
    """Scenario 27: Dynamic obstacle divergence triggers online recovery and safe navigation."""
    runner = FlightScenarioRunner()
    sc = get_scenario("prediction_divergence_recovery")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0, f"Collision occurred in scenario 27: {res.metrics}"
    assert res.metrics.success is True
    assert res.metrics.final_position_error <= sc.target_radius
