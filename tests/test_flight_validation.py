"""Unit tests for autonomous flight validation scenarios and scenario execution runner."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from adaptive_rl.controllers.pid import QuadrotorPIDController
from adaptive_rl.flight_testing import (
    FlightScenarioDefinition,
    FlightScenarioResult,
    FlightScenarioRunner,
    FlightStepTelemetry,
    get_standard_flight_scenarios,
)


def test_standard_flight_scenarios_definitions() -> None:
    """Verify that all 10 required flight scenarios exist and have valid specifications."""
    scenarios = get_standard_flight_scenarios()
    assert len(scenarios) == 10

    expected_names = [
        "hover_equilibrium",
        "straight_line_navigation",
        "vertical_ascent_descent",
        "horizontal_translation",
        "position_holding_disturbance",
        "obstacle_avoidance",
        "boundary_avoidance",
        "target_approach_arrival",
        "attitude_disturbance_recovery",
        "actuator_fault_degradation",
    ]

    actual_names = [s.name for s in scenarios]
    for exp_name in expected_names:
        assert exp_name in actual_names, f"Missing required scenario: {exp_name}"

    for sc in scenarios:
        assert isinstance(sc.name, str) and len(sc.name) > 0
        assert isinstance(sc.description, str) and len(sc.description) > 0
        assert sc.initial_position.shape == (3,)
        assert np.all(np.isfinite(sc.initial_position))
        assert sc.initial_velocity.shape == (3,)
        assert sc.initial_quaternion.shape == (4,)
        assert math.isclose(np.linalg.norm(sc.initial_quaternion), 1.0, abs_tol=1e-5)
        assert sc.max_steps > 0
        assert sc.target_radius > 0.0
        assert sc.collision_radius > 0.0
        assert len(sc.bounds) == 3


def test_flight_scenario_runner_hover() -> None:
    """Run hover equilibrium scenario and verify near-zero position and velocity error."""
    scenarios = get_standard_flight_scenarios()
    hover_sc = next(s for s in scenarios if s.name == "hover_equilibrium")

    # Shorten max_steps for fast unit test
    hover_sc.max_steps = 40

    runner = FlightScenarioRunner()
    ctrl = QuadrotorPIDController(dt=0.05)
    result = runner.run_scenario(hover_sc, controller=ctrl)

    assert isinstance(result, FlightScenarioResult)
    assert result.scenario_name == "hover_equilibrium"
    assert result.metrics.success is True
    assert result.metrics.outcome == "success"
    assert result.metrics.final_position_error < 0.05
    assert result.metrics.rms_position_error < 0.05
    assert result.metrics.collision_count == 0
    assert result.metrics.duration_steps > 0
    assert result.metrics.execution_wall_time_ms > 0.0

    # Verify telemetry details
    assert len(result.telemetry) == result.metrics.duration_steps
    first_step = result.telemetry[0]
    assert isinstance(first_step, FlightStepTelemetry)
    assert first_step.step == 1
    assert first_step.time_sec == 0.05
    assert first_step.position.shape == (3,)
    assert first_step.euler_angles_deg.shape == (3,)


def test_flight_scenario_obstacle_collision_detection() -> None:
    """Obstacle avoidance scenario with obstacles accurately records clearance and collision."""
    scenarios = get_standard_flight_scenarios()
    obs_sc = next(s for s in scenarios if s.name == "obstacle_avoidance")
    obs_sc.max_steps = 50

    runner = FlightScenarioRunner()
    result = runner.run_scenario(obs_sc)

    assert result.metrics.outcome in ("collision", "success", "timeout")
    # Verify obstacle clearance was evaluated and is finite
    assert math.isfinite(result.metrics.min_obstacle_clearance)
    if result.metrics.outcome == "collision":
        assert result.metrics.collision_count == 1


def test_flight_scenario_export_json_and_csv(tmp_path: Path) -> None:
    """Verify JSON record and CSV time-series export from FlightScenarioResult."""
    sc = FlightScenarioDefinition(
        name="test_export_scenario",
        description="Quick scenario to test artifact export serialization",
        initial_position=np.array([10.0, 10.0, 5.0]),
        waypoints=[np.array([12.0, 12.0, 6.0])],
        max_steps=10,
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc)

    # 1. Export JSON
    json_path = tmp_path / "test_scenario.json"
    result.export_json(json_path, include_telemetry=True)
    assert json_path.is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["scenario_name"] == "test_export_scenario"
    assert "metrics" in data
    assert "telemetry" in data
    assert len(data["telemetry"]) == result.metrics.duration_steps

    # 2. Export CSV
    csv_path = tmp_path / "test_scenario.csv"
    result.export_telemetry_csv(csv_path)
    assert csv_path.is_file()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == result.metrics.duration_steps
    assert "step" in rows[0]
    assert "pos_z" in rows[0]
    assert "speed" in rows[0]
    assert "distance_to_target" in rows[0]


def test_flight_scenario_runner_run_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run_all_scenarios returns a result for each scenario in the suite."""
    runner = FlightScenarioRunner()
    # Fast test with 2 small scenarios
    sc1 = FlightScenarioDefinition(
        name="fast_sc1",
        description="Fast 1",
        initial_position=np.array([5.0, 5.0, 5.0]),
        max_steps=5,
    )
    sc2 = FlightScenarioDefinition(
        name="fast_sc2",
        description="Fast 2",
        initial_position=np.array([10.0, 10.0, 5.0]),
        max_steps=5,
    )

    results = runner.run_all_scenarios(scenarios=[sc1, sc2])
    assert len(results) == 2
    assert results[0].scenario_name == "fast_sc1"
    assert results[1].scenario_name == "fast_sc2"
