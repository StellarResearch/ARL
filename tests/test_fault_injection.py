"""Unit tests for actuator fault injection, timing, thrust reduction, and classification (Issue #277, Phase III)."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

from adaptive_rl.flight_testing.scenarios import (
    FlightScenarioDefinition,
    FlightScenarioRunner,
    get_standard_flight_scenarios,
)


def test_canonical_fault_scenario_definition() -> None:
    """Verify that Scenario 10 is configured with the canonical 50% thrust fault."""
    scenarios = {s.name: s for s in get_standard_flight_scenarios()}
    assert "actuator_fault_degradation" in scenarios

    fault_sc = scenarios["actuator_fault_degradation"]
    assert fault_sc.actuator_fault_fn is not None

    # Before step 40: no reduction
    thrusts_nominal = np.array([3.0, 3.0, 3.0, 3.0])
    thrusts_step39 = fault_sc.actuator_fault_fn(39, thrusts_nominal)
    np.testing.assert_allclose(thrusts_step39, thrusts_nominal)

    # At and after step 40: canonical 50% thrust reduction on Motor 1
    thrusts_step40 = fault_sc.actuator_fault_fn(40, thrusts_nominal)
    expected_fault = np.array([1.5, 3.0, 3.0, 3.0])
    np.testing.assert_allclose(thrusts_step40, expected_fault)
    assert math.isclose(thrusts_step40[0], 0.5 * thrusts_nominal[0], rel_tol=1e-5)


def test_canonical_actuator_fault_telemetry_and_crash_classification() -> None:
    """Under canonical 50% thrust loss without re-allocation, drone crashes and is classified as crash_fault."""
    scenarios = {s.name: s for s in get_standard_flight_scenarios()}
    fault_sc = scenarios["actuator_fault_degradation"]

    runner = FlightScenarioRunner()
    result = runner.run_scenario(fault_sc)

    # Telemetry proves that standard PID cannot stabilize 50% rotor loss: outcome must be crash_fault
    assert result.metrics.outcome == "crash_fault"
    assert result.metrics.success is False

    # Verify telemetry tracking before and after fault injection
    assert len(result.telemetry) > 40

    # Before step 40: fault not active
    step_pre = result.telemetry[30]
    assert step_pre.actuator_fault_active is False

    # After step 40: fault active on Motor 1
    step_post = result.telemetry[45]
    assert step_post.actuator_fault_active is True
    assert step_post.commanded_thrusts is not None
    assert step_post.commanded_thrusts[0] > step_post.motor_thrusts[0]
    assert math.isclose(
        step_post.motor_thrusts[0], 0.5 * step_post.commanded_thrusts[0], rel_tol=1e-4
    )


def test_moderate_actuator_fault_recovers_degraded() -> None:
    """Under a survivable partial thrust loss, vehicle maintains level hover and classifies as recovered_degraded."""

    def moderate_fault(step: int, thrusts: np.ndarray) -> np.ndarray:
        if step >= 40:
            degraded = thrusts.copy()
            degraded[0] *= 0.98  # 2% thrust reduction on Motor 1 (within attitude recovery limits)
            return degraded
        return thrusts

    sc = FlightScenarioDefinition(
        name="moderate_fault_scenario",
        description="Moderate 2% thrust loss on Motor 1",
        initial_position=np.array([15.0, 15.0, 7.0]),
        waypoints=[np.array([15.0, 15.0, 7.0])],
        max_steps=120,
        actuator_fault_fn=moderate_fault,
        target_radius=1.5,
        settling_velocity_threshold=0.5,
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc)

    assert result.metrics.outcome == "recovered_degraded"
    assert result.metrics.success is True
    assert result.metrics.collision_count == 0
    assert result.metrics.max_attitude_error_deg < 35.0


def test_surviving_fault_without_reaching_goal_is_not_recovered_degraded() -> None:
    """If vehicle survives fault without crashing but does not settle at goal, outcome is max_steps, NOT recovered_degraded."""

    def drift_fault(step: int, thrusts: np.ndarray) -> np.ndarray:
        if step >= 40:
            degraded = thrusts.copy()
            degraded[0] *= 0.90  # 10% thrust reduction causes lateral position drift
            return degraded
        return thrusts

    sc = FlightScenarioDefinition(
        name="drift_fault_scenario",
        description="10% thrust loss causing vehicle to drift away from target",
        initial_position=np.array([15.0, 15.0, 7.0]),
        waypoints=[np.array([15.0, 15.0, 7.0])],
        max_steps=120,
        actuator_fault_fn=drift_fault,
        target_radius=1.2,
        settling_velocity_threshold=0.5,
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc)

    # Vehicle survived without hitting floor or boundary, but drifted 10m away from goal:
    assert result.metrics.collision_count == 0
    assert result.metrics.outcome != "recovered_degraded"
    assert result.metrics.outcome == "max_steps"
    assert result.metrics.success is False


def test_high_velocity_flythrough_rejected_from_success() -> None:
    """A vehicle passing through the target radius at high speed (> settling_velocity_threshold) is rejected from SUCCESS."""

    class HighSpeedController:
        def compute_action(
            self, state: Any, target_position: np.ndarray, target_velocity: np.ndarray, dt: float
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            # Command high forward tilt and excess thrust to speed through waypoint
            return np.array([0.9, 0.9, -0.8, -0.8], dtype=np.float32), {}

    sc = FlightScenarioDefinition(
        name="flythrough_scenario",
        description="High speed flythrough test",
        initial_position=np.array([5.0, 15.0, 7.0]),
        waypoints=[np.array([15.0, 15.0, 7.0])],
        max_steps=40,
        target_radius=3.0,
        settling_velocity_threshold=0.3,
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc, controller=HighSpeedController())

    # Velocity at waypoint exceeds settling threshold; must not be classified as success
    assert result.metrics.outcome != "success"


def test_collision_precedence_over_max_steps() -> None:
    """When a collision occurs, collision outcome takes precedence over max_steps or success."""
    sc = FlightScenarioDefinition(
        name="wall_crash_scenario",
        description="Fly directly into wall",
        initial_position=np.array([29.0, 15.0, 7.0]),
        waypoints=[np.array([35.0, 15.0, 7.0])],  # Outside bounds
        bounds=(30.0, 30.0, 15.0),
        max_steps=20,
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc)

    assert result.metrics.outcome in ("collision", "boundary_breach")
    assert result.metrics.outcome != "max_steps"
    assert result.metrics.success is False


def test_skipping_intermediate_waypoints_fails_mission() -> None:
    """A vehicle located at the final coordinate without completing preceding waypoints is not SUCCESS."""

    # Scenario: Start at (5, 5, 3), must visit W0=(20, 20, 10), then W1=(5, 5, 3).
    # Stationary controller stays at (5, 5, 3). It never completed W0!
    class HoverInPlaceController:
        def compute_action(
            self, state: Any, target_position: np.ndarray, target_velocity: np.ndarray, dt: float
        ) -> Tuple[np.ndarray, Dict[str, Any]]:
            # Zero normalized action = hover thrust
            return np.zeros(4, dtype=np.float32), {}

    sc = FlightScenarioDefinition(
        name="waypoint_skipping_test",
        description="Verify all waypoints in sequence must be visited",
        initial_position=np.array([5.0, 5.0, 3.0]),
        waypoints=[
            np.array([20.0, 20.0, 10.0]),  # Distant waypoint that must be visited first
            np.array([5.0, 5.0, 3.0]),  # Final destination matching start coordinate
        ],
        max_steps=30,
        target_radius=1.2,
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc, controller=HoverInPlaceController())

    # Distance to final waypoint is 0, but W0 was never completed: must NOT be success!
    assert result.metrics.outcome != "success"
    assert result.metrics.outcome == "max_steps"
    assert result.metrics.success is False


def test_explicit_timeout_classification() -> None:
    """When a scenario has max_duration_seconds and time expires before completion, outcome is timeout."""
    sc = FlightScenarioDefinition(
        name="timeout_test",
        description="Verify explicit timeout classification",
        initial_position=np.array([5.0, 5.0, 5.0]),
        waypoints=[np.array([25.0, 25.0, 10.0])],
        max_steps=200,
        max_duration_seconds=0.5,  # Times out after 10 steps (0.5s at 0.05s/step)
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc)

    assert result.metrics.outcome == "timeout"
    assert result.metrics.success is False
