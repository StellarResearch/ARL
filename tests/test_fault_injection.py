"""Unit tests for actuator fault injection, timing, thrust reduction, and classification (Issue #277, Phase III)."""

from __future__ import annotations

import math

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
    """Under a survivable 10% partial thrust loss, vehicle maintains level hover and classifies as recovered_degraded."""

    def moderate_fault(step: int, thrusts: np.ndarray) -> np.ndarray:
        if step >= 40:
            degraded = thrusts.copy()
            degraded[0] *= 0.90  # 10% thrust reduction on Motor 1
            return degraded
        return thrusts

    sc = FlightScenarioDefinition(
        name="moderate_fault_scenario",
        description="Moderate 10% thrust loss on Motor 1",
        initial_position=np.array([15.0, 15.0, 7.0]),
        waypoints=[np.array([15.0, 15.0, 7.0])],
        max_steps=120,
        actuator_fault_fn=moderate_fault,
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc)

    assert result.metrics.outcome == "recovered_degraded"
    assert result.metrics.success is True
    assert result.metrics.collision_count == 0
    assert result.metrics.max_attitude_error_deg < 35.0
