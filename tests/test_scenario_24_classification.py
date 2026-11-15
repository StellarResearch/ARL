"""Tests verifying distinct classification for Scenario 24: Stress Survival vs Full Completion."""

from __future__ import annotations

from adaptive_rl.flight_testing.scenarios import FlightScenarioRunner, get_scenario


def test_scenario_24_stress_classification() -> None:
    """Evaluate repeated_dynamic_replanning_stress for sustained collision-free replanning."""
    runner = FlightScenarioRunner()
    sc = get_scenario("repeated_dynamic_replanning_stress")
    res = runner.run_scenario(sc)

    # Stress test must maintain zero collisions
    assert res.metrics.collision_count == 0, (
        f"Collisions during stress test: {res.metrics.collision_count}"
    )

    # Must actively replan to evade dynamic obstacles
    replan_count = getattr(res.metrics, "replan_count", 0)
    assert replan_count >= 1, f"Expected active dynamic replanning, got {replan_count}"

    # Outcome is classified as sustained stress survival (collision-free success)
    assert res.metrics.outcome in ("success", "max_steps")
    assert res.metrics.success is True, (
        f"Expected success classification for stress test: {res.metrics}"
    )


def test_scenario_24_completed_classification() -> None:
    """Evaluate repeated_dynamic_replanning_completed for verified arrival at destination."""
    runner = FlightScenarioRunner()
    sc = get_scenario("repeated_dynamic_replanning_completed")
    res = runner.run_scenario(sc)

    assert res.metrics.collision_count == 0, (
        f"Collisions during completion test: {res.metrics.collision_count}"
    )
    assert res.metrics.outcome == "success", f"Expected success outcome, got {res.metrics.outcome}"
    assert res.metrics.success is True
    assert res.metrics.final_position_error <= sc.target_radius, (
        f"Final position error {res.metrics.final_position_error} exceeded target radius {sc.target_radius}"
    )


def test_scenario_24_alias_resolution() -> None:
    """Verify legacy name repeated_dynamic_replanning resolves correctly."""
    sc_alias = get_scenario("repeated_dynamic_replanning")
    sc_direct = get_scenario("repeated_dynamic_replanning_stress")
    assert sc_alias.name == sc_direct.name
