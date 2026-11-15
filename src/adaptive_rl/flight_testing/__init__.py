"""Autonomous flight validation, test scenarios, and reproducible flight recording."""

from __future__ import annotations

from adaptive_rl.flight_testing.recorder import (
    ExperimentMetadata,
    FlightRecord,
    FlightRecorder,
    PhysicsSimulationFingerprint,
    canonical_json_dumps,
    compute_config_fingerprint,
)
from adaptive_rl.flight_testing.scenarios import (
    FlightScenarioDefinition,
    FlightScenarioMetrics,
    FlightScenarioResult,
    FlightScenarioRunner,
    FlightStepTelemetry,
    get_advanced_navigation_scenarios,
    get_all_flight_scenarios,
    get_scenario,
    get_standard_flight_scenarios,
)

__all__ = [
    "ExperimentMetadata",
    "FlightRecord",
    "FlightRecorder",
    "FlightScenarioDefinition",
    "FlightScenarioMetrics",
    "FlightScenarioResult",
    "FlightScenarioRunner",
    "FlightStepTelemetry",
    "PhysicsSimulationFingerprint",
    "canonical_json_dumps",
    "compute_config_fingerprint",
    "get_advanced_navigation_scenarios",
    "get_all_flight_scenarios",
    "get_scenario",
    "get_standard_flight_scenarios",
]
