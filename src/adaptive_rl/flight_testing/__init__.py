"""Autonomous flight validation, test scenarios, and reproducible flight recording."""

from __future__ import annotations

from adaptive_rl.flight_testing.recorder import (
    ExperimentConfigurationFingerprint,
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
    extract_callable_provenance,
    get_advanced_navigation_scenarios,
    get_all_flight_scenarios,
    get_scenario,
    get_standard_flight_scenarios,
)

__all__ = [
    "ExperimentConfigurationFingerprint",
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
    "extract_callable_provenance",
    "get_advanced_navigation_scenarios",
    "get_all_flight_scenarios",
    "get_scenario",
    "get_standard_flight_scenarios",
]
