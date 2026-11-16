"""Unit tests for FlightRecorder, run fingerprinting, and security validation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from adaptive_rl.algorithms.registry import inspect_model_checkpoint
from adaptive_rl.flight_testing.recorder import (
    FlightRecorder,
    canonical_json_dumps,
    compute_config_fingerprint,
)
from adaptive_rl.flight_testing.scenarios import FlightScenarioMetrics


def test_canonical_json_dumps_determinism() -> None:
    """Verify key sorting and deterministic float formatting in canonical serialization."""
    d1 = {"b": 2.12345678, "a": 1, "nested": {"z": 10.0, "m": 5.0}}
    d2 = {"nested": {"m": 5.0, "z": 10.0}, "a": 1, "b": 2.12345678}

    s1 = canonical_json_dumps(d1)
    s2 = canonical_json_dumps(d2)
    assert s1 == s2
    # Exact finite floating-point representation is preserved without lossy rounding
    assert "2.12345678" in s1

    h1 = compute_config_fingerprint(d1)
    h2 = compute_config_fingerprint(d2)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256


def test_flight_record_creation_and_fingerprint_verification(tmp_path: Path) -> None:
    """Verify FlightRecord creation, provenance collection, and SHA-256 verification."""
    recorder = FlightRecorder(output_dir=tmp_path)
    physics_params = {
        "mass": 1.25,
        "gravity": 9.81,
        "arm_length": 0.22,
        "arm_span": 0.44,
        "rotor_offset_d": 0.15556,
        "rotor_radius": 0.099,
        "max_thrust_per_motor": 6.0,
        "torque_to_thrust_ratio": 0.015,
        "linear_damping": 0.12,
        "angular_damping": 0.025,
        "inertia": [0.012, 0.012, 0.024],
        "dt": 0.05,
        "substeps": 5,
        "substep_dt": 0.01,
        "body_frame_lidar": False,
        "num_lidar_rays": 16,
        "lidar_range": 20.0,
    }

    dummy_metric = FlightScenarioMetrics(
        scenario_name="test_sc",
        success=True,
        outcome="success",
        duration_steps=100,
        duration_seconds=5.0,
        final_position_error=0.1,
        mean_position_error=0.2,
        max_position_error=0.5,
        rms_position_error=0.25,
        mean_velocity_error=0.05,
        max_velocity_error=0.2,
        rms_velocity_error=0.08,
        mean_attitude_error_deg=1.2,
        max_attitude_error_deg=3.5,
        mean_angular_speed=0.4,
        max_angular_speed=1.1,
        initial_goal_distance=10.0,
        final_goal_distance=0.1,
        min_obstacle_clearance=1.5,
        collision_count=0,
        control_effort=12.5,
        execution_wall_time_ms=45.0,
    )

    record = recorder.create_record(
        experiment_id="test_exp_001",
        physics_params=physics_params,
        scenario_results=[dummy_metric],
        algorithm_name="test_algo",
        random_seed=42,
    )

    assert record.experiment_id == "test_exp_001"
    assert record.verify_fingerprint() is True
    assert record.provenance.git_commit is not None
    assert record.provenance.os_name is not None
    assert record.aggregate_summary["success_rate"] == 1.0


def test_tampered_physics_parameter_invalidates_fingerprint(tmp_path: Path) -> None:
    """Modifying any behavior-affecting physics parameter invalidates stored fingerprint."""
    recorder = FlightRecorder(output_dir=tmp_path)
    physics_params = {
        "mass": 1.0,
        "gravity": 9.81,
        "arm_length": 0.2,
        "max_thrust_per_motor": 5.0,
        "dt": 0.05,
        "substeps": 5,
    }

    record = recorder.create_record(
        experiment_id="tamper_test",
        physics_params=physics_params,
        scenario_results=[],
    )
    assert record.verify_fingerprint() is True

    json_path = recorder.save_record_json(record)
    assert json_path.is_file()

    # Tamper with the saved JSON on disk: change mass from 1.0 to 2.5
    raw_data = json.loads(json_path.read_text(encoding="utf-8"))
    raw_data["physics"]["mass"] = 2.5
    json_path.write_text(json.dumps(raw_data), encoding="utf-8")

    # load_record_json must reject the tampered record
    with pytest.raises(ValueError, match="Config fingerprint mismatch"):
        FlightRecorder.load_record_json(json_path)


def test_flight_record_export_and_reload(tmp_path: Path) -> None:
    """Save record to JSON and CSV, then reload and verify roundtrip integrity."""
    recorder = FlightRecorder(output_dir=tmp_path)
    physics_params = {"mass": 1.1, "dt": 0.04, "substeps": 4}

    dummy_m = FlightScenarioMetrics(
        scenario_name="sc_roundtrip",
        success=False,
        outcome="timeout",
        duration_steps=200,
        duration_seconds=8.0,
        final_position_error=1.5,
        mean_position_error=2.0,
        max_position_error=3.0,
        rms_position_error=2.2,
        mean_velocity_error=0.3,
        max_velocity_error=1.0,
        rms_velocity_error=0.4,
        mean_attitude_error_deg=2.5,
        max_attitude_error_deg=6.0,
        mean_angular_speed=0.6,
        max_angular_speed=1.8,
        initial_goal_distance=15.0,
        final_goal_distance=1.5,
        min_obstacle_clearance=float("inf"),
        collision_count=0,
        control_effort=35.0,
        execution_wall_time_ms=80.0,
    )

    record = recorder.create_record(
        experiment_id="exp_roundtrip",
        physics_params=physics_params,
        scenario_results=[dummy_m],
    )

    json_p = recorder.save_record_json(record)
    csv_p = recorder.save_summary_csv(record)

    assert json_p.is_file()
    assert csv_p.is_file()

    loaded = FlightRecorder.load_record_json(json_p)
    assert loaded.experiment_id == "exp_roundtrip"
    assert loaded.config_fingerprint == record.config_fingerprint
    assert len(loaded.scenario_metrics) == 1
    assert loaded.scenario_metrics[0].scenario_name == "sc_roundtrip"
    assert loaded.scenario_metrics[0].outcome == "timeout"


def test_corrupted_or_missing_record_handling(tmp_path: Path) -> None:
    """Missing files, non-JSON content, or missing required schema keys raise appropriate errors."""
    with pytest.raises(FileNotFoundError):
        FlightRecorder.load_record_json(tmp_path / "non_existent.json")

    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{ incomplete json content ...", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt or non-JSON"):
        FlightRecorder.load_record_json(corrupt_file)

    missing_keys_file = tmp_path / "missing_keys.json"
    missing_keys_file.write_text(json.dumps({"experiment_id": "test"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required schema key"):
        FlightRecorder.load_record_json(missing_keys_file)


def test_checkpoint_zip_slip_security_rejection(tmp_path: Path) -> None:
    """Security audit: inspect_model_checkpoint rejects archives containing path traversal entries."""
    malicious_zip = tmp_path / "malicious_checkpoint.zip"
    with zipfile.ZipFile(malicious_zip, "w") as z:
        # Malicious entry attempting path traversal out of extraction root
        z.writestr("../../etc/passwd", "malicious_payload")
        z.writestr("data", json.dumps({"policy_class": {}}))

    info = inspect_model_checkpoint(malicious_zip)
    assert info["valid"] is False
    assert info["compatibility_status"] == "incompatible"
    assert "Security rejection" in info.get("error", "")
