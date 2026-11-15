"""Unit tests for FlightBenchmarkSuite and PPO/SAC matrix evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from adaptive_rl.benchmarking.flight_benchmark import (
    SUPPORTED_COMBINATIONS,
    BenchmarkCombinationResult,
    BenchmarkSuiteReport,
    FlightBenchmarkSuite,
)


def test_supported_combinations_matrix() -> None:
    """Verify that all 4 required combinations are represented in the matrix."""
    combos = [list(c) for c in SUPPORTED_COMBINATIONS]
    assert ["ppo", "drone"] in combos  # PPO + 3-DOF
    assert ["ppo", "drone-6dof"] in combos  # PPO + 6-DOF
    assert ["sac", "drone"] in combos  # SAC + 3-DOF
    assert ["sac", "drone-6dof"] in combos  # SAC + 6-DOF


def test_quick_benchmark_single_combination(tmp_path: Path) -> None:
    """Run a quick validation profile on a single combination and verify computed metrics."""
    suite = FlightBenchmarkSuite(output_dir=tmp_path)
    res = suite.evaluate_combination(
        algorithm_name="ppo",
        env_name="drone",
        profile="quick",
        deterministic=True,
        seeds=[42, 43],
    )

    assert isinstance(res, BenchmarkCombinationResult)
    assert res.algorithm == "PPO"
    assert res.environment == "drone"
    assert res.mode == "3-DOF"
    assert res.total_episodes == 2
    assert 0.0 <= res.success_rate <= 1.0
    assert 0.0 <= res.collision_rate <= 1.0
    assert 0.0 <= res.timeout_rate <= 1.0
    assert res.evaluation_runtime_seconds > 0.0
    assert len(res.config_fingerprint) == 64

    # Sample size N=2 supports Student's t confidence interval
    assert "reward" in res.confidence_intervals
    assert "duration_seconds" in res.confidence_intervals


def test_incompatible_checkpoint_rejection(tmp_path: Path) -> None:
    """Evaluating a checkpoint trained for 3-DOF on 6-DOF raises a clean ValueError."""
    # Create a mock 3-DOF checkpoint with observation_space shape (29,)
    mock_zip = tmp_path / "mock_3dof_policy.zip"
    import zipfile

    data_payload = {
        "policy_class": {"__module__": "stable_baselines3.ppo.policies"},
        "observation_space": {"_shape": [29]},
        "action_space": {"_shape": [3]},
    }
    with zipfile.ZipFile(mock_zip, "w") as z:
        z.writestr("data", json.dumps(data_payload))

    suite = FlightBenchmarkSuite(output_dir=tmp_path)
    with pytest.raises(ValueError, match="was trained for 3-DOF, but environment is 6-DOF"):
        suite.evaluate_combination(
            algorithm_name="ppo",
            env_name="drone-6dof",
            checkpoint_path=mock_zip,
        )


def test_benchmark_suite_report_export(tmp_path: Path) -> None:
    """Run quick benchmark suite across 2 combinations and verify JSON & CSV report exports."""
    suite = FlightBenchmarkSuite(output_dir=tmp_path)
    report = suite.run_suite(
        combinations=[("ppo", "drone"), ("sac", "drone-6dof")],
        profile="quick",
        deterministic=True,
    )

    assert isinstance(report, BenchmarkSuiteReport)
    assert len(report.combinations_evaluated) == 2

    # Export JSON
    json_path = tmp_path / "benchmark_report.json"
    report.export_json(json_path)
    assert json_path.is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["profile"] == "quick"
    assert len(data["combinations_evaluated"]) == 2

    # Export CSV
    csv_path = tmp_path / "benchmark_report.csv"
    report.export_csv(csv_path)
    assert csv_path.is_file()

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["algorithm"] == "PPO"
    assert rows[1]["algorithm"] == "SAC"
