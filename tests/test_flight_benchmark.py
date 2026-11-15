"""Unit tests for FlightBenchmarkSuite, statistical integrity, and PPO/SAC verification."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from adaptive_rl.benchmarking.flight_benchmark import (
    MINIMUM_SAMPLES_FOR_CI,
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


def test_missing_checkpoint_rejects_silent_random_evaluation(tmp_path: Path) -> None:
    """PPO and SAC evaluation must reject missing checkpoints rather than silently substituting random actions."""
    suite = FlightBenchmarkSuite(output_dir=tmp_path)

    # Calling PPO without checkpoint or train_if_missing must raise ValueError
    with pytest.raises(
        ValueError, match="requires a valid model checkpoint or train_if_missing=True"
    ):
        suite.evaluate_combination(
            algorithm_name="ppo",
            env_name="drone",
            checkpoint_path=None,
            train_if_missing=False,
        )

    # Calling SAC without checkpoint or train_if_missing must raise ValueError
    with pytest.raises(
        ValueError, match="requires a valid model checkpoint or train_if_missing=True"
    ):
        suite.evaluate_combination(
            algorithm_name="sac",
            env_name="drone-6dof",
            checkpoint_path=None,
            train_if_missing=False,
        )


def test_explicit_random_policy_identity(tmp_path: Path) -> None:
    """Random baseline must explicitly report RANDOM identity and policy_source='random'."""
    suite = FlightBenchmarkSuite(output_dir=tmp_path)
    res = suite.evaluate_combination(
        algorithm_name="random",
        env_name="drone",
        profile="quick",
        deterministic=True,
        seeds=[42, 43],
    )

    assert isinstance(res, BenchmarkCombinationResult)
    assert res.algorithm == "RANDOM"
    assert res.policy_source == "random"
    assert res.checkpoint_hash is None
    assert res.total_episodes == 2
    assert res.is_smoke_test is True
    # For N=2 < 10, confidence intervals must be omitted to prevent invalid inferential claims
    assert res.confidence_intervals == {}
    assert "below minimum threshold" in (res.statistical_note or "")


def test_sample_threshold_and_confidence_interval_enforcement(tmp_path: Path) -> None:
    """Statistical sample threshold N >= 10 must be enforced for confidence intervals."""
    suite = FlightBenchmarkSuite(output_dir=tmp_path)

    # 1. N < 10 (e.g. 5 seeds): CIs must NOT be generated
    res_under = suite.evaluate_combination(
        algorithm_name="random",
        env_name="drone",
        profile="quick",
        seeds=[10, 20, 30, 40, 50],
    )
    assert res_under.total_episodes == 5
    assert res_under.confidence_intervals == {}
    assert f"N >= {MINIMUM_SAMPLES_FOR_CI}" in (res_under.statistical_note or "")

    # 2. N >= 10 (10 distinct seeds): CIs must be computed across independent seeds
    distinct_seeds = list(range(200, 210))
    res_full = suite.evaluate_combination(
        algorithm_name="random",
        env_name="drone",
        profile="full",
        seeds=distinct_seeds,
    )
    assert res_full.total_episodes == 10
    assert "reward" in res_full.confidence_intervals
    assert "duration_seconds" in res_full.confidence_intervals
    assert res_full.confidence_intervals["reward"]["mean"] is not None
    assert res_full.confidence_intervals["reward"]["ci95_lower"] is not None
    assert res_full.confidence_intervals["reward"]["ci95_upper"] is not None
    assert "satisfies research criteria" in (res_full.statistical_note or "")


def test_duplicate_seeds_and_empty_seeds_rejection(tmp_path: Path) -> None:
    """Repeated or duplicate seeds and empty seeds lists must be rejected."""
    suite = FlightBenchmarkSuite(output_dir=tmp_path)

    with pytest.raises(ValueError, match="must be distinct"):
        suite.evaluate_combination(
            algorithm_name="random",
            env_name="drone",
            seeds=[42, 42, 43],
        )

    with pytest.raises(ValueError, match="cannot be empty"):
        suite.evaluate_combination(
            algorithm_name="random",
            env_name="drone",
            seeds=[],
        )


def test_incompatible_checkpoint_rejection(tmp_path: Path) -> None:
    """Evaluating a checkpoint trained for 3-DOF on 6-DOF raises a clean ValueError."""
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
    """Run random baseline benchmark suite and verify JSON & CSV report exports with policy provenance."""
    suite = FlightBenchmarkSuite(output_dir=tmp_path)
    report = suite.run_suite(
        combinations=[("random", "drone"), ("random", "drone-6dof")],
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
    assert data["combinations_evaluated"][0]["policy_source"] == "random"

    # Export CSV
    csv_path = tmp_path / "benchmark_report.csv"
    report.export_csv(csv_path)
    assert csv_path.is_file()

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["algorithm"] == "RANDOM"
    assert rows[0]["policy_source"] == "random"
    assert rows[0]["is_smoke_test"] == "True"
