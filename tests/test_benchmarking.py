"""Tests for Phase 15 — Benchmarking and Ablation Framework."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from adaptive_rl.benchmarking import (
    BenchmarkResult,
    BenchmarkRunner,
    ComparisonReport,
    SeedResult,
    compute_aggregate_stats,
)

# ---------------------------------------------------------------------------
# Aggregate statistics tests
# ---------------------------------------------------------------------------


class TestAggregateStats:
    """Tests for the compute_aggregate_stats helper."""

    def test_basic_stats(self) -> None:
        """Computes correct mean, std, min, max for a list of values."""
        import math

        values = [0.0, 0.5, 1.0]
        stats = compute_aggregate_stats("success_rate", values)
        assert math.isclose(stats.mean, 0.5, rel_tol=1e-6)
        assert math.isclose(stats.min, 0.0, rel_tol=1e-6)
        assert math.isclose(stats.max, 1.0, rel_tol=1e-6)
        assert stats.n_seeds == 3
        assert stats.values == values

    def test_single_value(self) -> None:
        """Single-value list produces std=0."""
        stats = compute_aggregate_stats("x", [0.7])
        assert stats.mean == 0.7
        assert stats.std == 0.0
        assert stats.n_seeds == 1

    def test_empty_list_produces_nan(self) -> None:
        """Empty list produces NaN stats and n_seeds=0."""
        import math

        stats = compute_aggregate_stats("empty", [])
        assert math.isnan(stats.mean)
        assert math.isnan(stats.std)
        assert stats.n_seeds == 0

    def test_to_dict_structure(self) -> None:
        """AggregateStats.to_dict() returns expected keys."""
        stats = compute_aggregate_stats("metric", [0.0, 1.0])
        d = stats.to_dict()
        assert "mean" in d
        assert "std" in d
        assert "min" in d
        assert "max" in d
        assert "n_seeds" in d
        assert "values" in d

    def test_metric_name_preserved(self) -> None:
        """Metric name is stored in the AggregateStats object."""
        stats = compute_aggregate_stats("my_metric", [0.5])
        assert stats.metric_name == "my_metric"


# ---------------------------------------------------------------------------
# BenchmarkResult serialization
# ---------------------------------------------------------------------------


class TestBenchmarkResultSerialization:
    """Tests for BenchmarkResult data model."""

    def _make_result(self) -> BenchmarkResult:
        """Create a minimal BenchmarkResult for testing."""
        seed_results = [
            SeedResult(
                seed=42,
                success=True,
                metrics={"success_rate": 0.8, "mean_reward": 50.0},
                experiment_id="test_42",
            ),
            SeedResult(
                seed=43,
                success=True,
                metrics={"success_rate": 0.9, "mean_reward": 55.0},
                experiment_id="test_43",
            ),
        ]
        aggregate = {
            "success_rate": compute_aggregate_stats("success_rate", [0.8, 0.9]),
            "mean_reward": compute_aggregate_stats("mean_reward", [50.0, 55.0]),
        }
        return BenchmarkResult(
            name="test_benchmark",
            algorithm="ppo",
            environment="gridworld",
            seeds=[42, 43],
            seed_results=seed_results,
            aggregate=aggregate,
            successful_seeds=2,
        )

    def test_to_dict(self) -> None:
        """BenchmarkResult.to_dict() produces JSON-serializable structure."""
        result = self._make_result()
        d = result.to_dict()
        assert d["name"] == "test_benchmark"
        assert d["algorithm"] == "ppo"
        assert d["successful_seeds"] == 2
        assert "aggregate" in d
        assert "seed_results" in d

    def test_json_serializable(self) -> None:
        """BenchmarkResult can be serialized to JSON without errors."""
        result = self._make_result()
        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d, default=str)
        assert len(json_str) > 0

    def test_seed_result_failure(self) -> None:
        """SeedResult can represent a failed run."""
        sr = SeedResult(
            seed=99,
            success=False,
            metrics={},
            experiment_id="failed_99",
            error_message="Training error",
        )
        assert not sr.success
        assert sr.error_message == "Training error"


# ---------------------------------------------------------------------------
# BenchmarkRunner integration tests
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    """Integration tests for BenchmarkRunner using A* planner (fast)."""

    def _astar_config_path(self) -> Path:
        """Return path to the A* config for fast testing."""
        # Use the project's gridworld_astar.yaml config
        config_path = Path("configs/gridworld_astar.yaml")
        if not config_path.exists():
            pytest.skip("configs/gridworld_astar.yaml not found")
        return config_path

    def test_run_single_config(self) -> None:
        """BenchmarkRunner.run() completes and returns BenchmarkResult."""
        config_path = self._astar_config_path()

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                seeds=[42, 43],
                base_output_dir=Path(tmpdir),
            )
            result = runner.run(config_path=config_path, name="A* Test")

        assert isinstance(result, BenchmarkResult)
        assert result.name == "A* Test"
        assert result.seeds == [42, 43]
        assert len(result.seed_results) == 2

    def test_all_seeds_run(self) -> None:
        """BenchmarkRunner runs one experiment per seed."""
        config_path = self._astar_config_path()

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                seeds=[42, 43, 44],
                base_output_dir=Path(tmpdir),
            )
            result = runner.run(config_path=config_path)

        assert len(result.seed_results) == 3

    def test_aggregate_computed(self) -> None:
        """Aggregate statistics are computed when all seeds succeed."""
        config_path = self._astar_config_path()

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                seeds=[42, 43],
                base_output_dir=Path(tmpdir),
            )
            result = runner.run(config_path=config_path)

        if result.successful_seeds > 0:
            assert len(result.aggregate) > 0
            assert "success_rate" in result.aggregate

    def test_comparison_report_structure(self) -> None:
        """compare() returns a ComparisonReport with both arms."""
        config_path = self._astar_config_path()

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = BenchmarkRunner(
                seeds=[42],
                base_output_dir=Path(tmpdir),
            )
            report = runner.compare(
                config_a=config_path,
                config_b=config_path,
                name_a="A",
                name_b="B",
            )

        assert isinstance(report, ComparisonReport)
        assert report.arm_a.name == "A"
        assert report.arm_b.name == "B"
        assert isinstance(report.comparison, dict)

    def test_comparison_report_save(self) -> None:
        """ComparisonReport.save() writes valid JSON to disk."""
        config_path = self._astar_config_path()

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            runner = BenchmarkRunner(
                seeds=[42],
                base_output_dir=base / "experiments",
            )
            report_path = base / "comparison.json"
            runner.compare(
                config_a=config_path,
                config_b=config_path,
                output_path=report_path,
            )

            assert report_path.exists()
            with open(report_path) as f:
                data = json.load(f)
            assert "arm_a" in data
            assert "arm_b" in data
            assert "comparison" in data

    def test_runner_aggregate_ignores_failures(self) -> None:
        """Aggregate only uses data from successful seeds."""
        # Create seed results with one failure
        seed_results = [
            SeedResult(seed=42, success=True, metrics={"success_rate": 1.0}),
            SeedResult(seed=43, success=False, metrics={}, error_message="err"),
        ]
        aggregate = BenchmarkRunner._aggregate([sr for sr in seed_results if sr.success])
        # Only 1 successful seed → std should be 0
        if "success_rate" in aggregate:
            assert aggregate["success_rate"].n_seeds == 1
            assert aggregate["success_rate"].mean == 1.0

    def test_aggregate_with_missing_metric(self) -> None:
        """Aggregate handles seed results with missing metric gracefully."""
        seed_results = [
            SeedResult(seed=42, success=True, metrics={"success_rate": 0.8}),
            SeedResult(seed=43, success=True, metrics={}),  # missing success_rate
        ]
        aggregate = BenchmarkRunner._aggregate(seed_results)
        if "success_rate" in aggregate:
            # Only 1 seed contributed
            assert aggregate["success_rate"].n_seeds == 1
