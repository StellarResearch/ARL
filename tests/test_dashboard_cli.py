"""Tests for the dashboard and CLI integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.visualization.dashboard import (
    _fmt_float,
    _fmt_pct,
    _fmt_time,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Dashboard formatting helpers
# ---------------------------------------------------------------------------


class TestDashboardFormatters:
    """Tests for Rich dashboard formatting utilities."""

    def test_fmt_pct_normal(self) -> None:
        assert _fmt_pct(0.75) == "75.0%"
        assert _fmt_pct(1.0) == "100.0%"
        assert _fmt_pct(0.0) == "0.0%"

    def test_fmt_pct_none(self) -> None:
        assert _fmt_pct(None) == "N/A"

    def test_fmt_pct_nan(self) -> None:
        assert _fmt_pct(float("nan")) == "N/A"

    def test_fmt_float_normal(self) -> None:
        assert _fmt_float(3.14159, 2) == "3.14"
        assert _fmt_float(0.0, 1) == "0.0"

    def test_fmt_float_none(self) -> None:
        assert _fmt_float(None) == "N/A"

    def test_fmt_time_microseconds(self) -> None:
        result = _fmt_time(0.0001)
        assert "µs" in result or "ms" in result

    def test_fmt_time_milliseconds(self) -> None:
        result = _fmt_time(0.005)
        assert "ms" in result

    def test_fmt_time_seconds(self) -> None:
        result = _fmt_time(1.5)
        assert "s" in result

    def test_fmt_time_none(self) -> None:
        assert _fmt_time(None) == "N/A"

    def test_fmt_non_numeric_safety(self) -> None:
        assert _fmt_pct("bad") == "N/A"  # type: ignore
        assert _fmt_float("bad") == "N/A"  # type: ignore
        assert _fmt_time("bad") == "N/A"  # type: ignore

    def test_status_color_safety(self) -> None:
        from adaptive_rl.visualization.dashboard import _status_color

        assert "completed" in _status_color("completed")
        assert "unknown" in _status_color(None)
        assert "unknown" in _status_color("")


# ---------------------------------------------------------------------------
# CLI: algorithm commands
# ---------------------------------------------------------------------------


class TestAlgorithmCLI:
    """Tests for the adaptive-rl algorithm sub-commands."""

    def test_algorithm_list_success(self) -> None:
        """algorithm list returns 0 exit code and shows PPO, SAC, A*."""
        result = runner.invoke(app, ["algorithm", "list"])
        assert result.exit_code == 0
        output = result.output
        assert "ppo" in output.lower()
        assert "sac" in output.lower()
        assert "astar" in output.lower()

    def test_algorithm_inspect_ppo(self) -> None:
        """algorithm inspect ppo shows PPO details."""
        result = runner.invoke(app, ["algorithm", "inspect", "ppo"])
        assert result.exit_code == 0
        assert "PPO" in result.output or "ppo" in result.output.lower()
        assert "learning_rate" in result.output

    def test_algorithm_inspect_sac(self) -> None:
        """algorithm inspect sac shows SAC details."""
        result = runner.invoke(app, ["algorithm", "inspect", "sac"])
        assert result.exit_code == 0
        assert "SAC" in result.output or "sac" in result.output.lower()
        assert "buffer_size" in result.output

    def test_algorithm_inspect_astar(self) -> None:
        """algorithm inspect astar shows A* planner details."""
        result = runner.invoke(app, ["algorithm", "inspect", "astar"])
        assert result.exit_code == 0
        assert "astar" in result.output.lower() or "A*" in result.output
        assert "Planner" in result.output

    def test_algorithm_inspect_unknown_exits_1(self) -> None:
        """algorithm inspect <unknown> returns exit code 1."""
        result = runner.invoke(app, ["algorithm", "inspect", "nonexistent_algo"])
        assert result.exit_code == 1

    def test_algorithm_list_shows_trainable_info(self) -> None:
        """algorithm list output distinguishes trainable from planners."""
        result = runner.invoke(app, ["algorithm", "list"])
        assert result.exit_code == 0
        # Should show both RL Policy and Planner kinds
        assert "RL Policy" in result.output or "planner" in result.output.lower()


# ---------------------------------------------------------------------------
# CLI: experiment commands
# ---------------------------------------------------------------------------


class TestExperimentCLI:
    """Tests for adaptive-rl experiment sub-commands."""

    def test_experiment_list_empty(self) -> None:
        """experiment list with no experiments shows appropriate message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["experiment", "list", "--output-dir", tmpdir])
            assert result.exit_code == 0
            assert "No experiments" in result.output or "experiment" in result.output.lower()

    def test_experiment_run_astar(self) -> None:
        """experiment run with A* config completes successfully."""
        config_path = Path("configs/gridworld_astar.yaml")
        if not config_path.exists():
            pytest.skip("configs/gridworld_astar.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "experiment",
                    "run",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    tmpdir,
                ],
            )
            assert result.exit_code == 0, f"Experiment run failed:\n{result.output}"
            assert "Completed" in result.output or "completed" in result.output.lower()

    def test_experiment_run_seed_zero(self) -> None:
        """experiment run with --seed 0 preserves seed=0 and displays Seed: 0."""
        config_path = Path("configs/gridworld_astar.yaml")
        if not config_path.exists():
            pytest.skip("configs/gridworld_astar.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "experiment",
                    "run",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    tmpdir,
                    "--seed",
                    "0",
                ],
            )
            assert result.exit_code == 0, f"Experiment run failed:\n{result.output}"
            assert "Seed: 0" in result.output
            manifest_files = list(Path(tmpdir).rglob("manifest.json"))
            assert len(manifest_files) >= 1
            with open(manifest_files[0], encoding="utf-8") as f:
                manifest_data = json.load(f)
            assert manifest_data["seed"] == 0

    def test_experiment_run_invalid_config(self) -> None:
        """experiment run with missing config returns exit code 1."""
        result = runner.invoke(
            app,
            ["experiment", "run", "--config", "/nonexistent/config.yaml"],
        )
        assert result.exit_code == 1

    def test_experiment_list_after_run(self) -> None:
        """experiment list shows experiment after run."""
        config_path = Path("configs/gridworld_astar.yaml")
        if not config_path.exists():
            pytest.skip("configs/gridworld_astar.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run experiment
            runner.invoke(
                app,
                [
                    "experiment",
                    "run",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    tmpdir,
                ],
            )
            # List experiments
            result = runner.invoke(app, ["experiment", "list", "--output-dir", tmpdir])
            assert result.exit_code == 0

    def test_experiment_inspect_missing_id(self) -> None:
        """experiment inspect with unknown ID shows error message gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "experiment",
                    "inspect",
                    "unknown_exp_id",
                    "--output-dir",
                    tmpdir,
                ],
            )
            # Should exit 0 and print informative message (not crash)
            assert result.exit_code == 0
            assert "not found" in result.output.lower()
            assert "unknown_exp_id" in result.output.lower()


# ---------------------------------------------------------------------------
# CLI: dashboard command
# ---------------------------------------------------------------------------


class TestDashboardCLI:
    """Tests for the adaptive-rl dashboard command."""

    def test_dashboard_empty_results(self) -> None:
        """dashboard with no experiments shows overview gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["dashboard", "--output-dir", tmpdir])
            assert result.exit_code == 0
            # Should show "No experiments found" or similar
            assert "No experiments" in result.output or "dashboard" in result.output.lower()

    def test_dashboard_after_run(self) -> None:
        """dashboard shows experiments after a run completes."""
        config_path = Path("configs/gridworld_astar.yaml")
        if not config_path.exists():
            pytest.skip("configs/gridworld_astar.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run an experiment first
            runner.invoke(
                app,
                [
                    "experiment",
                    "run",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    tmpdir,
                ],
            )
            # Dashboard should show it
            result = runner.invoke(app, ["dashboard", "--output-dir", tmpdir])
            assert result.exit_code == 0

    def test_dashboard_with_corrupt_or_incomplete_manifest(self) -> None:
        """Dashboard renders cleanly even with missing or None fields in manifest."""
        import json

        from adaptive_rl.experiments.manager import ExperimentManager
        from adaptive_rl.visualization.dashboard import render_metrics, render_overview

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "corrupt_exp"
            exp_dir.mkdir()
            # Incomplete manifest with None fields
            manifest_path = exp_dir / "manifest.json"
            with open(manifest_path, "w") as f:
                json.dump(
                    {
                        "experiment_id": "corrupt_exp",
                        "algorithm": None,
                        "environment": None,
                        "seed": None,
                        "evaluation_status": None,
                        "python_version": None,
                    },
                    f,
                )

            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            # Must not crash
            render_overview(manager)
            render_metrics("corrupt_exp", manager)

    def test_sparkline_with_nans(self) -> None:
        """_render_sparkline handles NaNs and non-floats without raising."""
        from adaptive_rl.visualization.dashboard import _render_sparkline

        # Should not raise exception
        _render_sparkline([float("nan"), 1.0, 2.0, float("nan")])
        _render_sparkline([float("nan")])


# ---------------------------------------------------------------------------
# CLI: benchmark command
# ---------------------------------------------------------------------------


class TestBenchmarkCLI:
    """Tests for the adaptive-rl benchmark command."""

    def test_benchmark_astar_single_seed(self) -> None:
        """benchmark with A* config and single seed completes."""
        config_path = Path("configs/gridworld_astar.yaml")
        if not config_path.exists():
            pytest.skip("configs/gridworld_astar.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "benchmark",
                    "--config",
                    str(config_path),
                    "--seeds",
                    "42",
                    "--output-dir",
                    tmpdir,
                ],
            )
            assert result.exit_code == 0

    def test_benchmark_missing_config(self) -> None:
        """benchmark with missing config returns exit code 1."""
        result = runner.invoke(
            app,
            ["benchmark", "--config", "/nonexistent/config.yaml", "--seeds", "42"],
        )
        # Should fail gracefully with exit code 1
        assert result.exit_code == 1

    def test_benchmark_fatal_failure_exits_1(self) -> None:
        """benchmark exits with code 1 when all seeds fail."""
        from unittest.mock import patch

        from adaptive_rl.benchmarking import BenchmarkResult

        fake_res = BenchmarkResult(
            name="fatal_bench",
            algorithm="astar",
            environment="gridworld",
            seeds=[42],
            successful_seeds=0,
            requested_seeds=1,
            failed_seeds=1,
            failed_seed_ids=[42],
            failure_reasons={42: "Fatal error"},
            complete=False,
        )

        config_path = Path("configs/gridworld_astar.yaml")
        if not config_path.exists():
            pytest.skip("configs/gridworld_astar.yaml not found")

        with patch("adaptive_rl.benchmarking.BenchmarkRunner.run", return_value=fake_res):
            result = runner.invoke(
                app,
                ["benchmark", "--config", str(config_path), "--seeds", "42"],
            )
            assert result.exit_code == 1
            assert "Fatal Benchmark Failure" in result.output

    def test_benchmark_strict_mode_exits_1_on_partial(self) -> None:
        """benchmark with --strict exits with code 1 when some seeds fail."""
        from unittest.mock import patch

        from adaptive_rl.benchmarking import BenchmarkResult

        fake_res = BenchmarkResult(
            name="partial_bench",
            algorithm="astar",
            environment="gridworld",
            seeds=[42, 43],
            successful_seeds=1,
            requested_seeds=2,
            failed_seeds=1,
            failed_seed_ids=[43],
            failure_reasons={43: "Timeout"},
            complete=False,
        )

        config_path = Path("configs/gridworld_astar.yaml")
        if not config_path.exists():
            pytest.skip("configs/gridworld_astar.yaml not found")

        with patch("adaptive_rl.benchmarking.BenchmarkRunner.run", return_value=fake_res):
            # Without strict, exits 0
            res_normal = runner.invoke(
                app,
                ["benchmark", "--config", str(config_path), "--seeds", "42,43"],
            )
            assert res_normal.exit_code == 0
            assert "Partial Completion" in res_normal.output

            # With strict, exits 1
            res_strict = runner.invoke(
                app,
                ["benchmark", "--config", str(config_path), "--seeds", "42,43", "--strict"],
            )
            assert res_strict.exit_code == 1
            assert "Strict mode enabled" in res_strict.output


# ---------------------------------------------------------------------------
# CLI: info and version commands
# ---------------------------------------------------------------------------


class TestInfoAndVersion:
    """Tests for current info and version output."""

    def test_version_shows_version(self) -> None:
        """version command shows the package version."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_info_shows_planning_support(self) -> None:
        """info command shows classical planning support."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Planning" in result.output
        assert "A*" in result.output

    def test_info_shows_experiment_support(self) -> None:
        """info command shows experiment provenance support."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Experiments" in result.output
        assert "Provenance" in result.output

    def test_algorithm_help(self) -> None:
        """algorithm --help returns exit code 0."""
        result = runner.invoke(app, ["algorithm", "--help"])
        assert result.exit_code == 0

    def test_experiment_help(self) -> None:
        """experiment --help returns exit code 0."""
        result = runner.invoke(app, ["experiment", "--help"])
        assert result.exit_code == 0

    def test_benchmark_help(self) -> None:
        """benchmark --help returns exit code 0."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0

    def test_dashboard_help(self) -> None:
        """dashboard --help returns exit code 0."""
        result = runner.invoke(app, ["dashboard", "--help"])
        assert result.exit_code == 0
