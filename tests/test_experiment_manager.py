"""Tests for Phase 14 — Experiment Manager."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from adaptive_rl.config import ExperimentConfig
from adaptive_rl.experiments.manager import (
    ExperimentManager,
    ExperimentManifest,
    ExperimentResult,
    _get_git_commit,
    _get_package_version,
    _make_experiment_id,
)

# ---------------------------------------------------------------------------
# Helper configs
# ---------------------------------------------------------------------------


def _make_minimal_config(
    algo: str = "astar", env: str = "gridworld", seed: int = 42
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing."""
    from adaptive_rl.config import (
        AlgorithmConfig,
        EnvironmentConfig,
        EvaluationConfig,
        TrainingConfig,
    )

    is_planner = algo in ("astar", "rrt_star", "rrt")
    algo_cfg = (
        AlgorithmConfig(name=algo)
        if is_planner
        else AlgorithmConfig(name=algo, learning_rate=3e-4, gamma=0.99, batch_size=64)
    )
    training_cfg = (
        None if is_planner else TrainingConfig(total_timesteps=1, checkpoint_freq=0, log_interval=1)
    )

    return ExperimentConfig(
        name=f"{env}_{algo}_test",
        seed=seed,
        algorithm=algo_cfg,
        environment=EnvironmentConfig(name=env, max_steps=50),
        training=training_cfg,
        evaluation=EvaluationConfig(eval_episodes=3, deterministic=True),
    )


# ---------------------------------------------------------------------------
# Experiment ID generation
# ---------------------------------------------------------------------------


class TestExperimentIDGeneration:
    """Tests for experiment ID generation."""

    def test_id_contains_date(self) -> None:
        """Experiment ID starts with a date in YYYY-MM-DD format."""
        config = _make_minimal_config()
        exp_id = _make_experiment_id(config)
        # Should start with 4-digit year
        import re

        assert re.match(r"\d{4}-\d{2}-\d{2}", exp_id)

    def test_id_contains_env_name(self) -> None:
        """Experiment ID includes the environment name."""
        config = _make_minimal_config(env="gridworld")
        exp_id = _make_experiment_id(config)
        assert "gridworld" in exp_id

    def test_id_contains_algo_name(self) -> None:
        """Experiment ID includes the algorithm name."""
        config = _make_minimal_config(algo="ppo")
        exp_id = _make_experiment_id(config)
        assert "ppo" in exp_id

    def test_id_contains_seed(self) -> None:
        """Experiment ID includes the seed value."""
        config = _make_minimal_config(seed=99)
        exp_id = _make_experiment_id(config)
        assert "seed99" in exp_id

    def test_id_filesystem_safe(self) -> None:
        """Experiment ID contains no characters unsafe for directory names."""
        config = _make_minimal_config()
        exp_id = _make_experiment_id(config)
        invalid_chars = set('\\ / : * ? " < > |')
        assert not any(ch in exp_id for ch in invalid_chars)

    def test_different_seeds_different_ids(self) -> None:
        """Different seeds produce different IDs."""
        config1 = _make_minimal_config(seed=42)
        config2 = _make_minimal_config(seed=43)
        assert _make_experiment_id(config1) != _make_experiment_id(config2)


# ---------------------------------------------------------------------------
# Manifest creation
# ---------------------------------------------------------------------------


class TestExperimentManifest:
    """Tests for ExperimentManifest data model."""

    def test_manifest_to_dict(self) -> None:
        """Manifest serializes to a JSON-compatible dictionary."""
        manifest = ExperimentManifest(
            experiment_id="test_123",
            created_at="2026-09-17T00:00:00+00:00",
            algorithm="ppo",
            environment="gridworld",
            seed=42,
            config_path="configs/test.yaml",
            training_timesteps=1000,
            git_commit="abc1234",
            python_version="3.12.0",
            platform_info="Linux x86_64",
            package_versions={"adaptive-rl": "0.1.0"},
        )
        d = manifest.to_dict()
        assert d["experiment_id"] == "test_123"
        assert d["algorithm"] == "ppo"
        assert d["seed"] == 42

    def test_manifest_save_and_reload(self) -> None:
        """Manifest saves to JSON and can be reloaded."""
        manifest = ExperimentManifest(
            experiment_id="test_save",
            created_at="2026-09-17T00:00:00+00:00",
            algorithm="astar",
            environment="gridworld",
            seed=42,
            config_path="configs/test.yaml",
            training_timesteps=None,
            git_commit="def5678",
            python_version="3.12.0",
            platform_info="Linux x86_64",
            package_versions={},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            manifest.save(path)
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["experiment_id"] == "test_save"
            assert data["algorithm"] == "astar"
            assert data["training_timesteps"] is None


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------


class TestExperimentManager:
    """Tests for ExperimentManager orchestration."""

    def test_list_experiments_empty(self) -> None:
        """list_experiments returns empty list when no experiments exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            experiments = manager.list_experiments()
            assert experiments == []

    def test_get_experiment_missing(self) -> None:
        """get_experiment returns None for unknown experiment ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.get_experiment("nonexistent_id")
            assert result is None

    def test_get_metrics_missing(self) -> None:
        """get_metrics returns None for unknown experiment ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.get_metrics("nonexistent_id")
            assert result is None

    def test_run_astar_experiment(self) -> None:
        """ExperimentManager runs A* planner experiment end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld", seed=42)

            result = manager.run(config=config)

            assert isinstance(result, ExperimentResult)
            assert result.success, f"Experiment failed: {result.error_message}"
            assert result.experiment_id != ""
            assert result.output_dir.exists()
            assert (result.output_dir / "manifest.json").exists()
            assert (result.output_dir / "config.yaml").exists()
            assert (result.output_dir / "metrics.json").exists()

    def test_experiment_manifest_fields(self) -> None:
        """Manifest contains required provenance fields after experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld", seed=42)

            result = manager.run(config=config)
            manifest = result.manifest

            assert manifest.algorithm == "astar"
            assert manifest.environment == "gridworld"
            assert manifest.seed == 42
            assert manifest.python_version != ""
            assert manifest.platform_info != ""
            assert "adaptive-rl" in manifest.package_versions

    def test_run_creates_subdirectories(self) -> None:
        """ExperimentManager creates model, logs, and plots subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld")

            result = manager.run(config=config)

            assert (result.output_dir / "model").exists()
            assert (result.output_dir / "logs").exists()
            assert (result.output_dir / "plots").exists()

    def test_metrics_json_valid(self) -> None:
        """metrics.json is valid JSON with expected planner metric keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld")

            result = manager.run(config=config)

            metrics_path = result.output_dir / "metrics.json"
            with open(metrics_path) as f:
                metrics = json.load(f)

            assert "success_rate" in metrics
            assert "episodes" in metrics

    def test_metrics_csv_created(self) -> None:
        """metrics.csv is created with scalar metric columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld")

            result = manager.run(config=config)

            csv_path = result.output_dir / "metrics.csv"
            assert csv_path.exists()
            content = csv_path.read_text()
            assert len(content.strip()) > 0  # Has content
            assert "," in content  # Has CSV structure

    def test_list_experiments_after_run(self) -> None:
        """list_experiments returns the experiment after it completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld", seed=42)

            result = manager.run(config=config)
            experiments = manager.list_experiments()

            assert len(experiments) == 1
            assert experiments[0]["experiment_id"] == result.experiment_id

    def test_get_experiment_after_run(self) -> None:
        """get_experiment retrieves manifest by experiment ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            config = _make_minimal_config(algo="astar", env="gridworld", seed=42)

            result = manager.run(config=config)
            manifest_data = manager.get_experiment(result.experiment_id)

            assert manifest_data is not None
            assert manifest_data["experiment_id"] == result.experiment_id
            assert manifest_data["algorithm"] == "astar"

    def test_invalid_config_path_returns_failure(self) -> None:
        """run_from_config returns failure result for nonexistent config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(base_output_dir=Path(tmpdir))
            result = manager.run_from_config(Path("/nonexistent/path/config.yaml"))
            assert not result.success
            assert result.error_message != ""

    def test_helper_git_commit(self) -> None:
        """_get_git_commit returns a non-empty string."""
        commit = _get_git_commit()
        assert isinstance(commit, str)
        assert len(commit) > 0

    def test_helper_package_version(self) -> None:
        """_get_package_version returns a version string or 'not_installed'."""
        version = _get_package_version("pydantic")
        assert isinstance(version, str)
        assert len(version) > 0
        # Should be a valid version or 'not_installed'
        assert version == "not_installed" or "." in version
