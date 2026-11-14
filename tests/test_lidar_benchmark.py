"""Tests for the LiDAR robustness benchmark configuration."""

from pathlib import Path

import numpy as np

from adaptive_rl.benchmarking.lidar_benchmark import LIDAR_CONDITIONS
from adaptive_rl.environments.drone import DroneNavigation3DEnv


def test_lidar_conditions_match_protocol() -> None:
    expected = {
        "LIDAR_IDEAL": (0.0, 0.0, 0.0),
        "LIDAR_NOISE_MODERATE": (0.15, 0.0, 0.0),
        "LIDAR_DROPOUT_MODERATE": (0.0, 0.20, 0.0),
        "LIDAR_COMBINED": (0.15, 0.20, 0.0),
        "LIDAR_SEVERE": (0.35, 0.40, 0.5),
    }

    actual = {
        condition.name: (
            condition.noise_std,
            condition.dropout_prob,
            condition.min_range,
        )
        for condition in LIDAR_CONDITIONS
    }

    assert actual == expected


def test_matched_seeds_preserve_scenario_geometry() -> None:
    reference_layout = None

    for condition in LIDAR_CONDITIONS:
        env = DroneNavigation3DEnv(**condition.environment_kwargs())
        try:
            observation, _ = env.reset(seed=1234)

            layout = [(obstacle.center.copy(), obstacle.radius) for obstacle in env.obstacles]

            assert observation.shape == (29,)

            if reference_layout is None:
                reference_layout = layout
            else:
                assert len(layout) == len(reference_layout)
                for current, reference in zip(layout, reference_layout):
                    np.testing.assert_array_equal(current[0], reference[0])
                    assert current[1] == reference[1]
        finally:
            env.close()


def test_lidar_benchmark_runs_two_episodes_per_condition() -> None:
    from adaptive_rl.benchmarking.lidar_benchmark import run_lidar_benchmark

    class FixedPolicy:
        def predict(
            self, observation: np.ndarray, deterministic: bool = True
        ) -> tuple[np.ndarray, None]:
            assert deterministic
            return np.zeros(3, dtype=np.float32), None

    result = run_lidar_benchmark(
        FixedPolicy(),  # type: ignore[arg-type]
        num_episodes=2,
        base_seed=1000,
        env_kwargs={"max_steps": 8, "num_obstacles": 1},
    )

    assert len(result["conditions"]) == 5
    assert len(result["episodes"]) == 10
    assert result["metadata"]["seeds"] == [1000, 1001]

    for condition in LIDAR_CONDITIONS:
        records = [record for record in result["episodes"] if record["condition"] == condition.name]
        assert len(records) == 2
        assert [record["episode_reset_seed"] for record in records] == [
            1000,
            1001,
        ]


def test_lidar_benchmark_exports(tmp_path: Path) -> None:
    import csv
    import json

    from adaptive_rl.benchmarking.lidar_benchmark import (
        export_lidar_benchmark,
        run_lidar_benchmark,
    )

    class FixedPolicy:
        def predict(
            self, observation: np.ndarray, deterministic: bool = True
        ) -> tuple[np.ndarray, None]:
            return np.zeros(3, dtype=np.float32), None

    result = run_lidar_benchmark(
        FixedPolicy(),  # type: ignore[arg-type]
        num_episodes=2,
        env_kwargs={"max_steps": 5, "num_obstacles": 1},
    )

    paths = export_lidar_benchmark(result, str(tmp_path))

    assert all(Path(path).is_file() for path in paths.values())
    assert Path(paths["plot"]).stat().st_size > 0

    with open(paths["json"], encoding="utf-8") as handle:
        data = json.load(handle)

    assert len(data["conditions"]) == 5
    assert len(data["episodes"]) == 10

    with open(paths["csv"], newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    with open(paths["episodes_csv"], newline="", encoding="utf-8") as handle:
        episode_rows = list(csv.DictReader(handle))

    assert len(summary_rows) == 5
    assert len(episode_rows) == 10
    assert "collision_rate" in summary_rows[0]
    assert "episode_reset_seed" in episode_rows[0]


def test_timeout_trajectory_classification() -> None:
    from adaptive_rl.benchmarking.lidar_benchmark import (
        classify_timeout_trajectory,
    )

    stationary = [np.zeros(3) for _ in range(21)]
    oscillating = [np.array([float(i % 2) * 3.0, 0.0, 0.0]) for i in range(21)]
    advancing = [np.array([float(i), 0.0, 0.0]) for i in range(21)]

    assert classify_timeout_trajectory(stationary) == "freeze_timeout"
    assert classify_timeout_trajectory(oscillating) == "oscillation_timeout"
    assert classify_timeout_trajectory(advancing) == "unclassified_timeout"
    assert classify_timeout_trajectory(stationary[:10]) == "unclassified_timeout"


def test_frozen_policy_timeouts_are_classified() -> None:
    from adaptive_rl.benchmarking.lidar_benchmark import run_lidar_benchmark

    class FixedPolicy:
        def predict(
            self, observation: np.ndarray, deterministic: bool = True
        ) -> tuple[np.ndarray, None]:
            return np.zeros(3, dtype=np.float32), None

    result = run_lidar_benchmark(
        FixedPolicy(),  # type: ignore[arg-type]
        num_episodes=2,
        env_kwargs={"max_steps": 25, "num_obstacles": 0},
    )

    for summary in result["conditions"]:
        assert summary["freeze_timeouts"] == 2
        assert summary["oscillation_timeouts"] == 0
        assert summary["unclassified_timeouts"] == 0

    assert all(episode["failure_mode"] == "freeze_timeout" for episode in result["episodes"])


def test_lidar_cli_help_and_missing_checkpoint(tmp_path: Path) -> None:
    import re

    from typer.testing import CliRunner

    from adaptive_rl.cli import app

    runner = CliRunner()

    help_result = runner.invoke(app, ["benchmark", "lidar", "--help"])
    assert help_result.exit_code == 0
    clean_help = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", help_result.output)
    assert "--model" in clean_help
    assert "--episodes" in clean_help

    missing_result = runner.invoke(
        app,
        [
            "benchmark",
            "lidar",
            "--model",
            str(tmp_path / "missing.zip"),
        ],
    )
    assert missing_result.exit_code == 1
    clean_missing = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", missing_result.output)
    assert "Model file not found" in clean_missing


def test_lidar_benchmark_rejects_invalid_protocol_inputs() -> None:
    import pytest

    from adaptive_rl.benchmarking.lidar_benchmark import run_lidar_benchmark

    class FixedPolicy:
        def predict(
            self, observation: np.ndarray, deterministic: bool = True
        ) -> tuple[np.ndarray, None]:
            return np.zeros(3, dtype=np.float32), None

    policy = FixedPolicy()

    with pytest.raises(ValueError, match="unique"):
        run_lidar_benchmark(
            policy,
            seeds=[1000, 1000],  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="non-negative"):
        run_lidar_benchmark(
            policy,
            seeds=[-1],  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Sensor parameters"):
        run_lidar_benchmark(
            policy,  # type: ignore[arg-type]
            num_episodes=2,
            env_kwargs={"lidar_noise_std": 0.2},
        )


def test_lidar_cli_success_exports(tmp_path: Path, monkeypatch) -> None:
    import json
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from adaptive_rl import cli as cli_module
    from adaptive_rl.algorithms import registry

    class FixedPolicy:
        def predict(
            self, observation: np.ndarray, deterministic: bool = True
        ) -> tuple[np.ndarray, None]:
            return np.zeros(3, dtype=np.float32), None

    monkeypatch.setattr(
        registry,
        "load_algorithm_from_pretrained",
        lambda *args, **kwargs: FixedPolicy(),
    )
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda path: SimpleNamespace(
            environment=SimpleNamespace(
                name="drone",
                parameters={"num_obstacles": 1},
                max_steps=5,
            )
        ),
    )

    model = tmp_path / "model.zip"
    model.write_bytes(b"test-checkpoint")

    output_dir = tmp_path / "results"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "benchmark",
            "lidar",
            "--model",
            str(model),
            "--episodes",
            "2",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)

    data = json.loads((output_dir / "lidar_robustness.json").read_text(encoding="utf-8"))

    assert len(data["conditions"]) == 5
    assert len(data["episodes"]) == 10
    assert data["metadata"]["model_sha256"]
    assert (output_dir / "lidar_robustness.csv").is_file()
    assert (output_dir / "lidar_robustness_episodes.csv").is_file()
    assert (output_dir / "lidar_robustness.png").is_file()
