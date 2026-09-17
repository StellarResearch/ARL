"""Tests verifying Typer CLI commands and execution."""

from pathlib import Path

from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.environments.registry import register, registry
from adaptive_rl.environments.testing import DummyTestEnv

runner = CliRunner()


def test_cli_help() -> None:
    """Verify adaptive-rl --help prints help and available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AdaptiveRL" in result.output
    assert "config" in result.output
    assert "env" in result.output
    assert "curriculum" in result.output
    assert "train" in result.output
    assert "evaluate" in result.output
    assert "generalization" in result.output
    assert "benchmark-planners" in result.output


def test_cli_version() -> None:
    """Verify adaptive-rl version displays package version and phase."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "AdaptiveRL" in result.output
    assert "Phase" in result.output


def test_cli_info() -> None:
    """Verify adaptive-rl info displays the roadmap table with completed phases."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Roadmap" in result.output
    assert "Phase 1" in result.output
    assert "Phase 2" in result.output
    assert "Phase 3" in result.output
    assert "Phase 4" in result.output
    assert "Phase 5" in result.output
    assert "Phase 6" in result.output
    assert "Phase 7" in result.output
    assert "Phase 8" in result.output
    assert "Phase 9" in result.output
    assert "Phase 10" in result.output
    assert "Phase 11" in result.output
    assert "Phase 12" in result.output


def test_cli_config_validate_success() -> None:
    """Verify adaptive-rl config validate succeeds for valid YAML configuration."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ppo.yaml"
    result = runner.invoke(app, ["config", "validate", str(config_path)])
    assert result.exit_code == 0
    assert "Configuration is valid" in result.output


def test_cli_config_validate_failure(tmp_path: Path) -> None:
    """Verify adaptive-rl config validate fails with code 1 for invalid YAML."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("invalid: yaml: syntax: [", encoding="utf-8")
    result = runner.invoke(app, ["config", "validate", str(bad_config)])
    assert result.exit_code == 1
    assert "Configuration validation error" in result.output


def test_cli_env_list_empty() -> None:
    """Verify adaptive-rl env list displays informative notice when empty."""
    from adaptive_rl.environments import register_default_environments

    registry.clear()
    try:
        result = runner.invoke(app, ["env", "list"])
        assert result.exit_code == 0
        assert "Gymnasium environments" in result.output
    finally:
        register_default_environments()


def test_cli_env_list_with_registered_env() -> None:
    """Verify adaptive-rl env list displays registered environment metadata."""
    from adaptive_rl.environments import register_default_environments

    registry.clear()
    try:
        register(
            "cli_dummy",
            lambda: DummyTestEnv(),
            metadata={"observation_type": "box", "action_type": "discrete"},
        )
        result = runner.invoke(app, ["env", "list"])
        assert result.exit_code == 0
        assert "cli_dummy" in result.output
    finally:
        register_default_environments()


def test_cli_env_inspect_success() -> None:
    """Verify adaptive-rl env inspect inspects spaces for a standard Gymnasium environment."""
    result = runner.invoke(app, ["env", "inspect", "CartPole-v1"])
    assert result.exit_code == 0
    assert "verified successfully" in result.output
    assert "Observation Space" in result.output
    assert "Action Space" in result.output


def test_cli_env_inspect_failure() -> None:
    """Verify adaptive-rl env inspect fails gracefully for unknown environment."""
    result = runner.invoke(app, ["env", "inspect", "NonExistentEnv-v999"])
    assert result.exit_code == 1
    assert "Environment inspection failed" in result.output


def test_cli_env_run_gridworld() -> None:
    """Verify adaptive-rl env run executes GridWorld rollouts and displays metrics."""
    result = runner.invoke(app, ["env", "run", "gridworld", "--steps", "10", "--seed", "42"])
    assert result.exit_code == 0
    assert "Starting simulation for 'gridworld'" in result.output
    assert "Simulation Summary" in result.output
    assert "Total Steps:" in result.output
    assert "Cumulative Reward:" in result.output


def test_cli_env_run_navigation() -> None:
    """Verify adaptive-rl env run executes continuous Navigation rollouts and ASCII rendering."""
    result = runner.invoke(app, ["env", "run", "navigation", "--steps", "5", "--seed", "42"])
    assert result.exit_code == 0
    assert "Starting simulation for 'navigation'" in result.output
    assert "Simulation Summary" in result.output
    assert "Total Steps:" in result.output
    assert "Cumulative Reward:" in result.output


def test_cli_train_execution(tmp_path: Path) -> None:
    """Verify adaptive-rl train executes PPO training with overridden timesteps."""
    test_config = tmp_path / "test_ppo.yaml"
    test_config.write_text(
        f"""
name: "cli_test_run"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 32
  parameters:
    n_steps: 64
environment:
  name: "gridworld"
  max_steps: 20
  parameters:
    width: 5
    height: 5
    num_obstacles: 2
training:
  total_timesteps: 64
  checkpoint_freq: 0
  log_interval: 10
evaluation:
  eval_episodes: 2
output_dir: "{tmp_path / "results"}"
log_dir: "{tmp_path / "logs"}"
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "--config", str(test_config), "--timesteps", "64"])
    assert result.exit_code == 0
    assert "Starting Training: cli_test_run" in result.output
    assert "Training Completed Successfully!" in result.output
    assert "Total Timesteps Trained: 64" in result.output


def test_cli_evaluate_missing_model(tmp_path: Path) -> None:
    """Verify adaptive-rl evaluate displays informative error when model file is missing."""
    config_path = tmp_path / "eval_cfg.yaml"
    config_path.write_text(
        """
name: "no_model_test"
seed: 42
algorithm:
  name: "ppo"
environment:
  name: "gridworld"
  max_steps: 10
training:
  total_timesteps: 64
evaluation:
  eval_episodes: 2
output_dir: "non_existent_results_dir"
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["evaluate", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "No model weights provided" in result.output
