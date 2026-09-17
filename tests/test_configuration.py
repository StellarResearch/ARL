"""Tests verifying configuration schema validation, loading, and serialization."""

from pathlib import Path

import pytest

from adaptive_rl.config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    save_config,
)


def test_load_committed_configs() -> None:
    """Verify all repository sample configurations pass schema validation."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    sample_files = ["ppo.yaml", "sac.yaml", "navigation.yaml", "drone.yaml"]

    for filename in sample_files:
        filepath = config_dir / filename
        assert filepath.is_file(), f"Expected configuration file missing: {filepath}"
        cfg = load_config(filepath)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name
        assert cfg.seed >= 0
        assert cfg.algorithm.learning_rate > 0.0
        assert cfg.training.total_timesteps > 0
        assert cfg.evaluation.eval_episodes > 0


def test_missing_config_file_raises_error(tmp_path: Path) -> None:
    """Verify non-existent config path raises descriptive ConfigError."""
    non_existent = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config(non_existent)


def test_invalid_yaml_syntax_raises_error(tmp_path: Path) -> None:
    """Verify malformed YAML syntax raises ConfigError."""
    bad_yaml = tmp_path / "syntax_error.yaml"
    bad_yaml.write_text("name: test\nalgorithm: [unclosed list", encoding="utf-8")
    with pytest.raises(ConfigError, match="Failed to parse YAML file"):
        load_config(bad_yaml)


def test_non_dict_yaml_raises_error(tmp_path: Path) -> None:
    """Verify YAML that parses to a non-dictionary raises ConfigError."""
    list_yaml = tmp_path / "list_data.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(list_yaml)


def test_extra_forbidden_fields_raises_error(tmp_path: Path) -> None:
    """Verify unknown fields trigger validation error due to extra='forbid'."""
    invalid_yaml = tmp_path / "extra_field.yaml"
    invalid_yaml.write_text(
        """
name: "test_exp"
seed: 42
unknown_extra_field: "should_fail"
algorithm:
  name: "ppo"
environment:
  name: "CartPole-v1"
training:
  total_timesteps: 1000
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Configuration validation failed"):
        load_config(invalid_yaml)


def test_invalid_ranges_raises_error(tmp_path: Path) -> None:
    """Verify out-of-bounds parameters (e.g. negative learning rate) trigger validation errors."""
    invalid_yaml = tmp_path / "bad_range.yaml"
    invalid_yaml.write_text(
        """
name: "test_exp"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: -0.001
environment:
  name: "CartPole-v1"
training:
  total_timesteps: 1000
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Configuration validation failed"):
        load_config(invalid_yaml)


def test_config_serialization_roundtrip(tmp_path: Path) -> None:
    """Verify an ExperimentConfig can be serialized and re-loaded identically."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "ppo.yaml"
    original_cfg = load_config(config_path)

    saved_path = tmp_path / "roundtrip.yaml"
    save_config(original_cfg, saved_path)

    reloaded_cfg = load_config(saved_path)
    assert original_cfg.name == reloaded_cfg.name
    assert original_cfg.seed == reloaded_cfg.seed
    assert original_cfg.algorithm.name == reloaded_cfg.algorithm.name
    assert original_cfg.algorithm.learning_rate == reloaded_cfg.algorithm.learning_rate
    assert original_cfg.training.total_timesteps == reloaded_cfg.training.total_timesteps


def test_planner_configs_valid() -> None:
    """Verify clean A* and RRT* planner configs validate without RL hyperparameters or training blocks."""
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    for filename in ("gridworld_astar.yaml", "navigation_rrt_star.yaml"):
        cfg = load_config(config_dir / filename)
        assert cfg.algorithm.is_planner
        assert cfg.algorithm.learning_rate is None
        assert cfg.algorithm.gamma is None
        assert cfg.algorithm.batch_size is None
        assert cfg.training is None


def test_planner_rejects_rl_hyperparameters(tmp_path: Path) -> None:
    """Verify that specifying learning_rate or gamma for a planner fails validation clearly."""
    bad_planner_yaml = tmp_path / "bad_astar.yaml"
    bad_planner_yaml.write_text(
        """
name: "bad_astar"
seed: 42
algorithm:
  name: "astar"
  learning_rate: 0.001
environment:
  name: "gridworld"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="does not accept RL hyperparameter"):
        load_config(bad_planner_yaml)


def test_planner_rejects_training_block(tmp_path: Path) -> None:
    """Verify that including a training block for a classical planner fails validation."""
    bad_planner_yaml = tmp_path / "bad_astar_training.yaml"
    bad_planner_yaml.write_text(
        """
name: "bad_astar"
seed: 42
algorithm:
  name: "astar"
environment:
  name: "gridworld"
training:
  total_timesteps: 1000
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="does not support a 'training' configuration block"):
        load_config(bad_planner_yaml)


def test_rl_algorithm_requires_training_block(tmp_path: Path) -> None:
    """Verify that RL algorithms require a training configuration block."""
    bad_rl_yaml = tmp_path / "bad_ppo.yaml"
    bad_rl_yaml.write_text(
        """
name: "bad_ppo"
seed: 42
algorithm:
  name: "ppo"
environment:
  name: "gridworld"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Training configuration \\('training'\\) is required"):
        load_config(bad_rl_yaml)


def test_experiment_wrapper_block_unpacking(tmp_path: Path) -> None:
    """Verify that YAML configs using the experiment: namespace unpack name and seed."""
    exp_yaml = tmp_path / "namespaced_exp.yaml"
    exp_yaml.write_text(
        """
experiment:
  name: "namespaced_astar"
  seed: 99
algorithm:
  name: "astar"
  params:
    heuristic: "manhattan"
environment:
  name: "gridworld"
evaluation:
  episodes: 5
""",
        encoding="utf-8",
    )
    cfg = load_config(exp_yaml)
    assert cfg.name == "namespaced_astar"
    assert cfg.seed == 99
    assert cfg.algorithm.name == "astar"
    assert cfg.algorithm.parameters == {"heuristic": "manhattan"}
    assert cfg.evaluation.eval_episodes == 5


def test_astar_invalid_heuristic_rejected(tmp_path: Path) -> None:
    """Invalid A* heuristic raises ConfigError at load time."""
    bad_yaml = tmp_path / "bad_astar.yaml"
    bad_yaml.write_text(
        """
name: "bad_astar"
algorithm:
  name: "astar"
  parameters:
    heuristic: "teleportation"
environment:
  name: "gridworld"
""",
        encoding="utf-8",
    )
    with pytest.raises(
        ConfigError, match="Input should be 'manhattan', 'euclidean' or 'chebyshev'"
    ):
        load_config(bad_yaml)


def test_astar_valid_heuristics_accepted(tmp_path: Path) -> None:
    """All valid A* heuristics load and validate successfully."""
    for h in ("manhattan", "euclidean", "chebyshev"):
        yaml_file = tmp_path / f"astar_{h}.yaml"
        yaml_file.write_text(
            f"""
name: "astar_{h}"
algorithm:
  name: "astar"
  parameters:
    heuristic: "{h}"
environment:
  name: "gridworld"
""",
            encoding="utf-8",
        )
        cfg = load_config(yaml_file)
        assert cfg.algorithm.parameters["heuristic"] == h


def test_rrt_star_invalid_parameters_rejected(tmp_path: Path) -> None:
    """Invalid RRT* parameters raise ConfigError at load time."""
    bad_yaml = tmp_path / "bad_rrt.yaml"
    bad_yaml.write_text(
        """
name: "bad_rrt"
algorithm:
  name: "rrt_star"
  parameters:
    step_size: -0.5
environment:
  name: "navigation"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="step_size"):
        load_config(bad_yaml)


def test_planner_extra_forbidden_parameters_rejected(tmp_path: Path) -> None:
    """Unknown extra parameters in planner parameters block trigger validation error."""
    bad_yaml = tmp_path / "extra_param_planner.yaml"
    bad_yaml.write_text(
        """
name: "extra_param"
algorithm:
  name: "astar"
  parameters:
    heuristic: "manhattan"
    unsupported_field: 123
environment:
  name: "gridworld"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Extra inputs are not permitted"):
        load_config(bad_yaml)
