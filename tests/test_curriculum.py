"""Comprehensive unit and integration tests for Curriculum Learning subsystem."""

import json
from pathlib import Path

import pytest
from gymnasium.utils.env_checker import check_env
from typer.testing import CliRunner

from adaptive_rl.cli import app
from adaptive_rl.config import (
    AlgorithmConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    EnvironmentConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.curriculum.callbacks import CurriculumCallback
from adaptive_rl.curriculum.curriculum import Curriculum
from adaptive_rl.curriculum.presets import (
    create_navigation_curriculum,
    get_curriculum_preset,
)
from adaptive_rl.curriculum.stage import CurriculumStage
from adaptive_rl.curriculum.trainer import CurriculumTrainer
from adaptive_rl.curriculum.wrapper import CurriculumEnvWrapper
from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
from adaptive_rl.training.trainer import get_trainer

runner = CliRunner()


def test_curriculum_stage_advancement_logic() -> None:
    """Verify CurriculumStage criteria evaluation."""
    stage = CurriculumStage(
        stage_id=0,
        name="Test Stage",
        environment_parameters={"num_obstacles": 1},
        success_threshold=0.8,
        mean_reward_threshold=10.0,
        min_episodes=5,
        max_timesteps=100,
    )

    # 1. Under min episodes: cannot advance
    assert not stage.can_advance(
        rolling_metrics={"success_rate": 1.0, "mean_reward": 50.0},
        stage_episodes=4,
        stage_timesteps=50,
    )

    # 2. Reached min episodes but below success threshold
    assert not stage.can_advance(
        rolling_metrics={"success_rate": 0.7, "mean_reward": 50.0},
        stage_episodes=5,
        stage_timesteps=50,
    )

    # 3. Reached min episodes, meets success rate but below mean reward threshold
    assert not stage.can_advance(
        rolling_metrics={"success_rate": 0.85, "mean_reward": 5.0},
        stage_episodes=5,
        stage_timesteps=50,
    )

    # 4. Reached min episodes and satisfies all thresholds
    assert stage.can_advance(
        rolling_metrics={"success_rate": 0.85, "mean_reward": 15.0},
        stage_episodes=5,
        stage_timesteps=50,
    )

    # 5. Max timesteps override triggers advance even if min episodes not met
    assert stage.can_advance(
        rolling_metrics={"success_rate": 0.0, "mean_reward": -100.0},
        stage_episodes=1,
        stage_timesteps=100,
    )


def test_curriculum_lifecycle_and_transitions() -> None:
    """Verify Curriculum stage index management, advance, and serialization."""
    stages = [
        CurriculumStage(stage_id=0, name="Stage 0", min_episodes=2, success_threshold=0.5),
        CurriculumStage(stage_id=1, name="Stage 1", min_episodes=2, success_threshold=0.8),
    ]
    curriculum = Curriculum(name="test_curr", stages=stages, eval_window=5)

    assert curriculum.current_stage_index == 0
    assert curriculum.current_stage.name == "Stage 0"
    assert not curriculum.is_complete

    # Advance to Stage 1
    next_stage = curriculum.advance(timesteps=50, metrics={"success_rate": 0.6})
    assert next_stage is not None
    assert next_stage.stage_id == 1
    assert curriculum.current_stage_index == 1
    assert curriculum.is_complete

    # Cannot advance past last stage
    assert curriculum.advance(timesteps=100) is None
    assert len(curriculum.history) == 1
    assert curriculum.history[0]["from_stage_name"] == "Stage 0"
    assert curriculum.history[0]["to_stage_name"] == "Stage 1"

    # Serialization round-trip
    data = curriculum.to_dict()
    restored = Curriculum.from_dict(data)
    assert restored.name == curriculum.name
    assert restored.current_stage_index == 1
    assert len(restored.stages) == 2
    assert restored.is_complete

    # Reset
    curriculum.reset()
    assert curriculum.current_stage_index == 0
    assert len(curriculum.history) == 0


def test_curriculum_empty_stages_raises() -> None:
    """Verify initializing Curriculum with empty stages raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        Curriculum(name="invalid", stages=[])


def test_curriculum_env_wrapper_gymnasium_compliance() -> None:
    """Verify CurriculumEnvWrapper strictly passes Farama Gymnasium check_env."""
    raw_env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, max_steps=20)
    curriculum = create_navigation_curriculum(eval_window=5)
    wrapped_env = CurriculumEnvWrapper(env=raw_env, curriculum=curriculum)

    check_env(wrapped_env)
    wrapped_env.close()


def test_curriculum_env_wrapper_parameter_injection() -> None:
    """Verify CurriculumEnvWrapper updates environment parameters and telemetry."""
    raw_env = ContinuousNavigation2DEnv(arena_width=20.0, arena_height=20.0, num_obstacles=10)
    curriculum = create_navigation_curriculum()
    wrapped_env = CurriculumEnvWrapper(env=raw_env, curriculum=curriculum)

    # Initial stage (Clear Corridor) sets num_obstacles to 0
    obs, info = wrapped_env.reset(seed=42)
    assert info["curriculum_stage_id"] == 0
    assert info["curriculum_stage_name"] == "Clear Corridor"
    assert not info["curriculum_complete"]
    assert raw_env.num_obstacles == 0

    # Advance curriculum to Stage 1 (Sparse Clutter: num_obstacles=2)
    curriculum.advance(timesteps=100)
    obs, info = wrapped_env.reset(seed=42)
    assert info["curriculum_stage_id"] == 1
    assert info["curriculum_stage_name"] == "Sparse Clutter"
    assert raw_env.num_obstacles == 2

    # Step telemetry check
    obs, reward, term, trunc, step_info = wrapped_env.step(wrapped_env.action_space.sample())
    assert step_info["curriculum_stage_id"] == 1
    assert step_info["curriculum_stage_name"] == "Sparse Clutter"
    wrapped_env.close()


def test_curriculum_callback_trigger() -> None:
    """Verify CurriculumCallback triggers stage progression upon satisfying thresholds."""
    stages = [
        CurriculumStage(stage_id=0, name="Easy", success_threshold=0.8, min_episodes=2),
        CurriculumStage(stage_id=1, name="Hard", success_threshold=0.9, min_episodes=2),
    ]
    curriculum = Curriculum(name="test_cb", stages=stages, eval_window=2)
    raw_env = GridWorldEnv(width=5, height=5, num_obstacles=0)
    wrapped_env = CurriculumEnvWrapper(env=raw_env, curriculum=curriculum)

    cb = CurriculumCallback(curriculum=curriculum, env_wrapper=wrapped_env, verbose=0)

    # Episode 1: success
    cb.on_step(step=10)
    cb.on_episode_end(episode=1, episode_reward=100.0, episode_length=5, info={"success": True})
    assert curriculum.current_stage_index == 0

    # Episode 2: success -> rolling success rate = 100% >= 80% with min_episodes=2 satisfied
    cb.on_step(step=20)
    cb.on_episode_end(episode=2, episode_reward=100.0, episode_length=5, info={"success": True})
    assert curriculum.current_stage_index == 1
    assert curriculum.current_stage.name == "Hard"
    wrapped_env.close()


def test_curriculum_presets() -> None:
    """Verify navigation and gridworld curriculum presets."""
    nav_curr = get_curriculum_preset("navigation")
    assert len(nav_curr.stages) == 4
    assert nav_curr.stages[0].environment_parameters["num_obstacles"] == 0
    assert nav_curr.stages[3].environment_parameters["num_obstacles"] == 8

    grid_curr = get_curriculum_preset("gridworld")
    assert len(grid_curr.stages) == 4
    assert grid_curr.stages[0].environment_parameters["width"] == 4

    with pytest.raises(ValueError, match="Unknown curriculum preset"):
        get_curriculum_preset("unknown_domain")


def test_curriculum_trainer_end_to_end(tmp_path: Path) -> None:
    """Verify CurriculumTrainer executes staged training, checkpointing, and exports reports."""
    config = ExperimentConfig(
        name="test_curriculum_run",
        seed=42,
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.001,
            gamma=0.99,
            batch_size=16,
            parameters={"n_steps": 32},
        ),
        environment=EnvironmentConfig(
            name="navigation",
            max_steps=15,
            parameters={"arena_width": 10.0, "arena_height": 10.0},
        ),
        curriculum=CurriculumConfig(
            enabled=True,
            stages=[
                CurriculumStageConfig(
                    name="Stage 1",
                    environment_parameters={"num_obstacles": 0},
                    max_timesteps=32,
                    min_episodes=1,
                ),
                CurriculumStageConfig(
                    name="Stage 2",
                    environment_parameters={"num_obstacles": 2},
                    min_episodes=1,
                ),
            ],
            eval_window=5,
        ),
        training=TrainingConfig(
            total_timesteps=64,
            checkpoint_freq=32,
            log_interval=5,
        ),
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
    )

    trainer = get_trainer(config=config)
    assert isinstance(trainer, CurriculumTrainer)

    result = trainer.fit()
    assert result.total_timesteps == 64
    assert result.final_model_path.exists()

    # Verify curriculum summary JSON was created
    report_path = tmp_path / "results" / "curriculum" / "test_curriculum_run_curriculum.json"
    assert report_path.exists()
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "history" in report_data
    assert "stages" in report_data


def test_cli_curriculum_commands() -> None:
    """Verify CLI curriculum commands (list, inspect) and platform info."""
    # version
    v_res = runner.invoke(app, ["version"])
    assert v_res.exit_code == 0
    assert "AdaptiveRL" in v_res.output

    # info
    i_res = runner.invoke(app, ["info"])
    assert i_res.exit_code == 0
    assert "Curriculum" in i_res.output

    # curriculum list
    list_res = runner.invoke(app, ["curriculum", "list"])
    assert list_res.exit_code == 0
    assert "navigation" in list_res.output
    assert "gridworld" in list_res.output

    # curriculum inspect
    inspect_res = runner.invoke(app, ["curriculum", "inspect", "navigation"])
    assert inspect_res.exit_code == 0
    assert "Clear" in inspect_res.output and "Corridor" in inspect_res.output
    assert "Dense" in inspect_res.output and "Hazard" in inspect_res.output

    # curriculum inspect invalid
    bad_res = runner.invoke(app, ["curriculum", "inspect", "invalid_preset"])
    assert bad_res.exit_code == 1
    assert "Curriculum lookup failed" in bad_res.output


def test_cli_validate_curriculum_configs() -> None:
    """Verify validate_config parses new curriculum YAML files."""
    nav_cfg = Path("configs/curriculum_navigation.yaml")
    assert nav_cfg.exists()
    res1 = runner.invoke(app, ["config", "validate", str(nav_cfg)])
    assert res1.exit_code == 0
    assert "Curriculum" in res1.output

    grid_cfg = Path("configs/curriculum_gridworld.yaml")
    assert grid_cfg.exists()
    res2 = runner.invoke(app, ["config", "validate", str(grid_cfg)])
    assert res2.exit_code == 0
    assert "Curriculum" in res2.output
