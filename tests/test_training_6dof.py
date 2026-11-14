"""Integration tests verifying PPO and SAC training, checkpoints, and evaluation on Drone6DOFEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.registry import load_algorithm_from_pretrained
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments.drone_6dof import Drone6DOFEnv
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.training.trainer import RLTrainer


def test_ppo_drone_6dof_train_save_load(tmp_path: Path) -> None:
    """Verify PPO can train on Drone6DOFEnv, save checkpoint, load, and predict valid actions."""
    output_dir = tmp_path / "artifacts_ppo"
    config = ExperimentConfig(
        name="test_ppo_6dof",
        seed=42,
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=0.001,
            batch_size=64,
            parameters={"n_steps": 128, "n_epochs": 2, "clip_range": 0.2},
        ),
        environment=EnvironmentConfig(
            name="drone-6dof",
            max_steps=50,
            parameters={"num_obstacles": 1},
        ),
        training=TrainingConfig(
            total_timesteps=256,
            checkpoint_freq=128,
            log_interval=1,
        ),
        evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        output_dir=output_dir,
    )

    trainer = RLTrainer(config=config)
    result = trainer.fit()

    assert result.total_timesteps >= 256
    assert result.final_model_path.exists()

    # Load trained model via load_algorithm_from_pretrained
    eval_env = make_env("drone-6dof", num_obstacles=1, max_steps=50)
    loaded_algo = load_algorithm_from_pretrained(
        result.final_model_path,
        env=eval_env,
        algorithm_name="ppo",
    )

    obs, info = eval_env.reset(seed=42)
    action, _ = loaded_algo.predict(obs, deterministic=True)

    assert isinstance(action, np.ndarray)
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    eval_env.close()


def test_sac_drone_6dof_train_save_load(tmp_path: Path) -> None:
    """Verify SAC can train on Drone6DOFEnv, save checkpoint, load, and predict valid actions."""
    output_dir = tmp_path / "artifacts_sac"
    config = ExperimentConfig(
        name="test_sac_6dof",
        seed=42,
        algorithm=AlgorithmConfig(
            name="sac",
            learning_rate=0.001,
            batch_size=64,
            parameters={"learning_starts": 64, "buffer_size": 5000, "train_freq": 1},
        ),
        environment=EnvironmentConfig(
            name="drone-6dof",
            max_steps=50,
            parameters={"num_obstacles": 1},
        ),
        training=TrainingConfig(
            total_timesteps=256,
            checkpoint_freq=128,
            log_interval=1,
        ),
        evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        output_dir=output_dir,
    )

    trainer = RLTrainer(config=config)
    result = trainer.fit()

    assert result.total_timesteps >= 256
    assert result.final_model_path.exists()

    eval_env = make_env("drone-6dof", num_obstacles=1, max_steps=50)
    loaded_algo = load_algorithm_from_pretrained(
        result.final_model_path,
        env=eval_env,
        algorithm_name="sac",
    )

    obs, info = eval_env.reset(seed=42)
    action, _ = loaded_algo.predict(obs, deterministic=True)

    assert isinstance(action, np.ndarray)
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    eval_env.close()


def test_evaluation_reproducibility_and_reward_inspection(tmp_path: Path) -> None:
    """Verify deterministic evaluation on Drone6DOFEnv produces reproducible metrics and inspectable rewards."""
    env = Drone6DOFEnv(max_steps=30, num_obstacles=2)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)

    evaluator = Evaluator(algorithm=algo, env=env)
    metrics_run1 = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=100)
    records1 = list(evaluator.last_episode_records)

    metrics_run2 = evaluator.evaluate(num_episodes=3, deterministic=True, base_seed=100)
    records2 = list(evaluator.last_episode_records)

    # Identical mean reward and episode length
    assert np.isclose(metrics_run1.mean_reward, metrics_run2.mean_reward, atol=1e-5)
    assert np.isclose(metrics_run1.mean_episode_length, metrics_run2.mean_episode_length, atol=1e-5)

    # Inspect records and verify episode-level determinism
    assert len(records1) == 3
    assert len(records2) == 3
    for r1, r2 in zip(records1, records2):
        assert isinstance(r1.return_value, float)
        assert r1.episode_length > 0
        assert np.isclose(r1.return_value, r2.return_value, atol=1e-5)
        assert r1.episode_length == r2.episode_length
    env.close()


def test_cli_demo_drone_with_6dof_config(tmp_path: Path) -> None:
    """Verify CLI demo-drone command respects configured 6-DOF environment name and spaces."""
    from typer.testing import CliRunner

    from adaptive_rl.cli import app

    runner = CliRunner()

    # Train a quick 6-DOF model
    output_dir = tmp_path / "cli_demo"
    config_file = tmp_path / "drone_6dof_quick.yaml"
    config_file.write_text(
        f"""
name: "quick_demo_6dof"
seed: 42
algorithm:
  name: "ppo"
  learning_rate: 0.001
  batch_size: 64
  parameters:
    n_steps: 64
    n_epochs: 1
environment:
  name: "drone-6dof"
  max_steps: 20
  parameters:
    num_obstacles: 1
training:
  total_timesteps: 128
  checkpoint_freq: 64
evaluation:
  eval_episodes: 1
output_dir: "{output_dir}"
""",
        encoding="utf-8",
    )

    train_res = runner.invoke(app, ["train", "--config", str(config_file), "--timesteps", "128"])
    assert train_res.exit_code == 0

    model_path = output_dir / "models" / "quick_demo_6dof_final.zip"
    assert model_path.exists()

    demo_res = runner.invoke(
        app,
        [
            "demo-drone",
            "--model",
            str(model_path),
            "--config",
            str(config_file),
            "--seed",
            "42",
            "--max-steps",
            "5",
        ],
    )
    assert demo_res.exit_code == 0
    assert "drone-6dof" in demo_res.output or "Demonstration Flight" in demo_res.output
