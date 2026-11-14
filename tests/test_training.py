"""Tests for PPO training pipeline and model persistence."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.training.trainer import PPOTrainer


def test_ppo_algorithm_init_and_train() -> None:
    """Verify PPO algorithm initializes and completes a short rollout."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)

    assert algo.model is not None
    algo.train(total_timesteps=64)
    assert algo.num_timesteps >= 64

    obs, _ = env.reset(seed=42)
    action, _ = algo.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert not np.isnan(action).any()
    assert not np.isinf(action).any()
    env.close()


def test_ppo_model_save_and_load(tmp_path: Path) -> None:
    """Verify trained PPO weights can be saved to disk and loaded back."""
    env = DroneNavigation3DEnv(bounds=(20.0, 20.0, 10.0), max_steps=20, num_obstacles=1)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    algo.train(total_timesteps=64)

    save_path = tmp_path / "saved_ppo.zip"
    algo.save(save_path)
    assert save_path.exists()

    loaded_algo = PPOAlgorithm.from_pretrained(save_path, env=env)
    obs, _ = env.reset(seed=10)
    action1, _ = algo.predict(obs, deterministic=True)
    action2, _ = loaded_algo.predict(obs, deterministic=True)

    np.testing.assert_allclose(action1, action2, rtol=1e-5)
    env.close()


def test_ppo_trainer_full_lifecycle(tmp_path: Path) -> None:
    """Verify PPOTrainer runs fit, creates models and metadata, and exits successfully."""
    config = ExperimentConfig(
        name="test_lifecycle",
        seed=100,
        output_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=3e-4,
            parameters={"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        ),
        environment=EnvironmentConfig(
            name="drone",
            max_steps=25,
            parameters={"bounds": [20.0, 20.0, 10.0], "num_obstacles": 1},
        ),
        training=TrainingConfig(
            total_timesteps=128,
            checkpoint_freq=64,
            log_interval=1,
        ),
        evaluation=EvaluationConfig(eval_episodes=2),
    )

    trainer = PPOTrainer(config=config)
    result = trainer.fit()

    assert result.total_timesteps == 128
    assert result.final_model_path.exists()
    assert result.metadata_path is not None and result.metadata_path.exists()
    assert isinstance(result.mean_reward, float)
    assert result.training_time_seconds >= 0.0
    trainer.close()


def test_training_duration_excludes_model_serialization(tmp_path: Path, monkeypatch) -> None:
    from adaptive_rl.training import trainer as trainer_module

    clock = [0.0]

    class FakeEnv:
        def close(self) -> None:
            pass

    class FakeAlgorithm:
        def __init__(self, **kwargs) -> None:
            self.num_timesteps = 64

        def train(self, total_timesteps: int, callback=None) -> None:
            clock[0] += 2.5

        def save(self, path: str | Path) -> None:
            clock[0] += 100.0
            Path(path).write_bytes(b"model")

    monkeypatch.setattr(trainer_module, "PPOAlgorithm", FakeAlgorithm)
    monkeypatch.setattr(trainer_module, "SB3CallbackAdapter", lambda **kwargs: object())
    monkeypatch.setattr(trainer_module.time, "perf_counter", lambda: clock[0])
    config = ExperimentConfig(
        name="timing_test",
        output_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            parameters={"n_steps": 64, "batch_size": 32},
        ),
        environment=EnvironmentConfig(name="drone"),
        training=TrainingConfig(total_timesteps=64, checkpoint_freq=0),
        evaluation=EvaluationConfig(eval_episodes=1),
    )

    trainer = PPOTrainer(config=config, env=FakeEnv())  # type: ignore[arg-type]
    result = trainer.fit()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.training_time_seconds == 2.5
    assert clock[0] == 102.5
    assert metadata["training_time_seconds"] == 2.5
    assert result.final_model_path.is_file()
    trainer.close()


def test_trainer_close_logs_environment_teardown_failures(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup failures are logged with context instead of being swallowed."""
    from adaptive_rl.training import trainer as trainer_module

    class BrokenEnv:
        def close(self) -> None:
            raise RuntimeError("teardown exploded")

    monkeypatch.setattr(trainer_module, "PPOAlgorithm", lambda **kwargs: object())

    config = ExperimentConfig(
        name="close_logging",
        seed=7,
        output_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            parameters={"n_steps": 64, "batch_size": 32},
        ),
        environment=EnvironmentConfig(name="drone"),
        training=TrainingConfig(total_timesteps=64, checkpoint_freq=0),
        evaluation=EvaluationConfig(eval_episodes=1),
    )

    trainer = PPOTrainer(config=config, env=BrokenEnv())  # type: ignore[arg-type]
    with caplog.at_level(logging.ERROR, logger="adaptive_rl.training.trainer"):
        trainer.close()  # must not raise

    records = [record for record in caplog.records if record.name == "adaptive_rl.training.trainer"]
    assert records, "expected an ERROR record from the trainer logger"
    record = records[-1]
    message = record.getMessage()
    assert "BrokenEnv" in message
    assert "close_logging" in message
    assert record.exc_info is not None
    assert "teardown exploded" in str(record.exc_info[1])


def test_trainer_close_succeeds_for_well_behaved_environments(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive_rl.training import trainer as trainer_module

    class QuietEnv:
        def close(self) -> None:
            return None

    monkeypatch.setattr(trainer_module, "PPOAlgorithm", lambda **kwargs: object())

    config = ExperimentConfig(
        name="quiet_close",
        seed=7,
        output_dir=tmp_path / "artifacts",
        log_dir=tmp_path / "logs",
        algorithm=AlgorithmConfig(
            name="ppo",
            parameters={"n_steps": 64, "batch_size": 32},
        ),
        environment=EnvironmentConfig(name="drone"),
        training=TrainingConfig(total_timesteps=64, checkpoint_freq=0),
        evaluation=EvaluationConfig(eval_episodes=1),
    )

    trainer = PPOTrainer(config=config, env=QuietEnv())  # type: ignore[arg-type]
    with caplog.at_level(logging.ERROR, logger="adaptive_rl.training.trainer"):
        trainer.close()
    assert not [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR and record.name == "adaptive_rl.training.trainer"
    ]


def test_training_progress_callback_lifecycle() -> None:
    """Verify TrainingProgressCallback correctly captures steps, throughput, and rolling reward."""
    from adaptive_rl.training.callbacks import StreamlitProgressCallback, TrainingProgressCallback

    updates = []

    def on_update(data: dict) -> None:
        updates.append(data.copy())

    cb = TrainingProgressCallback(
        total_timesteps=1000,
        on_update=on_update,
        update_frequency=100,
        rolling_window=5,
    )
    # Verify Streamlit alias points to same class
    assert StreamlitProgressCallback is TrainingProgressCallback

    cb.on_training_start()
    assert cb.current_step == 0
    assert cb.total_episodes == 0
    assert len(updates) == 1
    assert updates[0]["progress_ratio"] == 0.0

    # Step up to 100 -> triggers progress update
    for s in range(1, 101):
        cb.on_step(step=s)
    assert len(updates) == 2
    assert updates[1]["current_step"] == 100
    assert updates[1]["progress_ratio"] == 0.1

    # Episode completion
    cb.on_episode_end(episode=1, episode_reward=25.0, episode_length=50)
    assert cb.total_episodes == 1
    assert cb.episode_rewards == [25.0]
    data = cb.get_progress_data()
    assert data["latest_reward"] == 25.0
    assert data["rolling_reward"] == 25.0

    # Add 4 more episodes to verify rolling return
    for ep, r in enumerate([10.0, 15.0, 20.0, 30.0], start=2):
        cb.on_episode_end(episode=ep, episode_reward=r, episode_length=50)
    data = cb.get_progress_data()
    assert data["episodes"] == 5
    # Mean of [25, 10, 15, 20, 30] = 100 / 5 = 20.0
    assert data["rolling_reward"] == 20.0

    cb.on_training_end()
    assert cb.is_complete is True
    assert cb.current_step == 1000
    data_final = cb.get_progress_data()
    assert data_final["progress_ratio"] == 1.0
    assert data_final["status"] == "completed"


def test_ppo_hyperparameter_validation() -> None:
    """Verify PPO parameter validation catches erroneous or contradictory settings."""
    env = DroneNavigation3DEnv(max_steps=5)

    # 1. batch_size > n_steps
    with pytest.raises(ValueError, match="cannot exceed n_steps"):
        PPOAlgorithm(env=env, n_steps=64, batch_size=128)

    # 2. Non-positive learning rate
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        PPOAlgorithm(env=env, learning_rate=-0.001)

    # 3. Invalid gamma
    with pytest.raises(ValueError, match="gamma must be between"):
        PPOAlgorithm(env=env, gamma=1.5)

    # 4. Invalid gae_lambda
    with pytest.raises(ValueError, match="gae_lambda must be between"):
        PPOAlgorithm(env=env, gae_lambda=-0.1)

    # 5. Unsupported keyword argument
    with pytest.raises(ValueError, match="Unsupported hyperparameter"):
        PPOAlgorithm(env=env, non_existent_hyperparameter=42)

    env.close()


def test_sac_hyperparameter_validation() -> None:
    """Verify SAC parameter validation catches erroneous or out-of-range settings."""
    from adaptive_rl.algorithms.sac import SACAlgorithm

    env = DroneNavigation3DEnv(max_steps=5)

    # 1. Invalid learning rate
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        SACAlgorithm(env=env, learning_rate=0.0)

    # 2. Invalid tau
    with pytest.raises(ValueError, match="tau must be in range"):
        SACAlgorithm(env=env, tau=1.5)

    # 3. Invalid ent_coef string
    with pytest.raises(ValueError, match="Invalid ent_coef string"):
        SACAlgorithm(env=env, ent_coef="invalid_ent_str")

    # 4. Negative learning_starts
    with pytest.raises(ValueError, match="learning_starts must be non-negative"):
        SACAlgorithm(env=env, learning_starts=-5)

    # 5. Unsupported keyword argument
    with pytest.raises(ValueError, match="Unsupported hyperparameter"):
        SACAlgorithm(env=env, invalid_sac_kwarg="bogus")

    env.close()
