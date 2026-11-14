"""Comprehensive tests for checkpoint inspection, metadata provenance, and compatibility validation."""

import zipfile
from pathlib import Path

import pytest
from stable_baselines3 import PPO, SAC

from adaptive_rl.algorithms.registry import (
    AlgorithmRegistryError,
    inspect_model_checkpoint,
    load_algorithm_from_pretrained,
)
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.environments.drone_6dof import Drone6DOFEnv


def test_checkpoint_valid_ppo(tmp_path: Path) -> None:
    """Verify inspection and loading of valid 3-DOF PPO checkpoint."""
    env = DroneNavigation3DEnv(max_steps=10)
    model_path = tmp_path / "valid_ppo.zip"
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32)
    model.save(str(model_path))
    env.close()

    info = inspect_model_checkpoint(model_path)
    assert info["valid"] is True
    assert info["algorithm"] == "ppo"
    assert info["obs_shape"] == (29,)
    assert info["action_shape"] == (3,)
    assert info["compatible_env"] == "3-DOF"
    assert info["compatibility_status"] in ("verified", "unverified_shape_match")

    eval_env = DroneNavigation3DEnv(max_steps=10)
    loaded = load_algorithm_from_pretrained(model_path, env=eval_env)
    assert loaded.__class__.__name__ == "PPOAlgorithm"
    eval_env.close()


def test_checkpoint_valid_sac(tmp_path: Path) -> None:
    """Verify inspection and loading of valid 6-DOF SAC checkpoint."""
    env = Drone6DOFEnv(max_steps=10)
    model_path = tmp_path / "valid_sac.zip"
    model = SAC("MlpPolicy", env, buffer_size=1000, learning_starts=10, batch_size=32)
    model.save(str(model_path))
    env.close()

    info = inspect_model_checkpoint(model_path)
    assert info["valid"] is True
    assert info["algorithm"] == "sac"
    assert info["obs_shape"] == (36,)
    assert info["action_shape"] == (4,)
    assert info["compatible_env"] == "6-DOF"
    assert info["compatibility_status"] in ("verified", "unverified_shape_match")

    eval_env = Drone6DOFEnv(max_steps=10)
    loaded = load_algorithm_from_pretrained(model_path, env=eval_env)
    assert loaded.__class__.__name__ == "SACAlgorithm"
    eval_env.close()


def test_checkpoint_invalid_archive(tmp_path: Path) -> None:
    """Verify corrupted non-zip archive is rejected."""
    bad_file = tmp_path / "not_a_zip.zip"
    bad_file.write_bytes(b"corrupted binary header that is not zip")

    info = inspect_model_checkpoint(bad_file)
    assert info["valid"] is False
    assert info["compatibility_status"] == "incompatible"
    assert "not a valid zip archive" in str(info["error"]).lower()

    with pytest.raises(AlgorithmRegistryError):
        load_algorithm_from_pretrained(bad_file)


def test_checkpoint_corrupt_metadata(tmp_path: Path) -> None:
    """Verify archive with corrupted metadata gracefully falls back without crashing."""
    corrupt_meta_zip = tmp_path / "corrupt_meta.zip"
    with zipfile.ZipFile(corrupt_meta_zip, "w") as z:
        z.writestr("arl_metadata.json", "{ malformed json :::")
        z.writestr(
            "data",
            b'{"policy_class": {"__module__": "stable_baselines3.ppo.policies"}, "observation_space": {"_shape": [29]}, "action_space": {"_shape": [3]}}',
        )

    info = inspect_model_checkpoint(corrupt_meta_zip)
    # Corrupt embedded json should not crash; it should fall back to shape inspection
    assert info["valid"] is True
    assert info["algorithm"] == "ppo"
    assert info["compatible_env"] == "3-DOF"
    assert info["compatibility_status"] == "unverified_shape_match"


def test_checkpoint_missing_metadata_legacy_compatibility(tmp_path: Path) -> None:
    """Verify legacy checkpoint with no metadata is loaded via shape matching with clear warning status."""
    legacy_zip = tmp_path / "legacy_model.zip"
    with zipfile.ZipFile(legacy_zip, "w") as z:
        z.writestr(
            "data",
            b'{"policy_class": {"__module__": "stable_baselines3.ppo.policies"}, "observation_space": {"_shape": [29]}, "action_space": {"_shape": [3]}}',
        )

    info = inspect_model_checkpoint(legacy_zip)
    assert info["valid"] is True
    assert info["algorithm"] == "ppo"
    assert info["compatible_env"] == "3-DOF"
    assert info["compatibility_status"] == "unverified_shape_match"
    assert info["metadata_source"] == "shape_inferred"


def test_checkpoint_environment_mismatch(tmp_path: Path) -> None:
    """Verify loading a 3-DOF model into a 6-DOF environment raises an explicit ValueError."""
    env3 = DroneNavigation3DEnv(max_steps=10)
    model_path = tmp_path / "ppo_3dof.zip"
    model = PPO("MlpPolicy", env3, n_steps=64, batch_size=32)
    model.save(str(model_path))
    env3.close()

    env6 = Drone6DOFEnv(max_steps=10)
    with pytest.raises(ValueError, match="does not match environment observation space"):
        load_algorithm_from_pretrained(model_path, env=env6)
    env6.close()


def test_checkpoint_non_existent_file(tmp_path: Path) -> None:
    """Verify nonexistent file returns structured failure."""
    missing = tmp_path / "missing_file.zip"
    info = inspect_model_checkpoint(missing)
    assert info["valid"] is False
    assert info["compatibility_status"] == "incompatible"
    assert "not found" in str(info["error"]).lower()

    with pytest.raises(FileNotFoundError):
        load_algorithm_from_pretrained(missing)


def test_checkpoint_manager_metadata_embedding(tmp_path: Path) -> None:
    """Verify CheckpointManager and CheckpointCallback embed arl_metadata.json for verified provenance."""
    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.training.callbacks import CheckpointCallback
    from adaptive_rl.training.checkpointing import CheckpointManager

    env = DroneNavigation3DEnv(max_steps=10)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)
    cp_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir=cp_dir)

    metadata = {
        "environment_id": "drone",
        "algorithm": "ppo",
        "seed": 42,
        "config_fingerprint": "abcdef1234567890",
    }

    # 1. Direct save_checkpoint with metadata
    cp_path = manager.save_checkpoint(model=algo, step=100, metadata=metadata)
    assert cp_path.is_file()
    info = inspect_model_checkpoint(cp_path)
    assert info["valid"] is True
    assert info["algorithm"] == "ppo"
    assert info["compatible_env"] == "3-DOF"
    assert info["compatibility_status"] == "verified"
    assert info["metadata_source"] == "embedded"
    assert info["config_fingerprint"] == "abcdef1234567890"

    # 2. CheckpointCallback with metadata
    cb = CheckpointCallback(checkpoint_manager=manager, save_freq=50, model=algo, metadata=metadata)
    cb.on_step(step=150)
    history = manager.list_checkpoints()
    assert len(history) == 2
    step150_info = inspect_model_checkpoint(history[-1]["path"])
    assert step150_info["valid"] is True
    assert step150_info["compatibility_status"] == "verified"
    assert step150_info["metadata_source"] == "embedded"
    env.close()

