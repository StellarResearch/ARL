"""Tests for evaluation seeding protocol and benchmark fairness.

Verifies:
1. RL and classical planners evaluate on the identical environment seeds given the same experiment seed.
2. Changing experiment seed changes the generated evaluation sequence.
3. Planner-internal randomness is decoupled from the shared environment seed.
4. Experiment manifests store the actual evaluation seeds used.
5. No arbitrary offsets (e.g. seed + 10000) are silently applied to RL evaluation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
)
from adaptive_rl.evaluation.seeding import (
    derive_evaluation_seed,
    derive_planner_seed,
    generate_evaluation_seeds,
)
from adaptive_rl.experiments.manager import ExperimentManager


class TestEvaluationSeedingProtocol:
    """Unit and integration tests for the evaluation seeding protocol."""

    def test_deterministic_seed_derivation(self) -> None:
        """derive_evaluation_seed produces deterministic seeds for same inputs."""
        s1 = derive_evaluation_seed(42, 0)
        s2 = derive_evaluation_seed(42, 0)
        assert s1 == s2 == 42

        assert derive_evaluation_seed(42, 5) == 47
        assert derive_evaluation_seed(100, 3) == 103

    def test_changing_experiment_seed_changes_sequence(self) -> None:
        """Changing the experiment seed changes the generated evaluation sequence."""
        seq_a = generate_evaluation_seeds(experiment_seed=42, num_episodes=5)
        seq_b = generate_evaluation_seeds(experiment_seed=43, num_episodes=5)
        seq_c = generate_evaluation_seeds(experiment_seed=100, num_episodes=5)

        assert seq_a != seq_b
        assert seq_a != seq_c
        assert seq_a == [42, 43, 44, 45, 46]
        assert seq_b == [43, 44, 45, 46, 47]
        assert seq_c == [100, 101, 102, 103, 104]

    def test_derive_evaluation_seed_input_validation(self) -> None:
        """Negative episode index raises ValueError."""
        with pytest.raises(ValueError, match="episode_index must be non-negative"):
            derive_evaluation_seed(42, -1)

    def test_generate_evaluation_seeds_input_validation(self) -> None:
        """Non-positive episode count raises ValueError."""
        with pytest.raises(ValueError, match="num_episodes must be positive"):
            generate_evaluation_seeds(42, 0)
        with pytest.raises(ValueError, match="num_episodes must be positive"):
            generate_evaluation_seeds(42, -5)

    def test_planner_seed_decoupled_from_env_seed(self) -> None:
        """Changing planner seed preserves the shared environment seed."""
        exp_seed = 42
        ep = 2

        env_seed = derive_evaluation_seed(exp_seed, ep)
        planner_seed_default = derive_planner_seed(exp_seed, ep, planner_seed=None)
        planner_seed_custom = derive_planner_seed(exp_seed, ep, planner_seed=999)

        # Environment seed remains strictly 44
        assert env_seed == 44

        # Planner seeds differ from each other and are distinct from env seed
        assert planner_seed_default is not None
        assert planner_seed_custom is not None
        assert planner_seed_default != env_seed
        assert planner_seed_custom != planner_seed_default
        assert planner_seed_custom == 999 + ep

    def test_regression_rl_and_classical_eval_receive_identical_seeds(self) -> None:
        """Regression test: RL Evaluator and Classical PlannerAdapter receive IDENTICAL environment seeds.

        Ensures no silent +10000 offset or divergence exists between RL and planners.
        """
        exp_seed = 42
        num_episodes = 5
        expected_seeds = [42, 43, 44, 45, 46]

        # 1. Classical PlannerAdapter on GridWorld
        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.astar import AStarPlanner

        env = GridWorldEnv(width=6, height=6, num_obstacles=2)
        captured_planner_env_seeds = []
        original_reset = env.reset

        def mocked_env_reset(seed: int | None = None, **kwargs):  # type: ignore
            captured_planner_env_seeds.append(seed)
            return original_reset(seed=seed, **kwargs)

        env.reset = mocked_env_reset  # type: ignore

        planner = AStarPlanner()
        adapter = PlannerAdapter(planner=planner, env=env)
        adapter.evaluate(num_episodes=num_episodes, base_seed=exp_seed)

        assert captured_planner_env_seeds == expected_seeds

        # 2. RL Evaluator
        from adaptive_rl.evaluation.evaluator import Evaluator

        captured_rl_env_seeds = []
        eval_env = GridWorldEnv(width=6, height=6, num_obstacles=2)
        original_eval_reset = eval_env.reset

        def mocked_eval_reset(seed: int | None = None, **kwargs):  # type: ignore
            captured_rl_env_seeds.append(seed)
            return original_eval_reset(seed=seed, **kwargs)

        eval_env.reset = mocked_eval_reset  # type: ignore

        dummy_algo = MagicMock()
        dummy_algo.predict.return_value = (0, {})

        evaluator = Evaluator(algorithm=dummy_algo, env=eval_env)
        evaluator.evaluate(num_episodes=num_episodes, deterministic=True, base_seed=exp_seed)

        assert captured_rl_env_seeds == expected_seeds

        # Assert parity: RL and Classical receive 100% IDENTICAL environment seeds
        assert captured_rl_env_seeds == captured_planner_env_seeds

    def test_experiment_manifest_records_evaluation_seeds(self) -> None:
        """ExperimentManager records the actual evaluation seeds in ExperimentManifest and manifest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manager = ExperimentManager(base_output_dir=tmp_path)

            config = ExperimentConfig(
                name="test_eval_seeds_manifest",
                seed=77,
                algorithm=AlgorithmConfig(name="astar"),
                environment=EnvironmentConfig(name="gridworld", max_steps=20),
                evaluation=EvaluationConfig(eval_episodes=4, deterministic=True),
            )

            result = manager.run(config)
            assert result.success is True
            assert result.manifest.evaluation_seeds == [77, 78, 79, 80]

            # Verify saved manifest.json contains evaluation_seeds
            manifest_json_path = result.output_dir / "manifest.json"
            assert manifest_json_path.exists()
            import json

            with open(manifest_json_path, encoding="utf-8") as f:
                data = json.load(f)
            assert data.get("evaluation_seeds") == [77, 78, 79, 80]
