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

    def test_evaluation_environment_state_parity_gridworld(self) -> None:
        """Verify full environment state parity between RL and Classical evaluation on GridWorld.

        Not only the seeds, but the generated start positions, goal positions,
        and obstacle configurations must be 100% identical episode-by-episode.
        """
        import numpy as np

        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.evaluation.evaluator import Evaluator
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.astar import AStarPlanner

        exp_seed = 123
        num_episodes = 4

        # 1. Classical Planner Evaluation
        planner_env = GridWorldEnv(width=8, height=8, num_obstacles=5)
        captured_planner_states = []
        orig_planner_reset = planner_env.reset

        def captured_planner_reset(seed: int | None = None, **kwargs):  # type: ignore
            obs, info = orig_planner_reset(seed=seed, **kwargs)
            captured_planner_states.append(
                {
                    "seed": seed,
                    "agent_pos": info["agent_pos"],
                    "goal_pos": info["goal_pos"],
                    "obstacles": set(planner_env._obstacles),
                    "obs": np.copy(obs),
                }
            )
            return obs, info

        planner_env.reset = captured_planner_reset  # type: ignore
        adapter = PlannerAdapter(planner=AStarPlanner(), env=planner_env)
        adapter.evaluate(num_episodes=num_episodes, base_seed=exp_seed)

        # 2. RL Evaluator Evaluation
        rl_env = GridWorldEnv(width=8, height=8, num_obstacles=5)
        captured_rl_states = []
        orig_rl_reset = rl_env.reset

        def captured_rl_reset(seed: int | None = None, **kwargs):  # type: ignore
            obs, info = orig_rl_reset(seed=seed, **kwargs)
            captured_rl_states.append(
                {
                    "seed": seed,
                    "agent_pos": info["agent_pos"],
                    "goal_pos": info["goal_pos"],
                    "obstacles": set(rl_env._obstacles),
                    "obs": np.copy(obs),
                }
            )
            return obs, info

        rl_env.reset = captured_rl_reset  # type: ignore
        dummy_algo = MagicMock()
        dummy_algo.predict.return_value = (0, {})
        evaluator = Evaluator(algorithm=dummy_algo, env=rl_env)
        evaluator.evaluate(num_episodes=num_episodes, deterministic=True, base_seed=exp_seed)

        # 3. Assert full episode-by-episode state parity
        assert len(captured_planner_states) == num_episodes
        assert len(captured_rl_states) == num_episodes

        for ep in range(num_episodes):
            p_state = captured_planner_states[ep]
            r_state = captured_rl_states[ep]

            assert p_state["seed"] == r_state["seed"] == exp_seed + ep
            assert p_state["agent_pos"] == r_state["agent_pos"]
            assert p_state["goal_pos"] == r_state["goal_pos"]
            assert p_state["obstacles"] == r_state["obstacles"]
            np.testing.assert_array_equal(p_state["obs"], r_state["obs"])

    def test_evaluation_environment_state_parity_continuous_nav(self) -> None:
        """Verify full environment state parity between RL and Classical evaluation on ContinuousNav2D.

        Start coordinates, goal coordinates, and obstacle geometries must be identical.
        """
        import numpy as np

        from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
        from adaptive_rl.evaluation.evaluator import Evaluator
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        exp_seed = 321
        num_episodes = 3

        # 1. Classical Planner Evaluation
        planner_env = ContinuousNavigation2DEnv(
            arena_width=10.0, arena_height=10.0, num_obstacles=3
        )
        captured_planner_states = []
        orig_p_reset = planner_env.reset

        def captured_p_reset(seed: int | None = None, **kwargs):  # type: ignore
            obs, info = orig_p_reset(seed=seed, **kwargs)
            captured_planner_states.append(
                {
                    "seed": seed,
                    "agent_pos": np.copy(info["agent_pos"]),
                    "goal_pos": np.copy(info["goal_pos"]),
                    "obstacles": list(info["obstacles"]),
                    "obs": np.copy(obs),
                }
            )
            return obs, info

        planner_env.reset = captured_p_reset  # type: ignore
        adapter = PlannerAdapter(
            planner=RRTStarPlanner(seed=55, max_iterations=50), env=planner_env
        )
        adapter.evaluate(num_episodes=num_episodes, base_seed=exp_seed)

        # 2. RL Evaluator Evaluation
        rl_env = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, num_obstacles=3)
        captured_rl_states = []
        orig_r_reset = rl_env.reset

        def captured_r_reset(seed: int | None = None, **kwargs):  # type: ignore
            obs, info = orig_r_reset(seed=seed, **kwargs)
            captured_rl_states.append(
                {
                    "seed": seed,
                    "agent_pos": np.copy(info["agent_pos"]),
                    "goal_pos": np.copy(info["goal_pos"]),
                    "obstacles": list(info["obstacles"]),
                    "obs": np.copy(obs),
                }
            )
            return obs, info

        rl_env.reset = captured_r_reset  # type: ignore
        dummy_algo = MagicMock()
        dummy_algo.predict.return_value = (np.array([0.0, 0.0], dtype=np.float32), {})
        evaluator = Evaluator(algorithm=dummy_algo, env=rl_env)
        evaluator.evaluate(num_episodes=num_episodes, deterministic=True, base_seed=exp_seed)

        # 3. Assert full episode-by-episode state parity
        assert len(captured_planner_states) == num_episodes
        assert len(captured_rl_states) == num_episodes

        for ep in range(num_episodes):
            p_state = captured_planner_states[ep]
            r_state = captured_rl_states[ep]

            assert p_state["seed"] == r_state["seed"] == exp_seed + ep
            np.testing.assert_allclose(p_state["agent_pos"], r_state["agent_pos"])
            np.testing.assert_allclose(p_state["goal_pos"], r_state["goal_pos"])
            assert len(p_state["obstacles"]) == len(r_state["obstacles"])
            for p_obs, r_obs in zip(p_state["obstacles"], r_state["obstacles"]):
                np.testing.assert_allclose(p_obs, r_obs)
            np.testing.assert_allclose(p_state["obs"], r_state["obs"], atol=1e-6)

    def test_planner_randomness_decoupled_from_environment_state(self) -> None:
        """Verify that altering planner seed does NOT change procedural environment state."""
        import numpy as np

        from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        exp_seed = 99
        num_episodes = 3

        # Run 1: Planner with seed=111
        env1 = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, num_obstacles=3)
        captured_env_1 = []
        orig_reset_1 = env1.reset

        def reset_1(seed: int | None = None, **kwargs):  # type: ignore
            obs, info = orig_reset_1(seed=seed, **kwargs)
            captured_env_1.append(
                {
                    "seed": seed,
                    "agent_pos": np.copy(info["agent_pos"]),
                    "goal_pos": np.copy(info["goal_pos"]),
                    "obstacles": list(info["obstacles"]),
                }
            )
            return obs, info

        env1.reset = reset_1  # type: ignore
        planner1 = RRTStarPlanner(seed=111, max_iterations=80)
        adapter1 = PlannerAdapter(planner=planner1, env=env1)
        res1 = adapter1.evaluate(num_episodes=num_episodes, base_seed=exp_seed)

        # Run 2: Planner with seed=777, identical environment seed
        env2 = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, num_obstacles=3)
        captured_env_2 = []
        orig_reset_2 = env2.reset

        def reset_2(seed: int | None = None, **kwargs):  # type: ignore
            obs, info = orig_reset_2(seed=seed, **kwargs)
            captured_env_2.append(
                {
                    "seed": seed,
                    "agent_pos": np.copy(info["agent_pos"]),
                    "goal_pos": np.copy(info["goal_pos"]),
                    "obstacles": list(info["obstacles"]),
                }
            )
            return obs, info

        env2.reset = reset_2  # type: ignore
        planner2 = RRTStarPlanner(seed=777, max_iterations=80)
        adapter2 = PlannerAdapter(planner=planner2, env=env2)
        res2 = adapter2.evaluate(num_episodes=num_episodes, base_seed=exp_seed)

        # Assert environment state is identical between Run 1 and Run 2
        for ep in range(num_episodes):
            assert captured_env_1[ep]["seed"] == captured_env_2[ep]["seed"] == exp_seed + ep
            np.testing.assert_allclose(
                captured_env_1[ep]["agent_pos"], captured_env_2[ep]["agent_pos"]
            )
            np.testing.assert_allclose(
                captured_env_1[ep]["goal_pos"], captured_env_2[ep]["goal_pos"]
            )
            for obs1, obs2 in zip(captured_env_1[ep]["obstacles"], captured_env_2[ep]["obstacles"]):
                np.testing.assert_allclose(obs1, obs2)

        # Assert planner-internal seeds were decoupled
        assert res1.additional_metrics["base_seed"] == exp_seed
        assert res2.additional_metrics["base_seed"] == exp_seed

    def test_same_env_and_planner_seed_produces_reproducible_planner_behavior(self) -> None:
        """Verify that same environment seed + same planner seed produces identical planner behavior."""
        import numpy as np

        from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        exp_seed = 105
        planner_seed = 42
        num_episodes = 3

        env1 = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, num_obstacles=3)
        planner1 = RRTStarPlanner(seed=planner_seed, max_iterations=100)
        adapter1 = PlannerAdapter(planner=planner1, env=env1)
        res1 = adapter1.evaluate(num_episodes=num_episodes, base_seed=exp_seed)

        env2 = ContinuousNavigation2DEnv(arena_width=10.0, arena_height=10.0, num_obstacles=3)
        planner2 = RRTStarPlanner(seed=planner_seed, max_iterations=100)
        adapter2 = PlannerAdapter(planner=planner2, env=env2)
        res2 = adapter2.evaluate(num_episodes=num_episodes, base_seed=exp_seed)

        # Assert identical metrics
        assert res1.episodes == res2.episodes == num_episodes
        assert res1.success_rate == res2.success_rate
        assert res1.all_path_lengths == res2.all_path_lengths
        if res1.mean_path_length is not None:
            assert res2.mean_path_length is not None
            np.testing.assert_allclose(res1.mean_path_length, res2.mean_path_length)
