"""Evaluation engine implementations for AdaptiveRL."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.scenarios import EvaluationScenario


class BaseEvaluator(ABC):
    """Abstract interface for agent/environment evaluation routines."""

    @abstractmethod
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        base_seed: Optional[int] = None,
    ) -> EvaluationMetrics:
        """Run evaluation benchmark over the specified number of episodes."""
        pass


class Evaluator(BaseEvaluator):
    """Standardized multi-episode evaluation benchmark engine."""

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: Optional[gym.Env] = None,
        env_name: Optional[str] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize evaluator with algorithm and evaluation environment.

        Args:
            algorithm: Trained agent algorithm instance implementing BaseAlgorithm.
            env: Pre-instantiated Gymnasium environment (optional).
            env_name: Registered environment name to instantiate via factory (optional).
            env_kwargs: Additional parameters forwarded to make_env when env_name is used.
        """
        self.algorithm = algorithm
        self.env_kwargs = dict(env_kwargs or {})

        if env_name is not None:
            self.env_name = env_name
            self.env = env if env is not None else make_env(env_name, **self.env_kwargs)
        elif env is not None:
            self.env = env
            spec = getattr(env, "spec", None)
            if spec is not None and getattr(spec, "id", None):
                self.env_name = str(spec.id)
            else:
                from adaptive_rl.environments.registry import list_environments

                registered = list_environments()
                cls_name = type(env).__name__
                simplified = cls_name.lower().replace("env", "")
                if simplified in registered:
                    self.env_name = simplified
                elif cls_name in registered:
                    self.env_name = cls_name
                else:
                    self.env_name = cls_name
        else:
            raise ValueError("Evaluator requires either 'env' or 'env_name'.")

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        base_seed: Optional[int] = None,
    ) -> EvaluationMetrics:
        """Execute deterministic or stochastic multi-episode evaluation benchmark.

        Args:
            num_episodes: Total evaluation episodes to run.
            deterministic: Whether to use deterministic action selection.
            base_seed: Base seed for reproducible evaluation episode initializations.

        Returns:
            EvaluationMetrics: Standardized aggregated performance metrics.
        """
        if num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}")

        rewards: List[float] = []
        lengths: List[int] = []
        successes = 0
        collisions = 0

        from adaptive_rl.evaluation.seeding import derive_evaluation_seed

        for ep in range(num_episodes):
            seed = derive_evaluation_seed(base_seed, ep) if base_seed is not None else None
            obs, info = self.env.reset(seed=seed)
            ep_reward = 0.0
            ep_length = 0
            done = False

            ep_success = False
            ep_collision = False

            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                ep_reward += float(reward)
                ep_length += 1

                if step_info.get("success", False):
                    ep_success = True
                if step_info.get("collision", False):
                    ep_collision = True

                done = terminated or truncated

            rewards.append(ep_reward)
            lengths.append(ep_length)
            if ep_success:
                successes += 1
            if ep_collision:
                collisions += 1

        mean_rew = float(np.mean(rewards))
        std_rew = float(np.std(rewards))
        min_rew = float(np.min(rewards))
        max_rew = float(np.max(rewards))
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        return EvaluationMetrics(
            episodes=num_episodes,
            mean_reward=mean_rew,
            std_reward=std_rew,
            min_reward=min_rew,
            max_reward=max_rew,
            success_rate=successes / num_episodes,
            collision_rate=collisions / num_episodes,
            mean_episode_length=mean_len,
            std_episode_length=std_len,
            additional_metrics={
                "all_rewards": rewards,
                "all_lengths": lengths,
                "deterministic": deterministic,
                "base_seed": base_seed,
            },
        )

    def evaluate_scenarios(
        self,
        scenarios: List[EvaluationScenario],
        deterministic: bool = True,
    ) -> Dict[str, EvaluationMetrics]:
        """Benchmark the agent across a curated collection of evaluation scenarios.

        Args:
            scenarios: List of EvaluationScenario specifications.
            deterministic: Whether to evaluate deterministically.

        Returns:
            Dict[str, EvaluationMetrics]: Mapping from scenario name to evaluation metrics.
        """
        results: Dict[str, EvaluationMetrics] = {}

        for sc in scenarios:
            scenario_kwargs = dict(self.env_kwargs)
            scenario_kwargs.update(sc.environment_overrides)

            # Create environment for this specific scenario
            sc_env = make_env(self.env_name, **scenario_kwargs)
            sc_evaluator = Evaluator(algorithm=self.algorithm, env=sc_env)
            metrics = sc_evaluator.evaluate(
                num_episodes=1,
                deterministic=deterministic,
                base_seed=sc.seed,
            )
            results[sc.name] = metrics
            sc_env.close()

        return results

    @staticmethod
    def save_report(
        metrics: EvaluationMetrics | Dict[str, EvaluationMetrics],
        output_path: str | Path,
    ) -> Path:
        """Serialize evaluation metrics to a formatted JSON report.

        Args:
            metrics: EvaluationMetrics instance or dictionary of named metrics.
            output_path: Destination filepath.

        Returns:
            Path: Written report path.
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(metrics, EvaluationMetrics):
            data = metrics.model_dump()
        else:
            data = {name: m.model_dump() for name, m in metrics.items()}

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return target

    def close(self) -> None:
        """Close the evaluation environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
