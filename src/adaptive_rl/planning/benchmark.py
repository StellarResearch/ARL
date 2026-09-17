"""Benchmarking engine comparing classical motion planners against reinforcement learning policies."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.environments.registry import make_env
from adaptive_rl.planning.astar import AStarPlanner, AStarPlannerPolicy
from adaptive_rl.planning.base import PlannerPolicy
from adaptive_rl.planning.rrt import RRTPlannerPolicy, RRTStarPlanner


@dataclass
class PlannerComparisonResult:
    """Individual seed result comparing classical planner vs RL policy.

    Attributes:
        seed: Environment procedural generation seed.
        planner_success: Whether classical planner reached goal.
        rl_success: Whether RL policy reached goal.
        planner_collision: Whether classical planner collided with obstacle.
        rl_collision: Whether RL policy collided with obstacle.
        planner_reward: Cumulative episodic reward earned by classical planner.
        rl_reward: Cumulative episodic reward earned by RL policy.
        planner_steps: Episode duration in steps for classical planner.
        rl_steps: Episode duration in steps for RL policy.
        planner_path_length: Total Euclidean distance traversed by planner.
        rl_path_length: Total Euclidean distance traversed by RL policy.
        path_length_ratio: RL path length divided by planner path length.
        planner_planning_time_ms: Offline path planning duration in milliseconds.
        rl_mean_step_time_ms: Mean per-step policy inference latency in milliseconds.
    """

    seed: int
    planner_success: bool
    rl_success: bool
    planner_collision: bool
    rl_collision: bool
    planner_reward: float
    rl_reward: float
    planner_steps: int
    rl_steps: int
    planner_path_length: float
    rl_path_length: float
    path_length_ratio: float
    planner_planning_time_ms: float
    rl_mean_step_time_ms: float


@dataclass
class ClassicalBenchmarkReport:
    """Comprehensive comparative benchmark report over multiple seeds."""

    experiment_name: str
    environment_name: str
    planner_name: str
    rl_algorithm_name: str
    seeds: List[int]
    total_episodes: int
    planner_success_rate: float
    rl_success_rate: float
    planner_collision_rate: float
    rl_collision_rate: float
    planner_mean_reward: float
    rl_mean_reward: float
    planner_mean_steps: float
    rl_mean_steps: float
    planner_mean_path_length: float
    rl_mean_path_length: float
    mean_path_length_ratio: float
    planner_mean_planning_time_ms: float
    rl_mean_step_time_ms: float
    detailed_results: List[PlannerComparisonResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to serializable dictionary."""
        data = asdict(self)
        return data

    def save_json(self, output_path: str | Path) -> Path:
        """Persist benchmark report to JSON on disk."""
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target


class ClassicalBenchmarkRunner:
    """Orchestrates head-to-head empirical evaluations between classical planners and RL agents."""

    def __init__(
        self,
        environment_name: str,
        planner_type: str = "auto",
        environment_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize ClassicalBenchmarkRunner.

        Args:
            environment_name: Registered environment key (e.g. 'gridworld', 'navigation').
            planner_type: 'astar', 'rrt', 'rrt_star', or 'auto'.
            environment_parameters: Optional dictionary of environment kwargs.
        """
        self.environment_name = environment_name.lower()
        self.env_kwargs = dict(environment_parameters or {})

        if planner_type == "auto":
            if "grid" in self.environment_name:
                self.planner_type = "astar"
            else:
                self.planner_type = "rrt_star"
        else:
            self.planner_type = planner_type.lower()

    def _create_planner_policy(self, env: Any) -> PlannerPolicy:
        """Create configured planner policy adapter for the given environment."""
        if self.planner_type == "astar":
            return AStarPlannerPolicy(env=env)
        elif self.planner_type in ("rrt", "rrt_star"):
            return RRTPlannerPolicy(env=env)
        else:
            raise ValueError(f"Unsupported planner type: {self.planner_type}")

    def _extract_agent_position(self, env: Any, obs: Any) -> Tuple[float, ...]:
        """Extract spatial position coordinate from environment state."""
        if hasattr(env, "agent_pos"):
            pos = env.agent_pos
            return tuple(float(c) for c in pos)
        if hasattr(env, "position"):
            pos = env.position
            return tuple(float(c) for c in pos)
        # Fallback to observation head
        return (float(obs[0]), float(obs[1]))

    def _evaluate_agent(
        self,
        env: Any,
        policy: Any,
        seed: int,
        is_rl: bool = False,
    ) -> Tuple[bool, bool, float, int, float, float]:
        """Execute single episode on environment with specified policy.

        Returns:
            Tuple of:
            - success (bool)
            - collision (bool)
            - cumulative_reward (float)
            - step_count (int)
            - path_length (float)
            - elapsed_time_ms (float)
        """
        if hasattr(policy, "reset_policy"):
            policy.reset_policy()

        plan_start = time.perf_counter()
        obs, _ = env.reset(seed=seed)
        plan_duration_ms = (time.perf_counter() - plan_start) * 1000.0

        done = False
        total_reward = 0.0
        steps = 0
        success = False
        collision = False

        prev_pos = self._extract_agent_position(env, obs)
        total_path_dist = 0.0
        inference_times_ms: List[float] = []

        while not done:
            t0 = time.perf_counter()
            action, _ = policy.predict(obs, deterministic=True)
            inference_times_ms.append((time.perf_counter() - t0) * 1000.0)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            curr_pos = self._extract_agent_position(env, obs)
            dist_step = math.sqrt(
                sum((c2 - c1) ** 2 for c1, c2 in zip(prev_pos, curr_pos))
            )
            total_path_dist += dist_step
            prev_pos = curr_pos

            if info.get("success", False):
                success = True
            if info.get("collision", False):
                collision = True

            done = terminated or truncated

        timing_metric = (
            float(np.mean(inference_times_ms))
            if is_rl and inference_times_ms
            else plan_duration_ms
        )

        return (
            success,
            collision,
            total_reward,
            steps,
            total_path_dist,
            timing_metric,
        )

    def run_benchmark(
        self,
        seeds: Sequence[int],
        rl_algorithm: Optional[Any] = None,
        experiment_name: str = "classical_vs_rl_benchmark",
    ) -> ClassicalBenchmarkReport:
        """Execute head-to-head benchmark across identical environment seeds.

        Args:
            seeds: List of integer seeds to instantiate environments.
            rl_algorithm: Optional trained RL policy (e.g. PPO or SAC).
            experiment_name: Name identifier for this benchmark.

        Returns:
            ClassicalBenchmarkReport: Comprehensive comparative analysis.
        """
        test_env = make_env(self.environment_name, **self.env_kwargs)
        planner_policy = self._create_planner_policy(test_env)

        results: List[PlannerComparisonResult] = []
        planner_name = planner_policy.name
        rl_name = (
            getattr(rl_algorithm, "name", type(rl_algorithm).__name__)
            if rl_algorithm is not None
            else "None (Planner Only)"
        )

        for seed in seeds:
            # 1. Evaluate classical planner
            p_succ, p_coll, p_rew, p_steps, p_dist, p_plan_ms = self._evaluate_agent(
                env=test_env, policy=planner_policy, seed=seed, is_rl=False
            )

            # 2. Evaluate RL policy (if provided)
            if rl_algorithm is not None:
                r_succ, r_coll, r_rew, r_steps, r_dist, r_step_ms = self._evaluate_agent(
                    env=test_env, policy=rl_algorithm, seed=seed, is_rl=True
                )
            else:
                r_succ, r_coll, r_rew, r_steps, r_dist, r_step_ms = (
                    False,
                    False,
                    0.0,
                    0,
                    0.0,
                    0.0,
                )

            ratio = (r_dist / p_dist) if p_dist > 1e-6 and r_dist > 1e-6 else 1.0

            results.append(
                PlannerComparisonResult(
                    seed=int(seed),
                    planner_success=p_succ,
                    rl_success=r_succ,
                    planner_collision=p_coll,
                    rl_collision=r_coll,
                    planner_reward=p_rew,
                    rl_reward=r_rew,
                    planner_steps=p_steps,
                    rl_steps=r_steps,
                    planner_path_length=p_dist,
                    rl_path_length=r_dist,
                    path_length_ratio=ratio,
                    planner_planning_time_ms=p_plan_ms,
                    rl_mean_step_time_ms=r_step_ms,
                )
            )

        test_env.close()

        n = len(seeds)
        p_succ_rate = float(sum(1 for r in results if r.planner_success) / n) if n > 0 else 0.0
        r_succ_rate = float(sum(1 for r in results if r.rl_success) / n) if n > 0 else 0.0
        p_coll_rate = float(sum(1 for r in results if r.planner_collision) / n) if n > 0 else 0.0
        r_coll_rate = float(sum(1 for r in results if r.rl_collision) / n) if n > 0 else 0.0

        p_mean_rew = float(np.mean([r.planner_reward for r in results])) if n > 0 else 0.0
        r_mean_rew = float(np.mean([r.rl_reward for r in results])) if n > 0 else 0.0
        p_mean_steps = float(np.mean([r.planner_steps for r in results])) if n > 0 else 0.0
        r_mean_steps = float(np.mean([r.rl_steps for r in results])) if n > 0 else 0.0
        p_mean_dist = float(np.mean([r.planner_path_length for r in results])) if n > 0 else 0.0
        r_mean_dist = float(np.mean([r.rl_path_length for r in results])) if n > 0 else 0.0
        mean_ratio = float(np.mean([r.path_length_ratio for r in results])) if n > 0 else 1.0

        p_mean_plan_ms = (
            float(np.mean([r.planner_planning_time_ms for r in results])) if n > 0 else 0.0
        )
        r_mean_step_ms = (
            float(np.mean([r.rl_mean_step_time_ms for r in results])) if n > 0 else 0.0
        )

        return ClassicalBenchmarkReport(
            experiment_name=experiment_name,
            environment_name=self.environment_name,
            planner_name=planner_name,
            rl_algorithm_name=rl_name,
            seeds=[int(s) for s in seeds],
            total_episodes=n,
            planner_success_rate=p_succ_rate,
            rl_success_rate=r_succ_rate,
            planner_collision_rate=p_coll_rate,
            rl_collision_rate=r_coll_rate,
            planner_mean_reward=p_mean_rew,
            rl_mean_reward=r_mean_rew,
            planner_mean_steps=p_mean_steps,
            rl_mean_steps=r_mean_steps,
            planner_mean_path_length=p_mean_dist,
            rl_mean_path_length=r_mean_dist,
            mean_path_length_ratio=mean_ratio,
            planner_mean_planning_time_ms=p_mean_plan_ms,
            rl_mean_step_time_ms=r_mean_step_ms,
            detailed_results=results,
            metadata={
                "planner_type": self.planner_type,
                "environment_parameters": self.env_kwargs,
            },
        )
