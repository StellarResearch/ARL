"""Reproducible PPO and SAC Benchmarking Suite for 3-DOF and 6-DOF quadrotors.

Supports all standard matrix combinations:
- PPO + 3-DOF (Kinematic)
- PPO + 6-DOF (Rigid-body dynamics)
- SAC + 3-DOF (Kinematic)
- SAC + 6-DOF (Rigid-body dynamics)
- Explicit RANDOM baseline

Features:
- Configurable execution profiles:
    * 'quick': Lightweight verification (smoke-test budget, N < 10, no inferential CI claims).
    * 'full': Comprehensive statistical research benchmark (N >= 10, multi-seed Student-t 95% CIs).
- Evaluation of both deterministic and stochastic policy actions.
- Rigorous statistical integrity adhering to established sample size requirements (N >= 10).
- Explicit policy provenance: never silently substitutes random actions under PPO/SAC labels.
- Standardized safety, performance, and flight trajectory metrics.
- Disambiguated outcome semantics (SUCCESS, COLLISION, BOUNDARY_BREACH, TIMEOUT, MAX_STEPS).
- Canonical JSON and CSV report export.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.algorithms.registry import (
    get_algorithm_metadata,
    inspect_model_checkpoint,
    load_algorithm_from_pretrained,
)
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.statistics import summarize_seed_values
from adaptive_rl.flight_testing.recorder import compute_config_fingerprint
from adaptive_rl.training.trainer import get_trainer

logger = logging.getLogger(__name__)

MINIMUM_SAMPLES_FOR_CI: int = 10

SUPPORTED_COMBINATIONS = [
    ("ppo", "drone"),
    ("ppo", "drone-6dof"),
    ("sac", "drone"),
    ("sac", "drone-6dof"),
]

BENCHMARK_ARENA_PARAMS: Dict[str, Any] = {
    "bounds": [30.0, 30.0, 15.0],
    "start_pos": [5.0, 5.0, 5.0],
    "goal_pos": [25.0, 25.0, 10.0],
    "num_obstacles": 4,
    "obstacle_radius": 2.0,
    "target_radius": 1.5,
    "collision_radius": 0.8,
}


@dataclass
class BenchmarkCombinationResult:
    """Benchmark performance summary for one algorithm + environment pair."""

    algorithm: str
    environment: str
    mode: str  # "3-DOF" or "6-DOF"
    policy_source: str  # "checkpoint", "smoke_trained", or "random"
    checkpoint_hash: Optional[str]
    deterministic: bool
    evaluation_profile: str
    is_smoke_test: bool
    total_episodes: int
    evaluation_seeds: List[int]
    success_rate: float
    collision_rate: float
    timeout_rate: float
    max_step_termination_rate: float
    mean_reward: float
    median_reward: float
    std_reward: Optional[float]
    mean_flight_duration_steps: float
    median_flight_duration_steps: float
    mean_flight_duration_seconds: float
    average_goal_distance: float
    mean_position_error: float
    min_obstacle_clearance: float
    evaluation_runtime_seconds: float
    confidence_intervals: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    statistical_note: Optional[str] = None
    config_fingerprint: str = ""
    training_config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSuiteReport:
    """Complete collection of benchmark combination results and provenance metadata."""

    report_id: str
    timestamp_utc: str
    profile: str
    combinations_evaluated: List[BenchmarkCombinationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp_utc": self.timestamp_utc,
            "profile": self.profile,
            "combinations_evaluated": [c.to_dict() for c in self.combinations_evaluated],
            "metadata": self.metadata,
        }

    def export_json(self, filepath: Path | str) -> Path:
        """Export benchmark report to JSON."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target

    def export_csv(self, filepath: Path | str) -> Path:
        """Export summary table to CSV."""
        target = Path(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self.combinations_evaluated:
            target.write_text("no benchmark data\n", encoding="utf-8")
            return target

        fieldnames = [
            "algorithm",
            "environment",
            "mode",
            "policy_source",
            "deterministic",
            "evaluation_profile",
            "is_smoke_test",
            "total_episodes",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "max_step_termination_rate",
            "mean_reward",
            "median_reward",
            "mean_flight_duration_steps",
            "mean_flight_duration_seconds",
            "average_goal_distance",
            "mean_position_error",
            "min_obstacle_clearance",
            "evaluation_runtime_seconds",
        ]

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.combinations_evaluated:
                row = {k: getattr(c, k) for k in fieldnames}
                for k, v in row.items():
                    if isinstance(v, float):
                        row[k] = round(v, 4)
                writer.writerow(row)
        return target


class FlightBenchmarkSuite:
    """Orchestrator for reproducible autonomous flight evaluations across algorithms and dynamics models."""

    def __init__(
        self,
        output_dir: Path | str = "artifacts/benchmarks",
        env_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_params = dict(
            env_parameters if env_parameters is not None else BENCHMARK_ARENA_PARAMS
        )

    def evaluate_combination(
        self,
        algorithm_name: str,
        env_name: str,
        checkpoint_path: Optional[Path | str] = None,
        profile: str = "quick",
        deterministic: bool = True,
        seeds: Optional[Sequence[int]] = None,
        train_if_missing: bool = False,
        smoke_train_steps: int = 256,
    ) -> BenchmarkCombinationResult:
        """Run benchmark evaluation for a single (algorithm, environment) combination.

        Args:
            algorithm_name: 'ppo', 'sac', or 'random'.
            env_name: 'drone' (3-DOF) or 'drone-6dof' (6-DOF).
            checkpoint_path: Optional path to pre-trained model checkpoint.
            profile: 'quick' (smoke profile, N < 10) or 'full' (research benchmark, N >= 10).
            deterministic: Deterministic action selection (True) or stochastic (False).
            seeds: Custom seeds sequence (must be distinct).
            train_if_missing: If True and checkpoint is missing, train a lightweight policy.
            smoke_train_steps: Timesteps for quick training if required.
        """
        algo_clean = algorithm_name.strip().lower()
        env_clean = env_name.strip().lower()

        if algo_clean not in ("ppo", "sac", "random", "random_policy"):
            raise ValueError(
                f"Unsupported algorithm '{algorithm_name}'. Expected 'ppo', 'sac', or 'random'."
            )
        if env_clean not in ("drone", "drone_3d", "drone-6dof", "drone_6dof"):
            raise ValueError(
                f"Unsupported environment '{env_name}'. Expected 'drone' or 'drone-6dof'."
            )

        mode = "6-DOF" if "6dof" in env_clean or "6-dof" in env_clean else "3-DOF"

        # Statistical sample requirement enforcement
        is_smoke = profile == "quick"
        if seeds is not None:
            eval_seeds = [int(s) for s in seeds]
        elif is_smoke:
            eval_seeds = [101, 102]
        else:
            # Full benchmark requires N >= 10 distinct seeds
            eval_seeds = list(range(1001, 1011))

        if not eval_seeds:
            raise ValueError("Evaluation seeds list cannot be empty.")

        if len(eval_seeds) != len(set(eval_seeds)):
            raise ValueError(
                f"Evaluation seeds must be distinct independent observations, got duplicates: {eval_seeds}"
            )

        # 1. Resolve or train policy
        model = None
        policy_source = "random"
        chk_hash: Optional[str] = None
        training_cfg_dict: Optional[Dict[str, Any]] = None

        if algo_clean in ("ppo", "sac"):
            if checkpoint_path is not None:
                c_path = Path(checkpoint_path)
                if not c_path.is_file():
                    raise FileNotFoundError(f"Checkpoint not found: {c_path}")
                # Compute SHA-256 digest of checkpoint
                with open(c_path, "rb") as f:
                    chk_hash = hashlib.sha256(f.read()).hexdigest()

                # Verify checkpoint compatibility before running
                chk_info = inspect_model_checkpoint(c_path)
                if not chk_info.get("valid", True) and chk_info.get("error"):
                    raise ValueError(f"Corrupt checkpoint '{c_path.name}': {chk_info['error']}")

                expected_mode = "6-DOF" if mode == "6-DOF" else "3-DOF"
                compat_env = chk_info.get("compatible_env")
                if compat_env and compat_env != expected_mode:
                    raise ValueError(
                        f"Checkpoint '{c_path.name}' was trained for {compat_env}, but environment is {expected_mode}. "
                        "Cannot evaluate incompatible policy architecture."
                    )
                model = load_algorithm_from_pretrained(algo_clean, c_path)
                policy_source = "checkpoint"
            elif train_if_missing:
                logger.info(
                    "Training %s on %s for %d smoke steps...",
                    algo_clean,
                    env_clean,
                    smoke_train_steps,
                )
                algo_meta = get_algorithm_metadata(algo_clean)
                training_cfg_dict = {
                    "algorithm": algo_clean,
                    "timesteps": smoke_train_steps,
                    "seed": eval_seeds[0],
                }
                exp_cfg = ExperimentConfig(
                    name=f"smoke_{algo_clean}_{env_clean}",
                    seed=eval_seeds[0],
                    algorithm=AlgorithmConfig(
                        name=algo_clean, parameters=dict(algo_meta.hyperparameters)
                    ),
                    environment=EnvironmentConfig(name=env_clean, parameters=self.env_params),
                    training=TrainingConfig(total_timesteps=smoke_train_steps, checkpoint_freq=0),
                    evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
                    output_dir=self.output_dir / "temp_training",
                )
                trainer = get_trainer(config=exp_cfg)
                trainer.fit()
                model = trainer.algorithm
                policy_source = "smoke_trained"
            else:
                raise ValueError(
                    f"Evaluation of '{algorithm_name}' requires a valid model checkpoint or train_if_missing=True. "
                    f"Random actions cannot be silently evaluated under algorithm identity '{algorithm_name}'."
                )
        else:
            algo_clean = "random"
            policy_source = "random"

        # Separate evaluation environment instance
        eval_env = make_env(env_clean, **self.env_params)

        eval_start_time = time.perf_counter()
        rewards: List[float] = []
        durations_steps: List[float] = []
        durations_seconds: List[float] = []
        goal_distances: List[float] = []
        pos_errors: List[float] = []
        clearances: List[float] = []

        success_count = 0
        collision_count = 0
        boundary_breach_count = 0
        timeout_count = 0
        max_step_count = 0

        try:
            for s in eval_seeds:
                obs, info = eval_env.reset(seed=s)
                ep_reward = 0.0
                ep_steps = 0
                done = False
                unwrapped = getattr(eval_env, "unwrapped", eval_env)
                dt_val = getattr(getattr(unwrapped, "dynamics", None), "dt", 0.05)
                max_env_steps = getattr(unwrapped, "max_steps", 500)
                last_step_info: Dict[str, Any] = {}

                while not done:
                    if model is not None:
                        act, _ = model.predict(obs, deterministic=deterministic)
                    else:
                        act = eval_env.action_space.sample()

                    obs, reward, terminated, truncated, step_info = eval_env.step(act)
                    last_step_info = dict(step_info or {})
                    ep_reward += float(reward)
                    ep_steps += 1
                    done = terminated or truncated

                    # Extract physical tracking metrics
                    if "distance_to_goal" in last_step_info:
                        goal_distances.append(float(last_step_info["distance_to_goal"]))
                    if "position" in last_step_info and "goal" in last_step_info:
                        p = np.asarray(last_step_info["position"])
                        g = np.asarray(last_step_info["goal"])
                        pos_errors.append(float(np.linalg.norm(g - p)))
                    if "min_obstacle_distance" in last_step_info and math.isfinite(
                        last_step_info["min_obstacle_distance"]
                    ):
                        clearances.append(float(last_step_info["min_obstacle_distance"]))

                rewards.append(ep_reward)
                durations_steps.append(float(ep_steps))
                durations_seconds.append(float(ep_steps * dt_val))

                # Unambiguous outcome classification with strict collision precedence
                is_coll = bool(last_step_info.get("collision", False))
                coll_type = str(last_step_info.get("collision_type", "none"))
                is_succ = bool(
                    last_step_info.get("success", last_step_info.get("is_success", False))
                )
                is_timeout = bool(
                    last_step_info.get("timeout", last_step_info.get("is_timeout", False))
                )

                if is_coll:
                    if coll_type.startswith("boundary") or coll_type == "ground":
                        boundary_breach_count += 1
                    else:
                        collision_count += 1
                elif is_succ:
                    success_count += 1
                elif is_timeout:
                    timeout_count += 1
                elif truncated or ep_steps >= max_env_steps:
                    max_step_count += 1
                else:
                    max_step_count += 1
        finally:
            eval_env.close()

        eval_duration = time.perf_counter() - eval_start_time

        # Compute empirical descriptive and statistical metrics
        n_episodes = len(rewards)
        succ_rate = float(success_count / n_episodes) if n_episodes > 0 else 0.0
        coll_rate = (
            float((collision_count + boundary_breach_count) / n_episodes) if n_episodes > 0 else 0.0
        )
        timeout_rate = float(timeout_count / n_episodes) if n_episodes > 0 else 0.0
        max_step_rate = float(max_step_count / n_episodes) if n_episodes > 0 else 0.0

        mean_rew = float(np.mean(rewards)) if rewards else 0.0
        median_rew = float(np.median(rewards)) if rewards else 0.0
        std_rew = float(np.std(rewards, ddof=1)) if n_episodes > 1 else None

        mean_dur_steps = float(np.mean(durations_steps)) if durations_steps else 0.0
        median_dur_steps = float(np.median(durations_steps)) if durations_steps else 0.0
        mean_dur_sec = float(np.mean(durations_seconds)) if durations_seconds else 0.0

        avg_goal_d = float(np.mean(goal_distances)) if goal_distances else 0.0
        mean_pos_e = float(np.mean(pos_errors)) if pos_errors else 0.0
        min_clear = float(min(clearances)) if clearances else float("inf")

        # Statistical integrity: enforce MINIMUM_SAMPLES_FOR_CI
        confidence_intervals: Dict[str, Dict[str, Optional[float]]] = {}
        stat_note: Optional[str] = None

        if n_episodes >= MINIMUM_SAMPLES_FOR_CI:
            rew_stat = summarize_seed_values(rewards)
            confidence_intervals["reward"] = {
                "mean": rew_stat.mean,
                "ci95_lower": rew_stat.ci95_lower,
                "ci95_upper": rew_stat.ci95_upper,
            }
            dur_stat = summarize_seed_values(durations_seconds)
            confidence_intervals["duration_seconds"] = {
                "mean": dur_stat.mean,
                "ci95_lower": dur_stat.ci95_lower,
                "ci95_upper": dur_stat.ci95_upper,
            }
            stat_note = (
                f"Sample size N={n_episodes} satisfies research criteria (N >= {MINIMUM_SAMPLES_FOR_CI}). "
                "95% Student's t confidence intervals computed across independent seed evaluations."
            )
        else:
            stat_note = (
                f"Sample size N={n_episodes} is below minimum threshold (N >= {MINIMUM_SAMPLES_FOR_CI}) "
                "for inferential statistics. Confidence intervals are omitted. Only descriptive statistics are reported."
            )

        # Config fingerprint for verification
        cfg_fingerprint = compute_config_fingerprint(
            {
                "algorithm": algo_clean,
                "environment": env_clean,
                "deterministic": deterministic,
                "profile": profile,
                "env_params": self.env_params,
                "policy_source": policy_source,
                "checkpoint_hash": chk_hash,
                "evaluation_seeds": eval_seeds,
            }
        )

        return BenchmarkCombinationResult(
            algorithm=algo_clean.upper(),
            environment=env_clean,
            mode=mode,
            policy_source=policy_source,
            checkpoint_hash=chk_hash,
            deterministic=deterministic,
            evaluation_profile=profile,
            is_smoke_test=is_smoke,
            total_episodes=n_episodes,
            evaluation_seeds=eval_seeds,
            success_rate=succ_rate,
            collision_rate=coll_rate,
            timeout_rate=timeout_rate,
            max_step_termination_rate=max_step_rate,
            mean_reward=mean_rew,
            median_reward=median_rew,
            std_reward=std_rew,
            mean_flight_duration_steps=mean_dur_steps,
            median_flight_duration_steps=median_dur_steps,
            mean_flight_duration_seconds=mean_dur_sec,
            average_goal_distance=avg_goal_d,
            mean_position_error=mean_pos_e,
            min_obstacle_clearance=min_clear,
            evaluation_runtime_seconds=eval_duration,
            confidence_intervals=confidence_intervals,
            statistical_note=stat_note,
            config_fingerprint=cfg_fingerprint,
            training_config=training_cfg_dict,
        )

    def run_suite(
        self,
        combinations: Optional[Sequence[Tuple[str, str]]] = None,
        profile: str = "quick",
        deterministic: bool = True,
        checkpoints: Optional[Dict[Tuple[str, str], Path | str]] = None,
        train_if_missing: bool = False,
    ) -> BenchmarkSuiteReport:
        """Run complete benchmark across multiple algorithm + environment combinations."""
        from datetime import datetime, timezone

        now_utc = datetime.now(timezone.utc).isoformat()
        combo_list = list(combinations or SUPPORTED_COMBINATIONS)
        chk_map = checkpoints or {}

        results: List[BenchmarkCombinationResult] = []
        for algo, env in combo_list:
            c_path = chk_map.get((algo, env))
            res = self.evaluate_combination(
                algorithm_name=algo,
                env_name=env,
                checkpoint_path=c_path,
                profile=profile,
                deterministic=deterministic,
                train_if_missing=train_if_missing,
            )
            results.append(res)

        rep_id = f"benchmark_{profile}_{int(time.time())}"
        return BenchmarkSuiteReport(
            report_id=rep_id,
            timestamp_utc=now_utc,
            profile=profile,
            combinations_evaluated=results,
            metadata={
                "combinations_count": len(results),
                "deterministic": deterministic,
                "env_parameters": self.env_params,
            },
        )
