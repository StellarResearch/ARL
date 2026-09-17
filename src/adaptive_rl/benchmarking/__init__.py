"""Benchmarking and ablation framework for AdaptiveRL.

Provides the BenchmarkRunner for running controlled multi-seed experiments,
computing aggregate statistics, and producing structured comparison reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adaptive_rl.config import load_config
from adaptive_rl.experiments.manager import ExperimentManager

# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Result from a single seed evaluation.

    Attributes:
        seed: The random seed used.
        success: Whether the run completed without error.
        metrics: Evaluation metrics dictionary.
        experiment_id: Associated experiment identifier.
        error_message: Error description if success is False.
    """

    seed: int
    success: bool
    metrics: Dict[str, Any]
    experiment_id: str = ""
    error_message: str = ""


@dataclass
class AggregateStats:
    """Aggregate statistics over multiple seed evaluations.

    Attributes:
        metric_name: Name of the metric.
        mean: Mean value across seeds.
        std: Standard deviation across seeds.
        min: Minimum value.
        max: Maximum value.
        n_seeds: Number of seeds contributing.
        values: Raw per-seed values.
    """

    metric_name: str
    mean: float
    std: float
    min: float
    max: float
    n_seeds: int
    values: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary."""
        return {
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "n_seeds": self.n_seeds,
            "values": self.values,
        }


@dataclass
class BenchmarkResult:
    """Structured result of a benchmark run (one algorithm/config over multiple seeds).

    Attributes:
        name: Human-readable name for this benchmark arm.
        algorithm: Algorithm name.
        environment: Environment name.
        seeds: Seeds used.
        seed_results: Per-seed evaluation results.
        aggregate: Aggregated statistics per metric over successful seeds.
        successful_seeds: Number of seeds that completed without error.
        total_seeds: Alias for requested_seeds (backward compatibility).
        requested_seeds: Total number of seeds requested for this benchmark.
        failed_seeds: Number of seeds that failed.
        successful_seed_ids: List of seed values that succeeded.
        failed_seed_ids: List of seed values that failed.
        failure_reasons: Mapping from failed seed to error message.
        complete: True if all requested seeds succeeded without failure.
    """

    name: str
    algorithm: str
    environment: str
    seeds: List[int]
    seed_results: List[SeedResult] = field(default_factory=list)
    aggregate: Dict[str, AggregateStats] = field(default_factory=dict)
    successful_seeds: int = 0
    total_seeds: int = 0
    requested_seeds: int = 0
    failed_seeds: int = 0
    successful_seed_ids: List[int] = field(default_factory=list)
    failed_seed_ids: List[int] = field(default_factory=list)
    failure_reasons: Dict[int, str] = field(default_factory=dict)
    complete: bool = True

    def __post_init__(self) -> None:
        """Ensure seed counts, status flags, and seed ID lists are initialized."""
        if self.requested_seeds == 0 and self.seeds:
            self.requested_seeds = len(self.seeds)
        if self.total_seeds == 0:
            self.total_seeds = self.requested_seeds

        if self.seed_results:
            if not self.successful_seed_ids:
                self.successful_seed_ids = [sr.seed for sr in self.seed_results if sr.success]
            if not self.failed_seed_ids:
                self.failed_seed_ids = [sr.seed for sr in self.seed_results if not sr.success]
            if not self.failure_reasons:
                self.failure_reasons = {
                    sr.seed: sr.error_message
                    for sr in self.seed_results
                    if not sr.success and sr.error_message
                }
            self.successful_seeds = len(self.successful_seed_ids)
            self.failed_seeds = len(self.failed_seed_ids)
            self.complete = self.failed_seeds == 0 and self.successful_seeds == self.requested_seeds
        else:
            self.failed_seeds = self.requested_seeds - self.successful_seeds
            self.complete = self.failed_seeds == 0 and self.successful_seeds == self.requested_seeds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary (JSON-serializable)."""
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "environment": self.environment,
            "seeds": self.seeds,
            "requested_seeds": self.requested_seeds,
            "total_seeds": self.total_seeds,
            "successful_seeds": self.successful_seeds,
            "failed_seeds": self.failed_seeds,
            "successful_seed_ids": self.successful_seed_ids,
            "failed_seed_ids": self.failed_seed_ids,
            "failure_reasons": self.failure_reasons,
            "complete": self.complete,
            "seed_results": [
                {
                    "seed": sr.seed,
                    "success": sr.success,
                    "experiment_id": sr.experiment_id,
                    "metrics": sr.metrics,
                    "error_message": sr.error_message,
                }
                for sr in self.seed_results
            ],
            "aggregate": {k: v.to_dict() for k, v in self.aggregate.items()},
        }


@dataclass
class ComparisonReport:
    """Side-by-side comparison of two benchmark results.

    Attributes:
        arm_a: First benchmark arm.
        arm_b: Second benchmark arm.
        comparison: Per-metric comparison statistics.
    """

    arm_a: BenchmarkResult
    arm_b: BenchmarkResult
    comparison: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dictionary."""
        return {
            "arm_a": self.arm_a.to_dict(),
            "arm_b": self.arm_b.to_dict(),
            "comparison": self.comparison,
        }

    def save(self, path: Path) -> None:
        """Write comparison to JSON.

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Aggregate statistics helper
# ---------------------------------------------------------------------------


def compute_aggregate_stats(
    metric_name: str,
    values: List[float],
) -> AggregateStats:
    """Compute aggregate statistics for a metric across seeds.

    Filters out NaN or non-finite values safely without crashing.

    Args:
        metric_name: Name of the metric.
        values: List of per-seed scalar values.

    Returns:
        AggregateStats with mean, std, min, max.
    """
    if not values:
        return AggregateStats(
            metric_name=metric_name,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            n_seeds=0,
            values=[],
        )

    valid = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not np.isnan(v)
    ]
    if not valid:
        return AggregateStats(
            metric_name=metric_name,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            n_seeds=0,
            values=list(values),
        )

    arr = np.array(valid, dtype=float)
    return AggregateStats(
        metric_name=metric_name,
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        n_seeds=len(valid),
        values=list(values),
    )


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Multi-seed controlled experiment runner for benchmarking and ablations.

    Runs a single algorithm/environment configuration across multiple seeds,
    aggregates statistics (mean ± std, min, max), and optionally compares
    two configurations for ablation studies.

    Example::

        from pathlib import Path
        from adaptive_rl.benchmarking.runner import BenchmarkRunner

        runner = BenchmarkRunner(seeds=[42, 43, 44], timesteps=1000)
        result = runner.run(config_path=Path("configs/gridworld_ppo.yaml"), name="PPO")
        print(result.aggregate["success_rate"].mean)
    """

    # Metrics to aggregate (present in EvaluationMetrics or PlannerEvaluationMetrics)
    SCALAR_METRICS = [
        "success_rate",
        "collision_rate",
        "mean_reward",
        "std_reward",
        "min_reward",
        "max_reward",
        "mean_episode_length",
        "std_episode_length",
        "mean_path_length",
        "std_path_length",
        "mean_planning_time",
    ]

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        timesteps: Optional[int] = None,
        base_output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            seeds: List of random seeds to evaluate over. Defaults to [42, 43, 44].
            timesteps: Override training timesteps (useful for smoke tests).
            base_output_dir: Root directory for experiment artifacts.
        """
        self.seeds = seeds or [42, 43, 44]
        self.timesteps = timesteps
        self.manager = ExperimentManager(base_output_dir=base_output_dir)

    def run(
        self,
        config_path: Path,
        name: Optional[str] = None,
    ) -> BenchmarkResult:
        """Run the experiment across all configured seeds and aggregate results.

        Args:
            config_path: Path to the YAML configuration file.
            name: Human-readable name for this benchmark arm. Defaults to the
                experiment config name.

        Returns:
            BenchmarkResult with per-seed results and aggregate statistics.
        """
        # Load config to get algorithm/env names
        config = load_config(config_path)
        arm_name = name or f"{config.environment.name}_{config.algorithm.name}"

        seed_results: List[SeedResult] = []

        for seed in self.seeds:
            exp_result = self.manager.run_from_config(
                config_path=config_path,
                timesteps_override=self.timesteps,
                seed_override=seed,
            )
            seed_results.append(
                SeedResult(
                    seed=seed,
                    success=exp_result.success,
                    metrics=exp_result.metrics,
                    experiment_id=exp_result.experiment_id,
                    error_message=exp_result.error_message,
                )
            )

        # Aggregate
        successful = [sr for sr in seed_results if sr.success]
        failed = [sr for sr in seed_results if not sr.success]
        aggregate = self._aggregate(successful)

        return BenchmarkResult(
            name=arm_name,
            algorithm=config.algorithm.name,
            environment=config.environment.name,
            seeds=list(self.seeds),
            seed_results=seed_results,
            aggregate=aggregate,
            successful_seeds=len(successful),
            total_seeds=len(self.seeds),
            requested_seeds=len(self.seeds),
            failed_seeds=len(failed),
            successful_seed_ids=[sr.seed for sr in successful],
            failed_seed_ids=[sr.seed for sr in failed],
            failure_reasons={sr.seed: sr.error_message for sr in failed if sr.error_message},
            complete=(len(failed) == 0 and len(successful) == len(self.seeds)),
        )

    def compare(
        self,
        config_a: Path,
        config_b: Path,
        name_a: Optional[str] = None,
        name_b: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> ComparisonReport:
        """Compare two algorithm/config combinations across the same seeds.

        Args:
            config_a: Path to first YAML configuration.
            config_b: Path to second YAML configuration.
            name_a: Label for first arm.
            name_b: Label for second arm.
            output_path: If provided, save the comparison JSON here.

        Returns:
            ComparisonReport with results for both arms and delta statistics.
        """
        result_a = self.run(config_a, name=name_a)
        result_b = self.run(config_b, name=name_b)

        comparison: Dict[str, Dict[str, Any]] = {}
        all_metrics = set(result_a.aggregate.keys()) | set(result_b.aggregate.keys())

        for metric in sorted(all_metrics):
            a_stats = result_a.aggregate.get(metric)
            b_stats = result_b.aggregate.get(metric)

            entry: Dict[str, Any] = {}
            if a_stats is not None:
                entry["arm_a_mean"] = a_stats.mean
                entry["arm_a_std"] = a_stats.std
            else:
                entry["arm_a_mean"] = None
                entry["arm_a_std"] = None

            if b_stats is not None:
                entry["arm_b_mean"] = b_stats.mean
                entry["arm_b_std"] = b_stats.std
            else:
                entry["arm_b_mean"] = None
                entry["arm_b_std"] = None

            if a_stats is not None and b_stats is not None:
                if not (np.isnan(a_stats.mean) or np.isnan(b_stats.mean)):
                    entry["delta"] = b_stats.mean - a_stats.mean
                    entry["pct_change"] = (
                        (b_stats.mean - a_stats.mean) / abs(a_stats.mean) * 100
                        if a_stats.mean != 0
                        else None
                    )
                else:
                    entry["delta"] = None
                    entry["pct_change"] = None
            else:
                entry["delta"] = None
                entry["pct_change"] = None

            comparison[metric] = entry

        report = ComparisonReport(arm_a=result_a, arm_b=result_b, comparison=comparison)

        if output_path is not None:
            report.save(output_path)

        return report

    @classmethod
    def _aggregate(cls, seed_results: List[SeedResult]) -> Dict[str, AggregateStats]:
        """Compute aggregate statistics across successful seed results.

        Args:
            seed_results: List of successful seed result records.

        Returns:
            Dictionary mapping metric names to AggregateStats.
        """
        if not seed_results:
            return {}

        aggregated: Dict[str, List[float]] = {m: [] for m in cls.SCALAR_METRICS}

        for sr in seed_results:
            for metric in cls.SCALAR_METRICS:
                val = sr.metrics.get(metric)
                if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                    aggregated[metric].append(float(val))

        return {
            metric: compute_aggregate_stats(metric, values)
            for metric, values in aggregated.items()
            if values  # Only include metrics with at least one value
        }
