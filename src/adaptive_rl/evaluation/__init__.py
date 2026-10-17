"""Evaluation engine and metrics for AdaptiveRL."""

from adaptive_rl.evaluation.evaluator import (
    EpisodeEvaluationRecord,
    Evaluator,
    MultiSeedEvaluationResult,
    SeedEvaluationSummary,
    compare_policies,
    evaluate_ppo_policy,
    evaluate_random_policy,
    run_obstacle_density_experiment,
)
from adaptive_rl.evaluation.generalization import (
    TEST_SEED_END,
    TEST_SEED_START,
    TRAIN_SEED_END,
    TRAIN_SEED_START,
    VALID_SPLITS,
    GeneralizationBenchmarkResult,
    GeneralizationGap,
    compute_generalization_gap,
    evaluate_generalization,
    get_split_seeds,
    is_seed_in_split,
    validate_split_seed,
)
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.evaluation.statistics import (
    DescriptiveMetrics,
    MetricStatistics,
    student_t_critical_value,
    summarize_descriptive_episodes,
    summarize_seed_values,
)

__all__ = [
    "DescriptiveMetrics",
    "EpisodeEvaluationRecord",
    "EvaluationMetrics",
    "Evaluator",
    "MetricStatistics",
    "MultiSeedEvaluationResult",
    "SeedEvaluationSummary",
    "GeneralizationBenchmarkResult",
    "GeneralizationGap",
    "TEST_SEED_END",
    "TEST_SEED_START",
    "TRAIN_SEED_END",
    "TRAIN_SEED_START",
    "VALID_SPLITS",
    "compare_policies",
    "compute_generalization_gap",
    "evaluate_generalization",
    "evaluate_ppo_policy",
    "evaluate_random_policy",
    "get_split_seeds",
    "is_seed_in_split",
    "run_obstacle_density_experiment",
    "student_t_critical_value",
    "summarize_descriptive_episodes",
    "summarize_seed_values",
    "validate_split_seed",
]
