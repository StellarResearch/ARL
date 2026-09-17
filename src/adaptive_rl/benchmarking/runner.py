"""Benchmarking runner module for AdaptiveRL.

Re-exports from the benchmarking package init for backwards compatibility
and direct import convenience.
"""

from adaptive_rl.benchmarking import (
    AggregateStats,
    BenchmarkResult,
    BenchmarkRunner,
    ComparisonReport,
    SeedResult,
    compute_aggregate_stats,
)

__all__ = [
    "AggregateStats",
    "BenchmarkResult",
    "BenchmarkRunner",
    "ComparisonReport",
    "SeedResult",
    "compute_aggregate_stats",
]
