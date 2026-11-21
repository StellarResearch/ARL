"""Algorithm abstraction layer for AdaptiveRL."""

from adaptive_rl.algorithms.base import BaseAlgorithm

try:
    from adaptive_rl.algorithms.ppo import PPOAlgorithm
except ImportError:
    PPOAlgorithm = None  # type: ignore[assignment, misc]

try:
    from adaptive_rl.algorithms.random_policy import RandomPolicy
except ImportError:
    RandomPolicy = None  # type: ignore[assignment, misc]

try:
    from adaptive_rl.algorithms.sac import SACAlgorithm
except ImportError:
    SACAlgorithm = None  # type: ignore[assignment, misc]

from adaptive_rl.algorithms.registry import (
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmRegistryError,
    algorithm_registry,
    get_algorithm_factory,
    get_algorithm_metadata,
    list_algorithms,
    list_all_algorithm_metadata,
    load_algorithm_from_pretrained,
    register_algorithm,
)

__all__ = [
    "AlgorithmMetadata",
    "AlgorithmRegistry",
    "AlgorithmRegistryError",
    "BaseAlgorithm",
    "PPOAlgorithm",
    "RandomPolicy",
    "SACAlgorithm",
    "algorithm_registry",
    "get_algorithm_factory",
    "get_algorithm_metadata",
    "list_algorithms",
    "list_all_algorithm_metadata",
    "load_algorithm_from_pretrained",
    "register_algorithm",
]
