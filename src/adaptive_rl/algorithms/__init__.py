"""Algorithm abstraction layer for AdaptiveRL."""

from typing import TYPE_CHECKING, Any

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.registry import (
    AlgorithmKind,
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmRegistryError,
    algorithm_registry,
    get_algorithm_factory,
    get_algorithm_metadata,
    list_algorithms,
    list_algorithms_by_kind,
    list_all_algorithm_metadata,
    register_algorithm,
    register_algorithm_alias,
    resolve_algorithm,
    restore_defaults,
)

if TYPE_CHECKING:
    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.algorithms.sac import SACAlgorithm

__all__ = [
    "AlgorithmKind",
    "AlgorithmMetadata",
    "AlgorithmRegistry",
    "AlgorithmRegistryError",
    "BaseAlgorithm",
    "PPOAlgorithm",
    "SACAlgorithm",
    "algorithm_registry",
    "get_algorithm_factory",
    "get_algorithm_metadata",
    "list_algorithms",
    "list_algorithms_by_kind",
    "list_all_algorithm_metadata",
    "register_algorithm",
    "register_algorithm_alias",
    "resolve_algorithm",
    "restore_defaults",
]

_LAZY_EXPORTS = {
    "PPOAlgorithm": "adaptive_rl.algorithms.ppo",
    "SACAlgorithm": "adaptive_rl.algorithms.sac",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        try:
            module = __import__(_LAZY_EXPORTS[name], fromlist=[name])
            return getattr(module, name)
        except ImportError as err:
            raise ImportError(
                f"{name} requires optional RL dependencies (gymnasium, stable-baselines3, torch). "
                f"Install them with: pip install 'adaptive-rl[rl]'"
            ) from err
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
