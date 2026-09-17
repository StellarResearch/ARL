"""Authoritative evaluation seeding protocol for AdaptiveRL.

Guarantees deterministic, synchronized, and fair evaluation seed derivation
across reinforcement learning policies and classical deterministic/sampling planners.
"""

from __future__ import annotations

from typing import List, Optional


def derive_evaluation_seed(experiment_seed: int, episode_index: int) -> int:
    """Deterministically derive the environment reset seed for an evaluation episode.

    Contract:
        For a given (experiment_seed, episode_index), this function returns the EXACT same
        integer seed regardless of whether the evaluating entity is an RL policy,
        a discrete grid planner (A*), or a continuous motion planner (RRT*).

    Args:
        experiment_seed: Base experiment seed configured for the run.
        episode_index: 0-indexed evaluation episode number (0 <= episode_index).

    Returns:
        Deterministic integer seed for environment.reset(seed=...).
    """
    if episode_index < 0:
        raise ValueError(f"episode_index must be non-negative, got {episode_index}")
    return int(experiment_seed) + int(episode_index)


def generate_evaluation_seeds(experiment_seed: int, num_episodes: int) -> List[int]:
    """Generate the full list of evaluation environment seeds for an experiment.

    Args:
        experiment_seed: Base experiment seed.
        num_episodes: Total number of evaluation episodes.

    Returns:
        List of integer seeds of length `num_episodes`.
    """
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    return [derive_evaluation_seed(experiment_seed, ep) for ep in range(num_episodes)]


def derive_planner_seed(
    experiment_seed: Optional[int],
    episode_index: int,
    planner_seed: Optional[int] = None,
) -> Optional[int]:
    """Derive deterministic internal randomness seed for sampling-based planners.

    Separates planner-internal sampling randomness (e.g. RRT* tree expansion)
    from the shared environment seed (e.g. obstacle & start/goal generation).
    Preserves algorithm-specific randomness without altering the shared environment layout.

    Args:
        experiment_seed: Base experiment seed (or None if unseeded).
        episode_index: 0-indexed evaluation episode number.
        planner_seed: Optional explicit seed configured on the planner instance.

    Returns:
        Integer seed for planner internal RNG, or None if unseeded.
    """
    if episode_index < 0:
        raise ValueError(f"episode_index must be non-negative, got {episode_index}")

    if planner_seed is not None:
        return int(planner_seed) + int(episode_index)

    if experiment_seed is not None:
        # Deterministically derive a distinct seed space from environment seed
        return (int(experiment_seed) * 10007 + int(episode_index)) & 0x7FFFFFFF

    return None
