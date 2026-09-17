"""Abstract base classes for reinforcement learning algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple


class BaseAlgorithm(ABC):
    """Abstract interface for algorithm wrappers in AdaptiveRL.

    Concrete implementations wrap algorithms such as Stable-Baselines3 PPO and SAC
    to provide a unified training, prediction, and serialization contract.
    Concrete algorithm wrappers provide the train, predict, and persistence operations.
    """

    @abstractmethod
    def train(self, total_timesteps: int, callback: Any = None) -> None:
        """Train the algorithm for the specified number of timesteps."""
        pass

    @abstractmethod
    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[Any, Any]:
        """Generate an action given an environment observation.

        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic action selection.

        Returns:
            Tuple[Any, Any]: (action, internal_state)
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save algorithm weights and hyperparameters to disk."""
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load algorithm weights and hyperparameters from disk."""
        pass
