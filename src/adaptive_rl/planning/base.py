"""Base interfaces and data structures for classical motion planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PlanningResult:
    """Encapsulates the output of a motion planning query.

    Attributes:
        success: Whether a valid collision-free path was found to the goal.
        path: Ordered sequence of coordinates from start to goal.
        cost: Cumulative metric path length or graph traversal cost.
        nodes_expanded: Number of nodes or states explored during search.
        planning_time_sec: Duration of planning computation in seconds.
        metadata: Additional planner-specific metrics (e.g., iterations, rewirings).
    """

    success: bool
    path: List[Tuple[float, ...]]
    cost: float
    nodes_expanded: int
    planning_time_sec: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BasePlanner(ABC):
    """Abstract interface for classical path and motion planners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the planning algorithm."""
        pass

    @abstractmethod
    def plan(
        self,
        start: Tuple[float, ...],
        goal: Tuple[float, ...],
    ) -> PlanningResult:
        """Compute a collision-free path connecting start to goal.

        Args:
            start: Origin coordinates in environment space.
            goal: Target destination coordinates.

        Returns:
            PlanningResult: Planning outcome containing path and metrics.
        """
        pass


class PlannerPolicy(ABC):
    """Adapter interface allowing classical planners to act as Gymnasium policies.

    Exposes the `.predict(observation, deterministic)` method compatible with
    AdaptiveRL Evaluator and SB3 algorithm contracts.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the planner policy."""
        pass

    @abstractmethod
    def reset_policy(self) -> None:
        """Reset internal trajectory tracking state at the beginning of an episode."""
        pass

    @abstractmethod
    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> Tuple[Any, Optional[Any]]:
        """Generate the next environment action to follow the planned path.

        Args:
            observation: Current environment observation.
            deterministic: Whether action selection is deterministic (ignored for exact planners).

        Returns:
            Tuple[Any, Optional[Any]]: (action, internal_state)
        """
        pass
