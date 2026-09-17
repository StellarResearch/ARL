"""Abstract base interface for deterministic navigation planners.

Planners are distinct from RL algorithms: they compute complete paths
from start to goal using world-model knowledge, not learned policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

# A coordinate is a (col, row) integer grid position or (x, y) float continuous position
GridCoordinate = Tuple[int, int]
ContinuousCoordinate = Tuple[float, float]
Coordinate = Union[GridCoordinate, ContinuousCoordinate]


@dataclass
class PlannerResult:
    """Structured result returned by a deterministic or sampling-based planner.

    Attributes:
        success: Whether a valid path was found.
        path: Sequence of coordinates from start to goal (inclusive).
              Empty list if no path was found.
        path_length: Total number of steps or arc length (0 if failed).
        planning_time_seconds: Wall-clock seconds spent planning.
        nodes_explored: Number of search nodes expanded (algorithm-dependent).
        failure_reason: Human-readable reason for failure if success is False.
    """

    success: bool
    path: Sequence[Coordinate] = field(default_factory=list)
    path_length: float = 0.0
    planning_time_seconds: float = 0.0
    nodes_explored: int = 0
    failure_reason: Optional[str] = None

    def is_collision_free(self, obstacles: Union[set, List[Any]]) -> bool:
        """Verify that no path cell falls in the obstacle set.

        Args:
            obstacles: Set of obstacle coordinates or list of obstacle geometries.

        Returns:
            True if every path cell is free of obstacles, False otherwise.
        """
        obs_set = set(obstacles) if isinstance(obstacles, list) else obstacles
        return all(cell not in obs_set for cell in self.path)

    def is_valid(
        self,
        start: Coordinate,
        goal: Coordinate,
        obstacles: Union[set, List[Any]],
    ) -> bool:
        """Full validity check: path connects start to goal and is collision-free.

        Args:
            start: Expected starting coordinate.
            goal: Expected goal coordinate.
            obstacles: Obstacle set for collision checking.

        Returns:
            True if path is non-empty, starts at start, ends at goal, and is obstacle-free.
        """
        if not self.success or not self.path:
            return False
        if self.path[0] != start or self.path[-1] != goal:
            return False
        return self.is_collision_free(obstacles)


class BasePlanner(ABC):
    """Abstract interface for deterministic grid navigation planners.

    Implementations must be deterministic given the same inputs (same start,
    goal, obstacles, and seed where applicable).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize base planner with optional random seed."""
        self.seed = seed

    @abstractmethod
    def plan(
        self,
        start: GridCoordinate,
        goal: GridCoordinate,
        obstacles: set,
        width: int,
        height: int,
    ) -> PlannerResult:
        """Compute a path from start to goal on a grid.

        Args:
            start: Starting (col, row) coordinate.
            goal: Goal (col, row) coordinate.
            obstacles: Set of (col, row) impassable obstacle coordinates.
            width: Grid width in columns.
            height: Grid height in rows.

        Returns:
            PlannerResult containing path, length, planning time, and status.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the planner's canonical identifier string."""
        pass
