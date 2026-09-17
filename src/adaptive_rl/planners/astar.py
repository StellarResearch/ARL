"""A* grid navigation planner for AdaptiveRL.

Implements the A* shortest-path algorithm on a discrete 2D grid using the
Manhattan distance heuristic. Deterministic: given the same start, goal, and
obstacle set, always produces the same path.

Reference: Hart, Nilsson, Raphael (1968). "A Formal Basis for the Heuristic
Determination of Minimum Cost Paths".
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

from adaptive_rl.planners.base import BasePlanner, GridCoordinate, PlannerResult

# 4-connected grid: UP, DOWN, LEFT, RIGHT
_NEIGHBORS: List[Tuple[int, int]] = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def _manhattan(a: GridCoordinate, b: GridCoordinate) -> int:
    """Compute the Manhattan distance between two grid coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class AStarPlanner(BasePlanner):
    """A* shortest-path planner for discrete 2D grid environments.

    Operates on the same grid representation used by :class:`GridWorldEnv`,
    treating the obstacle set as impassable cells. Uses the Manhattan distance
    heuristic, which is admissible and consistent on 4-connected grids.

    The planner is fully deterministic: given identical inputs it always returns
    the same path. It does not use any stochastic components.

    Example::

        from adaptive_rl.planners.astar import AStarPlanner

        planner = AStarPlanner()
        result = planner.plan(
            start=(0, 0),
            goal=(5, 4),
            obstacles={(2, 2), (3, 2)},
            width=6,
            height=5,
        )
        if result.success:
            print("Path length:", result.path_length)
    """

    def __init__(self, heuristic: str = "manhattan") -> None:
        """Initialize the A* planner.

        Args:
            heuristic: Heuristic function to use ('manhattan', 'euclidean', 'chebyshev').
                'manhattan': Admissible and consistent for 4-connected grids with unit step costs.
                    Guarantees optimal (shortest) paths with minimal node expansions.
                'euclidean': Admissible on 4-connected grids (h_euclid <= h_manhattan <= h*),
                    but less informed, expanding more search nodes.
                'chebyshev': Admissible on 8-connected grids; underestimates Manhattan distance on
                    4-connected grids and does not reflect 4-connected movement geometry.

        Raises:
            ValueError: If an unsupported heuristic name is given.
        """
        valid_heuristics = ("manhattan", "euclidean", "chebyshev")
        if heuristic not in valid_heuristics:
            raise ValueError(
                f"Unsupported heuristic '{heuristic}'. Valid options: {', '.join(valid_heuristics)}."
            )
        self._heuristic_name = heuristic

    @property
    def name(self) -> str:
        """Return the planner's canonical identifier."""
        return "astar"

    @property
    def heuristic_name(self) -> str:
        """Return the active heuristic function name."""
        return self._heuristic_name

    def _heuristic(self, a: GridCoordinate, b: GridCoordinate) -> float:
        """Compute the heuristic estimate from a to b."""
        if self._heuristic_name == "manhattan":
            return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))
        elif self._heuristic_name == "euclidean":
            import math

            return math.hypot(a[0] - b[0], a[1] - b[1])
        elif self._heuristic_name == "chebyshev":
            return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    @staticmethod
    def validate_path(
        path: List[GridCoordinate],
        obstacles: set,
        width: int,
        height: int,
    ) -> bool:
        """Verify that a path is continuous, within bounds, and collision-free.

        Args:
            path: Ordered list of (col, row) coordinates.
            obstacles: Set of obstacle cells.
            width: Grid width.
            height: Grid height.

        Returns:
            True if path is valid, False otherwise.
        """
        if not path:
            return False
        obs_set = set(obstacles) if obstacles is not None else set()
        for i, coord in enumerate(path):
            if not (0 <= coord[0] < width and 0 <= coord[1] < height):
                return False
            if coord in obs_set:
                return False
            if i > 0:
                prev = path[i - 1]
                dx = abs(coord[0] - prev[0])
                dy = abs(coord[1] - prev[1])
                # Must be adjacent 4-neighbor on unit grid
                if (dx + dy) != 1:
                    return False
        return True

    def plan(
        self,
        start: GridCoordinate,
        goal: GridCoordinate,
        obstacles: set,
        width: int,
        height: int,
    ) -> PlannerResult:
        """Compute the shortest obstacle-free path using A*.

        Args:
            start: Starting (col, row) coordinate within the grid.
            goal: Goal (col, row) coordinate within the grid.
            obstacles: Set of (col, row) impassable grid cells.
            width: Grid width (number of columns, 0-indexed up to width-1).
            height: Grid height (number of rows, 0-indexed up to height-1).

        Returns:
            PlannerResult with the shortest path if found, or a failure result
            if the goal is unreachable (blocked by obstacles or out of bounds).

        Raises:
            ValueError: If dimensions are non-positive, coordinates/obstacles are invalid,
                start/goal are out of grid bounds, or either lies on an obstacle cell.
        """
        # Dimension validation
        if width <= 0 or height <= 0:
            raise ValueError(f"Grid dimensions must be positive integers, got {width}x{height}.")

        # Coordinate format validation
        if not (
            isinstance(start, (tuple, list))
            and len(start) == 2
            and isinstance(start[0], int)
            and isinstance(start[1], int)
        ):
            raise ValueError(f"Start coordinate {start} must be a 2-tuple of integers.")
        if not (
            isinstance(goal, (tuple, list))
            and len(goal) == 2
            and isinstance(goal[0], int)
            and isinstance(goal[1], int)
        ):
            raise ValueError(f"Goal coordinate {goal} must be a 2-tuple of integers.")

        # Obstacle validation
        try:
            obs_set = set(obstacles) if obstacles is not None else set()
        except TypeError as exc:
            raise ValueError(
                f"Obstacles must be an iterable collection of coordinates: {exc}"
            ) from exc

        for obs in obs_set:
            if not (
                isinstance(obs, (tuple, list))
                and len(obs) == 2
                and isinstance(obs[0], int)
                and isinstance(obs[1], int)
            ):
                raise ValueError(
                    f"Invalid obstacle coordinate {obs}: expected 2-tuple of integers."
                )

        # Bounds and collision validation
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            raise ValueError(f"Start position {start} is outside grid bounds {width}x{height}.")
        if not (0 <= goal[0] < width and 0 <= goal[1] < height):
            raise ValueError(f"Goal position {goal} is outside grid bounds {width}x{height}.")
        if start in obs_set:
            raise ValueError(f"Start position {start} is occupied by an obstacle.")
        if goal in obs_set:
            raise ValueError(f"Goal position {goal} is occupied by an obstacle.")

        t_start = time.perf_counter()

        # Trivial case: already at goal
        if start == goal:
            elapsed = time.perf_counter() - t_start
            return PlannerResult(
                success=True,
                path=[start],
                path_length=0,
                planning_time_seconds=elapsed,
                nodes_explored=1,
            )

        # A* search
        # Priority queue entries: (f_score, tie_break, node)
        counter = 0  # tie-breaking counter for deterministic ordering
        open_heap: List[Tuple[float, int, GridCoordinate]] = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), counter, start))

        came_from: Dict[GridCoordinate, Optional[GridCoordinate]] = {start: None}
        g_score: Dict[GridCoordinate, float] = {start: 0.0}
        nodes_explored = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            nodes_explored += 1

            if current == goal:
                # Reconstruct path
                path = self._reconstruct_path(came_from, goal)
                elapsed = time.perf_counter() - t_start
                return PlannerResult(
                    success=True,
                    path=path,
                    path_length=len(path) - 1,  # edges, not nodes
                    planning_time_seconds=elapsed,
                    nodes_explored=nodes_explored,
                )

            for dx, dy in _NEIGHBORS:
                neighbor = (current[0] + dx, current[1] + dy)

                # Bounds check
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                # Obstacle check
                if neighbor in obs_set:
                    continue

                tentative_g = g_score[current] + 1  # unit cost grid

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f_score, counter, neighbor))
                    came_from[neighbor] = current

        # No path found
        elapsed = time.perf_counter() - t_start
        return PlannerResult(
            success=False,
            path=[],
            path_length=0,
            planning_time_seconds=elapsed,
            nodes_explored=nodes_explored,
            failure_reason=(
                f"No path found from {start} to {goal} on {width}x{height} grid "
                f"with {len(obs_set)} obstacle(s)."
            ),
        )

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[GridCoordinate, Optional[GridCoordinate]],
        goal: GridCoordinate,
    ) -> List[GridCoordinate]:
        """Walk back through came_from to reconstruct the full path from start to goal.

        Args:
            came_from: Dictionary mapping each node to its predecessor.
            goal: Terminal node.

        Returns:
            Ordered list of coordinates from start to goal (inclusive).
        """
        path: List[GridCoordinate] = []
        node: Optional[GridCoordinate] = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path
