"""A* (A-Star) graph search algorithm for discrete grid environments."""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from adaptive_rl.planning.base import BasePlanner, PlannerPolicy, PlanningResult

Coordinate2D = Tuple[int, int]


class AStarPlanner(BasePlanner):
    """A* optimal path planning algorithm for discrete 2D grid environments.

    Computes the shortest collision-free path across a grid from start to goal
    using priority-guided search with admissible and consistent heuristics.
    """

    # 4-connected grid deltas: (dx, dy)
    DELTAS_4_CONNECTED: List[Tuple[int, int]] = [
        (0, -1),  # UP
        (0, 1),  # DOWN
        (-1, 0),  # LEFT
        (1, 0),  # RIGHT
    ]

    # 8-connected grid deltas (including diagonals)
    DELTAS_8_CONNECTED: List[Tuple[int, int]] = [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ]

    def __init__(
        self,
        width: int,
        height: int,
        obstacles: Optional[Set[Coordinate2D] | Sequence[Coordinate2D]] = None,
        allow_diagonal: bool = False,
        heuristic: str = "manhattan",
    ) -> None:
        """Initialize AStarPlanner.

        Args:
            width: Grid width (number of columns >= 2).
            height: Grid height (number of rows >= 2).
            obstacles: Set of static obstacle grid coordinates.
            allow_diagonal: If True, permits 8-directional diagonal movement.
            heuristic: Heuristic metric name ('manhattan', 'euclidean', 'chebyshev').
        """
        if width < 2 or height < 2:
            raise ValueError(f"Grid dimensions must be at least 2x2, got {width}x{height}")

        self.width = width
        self.height = height
        self.obstacles: Set[Coordinate2D] = set(obstacles) if obstacles else set()
        self.allow_diagonal = allow_diagonal
        self.heuristic_name = heuristic.lower()

        self._heuristic_fn: Callable[[Coordinate2D, Coordinate2D], float]
        if self.heuristic_name == "manhattan":
            self._heuristic_fn = self._heuristic_manhattan
        elif self.heuristic_name == "euclidean":
            self._heuristic_fn = self._heuristic_euclidean
        elif self.heuristic_name == "chebyshev":
            self._heuristic_fn = self._heuristic_chebyshev
        else:
            raise ValueError(
                f"Unknown heuristic '{heuristic}'. Expected 'manhattan', 'euclidean', or 'chebyshev'."
            )

    @property
    def name(self) -> str:
        """Name of the planner."""
        diag_suffix = "_8conn" if self.allow_diagonal else "_4conn"
        return f"AStar_{self.heuristic_name}{diag_suffix}"

    @staticmethod
    def _heuristic_manhattan(p1: Coordinate2D, p2: Coordinate2D) -> float:
        return float(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))

    @staticmethod
    def _heuristic_euclidean(p1: Coordinate2D, p2: Coordinate2D) -> float:
        return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def _heuristic_chebyshev(p1: Coordinate2D, p2: Coordinate2D) -> float:
        return float(max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])))

    def is_valid_coordinate(self, coord: Coordinate2D) -> bool:
        """Check whether coordinate is within grid bounds and not an obstacle."""
        x, y = coord
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return coord not in self.obstacles

    def get_neighbors(self, node: Coordinate2D) -> List[Tuple[Coordinate2D, float]]:
        """Return valid adjacent neighboring nodes and their transition step costs."""
        deltas = self.DELTAS_8_CONNECTED if self.allow_diagonal else self.DELTAS_4_CONNECTED
        neighbors: List[Tuple[Coordinate2D, float]] = []

        for dx, dy in deltas:
            neighbor = (node[0] + dx, node[1] + dy)
            if self.is_valid_coordinate(neighbor):
                cost = math.sqrt(dx * dx + dy * dy)
                neighbors.append((neighbor, cost))

        return neighbors

    def plan(
        self,
        start: Tuple[float, ...],
        goal: Tuple[float, ...],
    ) -> PlanningResult:
        """Compute optimal path from start to goal coordinates.

        Args:
            start: Discrete or rounded integer coordinates (x, y).
            goal: Target coordinates (gx, gy).

        Returns:
            PlanningResult containing optimal path, cost, and execution duration.
        """
        start_time = time.perf_counter()
        start_node: Coordinate2D = (int(round(start[0])), int(round(start[1])))
        goal_node: Coordinate2D = (int(round(goal[0])), int(round(goal[1])))

        if not self.is_valid_coordinate(start_node):
            return PlanningResult(
                success=False,
                path=[],
                cost=float("inf"),
                nodes_expanded=0,
                planning_time_sec=time.perf_counter() - start_time,
                metadata={"reason": "Start coordinate is invalid or blocked by obstacle."},
            )

        if not self.is_valid_coordinate(goal_node):
            return PlanningResult(
                success=False,
                path=[],
                cost=float("inf"),
                nodes_expanded=0,
                planning_time_sec=time.perf_counter() - start_time,
                metadata={"reason": "Goal coordinate is invalid or blocked by obstacle."},
            )

        if start_node == goal_node:
            return PlanningResult(
                success=True,
                path=[(float(start_node[0]), float(start_node[1]))],
                cost=0.0,
                nodes_expanded=0,
                planning_time_sec=time.perf_counter() - start_time,
            )

        # Priority queue entries: (f_score, counter, current_node)
        counter = 0
        open_set: List[Tuple[float, int, Coordinate2D]] = []
        heapq.heappush(
            open_set, (self._heuristic_fn(start_node, goal_node), counter, start_node)
        )

        came_from: Dict[Coordinate2D, Coordinate2D] = {}
        g_score: Dict[Coordinate2D, float] = {start_node: 0.0}
        closed_set: Set[Coordinate2D] = set()

        nodes_expanded = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            nodes_expanded += 1
            if current == goal_node:
                # Reconstruct path
                path: List[Tuple[float, ...]] = []
                curr: Optional[Coordinate2D] = current
                while curr is not None:
                    path.append((float(curr[0]), float(curr[1])))
                    curr = came_from.get(curr)
                path.reverse()

                duration = time.perf_counter() - start_time
                return PlanningResult(
                    success=True,
                    path=path,
                    cost=g_score[goal_node],
                    nodes_expanded=nodes_expanded,
                    planning_time_sec=duration,
                    metadata={"path_length_steps": len(path) - 1},
                )

            closed_set.add(current)
            current_g = g_score[current]

            for neighbor, step_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = current_g + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic_fn(neighbor, goal_node)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        # Open set exhausted without finding goal
        duration = time.perf_counter() - start_time
        return PlanningResult(
            success=False,
            path=[],
            cost=float("inf"),
            nodes_expanded=nodes_expanded,
            planning_time_sec=duration,
            metadata={"reason": "No path exists connecting start and goal."},
        )

    @classmethod
    def from_gridworld(cls, env: Any) -> AStarPlanner:
        """Create an AStarPlanner directly configured from a GridWorldEnv instance.

        Args:
            env: GridWorldEnv instance.

        Returns:
            AStarPlanner: Configured planner instance.
        """
        width = int(getattr(env, "width", 6))
        height = int(getattr(env, "height", 6))
        obstacles = set(getattr(env, "obstacles", set()))
        return cls(width=width, height=height, obstacles=obstacles, allow_diagonal=False)


class AStarPlannerPolicy(PlannerPolicy):
    """Adapter wrapping AStarPlanner as a Gymnasium-compatible policy for GridWorld.

    Maps current grid state to the optimal discrete action leading to the goal.
    """

    DELTA_TO_ACTION: Dict[Coordinate2D, int] = {
        (0, -1): 0,  # UP
        (0, 1): 1,  # DOWN
        (-1, 0): 2,  # LEFT
        (1, 0): 3,  # RIGHT
    }

    def __init__(
        self,
        planner: Optional[AStarPlanner] = None,
        env: Optional[Any] = None,
        width: int = 6,
        height: int = 6,
    ) -> None:
        """Initialize AStarPlannerPolicy.

        Args:
            planner: Optional pre-configured AStarPlanner.
            env: Optional GridWorldEnv instance to dynamically inspect state.
            width: Grid width (fallback if env not provided).
            height: Grid height (fallback if env not provided).
        """
        self.env = env
        self.width = int(getattr(env, "width", width))
        self.height = int(getattr(env, "height", height))
        self.planner: AStarPlanner = planner or (
            AStarPlanner.from_gridworld(env)
            if env is not None
            else AStarPlanner(width=self.width, height=self.height)
        )
        self._path: List[Tuple[float, ...]] = []
        self._planned_goal: Optional[Coordinate2D] = None
        self._planned_start: Optional[Coordinate2D] = None

    @property
    def name(self) -> str:
        return f"Policy_{self.planner.name}"

    def reset_policy(self) -> None:
        """Clear cached path plan at episode start."""
        self._path = []
        self._planned_goal = None
        self._planned_start = None
        if self.env is not None:
            self.planner = AStarPlanner.from_gridworld(self.env)

    def _extract_positions(self, observation: Any) -> Tuple[Coordinate2D, Coordinate2D]:
        """Extract current agent position and goal position."""
        if self.env is not None and hasattr(self.env, "agent_pos") and hasattr(self.env, "goal_pos"):
            agent_pos = (int(self.env.agent_pos[0]), int(self.env.agent_pos[1]))
            goal_pos = (int(self.env.goal_pos[0]), int(self.env.goal_pos[1]))
            return agent_pos, goal_pos

        # Denormalize from 4-dimensional Box observation:
        # [agent_x / (w - 1), agent_y / (h - 1), goal_x / (w - 1), goal_y / (h - 1)]
        ax = int(round(float(observation[0]) * (self.width - 1)))
        ay = int(round(float(observation[1]) * (self.height - 1)))
        gx = int(round(float(observation[2]) * (self.width - 1)))
        gy = int(round(float(observation[3]) * (self.height - 1)))
        return (ax, ay), (gx, gy)

    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[Any]]:
        """Generate the next discrete action leading along the optimal A* path.

        Args:
            observation: Current observation array.
            deterministic: Ignored for exact planning.

        Returns:
            Tuple[int, Optional[Any]]: (discrete_action, None)
        """
        curr_pos, goal_pos = self._extract_positions(observation)

        # Plan path if not yet generated or goal changed
        if not self._path or self._planned_goal != goal_pos:
            # Refresh obstacles from environment if available
            if self.env is not None:
                self.planner = AStarPlanner.from_gridworld(self.env)
            plan_res = self.planner.plan(curr_pos, goal_pos)
            if plan_res.success and len(plan_res.path) > 1:
                self._path = plan_res.path
                self._planned_goal = goal_pos
                self._planned_start = curr_pos
            else:
                # No valid path: default to UP
                return 0, None

        # Determine next waypoint along path from current position
        curr_float = (float(curr_pos[0]), float(curr_pos[1]))
        try:
            curr_idx = self._path.index(curr_float)
        except ValueError:
            # Agent diverged from path; re-plan from current position
            plan_res = self.planner.plan(curr_pos, goal_pos)
            if plan_res.success and len(plan_res.path) > 1:
                self._path = plan_res.path
                curr_idx = 0
            else:
                return 0, None

        if curr_idx + 1 < len(self._path):
            next_step = self._path[curr_idx + 1]
            dx = int(round(next_step[0] - curr_pos[0]))
            dy = int(round(next_step[1] - curr_pos[1]))
            action = self.DELTA_TO_ACTION.get((dx, dy), 0)
            return action, None

        # Reached goal
        return 0, None
