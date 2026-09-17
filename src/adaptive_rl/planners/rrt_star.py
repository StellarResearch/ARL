"""RRT* (Rapidly-exploring Random Tree Star) planner for continuous 2D navigation.

Implements optimal kinodynamic-free sampling-based path planning in continuous 2D space
against arena wall boundaries and circular obstacles, matching ContinuousNavigation2DEnv.

Reference: Karaman & Frazzoli (2011). "Sampling-based algorithms for optimal motion planning".
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from adaptive_rl.planners.base import PlannerResult

ContinuousCoordinate = Tuple[float, float]
CircularObstacle = Tuple[float, float, float]  # (x, y, radius)


@dataclass
class RRTNode:
    """Node in the RRT* search tree."""

    x: float
    y: float
    cost: float = 0.0
    parent: Optional[RRTNode] = None
    children: List[RRTNode] = field(default_factory=list)

    @property
    def pos(self) -> ContinuousCoordinate:
        return (self.x, self.y)


def _euclidean_dist(p1: ContinuousCoordinate, p2: ContinuousCoordinate) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class RRTStarPlanner:
    """RRT* path planner for continuous 2D environments.

    Navigates a 2D rectangular arena with circular obstacles and perimeter walls,
    matching the collision geometry of :class:`ContinuousNavigation2DEnv`.

    Features:
    - Configurable step size, goal bias, and maximum iterations
    - Near-neighbor rewiring for asymptotic optimality
    - Interpolated continuous collision checking along path edges
    - Fully deterministic execution when seeded
    - Path extraction, arc-length calculation, and validity verification

    Example::

        planner = RRTStarPlanner(max_iterations=1000, step_size=0.5, seed=42)
        result = planner.plan(
            start=(2.0, 2.0),
            goal=(18.0, 18.0),
            arena_width=20.0,
            arena_height=20.0,
            obstacles=[(10.0, 10.0, 2.0)],
            agent_radius=0.3,
            goal_radius=0.8,
        )
    """

    def __init__(
        self,
        step_size: float = 0.5,
        max_iterations: int = 1500,
        goal_bias: float = 0.1,
        search_radius: float = 1.5,
        collision_resolution: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize RRT* planner.

        Args:
            step_size: Maximum extension distance per tree branch step.
            max_iterations: Maximum random samples to expand.
            goal_bias: Probability in [0, 1] of sampling directly at the goal.
            search_radius: Radius for finding near neighbors during rewiring.
            collision_resolution: Step size for line-segment collision checking.
            seed: Random seed for deterministic reproducibility.
        """
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if not (0.0 <= goal_bias <= 1.0):
            raise ValueError(f"goal_bias must be in [0.0, 1.0], got {goal_bias}")
        if search_radius <= 0:
            raise ValueError(f"search_radius must be positive, got {search_radius}")

        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.search_radius = search_radius
        self.collision_resolution = collision_resolution
        self.seed = seed

    @property
    def name(self) -> str:
        """Return the canonical planner identifier."""
        return "rrt_star"

    def is_point_valid(
        self,
        pos: ContinuousCoordinate,
        arena_width: float,
        arena_height: float,
        obstacles: List[CircularObstacle],
        agent_radius: float,
    ) -> bool:
        """Check if a point is within arena bounds and collision-free."""
        px, py = pos
        # Perimeter walls
        if (
            px - agent_radius < 0.0
            or px + agent_radius > arena_width
            or py - agent_radius < 0.0
            or py + agent_radius > arena_height
        ):
            return False

        # Circular obstacles
        for ox, oy, r in obstacles:
            dist = math.hypot(px - ox, py - oy)
            if dist <= (r + agent_radius):
                return False

        return True

    def is_segment_valid(
        self,
        p1: ContinuousCoordinate,
        p2: ContinuousCoordinate,
        arena_width: float,
        arena_height: float,
        obstacles: List[CircularObstacle],
        agent_radius: float,
    ) -> bool:
        """Interpolate along segment (p1, p2) to verify full collision freedom."""
        dist = _euclidean_dist(p1, p2)
        if dist == 0.0:
            return self.is_point_valid(p1, arena_width, arena_height, obstacles, agent_radius)

        n_steps = max(1, int(math.ceil(dist / self.collision_resolution)))
        dx = (p2[0] - p1[0]) / n_steps
        dy = (p2[1] - p1[1]) / n_steps

        for i in range(n_steps + 1):
            pt = (p1[0] + i * dx, p1[1] + i * dy)
            if not self.is_point_valid(pt, arena_width, arena_height, obstacles, agent_radius):
                return False
        return True

    def plan(
        self,
        start: ContinuousCoordinate,
        goal: ContinuousCoordinate,
        arena_width: float,
        arena_height: float,
        obstacles: Optional[List[CircularObstacle]] = None,
        agent_radius: float = 0.3,
        goal_radius: float = 0.8,
        seed_override: Optional[int] = None,
    ) -> PlannerResult:
        """Execute RRT* continuous path planning.

        Args:
            start: Start (x, y) coordinates.
            goal: Goal (x, y) coordinates.
            arena_width: Width of the 2D bounding arena.
            arena_height: Height of the 2D bounding arena.
            obstacles: List of circular obstacles as (ox, oy, radius).
            agent_radius: Collision margin around agent.
            goal_radius: Distance threshold to consider goal reached.
            seed_override: Optional seed overriding instance default.

        Returns:
            PlannerResult with success flag, coordinate path, length, and runtime.
        """
        obstacles = obstacles or []
        t_start = time.perf_counter()

        # Seed handling
        rng_seed = seed_override if seed_override is not None else self.seed
        rng = random.Random(rng_seed)

        # Validation
        if not self.is_point_valid(start, arena_width, arena_height, obstacles, agent_radius):
            return PlannerResult(
                success=False,
                failure_reason=f"Start position {start} collides with obstacles or bounds.",
                planning_time_seconds=time.perf_counter() - t_start,
            )
        if not self.is_point_valid(goal, arena_width, arena_height, obstacles, agent_radius):
            return PlannerResult(
                success=False,
                failure_reason=f"Goal position {goal} collides with obstacles or bounds.",
                planning_time_seconds=time.perf_counter() - t_start,
            )

        # Trivial check: start already in goal
        if _euclidean_dist(start, goal) <= goal_radius:
            elapsed = time.perf_counter() - t_start
            return PlannerResult(
                success=True,
                path=[start, goal],
                path_length=_euclidean_dist(start, goal),
                planning_time_seconds=elapsed,
                nodes_explored=1,
            )

        root = RRTNode(x=start[0], y=start[1], cost=0.0)
        tree: List[RRTNode] = [root]
        best_goal_node: Optional[RRTNode] = None
        best_goal_cost = float("inf")

        for _ in range(self.max_iterations):
            # Sample
            if rng.random() < self.goal_bias:
                x_rand = goal
            else:
                x_rand = (
                    rng.uniform(agent_radius, arena_width - agent_radius),
                    rng.uniform(agent_radius, arena_height - agent_radius),
                )

            # Nearest
            nearest = min(tree, key=lambda n: _euclidean_dist(n.pos, x_rand))

            # Steer
            dist = _euclidean_dist(nearest.pos, x_rand)
            if dist <= self.step_size:
                x_new = x_rand
            else:
                theta = math.atan2(x_rand[1] - nearest.y, x_rand[0] - nearest.x)
                x_new = (
                    nearest.x + self.step_size * math.cos(theta),
                    nearest.y + self.step_size * math.sin(theta),
                )

            # Segment check
            if not self.is_segment_valid(
                nearest.pos, x_new, arena_width, arena_height, obstacles, agent_radius
            ):
                continue

            # Find near neighbors for optimal parent selection
            near_nodes = [n for n in tree if _euclidean_dist(n.pos, x_new) <= self.search_radius]

            # Choose best parent
            min_node = nearest
            min_cost = nearest.cost + _euclidean_dist(nearest.pos, x_new)

            for near in near_nodes:
                d = _euclidean_dist(near.pos, x_new)
                if near.cost + d < min_cost:
                    if self.is_segment_valid(
                        near.pos, x_new, arena_width, arena_height, obstacles, agent_radius
                    ):
                        min_node = near
                        min_cost = near.cost + d

            new_node = RRTNode(x=x_new[0], y=x_new[1], cost=min_cost, parent=min_node)
            min_node.children.append(new_node)
            tree.append(new_node)

            # Rewire near neighbors
            for near in near_nodes:
                if near is min_node:
                    continue
                d = _euclidean_dist(new_node.pos, near.pos)
                if new_node.cost + d < near.cost:
                    if self.is_segment_valid(
                        new_node.pos, near.pos, arena_width, arena_height, obstacles, agent_radius
                    ):
                        # Rewire parent
                        if near.parent and near in near.parent.children:
                            near.parent.children.remove(near)
                        near.parent = new_node
                        new_node.children.append(near)
                        self._update_subtree_costs(near, new_node.cost + d)

            # Check goal reaching
            dist_to_goal = _euclidean_dist(new_node.pos, goal)
            if dist_to_goal <= goal_radius:
                total_cost = new_node.cost + dist_to_goal
                if total_cost < best_goal_cost:
                    if self.is_segment_valid(
                        new_node.pos, goal, arena_width, arena_height, obstacles, agent_radius
                    ):
                        best_goal_node = new_node
                        best_goal_cost = total_cost

        elapsed = time.perf_counter() - t_start

        if best_goal_node is None:
            return PlannerResult(
                success=False,
                path=[],
                path_length=0.0,
                planning_time_seconds=elapsed,
                nodes_explored=len(tree),
                failure_reason=f"No path found within {self.max_iterations} iterations.",
            )

        # Reconstruct path
        path: List[ContinuousCoordinate] = [goal]
        curr: Optional[RRTNode] = best_goal_node
        while curr is not None:
            path.append(curr.pos)
            curr = curr.parent
        path.reverse()

        # Compute total Euclidean path length
        path_length = 0.0
        for i in range(len(path) - 1):
            path_length += _euclidean_dist(path[i], path[i + 1])

        return PlannerResult(
            success=True,
            path=path,
            path_length=path_length,
            planning_time_seconds=elapsed,
            nodes_explored=len(tree),
        )

    def _update_subtree_costs(self, node: RRTNode, new_cost: float) -> None:
        """Recursively propagate cost update to descendants after rewiring."""
        delta = new_cost - node.cost
        node.cost = new_cost
        for child in node.children:
            self._update_subtree_costs(child, child.cost + delta)
