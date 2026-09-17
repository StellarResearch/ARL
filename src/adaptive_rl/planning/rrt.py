"""RRT and RRT* (Rapidly-exploring Random Trees) sampling-based motion planners."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.planning.base import BasePlanner, PlannerPolicy, PlanningResult


@dataclass
class RRTNode:
    """Node in an RRT tree representing a spatial configuration."""

    point: np.ndarray
    parent: Optional[RRTNode] = None
    cost: float = 0.0


class RRTPlanner(BasePlanner):
    """Rapidly-exploring Random Tree (RRT) for continuous continuous state spaces.

    Supports arbitrary continuous dimensions (e.g. 2D continuous navigation,
    3D drone translation) with spherical/circular obstacle collision checking.
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        obstacles: Optional[Sequence[Tuple[float, ...]]] = None,
        step_size: float = 0.8,
        max_iterations: int = 2500,
        goal_bias: float = 0.15,
        goal_tolerance: float = 0.8,
        agent_radius: float = 0.3,
        collision_resolution: float = 0.1,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Initialize RRTPlanner.

        Args:
            bounds: Spatial lower and upper bounds per dimension [(min_0, max_0), ...].
            obstacles: Sequence of obstacle tuples: (x, y, r) for 2D or (x, y, z, r) for 3D.
            step_size: Maximum extension distance per tree branch step.
            max_iterations: Maximum sampling iterations before termination.
            goal_bias: Probability of sampling the exact goal coordinate.
            goal_tolerance: Distance to goal considered successful arrival.
            agent_radius: Agent collision margin added to obstacle radii.
            collision_resolution: Step size for discretized edge collision checking.
            rng_seed: Optional seed for deterministic pseudo-random sampling.
        """
        if not bounds:
            raise ValueError("Bounds must have at least one dimension.")
        if step_size <= 0.0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if max_iterations < 10:
            raise ValueError(f"max_iterations must be >= 10, got {max_iterations}")

        self.bounds = [tuple(b) for b in bounds]
        self.dimension = len(self.bounds)
        self.obstacles: List[Tuple[float, ...]] = list(obstacles) if obstacles else []
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.goal_bias = float(goal_bias)
        self.goal_tolerance = float(goal_tolerance)
        self.agent_radius = float(agent_radius)
        self.collision_resolution = float(collision_resolution)
        self._rng = np.random.default_rng(rng_seed)

    @property
    def name(self) -> str:
        return f"RRT_{self.dimension}D"

    def is_point_valid(self, point: np.ndarray) -> bool:
        """Check whether point lies within bounds and clears all obstacles."""
        for dim in range(self.dimension):
            min_b, max_b = self.bounds[dim]
            if point[dim] < min_b + self.agent_radius or point[dim] > max_b - self.agent_radius:
                return False

        for obs in self.obstacles:
            center = np.array(obs[: self.dimension], dtype=np.float64)
            obs_radius = float(obs[self.dimension])
            dist = float(np.linalg.norm(point - center))
            if dist < (obs_radius + self.agent_radius):
                return False

        return True

    def is_edge_valid(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check whether straight-line trajectory between p1 and p2 is collision-free."""
        dist = float(np.linalg.norm(p2 - p1))
        if dist < 1e-6:
            return self.is_point_valid(p1)

        num_steps = max(int(math.ceil(dist / self.collision_resolution)), 2)
        alphas = np.linspace(0.0, 1.0, num_steps)
        for alpha in alphas:
            pt = (1.0 - alpha) * p1 + alpha * p2
            if not self.is_point_valid(pt):
                return False

        return True

    def sample_point(self, goal: np.ndarray) -> np.ndarray:
        """Sample random coordinate within domain with goal biasing."""
        if self._rng.random() < self.goal_bias:
            return goal.copy()

        point = np.zeros(self.dimension, dtype=np.float64)
        for dim in range(self.dimension):
            min_b, max_b = self.bounds[dim]
            point[dim] = self._rng.uniform(min_b + self.agent_radius, max_b - self.agent_radius)
        return point

    @staticmethod
    def nearest_node(tree: List[RRTNode], target: np.ndarray) -> RRTNode:
        """Find the existing node in the tree closest to target point."""
        best_node = tree[0]
        best_dist = float(np.linalg.norm(best_node.point - target))
        for node in tree[1:]:
            dist = float(np.linalg.norm(node.point - target))
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return best_node

    def steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Extend from from_point towards to_point by at most step_size."""
        vec = to_point - from_point
        dist = float(np.linalg.norm(vec))
        if dist <= self.step_size:
            return to_point.copy()
        direction = vec / dist
        return from_point + direction * self.step_size

    def plan(
        self,
        start: Tuple[float, ...],
        goal: Tuple[float, ...],
    ) -> PlanningResult:
        """Plan path from start to goal coordinates using basic RRT."""
        start_time = time.perf_counter()
        start_pt = np.array(start[: self.dimension], dtype=np.float64)
        goal_pt = np.array(goal[: self.dimension], dtype=np.float64)

        if not self.is_point_valid(start_pt):
            return PlanningResult(
                success=False,
                path=[],
                cost=float("inf"),
                nodes_expanded=0,
                planning_time_sec=time.perf_counter() - start_time,
                metadata={"reason": "Start position in collision or out of bounds."},
            )

        if not self.is_point_valid(goal_pt):
            return PlanningResult(
                success=False,
                path=[],
                cost=float("inf"),
                nodes_expanded=0,
                planning_time_sec=time.perf_counter() - start_time,
                metadata={"reason": "Goal position in collision or out of bounds."},
            )

        # Check trivial direct path
        if float(np.linalg.norm(start_pt - goal_pt)) <= self.goal_tolerance:
            return PlanningResult(
                success=True,
                path=[tuple(float(c) for c in start_pt), tuple(float(c) for c in goal_pt)],
                cost=float(np.linalg.norm(start_pt - goal_pt)),
                nodes_expanded=1,
                planning_time_sec=time.perf_counter() - start_time,
            )

        root = RRTNode(point=start_pt, parent=None, cost=0.0)
        tree: List[RRTNode] = [root]

        for iteration in range(1, self.max_iterations + 1):
            sample = self.sample_point(goal_pt)
            nearest = self.nearest_node(tree, sample)
            new_pt = self.steer(nearest.point, sample)

            if self.is_edge_valid(nearest.point, new_pt):
                step_dist = float(np.linalg.norm(new_pt - nearest.point))
                new_node = RRTNode(
                    point=new_pt,
                    parent=nearest,
                    cost=nearest.cost + step_dist,
                )
                tree.append(new_node)

                # Check if new node reaches goal tolerance
                dist_to_goal = float(np.linalg.norm(new_pt - goal_pt))
                if dist_to_goal <= self.goal_tolerance:
                    if self.is_edge_valid(new_pt, goal_pt):
                        goal_node = RRTNode(
                            point=goal_pt,
                            parent=new_node,
                            cost=new_node.cost + dist_to_goal,
                        )
                        tree.append(goal_node)
                        path = self._extract_path(goal_node)
                        duration = time.perf_counter() - start_time
                        return PlanningResult(
                            success=True,
                            path=path,
                            cost=goal_node.cost,
                            nodes_expanded=len(tree),
                            planning_time_sec=duration,
                            metadata={"iterations": iteration},
                        )

        # Max iterations reached without finding path
        duration = time.perf_counter() - start_time
        return PlanningResult(
            success=False,
            path=[],
            cost=float("inf"),
            nodes_expanded=len(tree),
            planning_time_sec=duration,
            metadata={"reason": "Max iterations reached without reaching goal."},
        )

    @staticmethod
    def _extract_path(goal_node: RRTNode) -> List[Tuple[float, ...]]:
        """Backtrack through parent pointers to construct start-to-goal path."""
        path: List[Tuple[float, ...]] = []
        curr: Optional[RRTNode] = goal_node
        while curr is not None:
            path.append(tuple(float(c) for c in curr.point))
            curr = curr.parent
        path.reverse()
        return path


class RRTStarPlanner(RRTPlanner):
    """RRT* asymptotically optimal motion planner.

    Extends standard RRT with neighborhood cost optimization and tree rewiring,
    guaranteeing convergence to the optimal collision-free trajectory as iterations grow.
    """

    def __init__(
        self,
        bounds: Sequence[Tuple[float, float]],
        obstacles: Optional[Sequence[Tuple[float, ...]]] = None,
        step_size: float = 0.8,
        max_iterations: int = 2500,
        goal_bias: float = 0.15,
        goal_tolerance: float = 0.8,
        agent_radius: float = 0.3,
        collision_resolution: float = 0.1,
        rewire_radius: float = 2.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        """Initialize RRTStarPlanner.

        Args:
            bounds: Spatial bounds per dimension.
            obstacles: Obstacle center and radius tuples.
            step_size: Tree branch expansion distance.
            max_iterations: Maximum sampling iterations.
            goal_bias: Probability of goal sampling.
            goal_tolerance: Arrival distance threshold.
            agent_radius: Collision safety buffer.
            collision_resolution: Step resolution for edge checking.
            rewire_radius: Search radius for finding near neighbors during rewiring.
            rng_seed: Optional random generator seed.
        """
        super().__init__(
            bounds=bounds,
            obstacles=obstacles,
            step_size=step_size,
            max_iterations=max_iterations,
            goal_bias=goal_bias,
            goal_tolerance=goal_tolerance,
            agent_radius=agent_radius,
            collision_resolution=collision_resolution,
            rng_seed=rng_seed,
        )
        self.rewire_radius = float(rewire_radius)

    @property
    def name(self) -> str:
        return f"RRTStar_{self.dimension}D"

    def find_near_nodes(
        self, tree: List[RRTNode], target: np.ndarray, radius: float
    ) -> List[Tuple[RRTNode, float]]:
        """Return all nodes within radius of target alongside their distance."""
        near: List[Tuple[RRTNode, float]] = []
        for node in tree:
            dist = float(np.linalg.norm(node.point - target))
            if dist <= radius:
                near.append((node, dist))
        return near

    def plan(
        self,
        start: Tuple[float, ...],
        goal: Tuple[float, ...],
    ) -> PlanningResult:
        """Compute asymptotically optimal path using RRT*."""
        start_time = time.perf_counter()
        start_pt = np.array(start[: self.dimension], dtype=np.float64)
        goal_pt = np.array(goal[: self.dimension], dtype=np.float64)

        if not self.is_point_valid(start_pt):
            return PlanningResult(
                success=False,
                path=[],
                cost=float("inf"),
                nodes_expanded=0,
                planning_time_sec=time.perf_counter() - start_time,
                metadata={"reason": "Start position in collision or out of bounds."},
            )

        if not self.is_point_valid(goal_pt):
            return PlanningResult(
                success=False,
                path=[],
                cost=float("inf"),
                nodes_expanded=0,
                planning_time_sec=time.perf_counter() - start_time,
                metadata={"reason": "Goal position in collision or out of bounds."},
            )

        root = RRTNode(point=start_pt, parent=None, cost=0.0)
        tree: List[RRTNode] = [root]
        best_goal_node: Optional[RRTNode] = None
        best_cost = float("inf")
        rewire_count = 0

        for iteration in range(1, self.max_iterations + 1):
            sample = self.sample_point(goal_pt)
            nearest = self.nearest_node(tree, sample)
            new_pt = self.steer(nearest.point, sample)

            if not self.is_edge_valid(nearest.point, new_pt):
                continue

            # Dynamic rewiring radius formula: r_star = min(r_max, gamma * (log n / n)^(1/d))
            n_nodes = len(tree)
            gamma = 2.5 * self.step_size
            dyn_radius = min(
                self.rewire_radius,
                gamma * math.pow(math.log(max(n_nodes, 2)) / n_nodes, 1.0 / self.dimension),
            )
            dyn_radius = max(dyn_radius, self.step_size)

            near_nodes = self.find_near_nodes(tree, new_pt, dyn_radius)

            # 1. Choose best parent among near nodes to minimize path cost
            min_parent = nearest
            min_cost = nearest.cost + float(np.linalg.norm(new_pt - nearest.point))

            for near_node, dist in near_nodes:
                if (near_node.cost + dist) < min_cost:
                    if self.is_edge_valid(near_node.point, new_pt):
                        min_parent = near_node
                        min_cost = near_node.cost + dist

            new_node = RRTNode(point=new_pt, parent=min_parent, cost=min_cost)
            tree.append(new_node)

            # 2. Rewire near neighbors if routing through new_node reduces their cost
            for near_node, dist in near_nodes:
                if (new_node.cost + dist) < near_node.cost:
                    if self.is_edge_valid(new_node.point, near_node.point):
                        near_node.parent = new_node
                        near_node.cost = new_node.cost + dist
                        rewire_count += 1

            # 3. Check goal reachability
            dist_to_goal = float(np.linalg.norm(new_pt - goal_pt))
            if dist_to_goal <= self.goal_tolerance:
                if self.is_edge_valid(new_pt, goal_pt):
                    potential_cost = new_node.cost + dist_to_goal
                    if potential_cost < best_cost:
                        best_cost = potential_cost
                        best_goal_node = RRTNode(
                            point=goal_pt, parent=new_node, cost=potential_cost
                        )

            # Early exit if a high quality path is found and refined
            if best_goal_node is not None and iteration > (self.max_iterations // 2):
                break

        duration = time.perf_counter() - start_time
        if best_goal_node is not None:
            path = self._extract_path(best_goal_node)
            return PlanningResult(
                success=True,
                path=path,
                cost=best_goal_node.cost,
                nodes_expanded=len(tree),
                planning_time_sec=duration,
                metadata={"iterations": iteration, "rewirings": rewire_count},
            )

        return PlanningResult(
            success=False,
            path=[],
            cost=float("inf"),
            nodes_expanded=len(tree),
            planning_time_sec=duration,
            metadata={"reason": "Goal could not be reached within iteration budget."},
        )

    @classmethod
    def from_navigation_env(
        cls,
        env: Any,
        max_iterations: int = 2500,
        rng_seed: Optional[int] = None,
    ) -> RRTStarPlanner:
        """Construct an RRTStarPlanner calibrated for ContinuousNavigation2DEnv.

        Args:
            env: ContinuousNavigation2DEnv instance.
            max_iterations: Search iteration budget.
            rng_seed: Reproducible sampling seed.

        Returns:
            RRTStarPlanner: Calibrated 2D planner instance.
        """
        width = float(getattr(env, "arena_width", 20.0))
        height = float(getattr(env, "arena_height", 20.0))
        bounds = [(0.0, width), (0.0, height)]

        raw_obstacles = getattr(env, "obstacles", [])
        obstacles: List[Tuple[float, ...]] = [
            (float(o[0]), float(o[1]), float(o[2])) for o in raw_obstacles
        ]

        agent_radius = float(getattr(env, "agent_radius", 0.3))
        goal_radius = float(getattr(env, "goal_radius", 0.8))

        return cls(
            bounds=bounds,
            obstacles=obstacles,
            step_size=0.8,
            max_iterations=max_iterations,
            goal_bias=0.2,
            goal_tolerance=goal_radius,
            agent_radius=agent_radius,
            collision_resolution=0.1,
            rewire_radius=2.5,
            rng_seed=rng_seed,
        )

    @classmethod
    def from_drone_env(
        cls,
        env: Any,
        max_iterations: int = 3000,
        rng_seed: Optional[int] = None,
    ) -> RRTStarPlanner:
        """Construct an RRTStarPlanner calibrated for 3D Drone environments.

        Args:
            env: Drone3DEnv instance.
            max_iterations: Search iteration budget.
            rng_seed: Reproducible sampling seed.

        Returns:
            RRTStarPlanner: Calibrated 3D planner instance.
        """
        bounds_raw = getattr(env, "bounds", (20.0, 20.0, 15.0))
        bounds = [(0.0, float(b)) for b in bounds_raw]

        raw_obstacles = getattr(env, "obstacles", [])
        obstacles: List[Tuple[float, ...]] = []
        for obs in raw_obstacles:
            if hasattr(obs, "center") and hasattr(obs, "radius"):
                obstacles.append(
                    (
                        float(obs.center[0]),
                        float(obs.center[1]),
                        float(obs.center[2]),
                        float(obs.radius),
                    )
                )

        collision_radius = float(getattr(env, "collision_radius", 0.3))
        target_radius = float(getattr(env, "target_radius", 0.8))

        return cls(
            bounds=bounds,
            obstacles=obstacles,
            step_size=1.2,
            max_iterations=max_iterations,
            goal_bias=0.2,
            goal_tolerance=target_radius,
            agent_radius=collision_radius,
            collision_resolution=0.15,
            rewire_radius=3.0,
            rng_seed=rng_seed,
        )


class RRTPlannerPolicy(PlannerPolicy):
    """Gymnasium policy adapter steering continuous navigation agents along planned RRT waypoints."""

    def __init__(
        self,
        planner: Optional[RRTPlanner] = None,
        env: Optional[Any] = None,
        waypoint_tolerance: float = 0.5,
    ) -> None:
        """Initialize RRTPlannerPolicy.

        Args:
            planner: Motion planner used to generate path trajectories.
            env: Reference continuous environment.
            waypoint_tolerance: Radius around waypoint to advance to next target.
        """
        self.env = env
        self.planner: RRTPlanner = planner or (
            RRTStarPlanner.from_navigation_env(env)
            if env is not None
            else RRTStarPlanner(bounds=[(0.0, 20.0), (0.0, 20.0)])
        )
        self.waypoint_tolerance = float(waypoint_tolerance)
        self._path: List[Tuple[float, ...]] = []
        self._current_waypoint_idx = 0

    @property
    def name(self) -> str:
        return f"Policy_{self.planner.name}"

    def reset_policy(self) -> None:
        """Clear cached trajectory waypoints at episode start."""
        self._path = []
        self._current_waypoint_idx = 0
        if self.env is not None and hasattr(self.env, "obstacles"):
            self.planner = RRTStarPlanner.from_navigation_env(self.env)

    def _extract_positions(self, observation: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Extract agent and goal coordinates from environment or observation."""
        if (
            self.env is not None
            and hasattr(self.env, "agent_pos")
            and hasattr(self.env, "goal_pos")
        ):
            return np.array(self.env.agent_pos, dtype=np.float64), np.array(
                self.env.goal_pos, dtype=np.float64
            )

        # Denormalize from 14-dim observation:
        # [x/w, y/h, gx/w, gy/h, ...]
        w = float(self.planner.bounds[0][1])
        h = float(self.planner.bounds[1][1])
        curr = np.array([float(observation[0]) * w, float(observation[1]) * h], dtype=np.float64)
        goal = np.array([float(observation[2]) * w, float(observation[3]) * h], dtype=np.float64)
        return curr, goal

    def predict(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Generate continuous velocity control [vx, vy] towards the next path waypoint.

        Args:
            observation: Current continuous observation array.
            deterministic: Ignored for geometric path following.

        Returns:
            Tuple[np.ndarray, Optional[Any]]: (action_vector, None)
        """
        curr_pos, goal_pos = self._extract_positions(observation)

        # Plan trajectory if not yet computed
        if not self._path:
            if self.env is not None and hasattr(self.env, "obstacles"):
                self.planner = RRTStarPlanner.from_navigation_env(self.env)
            plan_res = self.planner.plan(tuple(curr_pos), tuple(goal_pos))
            if plan_res.success and len(plan_res.path) > 1:
                self._path = plan_res.path
                self._current_waypoint_idx = 1
            else:
                # Direct fallback vector towards goal
                direction = goal_pos - curr_pos
                dist = float(np.linalg.norm(direction))
                action = (direction / dist) if dist > 1e-6 else np.zeros(2, dtype=np.float32)
                return np.clip(action, -1.0, 1.0).astype(np.float32), None

        # Advance along waypoint list
        while self._current_waypoint_idx < len(self._path):
            target_pt = np.array(self._path[self._current_waypoint_idx], dtype=np.float64)
            dist_to_target = float(np.linalg.norm(target_pt - curr_pos))
            if dist_to_target <= self.waypoint_tolerance:
                self._current_waypoint_idx += 1
            else:
                break

        # Compute velocity unit vector toward active waypoint
        if self._current_waypoint_idx < len(self._path):
            target_pt = np.array(self._path[self._current_waypoint_idx], dtype=np.float64)
        else:
            target_pt = goal_pos

        direction = target_pt - curr_pos
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            action = direction / norm
        else:
            action = np.zeros(len(curr_pos), dtype=np.float64)

        return np.clip(action, -1.0, 1.0).astype(np.float32), None
