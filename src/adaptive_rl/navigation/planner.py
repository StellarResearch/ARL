"""Deterministic 3D A* Path Planner with Spherical Obstacle Inflation and Line-of-Sight Simplification.

Features:
- Bounded 3D spatial grid planning (26-connectivity).
- Exact analytical obstacle inflation by vehicle collision radius + configurable safety margin.
- Start and destination collision validation with clean failure classifications.
- Guaranteed collision-free line-of-sight path shortcutting (string pulling).
- Deterministic path length, minimum obstacle clearance, and execution metrics.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D


@dataclass
class PlanningResult:
    """Outcome and trajectory of a 3D path planning request."""

    status: str  # "SUCCESS", "NO_PATH", "BLOCKED_START", "BLOCKED_GOAL", "OUT_OF_BOUNDS"
    waypoints: List[np.ndarray] = field(default_factory=list)
    raw_path_length: float = 0.0
    path_length: float = 0.0
    min_clearance: float = float("inf")
    planning_time_ms: float = 0.0
    nodes_expanded: int = 0
    raw_nodes_count: int = 0
    simplified_nodes_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "waypoints": [[round(float(c), 4) for c in w] for w in self.waypoints],
            "path_length": round(self.path_length, 4),
            "raw_path_length": round(self.raw_path_length, 4),
            "min_clearance": (
                round(self.min_clearance, 4) if math.isfinite(self.min_clearance) else -1.0
            ),
            "planning_time_ms": round(self.planning_time_ms, 2),
            "nodes_expanded": self.nodes_expanded,
            "raw_nodes_count": self.raw_nodes_count,
            "simplified_nodes_count": self.simplified_nodes_count,
        }


def _extract_obstacle(obs: Any) -> Tuple[np.ndarray, float]:
    """Extract (center, radius) from an ObstacleSphere3D, DynamicObstacleSphere3D, or array."""
    if hasattr(obs, "center") and hasattr(obs, "radius"):
        return np.asarray(obs.center, dtype=np.float64), float(obs.radius)
    if hasattr(obs, "position") and hasattr(obs, "radius"):
        return np.asarray(obs.position, dtype=np.float64), float(obs.radius)
    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim == 1 and len(arr) == 4:
        return arr[:3], float(arr[3])
    raise ValueError(f"Cannot extract obstacle geometry from {obs!r}")


class AStarPlanner3D:
    """Deterministic 3D grid-based A* path planner for aerial quadrotors."""

    # 26-connected discrete neighborhood offsets
    _NEIGHBOR_OFFSETS: List[Tuple[int, int, int]] = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0),
        resolution: float = 1.0,
        vehicle_radius: float = 0.8,
        safety_margin: float = 0.4,
    ) -> None:
        """Initialize the 3D A* path planner.

        Args:
            bounds: Dimensions (Bx, By, Bz) of the 3D arena.
            resolution: Grid cell edge length in meters.
            vehicle_radius: Quadrotor bounding collision radius in meters.
            safety_margin: Additional safety clearance buffer in meters.
        """
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.resolution = float(resolution)
        self.vehicle_radius = float(vehicle_radius)
        self.safety_margin = float(safety_margin)
        self.total_inflation_margin = self.vehicle_radius + self.safety_margin

    def is_point_valid(
        self,
        point: np.ndarray,
        obstacles: Sequence[ObstacleSphere3D],
    ) -> Tuple[bool, float]:
        """Check whether a 3D continuous point satisfies boundaries and obstacle inflation.

        Returns:
            (is_valid, min_clearance_to_obstacle_surface)
        """
        p = np.asarray(point, dtype=np.float64)

        # Arena boundary containment check (must maintain vehicle radius inside arena)
        if (
            p[0] < self.vehicle_radius
            or p[0] > self.bounds[0] - self.vehicle_radius
            or p[1] < self.vehicle_radius
            or p[1] > self.bounds[1] - self.vehicle_radius
            or p[2] < self.vehicle_radius
            or p[2] > self.bounds[2] - self.vehicle_radius
        ):
            return False, 0.0

        min_clear = float("inf")
        for obs in obstacles:
            center, radius = _extract_obstacle(obs)
            dist_to_center = float(np.linalg.norm(p - center))
            clearance_to_surface = dist_to_center - (radius + self.vehicle_radius)
            if clearance_to_surface < min_clear:
                min_clear = clearance_to_surface
            # Collision if clearance is below safety margin
            if dist_to_center < radius + self.total_inflation_margin:
                return False, clearance_to_surface

        return True, min_clear

    def is_segment_valid(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        obstacles: Sequence[ObstacleSphere3D],
    ) -> Tuple[bool, float]:
        """Check whether a continuous 3D line segment between p1 and p2 is collision-free.

        Uses exact analytical point-to-line-segment distance projection against all obstacles.

        Returns:
            (is_valid, min_clearance_along_segment)
        """
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        seg = b - a
        seg_len_sq = float(np.dot(seg, seg))

        # Check endpoints first
        v1, c1 = self.is_point_valid(a, obstacles)
        if not v1:
            return False, c1
        v2, c2 = self.is_point_valid(b, obstacles)
        if not v2:
            return False, c2

        min_clear = min(c1, c2)

        # Check arena boundaries along segment
        min_p = np.minimum(a, b)
        max_p = np.maximum(a, b)
        if (
            min_p[0] < self.vehicle_radius
            or max_p[0] > self.bounds[0] - self.vehicle_radius
            or min_p[1] < self.vehicle_radius
            or max_p[1] > self.bounds[1] - self.vehicle_radius
            or min_p[2] < self.vehicle_radius
            or max_p[2] > self.bounds[2] - self.vehicle_radius
        ):
            return False, 0.0

        if seg_len_sq < 1e-12:
            return True, min_clear

        # Test against all inflated obstacle spheres
        for obs in obstacles:
            center, radius = _extract_obstacle(obs)
            # Analytical projection parameter t in [0, 1]
            t = float(np.dot(center - a, seg) / seg_len_sq)
            t_clamped = max(0.0, min(1.0, t))
            closest_point = a + t_clamped * seg
            dist_to_center = float(np.linalg.norm(closest_point - center))
            clearance_to_surface = dist_to_center - (radius + self.vehicle_radius)

            if clearance_to_surface < min_clear:
                min_clear = clearance_to_surface

            if dist_to_center < radius + self.total_inflation_margin:
                return False, clearance_to_surface

        return True, min_clear

    def simplify_path(
        self,
        waypoints: List[np.ndarray],
        obstacles: Sequence[ObstacleSphere3D],
    ) -> Optional[List[np.ndarray]]:
        """Greedy line-of-sight shortcutting (string pulling) with strict collision revalidation.

        Returns None if no safe continuation exists between waypoints, ensuring an unvalidated
        segment is never appended.
        """
        if len(waypoints) <= 2:
            if len(waypoints) == 2:
                valid, _ = self.is_segment_valid(waypoints[0], waypoints[1], obstacles)
                if not valid:
                    return None
            return [w.copy() for w in waypoints]

        simplified = [waypoints[0].copy()]
        curr_idx = 0
        n = len(waypoints)

        while curr_idx < n - 1:
            furthest_valid_idx: Optional[int] = None
            for next_idx in range(n - 1, curr_idx, -1):
                valid, _ = self.is_segment_valid(
                    waypoints[curr_idx], waypoints[next_idx], obstacles
                )
                if valid:
                    furthest_valid_idx = next_idx
                    break

            if furthest_valid_idx is None:
                # No collision-free continuation exists from curr_idx; do not append unvalidated segment
                return None

            simplified.append(waypoints[furthest_valid_idx].copy())
            curr_idx = furthest_valid_idx

        # Invariant check: every consecutive segment must pass continuous collision validation
        for i in range(len(simplified) - 1):
            valid, _ = self.is_segment_valid(simplified[i], simplified[i + 1], obstacles)
            if not valid:
                return None

        return simplified

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: Sequence[ObstacleSphere3D],
        max_expansions: int = 15000,
    ) -> PlanningResult:
        """Compute an optimal collision-free 3D path from start to goal."""
        t0 = time.perf_counter()
        start_pt = np.asarray(start, dtype=np.float64)
        goal_pt = np.asarray(goal, dtype=np.float64)

        # 1. Validate start point
        valid_start, clear_start = self.is_point_valid(start_pt, obstacles)
        if not valid_start:
            elapsed = (time.perf_counter() - t0) * 1000.0
            status = "OUT_OF_BOUNDS" if clear_start == 0.0 else "BLOCKED_START"
            return PlanningResult(
                status=status,
                min_clearance=clear_start,
                planning_time_ms=elapsed,
            )

        # 2. Validate goal point
        valid_goal, clear_goal = self.is_point_valid(goal_pt, obstacles)
        if not valid_goal:
            elapsed = (time.perf_counter() - t0) * 1000.0
            status = "OUT_OF_BOUNDS" if clear_goal == 0.0 else "BLOCKED_GOAL"
            return PlanningResult(
                status=status,
                min_clearance=clear_goal,
                planning_time_ms=elapsed,
            )

        # 3. Direct line-of-sight shortcut check
        direct_valid, direct_clear = self.is_segment_valid(start_pt, goal_pt, obstacles)
        if direct_valid:
            elapsed = (time.perf_counter() - t0) * 1000.0
            dist = float(np.linalg.norm(goal_pt - start_pt))
            return PlanningResult(
                status="SUCCESS",
                waypoints=[start_pt.copy(), goal_pt.copy()],
                raw_path_length=dist,
                path_length=dist,
                min_clearance=direct_clear,
                planning_time_ms=elapsed,
                nodes_expanded=0,
                raw_nodes_count=2,
                simplified_nodes_count=2,
            )

        # 4. Grid-based 3D A* search
        res = self.resolution

        def to_grid(p: np.ndarray) -> Tuple[int, int, int]:
            return (
                int(round(p[0] / res)),
                int(round(p[1] / res)),
                int(round(p[2] / res)),
            )

        def to_world(coord: Tuple[int, int, int]) -> np.ndarray:
            return np.array([coord[0] * res, coord[1] * res, coord[2] * res], dtype=np.float64)

        base_start = to_grid(start_pt)
        goal_grid = to_grid(goal_pt)

        # 4. Search valid, collision-free candidate grid connections for start_pt
        # Examines base grid cell and all 26 spatial neighbors
        candidate_offsets = [(0, 0, 0)] + list(self._NEIGHBOR_OFFSETS)
        valid_start_nodes: List[Tuple[Tuple[int, int, int], float]] = []
        for off in candidate_offsets:
            cand_coord = (base_start[0] + off[0], base_start[1] + off[1], base_start[2] + off[2])
            cand_world = to_world(cand_coord)
            v_pt, _ = self.is_point_valid(cand_world, obstacles)
            if not v_pt:
                continue
            v_seg, _ = self.is_segment_valid(start_pt, cand_world, obstacles)
            if not v_seg:
                continue
            dist_cand = float(np.linalg.norm(cand_world - start_pt))
            valid_start_nodes.append((cand_coord, dist_cand))

        # If no neighboring grid node has a collision-free connection to start_pt, fail safely
        if not valid_start_nodes:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return PlanningResult(
                status="NO_PATH",
                planning_time_ms=elapsed,
                nodes_expanded=0,
            )

        valid_start_nodes.sort(key=lambda item: item[1])

        # Priority queue item: (f_score, g_score, grid_coord)
        open_set: List[Tuple[float, float, Tuple[int, int, int]]] = []
        came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {}
        g_scores: Dict[Tuple[int, int, int], float] = {}

        for cand_coord, d_start in valid_start_nodes:
            cand_world = to_world(cand_coord)
            h_score = float(np.linalg.norm(cand_world - goal_pt))
            g_scores[cand_coord] = d_start
            came_from[cand_coord] = None
            heapq.heappush(open_set, (d_start + h_score, d_start, cand_coord))

        visited: set[Tuple[int, int, int]] = set()
        goal_reached_coord: Optional[Tuple[int, int, int]] = None
        expansions = 0

        while open_set and expansions < max_expansions:
            f_curr, g_curr, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            expansions += 1

            curr_world = to_world(current)

            # Check if line-of-sight to non-grid goal is open
            if current == goal_grid or np.linalg.norm(curr_world - goal_pt) <= 2.0 * res:
                can_see_goal, _ = self.is_segment_valid(curr_world, goal_pt, obstacles)
                if can_see_goal:
                    goal_reached_coord = current
                    break

            for offset in self._NEIGHBOR_OFFSETS:
                neighbor = (current[0] + offset[0], current[1] + offset[1], current[2] + offset[2])
                if neighbor in visited:
                    continue

                neighbor_world = to_world(neighbor)
                valid_pt, _ = self.is_point_valid(neighbor_world, obstacles)
                if not valid_pt:
                    continue

                # Continuous segment check to prevent diagonal corner-cutting
                valid_seg, _ = self.is_segment_valid(curr_world, neighbor_world, obstacles)
                if not valid_seg:
                    continue

                step_cost = math.sqrt(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2) * res
                tentative_g = g_curr + step_cost

                if tentative_g < g_scores.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    h_score = float(np.linalg.norm(neighbor_world - goal_pt))
                    heapq.heappush(open_set, (tentative_g + h_score, tentative_g, neighbor))

        if goal_reached_coord is None:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return PlanningResult(
                status="NO_PATH",
                planning_time_ms=elapsed,
                nodes_expanded=expansions,
            )

        # 5. Reconstruct raw grid path with provable collision-free continuous connections
        raw_path: List[np.ndarray] = [goal_pt.copy()]
        curr_step: Optional[Tuple[int, int, int]] = goal_reached_coord
        while curr_step is not None:
            raw_path.append(to_world(curr_step))
            curr_step = came_from.get(curr_step)
        raw_path.append(start_pt.copy())
        raw_path.reverse()

        # Deduplicate consecutive waypoints that are identical within tolerance
        dedup_path: List[np.ndarray] = []
        for pt in raw_path:
            if not dedup_path or np.linalg.norm(pt - dedup_path[-1]) > 1e-6:
                dedup_path.append(pt)
        if len(dedup_path) >= 2:
            raw_path = dedup_path

        # Compute raw length
        raw_len = sum(
            float(np.linalg.norm(raw_path[i + 1] - raw_path[i])) for i in range(len(raw_path) - 1)
        )

        # 6. Apply line-of-sight shortcutting
        simplified = self.simplify_path(raw_path, obstacles)
        if simplified is None:
            # Fall back to raw grid path only if every raw segment is provably collision-free
            raw_valid = True
            for i in range(len(raw_path) - 1):
                v_seg, _ = self.is_segment_valid(raw_path[i], raw_path[i + 1], obstacles)
                if not v_seg:
                    raw_valid = False
                    break
            if raw_valid:
                simplified = raw_path
            else:
                elapsed = (time.perf_counter() - t0) * 1000.0
                return PlanningResult(
                    status="NO_PATH",
                    planning_time_ms=elapsed,
                    nodes_expanded=expansions,
                )

        # Compute simplified metrics
        simp_len = sum(
            float(np.linalg.norm(simplified[i + 1] - simplified[i]))
            for i in range(len(simplified) - 1)
        )
        min_clear = float("inf")
        for i in range(len(simplified) - 1):
            _, clear = self.is_segment_valid(simplified[i], simplified[i + 1], obstacles)
            if clear < min_clear:
                min_clear = clear

        elapsed = (time.perf_counter() - t0) * 1000.0

        return PlanningResult(
            status="SUCCESS",
            waypoints=simplified,
            raw_path_length=raw_len,
            path_length=simp_len,
            min_clearance=min_clear,
            planning_time_ms=elapsed,
            nodes_expanded=expansions,
            raw_nodes_count=len(raw_path),
            simplified_nodes_count=len(simplified),
        )
