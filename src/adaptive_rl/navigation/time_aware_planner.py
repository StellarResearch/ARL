"""Predictive Time-Aware 3D A* Path Planner and Route Evaluation Architecture.

Implements a bounded time-expanded 3D A* planner for aerial robotics operating
in dynamic environments with moving obstacles.

Key Features:
- 4D Time-expanded search state (x_grid, y_grid, z_grid, time_step).
- 26-connected spatial transitions + configurable in-place waiting / hover transitions.
- Continuous swept collision checking against moving obstacles across [t_k, t_{k+1}]
  using exact dynamic obstacle relative trajectory APIs and reflection event handling.
- Admissible time-aware Euclidean heuristic guaranteed never to overestimate remaining cost.
- Bounded search complexity: maximum node expansions, time limits, and discrete horizons.
- Generation of diverse alternative corridors with configurable multi-objective route evaluation.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.planner import _extract_obstacle


@dataclass
class TimedWaypoint:
    """Waypoints augmented with predicted arrival time and expected flight mode."""

    position: np.ndarray
    time_sec: float
    is_waiting: bool = False
    clearance: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": [round(float(c), 4) for c in self.position],
            "time_sec": round(float(self.time_sec), 4),
            "is_waiting": bool(self.is_waiting),
            "clearance": round(float(self.clearance), 4) if math.isfinite(self.clearance) else -1.0,
        }


@dataclass
class RouteEvaluationMetrics:
    """Quantitative performance and safety metrics for an evaluated route."""

    path_length: float
    travel_time: float
    waiting_time: float
    min_clearance: float
    mean_clearance: float
    risk_score: float
    curvature_cost: float
    total_objective_cost: float
    is_collision_free: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_length": round(self.path_length, 4),
            "travel_time": round(self.travel_time, 4),
            "waiting_time": round(self.waiting_time, 4),
            "min_clearance": round(self.min_clearance, 4)
            if math.isfinite(self.min_clearance)
            else -1.0,
            "mean_clearance": round(self.mean_clearance, 4)
            if math.isfinite(self.mean_clearance)
            else -1.0,
            "risk_score": round(self.risk_score, 4),
            "curvature_cost": round(self.curvature_cost, 4),
            "total_objective_cost": round(self.total_objective_cost, 4),
            "is_collision_free": self.is_collision_free,
        }


@dataclass
class TimeAwarePlanningResult:
    """Outcome, trajectory, and diagnostics of a predictive time-aware planning request."""

    status: str  # "SUCCESS", "NO_PATH", "BLOCKED_START", "BLOCKED_GOAL", "OUT_OF_BOUNDS", "SEARCH_BUDGET_EXCEEDED", "TIMEOUT"
    timed_waypoints: List[TimedWaypoint] = field(default_factory=list)
    spatial_waypoints: List[np.ndarray] = field(default_factory=list)
    metrics: Optional[RouteEvaluationMetrics] = None
    planning_time_ms: float = 0.0
    nodes_expanded: int = 0
    generated_nodes: int = 0
    search_depth: int = 0
    route_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "timed_waypoints": [tw.to_dict() for tw in self.timed_waypoints],
            "spatial_waypoints": [[round(float(c), 4) for c in w] for w in self.spatial_waypoints],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "planning_time_ms": round(self.planning_time_ms, 2),
            "nodes_expanded": self.nodes_expanded,
            "generated_nodes": self.generated_nodes,
            "search_depth": self.search_depth,
            "route_index": self.route_index,
        }


class RouteEvaluator:
    """Evaluates candidate 3D trajectories using configurable, normalized multi-objective costs.

    Objective formulation:
        J = w_d * (L / L_ref) + w_t * (T / T_ref) + w_w * (T_wait / T_ref)
            + w_r * C_risk + w_c * C_curvature

    Where all weights are non-negative, metrics are strictly non-negative, and
    clearance constraints act as hard safety thresholds.
    """

    def __init__(
        self,
        weight_distance: float = 1.0,
        weight_time: float = 1.0,
        weight_waiting: float = 0.8,
        weight_risk: float = 0.6,
        weight_curvature: float = 0.3,
        nominal_speed: float = 2.0,
        risk_distance_threshold: float = 2.5,
    ) -> None:
        self.w_dist = max(0.0, float(weight_distance))
        self.w_time = max(0.0, float(weight_time))
        self.w_wait = max(0.0, float(weight_waiting))
        self.w_risk = max(0.0, float(weight_risk))
        self.w_curv = max(0.0, float(weight_curvature))
        self.nominal_speed = max(0.1, float(nominal_speed))
        self.risk_distance_threshold = max(0.5, float(risk_distance_threshold))

    def evaluate(
        self,
        timed_waypoints: Sequence[TimedWaypoint],
        static_obstacles: Sequence[ObstacleSphere3D],
        dynamic_obstacles: Sequence[DynamicObstacleSphere3D],
        vehicle_radius: float = 0.8,
        reference_distance: Optional[float] = None,
    ) -> RouteEvaluationMetrics:
        """Compute comprehensive multi-objective metrics for a planned route."""
        if not timed_waypoints:
            return RouteEvaluationMetrics(
                path_length=0.0,
                travel_time=0.0,
                waiting_time=0.0,
                min_clearance=0.0,
                mean_clearance=0.0,
                risk_score=float("inf"),
                curvature_cost=0.0,
                total_objective_cost=float("inf"),
                is_collision_free=False,
            )

        pts = [tw.position for tw in timed_waypoints]
        times = [tw.time_sec for tw in timed_waypoints]
        n = len(pts)

        # 1. Path length & Travel time
        path_length = 0.0
        waiting_time = 0.0
        for i in range(n - 1):
            disp = float(np.linalg.norm(pts[i + 1] - pts[i]))
            path_length += disp
            dt = max(0.0, times[i + 1] - times[i])
            if timed_waypoints[i].is_waiting or disp < 1e-4:
                waiting_time += dt

        travel_time = max(0.0, times[-1] - times[0]) if n > 1 else 0.0

        # Reference distance for normalization
        if reference_distance is None or reference_distance <= 1e-3:
            direct_disp = float(np.linalg.norm(pts[-1] - pts[0])) if n > 1 else 1.0
            l_ref = max(1.0, direct_disp)
        else:
            l_ref = max(1.0, float(reference_distance))
        t_ref = max(0.5, l_ref / self.nominal_speed)

        # 2. Curvature cost (penalizing sharp changes in flight direction)
        curvature_cost = 0.0
        if n >= 3:
            for i in range(1, n - 1):
                v_in = pts[i] - pts[i - 1]
                v_out = pts[i + 1] - pts[i]
                norm_in = float(np.linalg.norm(v_in))
                norm_out = float(np.linalg.norm(v_out))
                if norm_in > 1e-3 and norm_out > 1e-3:
                    cos_theta = float(np.dot(v_in, v_out) / (norm_in * norm_out))
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    curvature_cost += 1.0 - cos_theta

        # 3. Continuous Trajectory Clearance and Risk Score against static & dynamic obstacles
        min_clear = float("inf")
        clearances: List[float] = []
        risk_score = 0.0
        is_safe = True

        if n == 1:
            # Single-waypoint trajectory (hover/hold at point)
            t_curr = times[0]
            p_curr = pts[0]
            point_min_clear = float("inf")
            for so in static_obstacles:
                c_stat, rad_stat = _extract_obstacle(so)
                clr = float(np.linalg.norm(p_curr - c_stat)) - (rad_stat + vehicle_radius)
                point_min_clear = min(point_min_clear, clr)
            for do in dynamic_obstacles:
                clr = float(np.linalg.norm(p_curr - do.center_at(t_curr))) - (
                    do.radius + vehicle_radius
                )
                point_min_clear = min(point_min_clear, clr)
            min_clear = point_min_clear
            clearances.append(point_min_clear)
            if point_min_clear < 0.0:
                is_safe = False
            if point_min_clear < self.risk_distance_threshold:
                pen = self.risk_distance_threshold - max(0.0, point_min_clear)
                risk_score += (pen / self.risk_distance_threshold) ** 2
        else:
            # Evaluate continuous trajectory chords across each waypoint transition
            for i in range(n - 1):
                p_a = pts[i]
                p_b = pts[i + 1]
                t_a = times[i]
                t_b = times[i + 1]
                dt_seg = max(0.0, t_b - t_a)
                seg_vec = p_b - p_a
                seg_len_sq = float(np.dot(seg_vec, seg_vec))

                # 3a. Exact analytical continuous clearance to static obstacles along chord
                for so in static_obstacles:
                    c_stat, rad_stat = _extract_obstacle(so)
                    if seg_len_sq < 1e-12:
                        dist_to_c = float(np.linalg.norm(p_a - c_stat))
                    else:
                        tau = max(0.0, min(1.0, float(np.dot(c_stat - p_a, seg_vec) / seg_len_sq)))
                        proj_pt = p_a + tau * seg_vec
                        dist_to_c = float(np.linalg.norm(proj_pt - c_stat))
                    chord_stat_clr = dist_to_c - (rad_stat + vehicle_radius)
                    if chord_stat_clr < min_clear:
                        min_clear = chord_stat_clr
                    if chord_stat_clr < 0.0:
                        is_safe = False

                # 3b. Continuous swept collision check against moving obstacles
                for do in dynamic_obstacles:
                    if do.check_continuous_collision(
                        prev_drone_pos=p_a,
                        curr_drone_pos=p_b,
                        prev_time=t_a,
                        curr_time=t_b,
                        vehicle_radius=vehicle_radius,
                        veh_acc_max=0.0,
                    ):
                        is_safe = False

                # 3c. Continuous temporal sampling for dynamic clearance, waiting intervals, and risk integration
                sample_count = max(3, min(25, int(math.ceil(dt_seg / 0.1))))
                for k in range(sample_count + 1):
                    s = k / float(sample_count)
                    t_k = t_a + s * dt_seg
                    p_k = p_a + s * seg_vec
                    sample_min_clr = float("inf")

                    for so in static_obstacles:
                        c_stat, rad_stat = _extract_obstacle(so)
                        clr = float(np.linalg.norm(p_k - c_stat)) - (rad_stat + vehicle_radius)
                        if clr < sample_min_clr:
                            sample_min_clr = clr

                    for do in dynamic_obstacles:
                        c_dyn = do.center_at(t_k)
                        clr = float(np.linalg.norm(p_k - c_dyn)) - (do.radius + vehicle_radius)
                        if clr < sample_min_clr:
                            sample_min_clr = clr

                    if sample_min_clr < min_clear:
                        min_clear = sample_min_clr
                    clearances.append(sample_min_clr)

                    if sample_min_clr < 0.0:
                        is_safe = False

                    # Proximity risk: penalty integrated continuously along flight path
                    if sample_min_clr < self.risk_distance_threshold:
                        pen = self.risk_distance_threshold - max(0.0, sample_min_clr)
                        risk_score += ((pen / self.risk_distance_threshold) ** 2) * (
                            1.0 / sample_count
                        )

        mean_clear = float(np.mean(clearances)) if clearances else 0.0

        # Normalized objective calculation
        norm_dist = path_length / l_ref
        norm_time = travel_time / t_ref
        norm_wait = waiting_time / t_ref
        norm_curv = curvature_cost / max(1.0, float(n - 2))

        total_cost = (
            self.w_dist * norm_dist
            + self.w_time * norm_time
            + self.w_wait * norm_wait
            + self.w_risk * risk_score
            + self.w_curv * norm_curv
        )

        return RouteEvaluationMetrics(
            path_length=path_length,
            travel_time=travel_time,
            waiting_time=waiting_time,
            min_clearance=min_clear,
            mean_clearance=mean_clear,
            risk_score=risk_score,
            curvature_cost=curvature_cost,
            total_objective_cost=total_cost,
            is_collision_free=is_safe,
        )


class TimeAwareAStarPlanner3D:
    """Deterministic Time-Expanded 3D A* Path Planner.

    Discretizes the spatio-temporal flight domain into (ix, iy, iz, it).
    Evaluates 26-connected spatial motion and stationary hover waiting transitions,
    guaranteeing that continuous-time moving obstacle trajectories are evaluated
    over the exact traversal intervals.
    """

    # 26 spatial neighbor directions
    _SPATIAL_NEIGHBORS: List[Tuple[int, int, int]] = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    def __init__(
        self,
        bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0),
        spatial_resolution: float = 1.0,
        time_resolution: float = 0.5,
        vehicle_radius: float = 0.8,
        safety_margin: float = 0.4,
        nominal_speed: float = 2.0,
        max_speed: float = 4.0,
        time_horizon_sec: float = 30.0,
        max_wait_steps: int = 10,
        wait_cost_factor: float = 1.05,
        evaluator: Optional[RouteEvaluator] = None,
    ) -> None:
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.spatial_resolution = float(spatial_resolution)
        self.time_resolution = float(time_resolution)
        self.vehicle_radius = float(vehicle_radius)
        self.safety_margin = float(safety_margin)
        self.total_inflation_margin = self.vehicle_radius + self.safety_margin
        self.nominal_speed = float(nominal_speed)
        self.max_speed = float(max_speed)
        self.time_horizon_sec = float(time_horizon_sec)
        self.max_wait_steps = int(max_wait_steps)
        self.wait_cost_factor = float(wait_cost_factor)
        self.evaluator = evaluator or RouteEvaluator(nominal_speed=self.nominal_speed)

    def _pos_to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        return (
            int(round(float(pos[0]) / self.spatial_resolution)),
            int(round(float(pos[1]) / self.spatial_resolution)),
            int(round(float(pos[2]) / self.spatial_resolution)),
        )

    def _grid_to_pos(self, grid: Tuple[int, int, int]) -> np.ndarray:
        return np.array(
            [
                grid[0] * self.spatial_resolution,
                grid[1] * self.spatial_resolution,
                grid[2] * self.spatial_resolution,
            ],
            dtype=np.float64,
        )

    def _is_in_bounds(self, p: np.ndarray) -> bool:
        r = self.vehicle_radius
        return bool(
            r <= p[0] <= self.bounds[0] - r
            and r <= p[1] <= self.bounds[1] - r
            and r <= p[2] <= self.bounds[2] - r
        )

    def is_static_point_valid(
        self,
        point: np.ndarray,
        static_obstacles: Sequence[ObstacleSphere3D],
    ) -> Tuple[bool, float]:
        """Check static boundary and static obstacle containment."""
        p = np.asarray(point, dtype=np.float64)
        if not self._is_in_bounds(p):
            return False, 0.0

        min_clr = float("inf")
        for so in static_obstacles:
            c, rad = _extract_obstacle(so)
            d = float(np.linalg.norm(p - c))
            clr = d - (rad + self.vehicle_radius)
            if clr < min_clr:
                min_clr = clr
            if d < rad + self.total_inflation_margin:
                return False, clr

        return True, min_clr

    def is_static_segment_valid(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        static_obstacles: Sequence[ObstacleSphere3D],
    ) -> Tuple[bool, float]:
        """Check continuous line segment against static obstacles."""
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)

        v1, c1 = self.is_static_point_valid(a, static_obstacles)
        if not v1:
            return False, c1
        v2, c2 = self.is_static_point_valid(b, static_obstacles)
        if not v2:
            return False, c2

        seg = b - a
        seg_len_sq = float(np.dot(seg, seg))
        min_clr = min(c1, c2)

        if seg_len_sq < 1e-12:
            return True, min_clr

        for so in static_obstacles:
            c, rad = _extract_obstacle(so)
            t = float(np.dot(c - a, seg) / seg_len_sq)
            t_clamped = max(0.0, min(1.0, t))
            proj = a + t_clamped * seg
            d = float(np.linalg.norm(proj - c))
            clr = d - (rad + self.vehicle_radius)
            if clr < min_clr:
                min_clr = clr
            if d < rad + self.total_inflation_margin:
                return False, clr

        return True, min_clr

    def check_dynamic_transition_collision(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        t_start: float,
        t_end: float,
        dynamic_obstacles: Sequence[DynamicObstacleSphere3D],
        veh_acc_max: float = 0.0,
    ) -> bool:
        """Evaluate continuous swept-sphere collision against all moving obstacles over [t_start, t_end].

        Employs a conservative O(1) analytical bounding-sphere pre-filter before invoking
        detailed continuous swept collision checks, maximizing planning throughput.
        """
        if not dynamic_obstacles or t_end <= t_start:
            return False

        effective_radius = self.vehicle_radius + self.safety_margin
        dt = t_end - t_start
        t_mid = 0.5 * (t_start + t_end)

        seg = p_end - p_start
        seg_len_sq = float(np.dot(seg, seg))

        for do in dynamic_obstacles:
            # 1. Conservative O(1) bounding sphere pre-filter:
            # Over [t_start, t_end], the obstacle trajectory is strictly bounded within a sphere
            # centered at do.center_at(t_mid) with radius = do.radius + 0.5 * v_max * dt
            v_max = do.max_speed()
            obs_bound_rad = do.radius + 0.5 * v_max * dt + effective_radius

            if seg_len_sq < 1e-12:
                dist_to_c = float(np.linalg.norm(p_start - do.center_at(t_mid)))
            else:
                t_proj = max(
                    0.0, min(1.0, float(np.dot(do.center_at(t_mid) - p_start, seg) / seg_len_sq))
                )
                proj_pt = p_start + t_proj * seg
                dist_to_c = float(np.linalg.norm(proj_pt - do.center_at(t_mid)))

            # If segment is clearly outside the obstacle swept envelope, no collision is possible
            if dist_to_c > obs_bound_rad:
                continue

            # 2. Detailed continuous swept collision check (with exact bounce splitting)
            if do.check_continuous_collision(
                prev_drone_pos=p_start,
                curr_drone_pos=p_end,
                prev_time=t_start,
                curr_time=t_end,
                vehicle_radius=effective_radius,
                veh_acc_max=veh_acc_max,
            ):
                return True

        return False

    def plan(
        self,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        static_obstacles: Sequence[ObstacleSphere3D],
        dynamic_obstacles: Sequence[DynamicObstacleSphere3D],
        t_start: float = 0.0,
        max_expansions: int = 15000,
        time_limit_sec: float = 5.0,
        spatial_exclusion_centers: Optional[Sequence[Tuple[np.ndarray, float]]] = None,
    ) -> TimeAwarePlanningResult:
        """Compute an optimal time-aware, collision-free 3D path considering moving obstacles."""
        start_wall_time = time.perf_counter()

        # Validate finite numerical inputs
        p_start_raw = np.asarray(start_position, dtype=np.float64)
        p_goal_raw = np.asarray(goal_position, dtype=np.float64)
        if (
            self.spatial_resolution <= 0.0
            or self.time_resolution <= 0.0
            or self.nominal_speed <= 0.0
            or self.bounds.shape != (3,)
            or np.any(self.bounds <= 0.0)
            or not np.all(np.isfinite(self.bounds))
            or p_start_raw.shape != (3,)
            or p_goal_raw.shape != (3,)
            or not np.all(np.isfinite(p_start_raw))
            or not np.all(np.isfinite(p_goal_raw))
            or not math.isfinite(t_start)
            or t_start < 0.0
            or not math.isfinite(time_limit_sec)
            or time_limit_sec <= 0.0
            or max_expansions <= 0
        ):
            return TimeAwarePlanningResult(
                status="INVALID_INPUT",
                planning_time_ms=(time.perf_counter() - start_wall_time) * 1000.0,
            )

        p_start = p_start_raw.copy()
        p_goal = p_goal_raw.copy()
        t0 = max(0.0, float(t_start))

        # Check bounds
        if not self._is_in_bounds(p_start) or not self._is_in_bounds(p_goal):
            return TimeAwarePlanningResult(
                status="OUT_OF_BOUNDS",
                planning_time_ms=(time.perf_counter() - start_wall_time) * 1000.0,
            )

        # Start and Goal static validity
        v_start, _ = self.is_static_point_valid(p_start, static_obstacles)
        if not v_start:
            return TimeAwarePlanningResult(
                status="BLOCKED_START",
                planning_time_ms=(time.perf_counter() - start_wall_time) * 1000.0,
            )

        v_goal, _ = self.is_static_point_valid(p_goal, static_obstacles)
        if not v_goal:
            return TimeAwarePlanningResult(
                status="BLOCKED_GOAL",
                planning_time_ms=(time.perf_counter() - start_wall_time) * 1000.0,
            )

        # Immediate trivial arrival
        dist_direct = float(np.linalg.norm(p_goal - p_start))
        if dist_direct < 0.2:
            single_tw = [TimedWaypoint(position=p_start, time_sec=t0)]
            metrics = self.evaluator.evaluate(
                single_tw, static_obstacles, dynamic_obstacles, self.vehicle_radius
            )
            return TimeAwarePlanningResult(
                status="SUCCESS",
                timed_waypoints=single_tw,
                spatial_waypoints=[p_start.copy()],
                metrics=metrics,
                planning_time_ms=(time.perf_counter() - start_wall_time) * 1000.0,
            )

        start_grid = self._pos_to_grid(p_start)
        goal_grid = self._pos_to_grid(p_goal)

        # Admissible time-aware heuristic: Euclidean distance divided by nominal flight speed
        def heuristic(p: np.ndarray) -> float:
            d = float(np.linalg.norm(p_goal - p))
            return d / self.nominal_speed

        # Priority queue entries: (f_score, g_cost, state_key, wait_streak)
        # state_key = (ix, iy, iz, it)
        start_state = (start_grid[0], start_grid[1], start_grid[2], 0)
        start_h = heuristic(p_start)

        open_set: List[Tuple[float, float, Tuple[int, int, int, int], int]] = []
        heapq.heappush(open_set, (start_h, 0.0, start_state, 0))

        g_scores: Dict[Tuple[int, int, int, int], float] = {start_state: 0.0}
        came_from: Dict[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], bool]] = {}

        nodes_expanded = 0
        generated_nodes = 1
        max_time_steps = int(math.ceil(self.time_horizon_sec / self.time_resolution))

        best_goal_state: Optional[Tuple[int, int, int, int]] = None

        while open_set:
            if (time.perf_counter() - start_wall_time) > time_limit_sec:
                return TimeAwarePlanningResult(
                    status="TIMEOUT",
                    planning_time_ms=(time.perf_counter() - start_wall_time) * 1000.0,
                    nodes_expanded=nodes_expanded,
                    generated_nodes=generated_nodes,
                )

            if nodes_expanded >= max_expansions:
                return TimeAwarePlanningResult(
                    status="SEARCH_BUDGET_EXCEEDED",
                    planning_time_ms=(time.perf_counter() - start_wall_time) * 1000.0,
                    nodes_expanded=nodes_expanded,
                    generated_nodes=generated_nodes,
                )

            _, current_g, current_state, wait_streak = heapq.heappop(open_set)

            # Skip suboptimal expansions
            if current_g > g_scores.get(current_state, float("inf")):
                continue

            nodes_expanded += 1
            curr_pos = self._grid_to_pos(current_state[:3])
            curr_t = t0 + current_state[3] * self.time_resolution

            # Check goal condition (within one cell of goal)
            if (
                current_state[:3] == goal_grid
                or float(np.linalg.norm(curr_pos - p_goal)) <= 1.2 * self.spatial_resolution
            ):
                # Validate direct final segment to exact goal with conservative arrival time
                final_dist = float(np.linalg.norm(p_goal - curr_pos))
                if final_dist > 1e-6:
                    final_dur = final_dist / self.nominal_speed
                    final_d_steps = max(1, int(math.ceil(final_dur / self.time_resolution - 1e-9)))
                    final_dt = final_d_steps * self.time_resolution
                else:
                    final_dt = 0.0
                stat_ok, _ = self.is_static_segment_valid(curr_pos, p_goal, static_obstacles)
                dyn_coll = self.check_dynamic_transition_collision(
                    curr_pos, p_goal, curr_t, curr_t + final_dt, dynamic_obstacles
                )
                if stat_ok and not dyn_coll:
                    best_goal_state = current_state
                    break

            # 1. Spatial neighbor transitions (26-connectivity)
            for dx, dy, dz in self._SPATIAL_NEIGHBORS:
                next_grid = (
                    current_state[0] + dx,
                    current_state[1] + dy,
                    current_state[2] + dz,
                )
                next_pos = self._grid_to_pos(next_grid)

                if not self._is_in_bounds(next_pos):
                    continue

                # Spatial exclusion penalty (for generating diverse alternative corridors)
                exclusion_penalty = 0.0
                if spatial_exclusion_centers:
                    for excl_c, excl_r in spatial_exclusion_centers:
                        if float(np.linalg.norm(next_pos - excl_c)) <= excl_r:
                            exclusion_penalty = 10.0
                            break

                # Continuous line validation against static obstacles
                stat_valid, _ = self.is_static_segment_valid(curr_pos, next_pos, static_obstacles)
                if not stat_valid:
                    continue

                step_disp = float(np.linalg.norm(next_pos - curr_pos))
                duration = step_disp / self.nominal_speed
                # Conservative discrete time steps: arrival time never precedes nominal physical transit time
                d_steps = max(1, int(math.ceil(duration / self.time_resolution - 1e-9)))
                next_t_idx = current_state[3] + d_steps

                if next_t_idx > max_time_steps:
                    continue

                t_trans_start = curr_t
                t_trans_end = t0 + next_t_idx * self.time_resolution

                # Continuous swept collision check against dynamic obstacles across transition interval
                if self.check_dynamic_transition_collision(
                    curr_pos, next_pos, t_trans_start, t_trans_end, dynamic_obstacles
                ):
                    continue

                # Consistent step cost matching actual time allocation and preserving admissibility
                step_cost = (d_steps * self.time_resolution) + exclusion_penalty
                new_g = current_g + step_cost
                next_state = (next_grid[0], next_grid[1], next_grid[2], next_t_idx)

                if new_g < g_scores.get(next_state, float("inf")):
                    g_scores[next_state] = new_g
                    came_from[next_state] = (current_state, False)
                    f_val = new_g + heuristic(next_pos)
                    heapq.heappush(open_set, (f_val, new_g, next_state, 0))
                    generated_nodes += 1

            # 2. Wait transition (hovering in-place for 1 time_resolution)
            if wait_streak < self.max_wait_steps:
                wait_t_idx = current_state[3] + 1
                if wait_t_idx <= max_time_steps:
                    t_wait_start = curr_t
                    t_wait_end = t0 + wait_t_idx * self.time_resolution

                    # Check if hovering in-place is safe from moving obstacles
                    if not self.check_dynamic_transition_collision(
                        curr_pos, curr_pos, t_wait_start, t_wait_end, dynamic_obstacles
                    ):
                        wait_state = (
                            current_state[0],
                            current_state[1],
                            current_state[2],
                            wait_t_idx,
                        )
                        wait_cost = self.time_resolution * self.wait_cost_factor
                        new_g = current_g + wait_cost

                        if new_g < g_scores.get(wait_state, float("inf")):
                            g_scores[wait_state] = new_g
                            came_from[wait_state] = (current_state, True)
                            f_val = new_g + heuristic(curr_pos)
                            heapq.heappush(open_set, (f_val, new_g, wait_state, wait_streak + 1))
                            generated_nodes += 1

        if best_goal_state is None:
            return TimeAwarePlanningResult(
                status="NO_PATH",
                planning_time_ms=(time.perf_counter() - start_wall_time) * 1000.0,
                nodes_expanded=nodes_expanded,
                generated_nodes=generated_nodes,
            )

        # Reconstruct path
        path_states: List[Tuple[Tuple[int, int, int, int], bool]] = []
        curr = best_goal_state
        while curr in came_from:
            prev, was_wait = came_from[curr]
            path_states.append((curr, was_wait))
            curr = prev
        path_states.append((start_state, False))
        path_states.reverse()

        timed_waypoints: List[TimedWaypoint] = []
        for state, was_wait in path_states:
            pos = self._grid_to_pos(state[:3])
            t_arrival = t0 + state[3] * self.time_resolution
            timed_waypoints.append(
                TimedWaypoint(position=pos, time_sec=t_arrival, is_waiting=was_wait)
            )

        # Adjust start and final endpoints to exact positions
        timed_waypoints[0].position = p_start.copy()
        final_disp = float(np.linalg.norm(timed_waypoints[-1].position - p_goal))
        if final_disp > 1e-3:
            final_duration = final_disp / self.nominal_speed
            final_d_steps = max(1, int(math.ceil(final_duration / self.time_resolution - 1e-9)))
            t_final = timed_waypoints[-1].time_sec + final_d_steps * self.time_resolution
            timed_waypoints.append(TimedWaypoint(position=p_goal.copy(), time_sec=t_final))
        else:
            timed_waypoints[-1].position = p_goal.copy()

        # Simplify redundant collinear waypoints while strictly preserving waiting events & clearance
        simplified_timed = self._simplify_timed_path(
            timed_waypoints, static_obstacles, dynamic_obstacles
        )
        spatial_waypoints = [tw.position.copy() for tw in simplified_timed]

        metrics = self.evaluator.evaluate(
            simplified_timed,
            static_obstacles,
            dynamic_obstacles,
            self.vehicle_radius,
            reference_distance=dist_direct,
        )

        elapsed_ms = (time.perf_counter() - start_wall_time) * 1000.0
        return TimeAwarePlanningResult(
            status="SUCCESS",
            timed_waypoints=simplified_timed,
            spatial_waypoints=spatial_waypoints,
            metrics=metrics,
            planning_time_ms=elapsed_ms,
            nodes_expanded=nodes_expanded,
            generated_nodes=generated_nodes,
            search_depth=len(simplified_timed),
        )

    def _simplify_timed_path(
        self,
        timed_path: List[TimedWaypoint],
        static_obstacles: Sequence[ObstacleSphere3D],
        dynamic_obstacles: Sequence[DynamicObstacleSphere3D],
    ) -> List[TimedWaypoint]:
        """Opportunistically shortcut non-waiting waypoints while guaranteeing continuous dynamic collision safety."""
        if len(timed_path) <= 2:
            return timed_path

        simplified: List[TimedWaypoint] = [timed_path[0]]
        curr_idx = 0
        n = len(timed_path)

        while curr_idx < n - 1:
            furthest_valid = curr_idx + 1
            for next_idx in range(n - 1, curr_idx, -1):
                # Do not shortcut across waiting waypoints
                has_wait = any(timed_path[k].is_waiting for k in range(curr_idx + 1, next_idx + 1))
                if has_wait:
                    continue

                p_a = timed_path[curr_idx].position
                p_b = timed_path[next_idx].position
                t_a = timed_path[curr_idx].time_sec
                t_b = timed_path[next_idx].time_sec

                # In confined passages with both static obstacles and crossing dynamic obstacles,
                # bound shortcut duration to ensure receding-horizon visibility and avoid overflying dynamic blockages
                if static_obstacles and dynamic_obstacles and (t_b - t_a) > 2.5:
                    continue

                stat_ok, _ = self.is_static_segment_valid(p_a, p_b, static_obstacles)
                if not stat_ok:
                    continue

                dyn_coll = self.check_dynamic_transition_collision(
                    p_a, p_b, t_a, t_b, dynamic_obstacles
                )
                if not dyn_coll:
                    furthest_valid = next_idx
                    break

            simplified.append(timed_path[furthest_valid])
            curr_idx = furthest_valid

        return simplified

    def plan_alternatives(
        self,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        static_obstacles: Sequence[ObstacleSphere3D],
        dynamic_obstacles: Sequence[DynamicObstacleSphere3D],
        t_start: float = 0.0,
        num_alternatives: int = 3,
        max_expansions: int = 15000,
        time_limit_sec: float = 5.0,
    ) -> List[TimeAwarePlanningResult]:
        """Generate multiple distinct candidate corridors ranked by multi-objective evaluation."""
        results: List[TimeAwarePlanningResult] = []
        exclusions: List[Tuple[np.ndarray, float]] = []

        # 1. Primary optimal plan
        res0 = self.plan(
            start_position,
            goal_position,
            static_obstacles,
            dynamic_obstacles,
            t_start=t_start,
            max_expansions=max_expansions,
            time_limit_sec=time_limit_sec,
        )
        if res0.status != "SUCCESS":
            return [res0]

        res0.route_index = 0
        results.append(res0)

        # 2. Derive alternative corridors by placing spatial penalty zones near intermediate waypoints
        for alt_idx in range(1, num_alternatives):
            # Extract midpoints of prior successful routes
            for r in results:
                if len(r.spatial_waypoints) >= 3:
                    mid_idx = len(r.spatial_waypoints) // 2
                    exclusions.append(
                        (r.spatial_waypoints[mid_idx].copy(), 2.0 * self.spatial_resolution)
                    )

            res_alt = self.plan(
                start_position,
                goal_position,
                static_obstacles,
                dynamic_obstacles,
                t_start=t_start,
                max_expansions=max_expansions,
                time_limit_sec=time_limit_sec,
                spatial_exclusion_centers=exclusions,
            )

            if res_alt.status == "SUCCESS":
                res_alt.route_index = alt_idx
                # Ensure the alternative route is actually distinct from previous routes
                is_distinct = True
                for prev_res in results:
                    disp_diff = sum(
                        float(np.linalg.norm(p1 - p2))
                        for p1, p2 in zip(res_alt.spatial_waypoints, prev_res.spatial_waypoints)
                    )
                    if disp_diff < 1.0:
                        is_distinct = False
                        break
                if is_distinct:
                    results.append(res_alt)

        # Rank all generated valid routes by total objective cost J
        results.sort(
            key=lambda r: (
                r.metrics.total_objective_cost
                if r.metrics is not None and r.status == "SUCCESS"
                else float("inf")
            )
        )
        return results
