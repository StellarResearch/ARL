"""Mission and Navigation Manager for Autonomous 3D Quadrotor Flight.

Coordinates high-level mission phases, global 3D A* planning, local dynamic
obstacle collision monitoring, and waypoint sequencing for the flight controller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import DroneState6DOF
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.planner import AStarPlanner3D
from adaptive_rl.navigation.time_aware_planner import TimeAwareAStarPlanner3D, TimedWaypoint

logger = logging.getLogger(__name__)


class MissionState(str, Enum):
    """High-level autonomous mission states."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    REPLANNING = "REPLANNING"
    WAITING = "WAITING"
    ARRIVED = "ARRIVED"
    BLOCKED = "FAILED_BLOCKED"
    FAILED_BLOCKED = "FAILED_BLOCKED"
    FAILED_COLLISION = "FAILED_COLLISION"
    ABORTED = "ABORTED"


@dataclass
class MissionTelemetry:
    """Status record of the active navigation mission."""

    state: MissionState
    active_waypoint_index: int
    total_waypoints: int
    current_target: np.ndarray
    distance_to_target: float
    distance_to_final_goal: float
    replan_count: int
    min_clearance_observed: float
    mission_waypoint_index: int = 0
    total_mission_waypoints: int = 0
    replan_reason: str = "none"
    waiting_duration: float = 0.0
    min_predicted_clearance: float = float("inf")
    state_source: str = "ground_truth"
    planned_vs_actual_deviation: float = 0.0
    active_route_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "active_waypoint_index": self.active_waypoint_index,
            "total_waypoints": self.total_waypoints,
            "mission_waypoint_index": self.mission_waypoint_index,
            "total_mission_waypoints": self.total_mission_waypoints,
            "current_target": [round(float(x), 4) for x in self.current_target],
            "distance_to_target": round(self.distance_to_target, 4),
            "distance_to_final_goal": round(self.distance_to_final_goal, 4),
            "replan_count": self.replan_count,
            "min_clearance_observed": (
                round(self.min_clearance_observed, 4)
                if self.min_clearance_observed < float("inf")
                else -1.0
            ),
            "replan_reason": self.replan_reason,
            "waiting_duration": round(self.waiting_duration, 4),
            "min_predicted_clearance": (
                round(self.min_predicted_clearance, 4)
                if self.min_predicted_clearance < float("inf")
                else -1.0
            ),
            "state_source": self.state_source,
            "planned_vs_actual_deviation": round(self.planned_vs_actual_deviation, 4),
            "active_route_cost": round(self.active_route_cost, 4),
        }


class MissionManager:
    """Orchestrates autonomous navigation, predictive time-aware planning, and dynamic obstacle avoidance.

    Supports:
    - Standard 3D A* planning or predictive Time-Aware 4D A* planning.
    - Receding-horizon obstacle clearance monitoring.
    - Replanning triggers with cooldown and route switching hysteresis.
    - Safe waiting / hover hold when corridor is temporarily obstructed, with deterministic timeout.
    - Explicit sensor-in-the-loop state estimation ingestion with fail-closed validation.
    """

    def __init__(
        self,
        planner: Optional[AStarPlanner3D] = None,
        time_aware_planner: Optional[TimeAwareAStarPlanner3D] = None,
        waypoint_acceptance_radius: float = 1.0,
        replan_clearance_threshold: float = 1.2,
        replan_cooldown_sec: float = 0.5,
        replan_hysteresis_cost: float = 0.1,
        max_waiting_duration_sec: float = 5.0,
        max_path_deviation_threshold: float = 2.0,
        state_source: str = "ground_truth",
    ) -> None:
        self.planner = planner or AStarPlanner3D()
        self.time_aware_planner = time_aware_planner
        self.waypoint_acceptance_radius = float(waypoint_acceptance_radius)
        self.replan_clearance_threshold = float(replan_clearance_threshold)
        self.replan_cooldown_sec = max(0.0, float(replan_cooldown_sec))
        self.replan_hysteresis_cost = max(0.0, float(replan_hysteresis_cost))
        self.max_waiting_duration_sec = max(0.5, float(max_waiting_duration_sec))
        self.max_path_deviation_threshold = max(0.5, float(max_path_deviation_threshold))
        self.state_source = str(state_source)

        self.state: MissionState = MissionState.IDLE
        self.mission_waypoints: List[np.ndarray] = []
        self.current_mission_waypoint_idx: int = 0
        self.active_path: List[np.ndarray] = []
        self.active_timed_waypoints: List[TimedWaypoint] = []
        self.active_path_idx: int = 0
        self.replan_count: int = 0
        self.min_clearance_observed: float = float("inf")
        self.min_predicted_clearance: float = float("inf")
        self.final_goal: Optional[np.ndarray] = None
        self.home_position: Optional[np.ndarray] = None

        self.replan_reason: str = "none"
        self.last_replan_time_sec: float = -100.0
        self.waiting_start_time_sec: Optional[float] = None
        self.cumulative_waiting_duration: float = 0.0
        self.planned_vs_actual_deviation: float = 0.0
        self.active_route_cost: float = 0.0

    def start_mission(
        self,
        start_position: np.ndarray,
        goals: Sequence[np.ndarray],
        static_obstacles: Sequence[ObstacleSphere3D],
        dynamic_obstacles: Optional[Sequence[DynamicObstacleSphere3D]] = None,
        time_sec: float = 0.0,
    ) -> bool:
        """Initialize mission and plan initial collision-free path through waypoints."""
        self.home_position = np.asarray(start_position, dtype=np.float64).copy()
        self.mission_waypoints = [np.asarray(g, dtype=np.float64).copy() for g in goals]
        if not self.mission_waypoints:
            raise ValueError("Mission must contain at least one destination waypoint.")

        self.current_mission_waypoint_idx = 0
        self.final_goal = self.mission_waypoints[-1].copy()
        self.replan_count = 0
        self.min_clearance_observed = float("inf")
        self.min_predicted_clearance = float("inf")
        self.cumulative_waiting_duration = 0.0
        self.waiting_start_time_sec = None
        self.replan_reason = "none"
        self.state = MissionState.PLANNING

        dyn_obs = dynamic_obstacles or []

        # 1. Use predictive time-aware planner if provided
        if self.time_aware_planner is not None:
            full_timed: List[TimedWaypoint] = [
                TimedWaypoint(position=self.home_position.copy(), time_sec=time_sec)
            ]
            curr_pt = self.home_position.copy()
            curr_t = time_sec
            total_cost = 0.0

            for goal in self.mission_waypoints:
                ta_res = self.time_aware_planner.plan(
                    start_position=curr_pt,
                    goal_position=goal,
                    static_obstacles=static_obstacles,
                    dynamic_obstacles=dyn_obs,
                    t_start=curr_t,
                )
                if ta_res.status != "SUCCESS" or not ta_res.timed_waypoints:
                    logger.warning("Time-aware planning failed to goal %s: %s", goal, ta_res.status)
                    self.state = MissionState.FAILED_BLOCKED
                    self.replan_reason = f"initial_plan_failed_{ta_res.status.lower()}"
                    return False

                for tw in ta_res.timed_waypoints[1:]:
                    full_timed.append(tw)
                curr_pt = goal.copy()
                curr_t = full_timed[-1].time_sec
                if ta_res.metrics:
                    total_cost += ta_res.metrics.total_objective_cost

            self.active_timed_waypoints = full_timed
            self.active_path = [tw.position.copy() for tw in full_timed]
            self.active_route_cost = total_cost
            self.active_path_idx = 1 if len(full_timed) > 1 else 0
            self.state = MissionState.EXECUTING
            return True

        # 2. Standard 3D A* planning with horizon-predicted occupied spheres
        current_obs = list(static_obstacles)
        for d in dyn_obs:
            current_obs.extend(d.predict_occupied_spheres(time_sec, horizon_sec=3.0, dt=0.5))

        full_path: List[np.ndarray] = [self.home_position.copy()]
        curr_pt = self.home_position.copy()

        for goal in self.mission_waypoints:
            plan_res = self.planner.plan(curr_pt, goal, current_obs)
            if plan_res.status != "SUCCESS" or not plan_res.waypoints:
                logger.warning("Planning failed to goal %s: %s", goal, plan_res.status)
                self.state = MissionState.FAILED_BLOCKED
                self.replan_reason = f"initial_plan_failed_{plan_res.status.lower()}"
                return False
            for wp in plan_res.waypoints[1:]:
                full_path.append(wp.copy())
            curr_pt = goal.copy()

        self.active_path = full_path
        self.active_path_idx = 1 if len(full_path) > 1 else 0
        self.state = MissionState.EXECUTING
        return True

    def update(
        self,
        drone_state: DroneState6DOF,
        time_sec: float,
        static_obstacles: Sequence[ObstacleSphere3D],
        dynamic_obstacles: Optional[Sequence[DynamicObstacleSphere3D]] = None,
    ) -> Tuple[np.ndarray, MissionTelemetry]:
        """Update navigation state machine, check clearance, replan if blocked, and return target."""
        pos = drone_state.position

        # Fail closed on non-finite state observations (sensor failure or crash)
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(drone_state.velocity)):
            self.state = MissionState.ABORTED
            self.replan_reason = "sensor_state_invalid_non_finite"
            return (
                self.home_position.copy() if self.home_position is not None else np.zeros(3)
            ), self.get_telemetry(drone_state)

        if self.state not in (
            MissionState.EXECUTING,
            MissionState.REPLANNING,
            MissionState.WAITING,
        ):
            current_target = self.active_path[-1] if self.active_path else pos.copy()
            return current_target, self.get_telemetry(drone_state)

        dyn_obs = dynamic_obstacles or []
        current_obs = list(static_obstacles) + [d.as_static_obstacle_at(time_sec) for d in dyn_obs]

        # Update observed clearance
        veh_rad = (
            self.time_aware_planner.vehicle_radius
            if self.time_aware_planner is not None
            else self.planner.vehicle_radius
        )
        for obs in current_obs:
            clear = float(np.linalg.norm(pos - obs.center)) - (obs.radius + veh_rad)
            if clear < self.min_clearance_observed:
                self.min_clearance_observed = clear

        # Current target waypoint along active path
        if self.active_path_idx < len(self.active_path):
            current_target = self.active_path[self.active_path_idx]
        else:
            current_target = self.active_path[-1] if self.active_path else pos.copy()

        dist_to_target = float(np.linalg.norm(current_target - pos))

        # Planned versus actual trajectory deviation
        prev_wp = (
            self.active_path[self.active_path_idx - 1]
            if (self.active_path_idx > 0 and self.active_path_idx <= len(self.active_path))
            else pos
        )
        seg_dir = current_target - prev_wp
        seg_len_sq = float(np.dot(seg_dir, seg_dir))
        if seg_len_sq > 1e-6:
            t_proj = max(0.0, min(1.0, float(np.dot(pos - prev_wp, seg_dir) / seg_len_sq)))
            proj_pos = prev_wp + t_proj * seg_dir
            self.planned_vs_actual_deviation = float(np.linalg.norm(pos - proj_pos))
        else:
            self.planned_vs_actual_deviation = float(np.linalg.norm(pos - current_target))

        # --- Receding-Horizon Route Clearance Check ---
        trigger_replan = False
        replan_reason = "none"
        critical_clearance = False

        if self.active_path_idx < len(self.active_path):
            # Check upcoming segment against moving obstacles over lookahead window
            dt_lookahead = 2.5
            t_eval_end = time_sec + dt_lookahead

            if self.time_aware_planner is not None:
                coll_ahead = self.time_aware_planner.check_dynamic_transition_collision(
                    pos, current_target, time_sec, t_eval_end, dyn_obs
                )
                stat_ahead, clr_stat = self.time_aware_planner.is_static_segment_valid(
                    pos, current_target, static_obstacles
                )
                if coll_ahead:
                    trigger_replan = True
                    replan_reason = "dynamic_obstacle_intersects_corridor"
                elif not stat_ahead:
                    trigger_replan = True
                    replan_reason = "static_corridor_obstructed"
            else:
                predicted_dyn_obs: List[ObstacleSphere3D] = []
                for d in dyn_obs:
                    predicted_dyn_obs.extend(
                        d.predict_occupied_spheres(time_sec, horizon_sec=dt_lookahead, dt=0.5)
                    )
                eval_obs = list(static_obstacles) + predicted_dyn_obs
                segment_clear, clr_val = self.planner.is_segment_valid(
                    pos, current_target, eval_obs
                )
                if not segment_clear:
                    trigger_replan = True
                    replan_reason = "predicted_obstacle_in_corridor"

            # Off-track deviation trigger
            if self.planned_vs_actual_deviation > self.max_path_deviation_threshold:
                trigger_replan = True
                replan_reason = "excessive_path_deviation"

            # Check if clearance is critically low
            for d in dyn_obs:
                c_now = d.center_at(time_sec)
                d_dist = float(np.linalg.norm(pos - c_now))
                d_clear = d_dist - (d.radius + veh_rad)
                if d_clear < self.min_predicted_clearance:
                    self.min_predicted_clearance = d_clear
                if d_clear < (veh_rad + 0.1):
                    critical_clearance = True

        # --- Replanning Execution with Cooldown & Hysteresis ---
        if trigger_replan:
            cooldown_active = (time_sec - self.last_replan_time_sec) < self.replan_cooldown_sec
            # Safety override: critical clearance bypasses cooldown immediately!
            if critical_clearance or not cooldown_active:
                logger.info(
                    "Replanning triggered at t=%.2fs (reason: %s)...", time_sec, replan_reason
                )
                self.state = MissionState.REPLANNING
                self.replan_count += 1
                self.replan_reason = replan_reason
                self.last_replan_time_sec = time_sec

                remaining_goals = self.mission_waypoints[self.current_mission_waypoint_idx :]
                if not remaining_goals:
                    remaining_goals = [
                        self.final_goal if self.final_goal is not None else current_target
                    ]

                replan_success = False

                # 1. Attempt Time-Aware Replanning
                if self.time_aware_planner is not None:
                    replan_timed: List[TimedWaypoint] = [
                        TimedWaypoint(position=pos.copy(), time_sec=time_sec)
                    ]
                    curr_pt = pos.copy()
                    curr_t = time_sec
                    total_cand_cost = 0.0
                    all_subgoals_ok = True

                    for goal in remaining_goals:
                        ta_res = self.time_aware_planner.plan(
                            start_position=curr_pt,
                            goal_position=goal,
                            static_obstacles=static_obstacles,
                            dynamic_obstacles=dyn_obs,
                            t_start=curr_t,
                        )
                        if ta_res.status != "SUCCESS" or not ta_res.timed_waypoints:
                            all_subgoals_ok = False
                            break
                        for tw in ta_res.timed_waypoints[1:]:
                            replan_timed.append(tw)
                        curr_pt = goal.copy()
                        curr_t = replan_timed[-1].time_sec
                        if ta_res.metrics:
                            total_cand_cost += ta_res.metrics.total_objective_cost

                    if all_subgoals_ok and len(replan_timed) > 1:
                        # Accept candidate route if current route is obstructed, critical, or candidate is cheaper
                        route_obstructed = replan_reason in (
                            "dynamic_obstacle_intersects_corridor",
                            "static_corridor_obstructed",
                            "excessive_path_deviation",
                        )
                        if (
                            critical_clearance
                            or route_obstructed
                            or (
                                total_cand_cost
                                < (self.active_route_cost + self.replan_hysteresis_cost)
                            )
                        ):
                            self.active_timed_waypoints = replan_timed
                            self.active_path = [tw.position.copy() for tw in replan_timed]
                            self.active_path_idx = 1
                            self.active_route_cost = total_cand_cost
                            current_target = self.active_path[self.active_path_idx]
                            dist_to_target = float(np.linalg.norm(current_target - pos))
                            self.state = MissionState.EXECUTING
                            self.waiting_start_time_sec = None
                            replan_success = True

                # 2. Fallback to Standard Sequential A* Replanning
                if not replan_success and self.planner is not None:
                    predicted_dyn_obs = []
                    for d in dyn_obs:
                        predicted_dyn_obs.extend(
                            d.predict_occupied_spheres(time_sec, horizon_sec=2.5, dt=0.5)
                        )
                    eval_obs = list(static_obstacles) + predicted_dyn_obs

                    replan_path: List[np.ndarray] = [pos.copy()]
                    curr_pt = pos.copy()
                    replan_ok = True

                    for goal in remaining_goals:
                        seg_plan = self.planner.plan(curr_pt, goal, eval_obs)
                        if seg_plan.status != "SUCCESS" or not seg_plan.waypoints:
                            replan_ok = False
                            break
                        for wp in seg_plan.waypoints[1:]:
                            replan_path.append(wp.copy())
                        curr_pt = goal.copy()

                    if replan_ok and len(replan_path) > 1:
                        self.active_path = replan_path
                        self.active_path_idx = 1
                        current_target = self.active_path[self.active_path_idx]
                        dist_to_target = float(np.linalg.norm(current_target - pos))
                        self.state = MissionState.EXECUTING
                        self.waiting_start_time_sec = None
                        replan_success = True

                # 3. If no route found, enter safe WAITING hold (or FAILED_BLOCKED on timeout)
                if not replan_success:
                    if self.waiting_start_time_sec is None:
                        self.waiting_start_time_sec = time_sec

                    wait_elapsed = time_sec - self.waiting_start_time_sec
                    self.cumulative_waiting_duration = wait_elapsed

                    if self.time_aware_planner is not None:
                        if wait_elapsed < self.max_waiting_duration_sec:
                            self.state = MissionState.WAITING
                            self.replan_reason = "waiting_for_corridor_clearance"
                            current_target = pos.copy()
                            return current_target, self.get_telemetry(drone_state)
                        else:
                            self.state = MissionState.FAILED_BLOCKED
                            self.replan_reason = "waiting_duration_exceeded_corridor_blocked"
                            current_target = pos.copy()
                            return current_target, self.get_telemetry(drone_state)
                    else:
                        # Standard planner maintains REPLANNING state hover hold
                        self.replan_reason = "hover_hold_awaiting_corridor_clearance"
                        current_target = pos.copy()
                        return current_target, self.get_telemetry(drone_state)

        # In WAITING state, check if corridor ahead has become clear
        if self.state == MissionState.WAITING and self.waiting_start_time_sec is not None:
            wait_elapsed = time_sec - self.waiting_start_time_sec
            self.cumulative_waiting_duration = wait_elapsed
            if wait_elapsed >= self.max_waiting_duration_sec:
                self.state = MissionState.FAILED_BLOCKED
                self.replan_reason = "waiting_duration_exceeded_corridor_blocked"
                return pos.copy(), self.get_telemetry(drone_state)

        # Advance active path waypoint if reached
        if dist_to_target <= self.waypoint_acceptance_radius:
            if self.active_path_idx < len(self.active_path) - 1:
                self.active_path_idx += 1
                current_target = self.active_path[self.active_path_idx]

        # Advance high-level mission waypoints sequentially as reached
        while self.current_mission_waypoint_idx < len(self.mission_waypoints):
            dist_to_mwp = float(
                np.linalg.norm(self.mission_waypoints[self.current_mission_waypoint_idx] - pos)
            )
            if dist_to_mwp <= self.waypoint_acceptance_radius:
                self.current_mission_waypoint_idx += 1
            else:
                break

        # Mission arrival check: all mission waypoints must be completed in order
        all_mission_wps_reached = self.current_mission_waypoint_idx >= len(self.mission_waypoints)
        final_goal_reached = (
            float(
                np.linalg.norm(
                    (self.mission_waypoints[-1] if self.mission_waypoints else current_target) - pos
                )
            )
            <= self.waypoint_acceptance_radius
        )
        at_end_of_path = self.active_path_idx >= len(self.active_path) - 1

        if all_mission_wps_reached and final_goal_reached and at_end_of_path:
            self.state = MissionState.ARRIVED

        return current_target, self.get_telemetry(drone_state)

    def trigger_return_to_home(
        self,
        current_position: np.ndarray,
        static_obstacles: Sequence[ObstacleSphere3D],
        dynamic_obstacles: Optional[Sequence[DynamicObstacleSphere3D]] = None,
        time_sec: float = 0.0,
    ) -> bool:
        """Trigger Return-To-Home (RTH) back to the initial start position."""
        if self.home_position is None:
            return False

        if self.time_aware_planner is not None:
            res_ta = self.time_aware_planner.plan(
                current_position,
                self.home_position,
                static_obstacles,
                dynamic_obstacles or [],
                t_start=time_sec,
            )
            if res_ta.status == "SUCCESS" and res_ta.spatial_waypoints:
                self.active_timed_waypoints = res_ta.timed_waypoints
                self.active_path = res_ta.spatial_waypoints
                self.active_path_idx = 1 if len(res_ta.spatial_waypoints) > 1 else 0
                self.mission_waypoints = [self.home_position.copy()]
                self.current_mission_waypoint_idx = 0
                self.final_goal = self.home_position.copy()
                self.state = MissionState.EXECUTING
                return True

        res = self.planner.plan(current_position, self.home_position, static_obstacles)
        if res.status == "SUCCESS" and res.waypoints:
            self.active_path = res.waypoints
            self.active_path_idx = 1 if len(res.waypoints) > 1 else 0
            self.mission_waypoints = [self.home_position.copy()]
            self.current_mission_waypoint_idx = 0
            self.final_goal = self.home_position.copy()
            self.state = MissionState.EXECUTING
            return True
        return False

    def get_telemetry(self, drone_state: DroneState6DOF) -> MissionTelemetry:
        """Return instantaneous mission telemetry."""
        pos = drone_state.position
        if self.active_path and self.active_path_idx < len(self.active_path):
            current_target = self.active_path[self.active_path_idx]
        elif self.active_path:
            current_target = self.active_path[-1]
        else:
            current_target = pos.copy()

        dist_target = float(np.linalg.norm(current_target - pos))
        final_goal = self.final_goal if self.final_goal is not None else current_target
        dist_final = float(np.linalg.norm(final_goal - pos))

        return MissionTelemetry(
            state=self.state,
            active_waypoint_index=self.active_path_idx,
            total_waypoints=len(self.active_path),
            current_target=current_target,
            distance_to_target=dist_target,
            distance_to_final_goal=dist_final,
            replan_count=self.replan_count,
            min_clearance_observed=self.min_clearance_observed,
            mission_waypoint_index=self.current_mission_waypoint_idx,
            total_mission_waypoints=len(self.mission_waypoints),
            replan_reason=self.replan_reason,
            waiting_duration=self.cumulative_waiting_duration,
            min_predicted_clearance=self.min_predicted_clearance,
            state_source=self.state_source,
            planned_vs_actual_deviation=self.planned_vs_actual_deviation,
            active_route_cost=self.active_route_cost,
        )
