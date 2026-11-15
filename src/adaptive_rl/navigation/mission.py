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

logger = logging.getLogger(__name__)


class MissionState(str, Enum):
    """High-level autonomous mission states."""

    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    REPLANNING = "REPLANNING"
    ARRIVED = "ARRIVED"
    FAILED_BLOCKED = "FAILED_BLOCKED"
    FAILED_COLLISION = "FAILED_COLLISION"


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "active_waypoint_index": self.active_waypoint_index,
            "total_waypoints": self.total_waypoints,
            "current_target": [round(float(x), 4) for x in self.current_target],
            "distance_to_target": round(self.distance_to_target, 4),
            "distance_to_final_goal": round(self.distance_to_final_goal, 4),
            "replan_count": self.replan_count,
            "min_clearance_observed": (
                round(self.min_clearance_observed, 4)
                if self.min_clearance_observed < float("inf")
                else -1.0
            ),
        }


class MissionManager:
    """Orchestrates autonomous navigation, global A* planning, and dynamic obstacle avoidance."""

    def __init__(
        self,
        planner: Optional[AStarPlanner3D] = None,
        waypoint_acceptance_radius: float = 1.0,
        replan_clearance_threshold: float = 1.2,
    ) -> None:
        self.planner = planner or AStarPlanner3D()
        self.waypoint_acceptance_radius = float(waypoint_acceptance_radius)
        self.replan_clearance_threshold = float(replan_clearance_threshold)

        self.state: MissionState = MissionState.IDLE
        self.mission_waypoints: List[np.ndarray] = []
        self.active_path: List[np.ndarray] = []
        self.active_path_idx: int = 0
        self.replan_count: int = 0
        self.min_clearance_observed: float = float("inf")
        self.final_goal: Optional[np.ndarray] = None
        self.home_position: Optional[np.ndarray] = None

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

        self.final_goal = self.mission_waypoints[-1].copy()
        self.replan_count = 0
        self.min_clearance_observed = float("inf")
        self.state = MissionState.PLANNING

        # Collect current obstacle set including horizon-predicted dynamic obstacles
        dyn_obs = dynamic_obstacles or []
        current_obs = list(static_obstacles)
        for d in dyn_obs:
            current_obs.extend(d.predict_occupied_spheres(time_sec, horizon_sec=3.0, dt=0.5))

        # Plan through each high-level mission waypoint
        full_path: List[np.ndarray] = [self.home_position.copy()]
        curr_pt = self.home_position.copy()

        for goal in self.mission_waypoints:
            plan_res = self.planner.plan(curr_pt, goal, current_obs)
            if plan_res.status != "SUCCESS" or not plan_res.waypoints:
                logger.warning("Planning failed to goal %s: %s", goal, plan_res.status)
                self.state = MissionState.FAILED_BLOCKED
                return False
            # Append waypoints (skipping duplicate connecting point)
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
        if self.state not in (MissionState.EXECUTING, MissionState.REPLANNING):
            current_target = self.active_path[-1] if self.active_path else drone_state.position
            return current_target, self.get_telemetry(drone_state)

        pos = drone_state.position
        dyn_obs = dynamic_obstacles or []
        current_obs = list(static_obstacles) + [d.as_static_obstacle_at(time_sec) for d in dyn_obs]

        # Update observed clearance
        for obs in current_obs:
            clear = float(np.linalg.norm(pos - obs.center)) - (
                obs.radius + self.planner.vehicle_radius
            )
            if clear < self.min_clearance_observed:
                self.min_clearance_observed = clear

        # Current target waypoint
        if self.active_path_idx < len(self.active_path):
            current_target = self.active_path[self.active_path_idx]
        else:
            current_target = self.active_path[-1]

        dist_to_target = float(np.linalg.norm(current_target - pos))

        # Check if current path segment ahead is blocked by dynamic obstacle in near future
        if self.active_path_idx < len(self.active_path):
            predicted_dyn_obs: List[ObstacleSphere3D] = []
            for d in dyn_obs:
                predicted_dyn_obs.extend(
                    d.predict_occupied_spheres(time_sec, horizon_sec=2.5, dt=0.5)
                )
            eval_obs = list(static_obstacles) + predicted_dyn_obs

            segment_clear, _ = self.planner.is_segment_valid(pos, current_target, eval_obs)
            if not segment_clear:
                logger.info(
                    "Dynamic obstacle predicted in flight corridor at t=%.2fs! Replanning...",
                    time_sec,
                )
                self.state = MissionState.REPLANNING
                self.replan_count += 1
                # Replan from current position to the destination
                dest = self.mission_waypoints[-1] if self.mission_waypoints else current_target
                replan_res = self.planner.plan(pos, dest, eval_obs)
                if replan_res.status == "SUCCESS" and replan_res.waypoints:
                    self.active_path = replan_res.waypoints
                    self.active_path_idx = 1 if len(replan_res.waypoints) > 1 else 0
                    current_target = self.active_path[self.active_path_idx]
                    self.state = MissionState.EXECUTING
                else:
                    # Safe stop in place (hover hold) while dynamic obstacle clears the corridor
                    current_target = pos.copy()
                    return current_target, self.get_telemetry(drone_state)

        # Advance waypoint if reached
        if dist_to_target <= self.waypoint_acceptance_radius:
            if self.active_path_idx < len(self.active_path) - 1:
                self.active_path_idx += 1
                current_target = self.active_path[self.active_path_idx]
            else:
                self.state = MissionState.ARRIVED

        return current_target, self.get_telemetry(drone_state)

    def trigger_return_to_home(
        self,
        current_position: np.ndarray,
        static_obstacles: Sequence[ObstacleSphere3D],
    ) -> bool:
        """Trigger Return-To-Home (RTH) back to the initial start position."""
        if self.home_position is None:
            return False
        res = self.planner.plan(current_position, self.home_position, static_obstacles)
        if res.status == "SUCCESS" and res.waypoints:
            self.active_path = res.waypoints
            self.active_path_idx = 1 if len(res.waypoints) > 1 else 0
            self.mission_waypoints = [self.home_position.copy()]
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
        )
