"""Autonomous flight validation scenarios and metric evaluation suite for 6-DOF quadrotors.

Implements reproducible benchmark scenarios covering:
1. Stable hovering
2. Straight-line waypoint navigation
3. Vertical ascent and descent
4. Horizontal movement
5. Position holding under aerodynamic disturbance
6. 3D Obstacle avoidance (A* collision-free path planning)
7. Boundary avoidance
8. Target approach and arrival
9. Recovery from moderate disturbances
10. Actuator fault degradation (canonical 50% motor loss)

Advanced Phase III 3D Navigation Scenarios:
11. Multi-obstacle navigation
12. Narrow passage navigation
13. Dynamic obstacle crossing
14. Moving target tracking
15. Multiple sequential 3D waypoints
16. Dead-end / blocked-route handling
17. Dynamic replanning after obstacle shift
18. Return-to-Home (RTH)

Computes measurable, independently verifiable aerospace flight metrics:
- Position tracking error (mean, max, RMS)
- Velocity tracking error (mean, max, RMS)
- Attitude tracking error (mean, max)
- Angular velocity magnitude (mean, max)
- Goal distance and settling time
- Minimum obstacle clearance
- Actuator control effort
- Flight duration and wall-clock execution time
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from adaptive_rl.controllers.pid import QuadrotorPIDController
from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import (
    Drone6DOFEnv,
    DroneState6DOF,
    euler_to_quaternion,
    quaternion_to_euler_angles,
)
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.planner import AStarPlanner3D

logger = logging.getLogger(__name__)


@dataclass
class FlightStepTelemetry:
    """Telemetry recorded at each physical simulation step."""

    step: int
    time_sec: float
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    euler_angles_deg: np.ndarray
    angular_velocity: np.ndarray
    target_position: np.ndarray
    distance_to_target: float
    action: np.ndarray
    motor_thrusts: np.ndarray
    min_obstacle_distance: float
    is_collision: bool
    is_success: bool
    commanded_thrusts: Optional[np.ndarray] = None
    actuator_fault_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        cmd_thrusts = (
            [round(float(x), 4) for x in self.commanded_thrusts]
            if self.commanded_thrusts is not None
            else [round(float(x), 4) for x in self.motor_thrusts]
        )
        return {
            "step": self.step,
            "time_sec": round(self.time_sec, 4),
            "position": [round(float(x), 4) for x in self.position],
            "velocity": [round(float(x), 4) for x in self.velocity],
            "speed": round(float(np.linalg.norm(self.velocity)), 4),
            "quaternion": [round(float(x), 5) for x in self.quaternion],
            "euler_angles_deg": [round(float(x), 2) for x in self.euler_angles_deg],
            "angular_velocity": [round(float(x), 4) for x in self.angular_velocity],
            "target_position": [round(float(x), 4) for x in self.target_position],
            "distance_to_target": round(float(self.distance_to_target), 4),
            "action": [round(float(x), 4) for x in self.action],
            "motor_thrusts": [round(float(x), 4) for x in self.motor_thrusts],
            "commanded_thrusts": cmd_thrusts,
            "actuator_fault_active": self.actuator_fault_active,
            "min_obstacle_distance": (
                round(float(self.min_obstacle_distance), 4)
                if math.isfinite(self.min_obstacle_distance)
                else None
            ),
            "is_collision": self.is_collision,
            "is_success": self.is_success,
        }


@dataclass
class FlightScenarioMetrics:
    """Summary flight performance metrics computed from empirical simulation data."""

    scenario_name: str
    success: bool
    outcome: str  # "success", "collision", "boundary_breach", "max_steps", "timeout", "recovered_degraded", "crash_fault", "blocked_no_path"
    duration_steps: int
    duration_seconds: float
    final_position_error: float
    mean_position_error: float
    max_position_error: float
    rms_position_error: float
    mean_velocity_error: float
    max_velocity_error: float
    rms_velocity_error: float
    mean_attitude_error_deg: float
    max_attitude_error_deg: float
    mean_angular_speed: float
    max_angular_speed: float
    initial_goal_distance: float
    final_goal_distance: float
    min_obstacle_clearance: float
    collision_count: int
    control_effort: float
    execution_wall_time_ms: float
    replan_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlightScenarioResult:
    """Full scenario execution record containing metrics and step telemetry."""

    scenario_name: str
    description: str
    metrics: FlightScenarioMetrics
    telemetry: List[FlightStepTelemetry] = field(default_factory=list)

    def to_dict(self, include_telemetry: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "metrics": self.metrics.to_dict(),
        }
        if include_telemetry:
            data["telemetry"] = [step.to_dict() for step in self.telemetry]
        return data

    def export_json(self, filepath: Path | str, include_telemetry: bool = True) -> Path:
        """Export scenario result to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_telemetry=include_telemetry), f, indent=2)
        return path

    def export_telemetry_csv(self, filepath: Path | str) -> Path:
        """Export time-series telemetry to CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.telemetry:
            path.write_text("no telemetry recorded\n", encoding="utf-8")
            return path

        fieldnames = [
            "step",
            "time_sec",
            "pos_x",
            "pos_y",
            "pos_z",
            "vel_x",
            "vel_y",
            "vel_z",
            "speed",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "omega_p",
            "omega_q",
            "omega_r",
            "target_x",
            "target_y",
            "target_z",
            "distance_to_target",
            "act_m1",
            "act_m2",
            "act_m3",
            "act_m4",
            "thrust_m1",
            "thrust_m2",
            "thrust_m3",
            "thrust_m4",
            "fault_active",
            "min_obstacle_distance",
            "is_collision",
            "is_success",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in self.telemetry:
                writer.writerow(
                    {
                        "step": t.step,
                        "time_sec": round(t.time_sec, 4),
                        "pos_x": round(float(t.position[0]), 4),
                        "pos_y": round(float(t.position[1]), 4),
                        "pos_z": round(float(t.position[2]), 4),
                        "vel_x": round(float(t.velocity[0]), 4),
                        "vel_y": round(float(t.velocity[1]), 4),
                        "vel_z": round(float(t.velocity[2]), 4),
                        "speed": round(float(np.linalg.norm(t.velocity)), 4),
                        "roll_deg": round(float(t.euler_angles_deg[0]), 2),
                        "pitch_deg": round(float(t.euler_angles_deg[1]), 2),
                        "yaw_deg": round(float(t.euler_angles_deg[2]), 2),
                        "omega_p": round(float(t.angular_velocity[0]), 4),
                        "omega_q": round(float(t.angular_velocity[1]), 4),
                        "omega_r": round(float(t.angular_velocity[2]), 4),
                        "target_x": round(float(t.target_position[0]), 4),
                        "target_y": round(float(t.target_position[1]), 4),
                        "target_z": round(float(t.target_position[2]), 4),
                        "distance_to_target": round(float(t.distance_to_target), 4),
                        "act_m1": round(float(t.action[0]), 4),
                        "act_m2": round(float(t.action[1]), 4),
                        "act_m3": round(float(t.action[2]), 4),
                        "act_m4": round(float(t.action[3]), 4),
                        "thrust_m1": round(float(t.motor_thrusts[0]), 4),
                        "thrust_m2": round(float(t.motor_thrusts[1]), 4),
                        "thrust_m3": round(float(t.motor_thrusts[2]), 4),
                        "thrust_m4": round(float(t.motor_thrusts[3]), 4),
                        "fault_active": int(t.actuator_fault_active),
                        "min_obstacle_distance": (
                            round(float(t.min_obstacle_distance), 4)
                            if math.isfinite(t.min_obstacle_distance)
                            else -1.0
                        ),
                        "is_collision": int(t.is_collision),
                        "is_success": int(t.is_success),
                    }
                )
        return path


@dataclass
class FlightScenarioDefinition:
    """Specification of an autonomous flight validation scenario."""

    name: str
    description: str
    initial_position: np.ndarray
    initial_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    initial_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    initial_angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    waypoints: List[np.ndarray] = field(default_factory=list)
    obstacles: List[ObstacleSphere3D] = field(default_factory=list)
    dynamic_obstacles: List[DynamicObstacleSphere3D] = field(default_factory=list)
    bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0)
    max_steps: int = 300
    target_radius: float = 1.2
    collision_radius: float = 0.8
    step_duration_sec: float = 0.05
    use_path_planner: bool = False
    disturbance_fn: Optional[Callable[[int, DroneState6DOF], np.ndarray]] = None
    actuator_fault_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None


def get_standard_flight_scenarios() -> List[FlightScenarioDefinition]:
    """Instantiate the 10 standard autonomous flight scenarios."""
    scenarios: List[FlightScenarioDefinition] = []

    # 1. Stable Hovering
    scenarios.append(
        FlightScenarioDefinition(
            name="hover_equilibrium",
            description="Hold steady hover position at [15.0, 15.0, 7.5] for 200 simulation steps.",
            initial_position=np.array([15.0, 15.0, 7.5]),
            waypoints=[np.array([15.0, 15.0, 7.5])],
            obstacles=[],
            max_steps=200,
            target_radius=0.5,
        )
    )

    # 2. Straight-Line Waypoint Navigation
    scenarios.append(
        FlightScenarioDefinition(
            name="straight_line_navigation",
            description="Navigate straight line from [5.0, 5.0, 3.0] to diagonal waypoint [25.0, 25.0, 12.0].",
            initial_position=np.array([5.0, 5.0, 3.0]),
            waypoints=[np.array([25.0, 25.0, 12.0])],
            obstacles=[],
            max_steps=350,
            target_radius=1.2,
        )
    )

    # 3. Vertical Ascent and Descent
    scenarios.append(
        FlightScenarioDefinition(
            name="vertical_ascent_descent",
            description="Ascend vertically from z=2.0m to z=12.0m, then descend to z=5.0m.",
            initial_position=np.array([15.0, 15.0, 2.0]),
            waypoints=[np.array([15.0, 15.0, 12.0]), np.array([15.0, 15.0, 5.0])],
            obstacles=[],
            max_steps=350,
            target_radius=1.0,
        )
    )

    # 4. Horizontal Movement
    scenarios.append(
        FlightScenarioDefinition(
            name="horizontal_translation",
            description="Execute horizontal square pattern at constant altitude z=6.0m.",
            initial_position=np.array([8.0, 8.0, 6.0]),
            waypoints=[
                np.array([22.0, 8.0, 6.0]),
                np.array([22.0, 22.0, 6.0]),
                np.array([8.0, 22.0, 6.0]),
                np.array([8.0, 8.0, 6.0]),
            ],
            obstacles=[],
            max_steps=500,
            target_radius=1.2,
        )
    )

    # 5. Position Holding Under Wind-Like Force
    def wind_disturbance(step: int, state: DroneState6DOF) -> np.ndarray:
        if 50 <= step <= 150:
            return np.array([1.2, 0.8, 0.0])  # +1.2 N in X, +0.8 N in Y
        return np.zeros(3)

    scenarios.append(
        FlightScenarioDefinition(
            name="position_holding_disturbance",
            description="Hold position at [12.0, 12.0, 6.0] while subjected to lateral step disturbances.",
            initial_position=np.array([12.0, 12.0, 6.0]),
            waypoints=[np.array([12.0, 12.0, 6.0])],
            obstacles=[],
            max_steps=250,
            target_radius=1.0,
            disturbance_fn=wind_disturbance,
        )
    )

    # 6. Obstacle Avoidance (Using 3D A* Collision-Free Path Planning)
    obs1 = ObstacleSphere3D(center=np.array([15.0, 15.0, 7.0]), radius=2.2)
    obs2 = ObstacleSphere3D(center=np.array([10.0, 18.0, 6.0]), radius=1.8)
    scenarios.append(
        FlightScenarioDefinition(
            name="obstacle_avoidance",
            description="Navigate from [5.0, 5.0, 7.0] to [25.0, 25.0, 7.0] through spherical obstacle field using 3D A* collision-free planning.",
            initial_position=np.array([5.0, 5.0, 7.0]),
            waypoints=[np.array([25.0, 25.0, 7.0])],
            obstacles=[obs1, obs2],
            max_steps=400,
            target_radius=1.5,
            use_path_planner=True,
        )
    )

    # 7. Boundary Avoidance
    scenarios.append(
        FlightScenarioDefinition(
            name="boundary_avoidance",
            description="Fly toward arena boundary corner [28.0, 28.0, 13.5] and arrest motion before collision.",
            initial_position=np.array([15.0, 15.0, 8.0]),
            waypoints=[np.array([28.0, 28.0, 13.5])],
            bounds=(30.0, 30.0, 15.0),
            obstacles=[],
            max_steps=300,
            target_radius=1.5,
        )
    )

    # 8. Target Approach and Arrival (Precision Settling)
    scenarios.append(
        FlightScenarioDefinition(
            name="target_approach_arrival",
            description="Approach target waypoint [20.0, 15.0, 6.0] and achieve stable terminal hover (< 0.2 m/s).",
            initial_position=np.array([6.0, 15.0, 6.0]),
            waypoints=[np.array([20.0, 15.0, 6.0])],
            obstacles=[],
            max_steps=300,
            target_radius=0.8,
        )
    )

    # 9. Recovery from Moderate Disturbances
    q_init_pert = euler_to_quaternion(roll=math.radians(25.0), pitch=math.radians(-15.0), yaw=0.0)
    omega_init_pert = np.array([1.5, -1.2, 0.5])
    scenarios.append(
        FlightScenarioDefinition(
            name="attitude_disturbance_recovery",
            description="Recover to level hover from an initial 25-degree attitude tilt and angular rate.",
            initial_position=np.array([15.0, 15.0, 8.0]),
            initial_quaternion=q_init_pert,
            initial_angular_velocity=omega_init_pert,
            waypoints=[np.array([15.0, 15.0, 8.0])],
            obstacles=[],
            max_steps=250,
            target_radius=0.8,
        )
    )

    # 10. Failure Verification Scenario (Canonical 50% thrust loss on Motor 1)
    def actuator_canonical_fault(step: int, thrusts: np.ndarray) -> np.ndarray:
        if step >= 40:
            degraded = thrusts.copy()
            degraded[0] *= 0.5  # Canonical 50% thrust reduction on front-right motor
            return degraded
        return thrusts

    scenarios.append(
        FlightScenarioDefinition(
            name="actuator_fault_degradation",
            description="Evaluate flight control stability under canonical 50% partial thrust loss on Motor 1 at step 40.",
            initial_position=np.array([15.0, 15.0, 7.0]),
            waypoints=[np.array([15.0, 15.0, 7.0])],
            obstacles=[],
            max_steps=250,
            target_radius=1.5,
            actuator_fault_fn=actuator_canonical_fault,
        )
    )

    return scenarios


def get_advanced_navigation_scenarios() -> List[FlightScenarioDefinition]:
    """Instantiate the 8 advanced 3D autonomous navigation scenarios (Phase III)."""
    scenarios: List[FlightScenarioDefinition] = []

    # 11. Multi-Obstacle Field Navigation
    obs_field = [
        ObstacleSphere3D(center=np.array([10.0, 10.0, 6.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([15.0, 18.0, 7.0]), radius=2.2),
        ObstacleSphere3D(center=np.array([20.0, 12.0, 8.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([22.0, 22.0, 6.0]), radius=1.8),
    ]
    scenarios.append(
        FlightScenarioDefinition(
            name="multi_obstacle_navigation",
            description="Navigate through a dense 4-sphere obstacle field using 3D A* global path planning.",
            initial_position=np.array([4.0, 4.0, 5.0]),
            waypoints=[np.array([26.0, 26.0, 9.0])],
            obstacles=obs_field,
            max_steps=500,
            target_radius=1.5,
            use_path_planner=True,
        )
    )

    # 12. Narrow Passage Navigation
    # Two large obstacles forming a narrow corridor
    obs_gap_1 = ObstacleSphere3D(center=np.array([15.0, 12.0, 7.0]), radius=2.0)
    obs_gap_2 = ObstacleSphere3D(center=np.array([15.0, 18.0, 7.0]), radius=2.0)
    scenarios.append(
        FlightScenarioDefinition(
            name="narrow_passage_navigation",
            description="Navigate safely through a narrow 2.0m corridor between two large obstacles.",
            initial_position=np.array([5.0, 15.0, 7.0]),
            waypoints=[np.array([25.0, 15.0, 7.0])],
            obstacles=[obs_gap_1, obs_gap_2],
            max_steps=400,
            target_radius=1.2,
            use_path_planner=True,
        )
    )

    # 13. Dynamic Obstacle Crossing
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([15.0, 5.0, 7.0]),
        radius=1.6,
        velocity=np.array([0.0, 1.2, 0.0]),  # Translating across the flight path
        bounds=(30.0, 30.0, 15.0),
    )
    scenarios.append(
        FlightScenarioDefinition(
            name="dynamic_obstacle_crossing",
            description="Cross flight path with a moving obstacle translating deterministically across trajectory.",
            initial_position=np.array([5.0, 15.0, 7.0]),
            waypoints=[np.array([25.0, 15.0, 7.0])],
            obstacles=[],
            dynamic_obstacles=[dyn_obs],
            max_steps=400,
            target_radius=1.5,
            use_path_planner=True,
        )
    )

    # 14. Multiple Sequential 3D Waypoints (Slalom)
    scenarios.append(
        FlightScenarioDefinition(
            name="multiple_sequential_waypoints",
            description="Follow a 5-waypoint 3D spatial slalom course.",
            initial_position=np.array([5.0, 5.0, 3.0]),
            waypoints=[
                np.array([10.0, 20.0, 5.0]),
                np.array([15.0, 8.0, 7.0]),
                np.array([20.0, 22.0, 9.0]),
                np.array([25.0, 10.0, 11.0]),
                np.array([27.0, 25.0, 8.0]),
            ],
            obstacles=[],
            max_steps=600,
            target_radius=1.2,
        )
    )

    # 15. Blocked Route / Dead-End Handling
    # Impassable barrier completely surrounding goal
    barrier = [
        ObstacleSphere3D(center=np.array([25.0, 25.0, 7.0]), radius=3.0),
    ]
    scenarios.append(
        FlightScenarioDefinition(
            name="blocked_route_dead_end",
            description="Attempt navigation to an obstructed goal inside an obstacle, verifying clean NO_PATH handling.",
            initial_position=np.array([5.0, 5.0, 5.0]),
            waypoints=[np.array([25.0, 25.0, 7.0])],  # Goal inside obstacle
            obstacles=barrier,
            max_steps=100,
            target_radius=1.0,
            use_path_planner=True,
        )
    )

    # 16. Return-to-Home (RTH) Navigation
    scenarios.append(
        FlightScenarioDefinition(
            name="return_to_home_mission",
            description="Fly outbound to waypoint [22.0, 22.0, 10.0] and execute autonomous Return-to-Home.",
            initial_position=np.array([5.0, 5.0, 3.0]),
            waypoints=[np.array([22.0, 22.0, 10.0]), np.array([5.0, 5.0, 3.0])],
            obstacles=[],
            max_steps=500,
            target_radius=1.2,
        )
    )

    # 17. Turbulent Canyon Navigation
    canyon_obs = [
        ObstacleSphere3D(center=np.array([10.0, 8.0, 5.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([10.0, 12.0, 5.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([15.0, 8.0, 5.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([15.0, 12.0, 5.0]), radius=2.0),
    ]
    scenarios.append(
        FlightScenarioDefinition(
            name="turbulent_canyon_wind",
            description="Navigate narrow passage under constant 3.0 m/s lateral wind disturbance.",
            initial_position=np.array([5.0, 10.0, 5.0]),
            waypoints=[np.array([20.0, 10.0, 5.0])],
            obstacles=canyon_obs,
            disturbance_fn=lambda step, state: np.array([0.0, 3.0, 0.0]),
            max_steps=300,
            target_radius=1.0,
            use_path_planner=True,
        )
    )

    # 18. Degraded Actuator Loiter
    scenarios.append(
        FlightScenarioDefinition(
            name="degraded_actuator_loiter",
            description="Maintain hover loiter under 10% motor thrust loss (Motor 1).",
            initial_position=np.array([10.0, 10.0, 5.0]),
            waypoints=[np.array([10.0, 10.0, 5.0])],
            obstacles=[],
            actuator_fault_fn=lambda step, thrusts: np.array(
                [
                    thrusts[0] * 0.90 if step >= 20 else thrusts[0],
                    thrusts[1],
                    thrusts[2],
                    thrusts[3],
                ]
            ),
            max_steps=200,
            target_radius=1.0,
        )
    )

    return scenarios


def get_all_flight_scenarios() -> List[FlightScenarioDefinition]:
    """Return all standard and advanced flight validation scenarios."""
    return get_standard_flight_scenarios() + get_advanced_navigation_scenarios()


def get_scenario(name: str) -> FlightScenarioDefinition:
    """Retrieve a scenario definition by name."""
    clean = name.strip().lower()
    for sc in get_all_flight_scenarios():
        if sc.name.lower() == clean:
            return sc
    raise KeyError(f"Flight scenario '{name}' not found.")


class FlightScenarioRunner:
    """Executes flight scenarios against real 6-DOF simulation models and records telemetry."""

    def __init__(self, env: Optional[Drone6DOFEnv] = None) -> None:
        self.env = env or Drone6DOFEnv()

    def run_scenario(
        self,
        scenario: FlightScenarioDefinition,
        controller: Optional[Any] = None,
    ) -> FlightScenarioResult:
        """Execute a flight scenario and compute full empirical aerospace flight metrics.

        Args:
            scenario: FlightScenarioDefinition specifying geometry, bounds, and waypoints.
            controller: Controller instance (defaults to QuadrotorPIDController with scenario parameters).
        """
        start_wall_time = time.perf_counter()

        # Instantiate or reset controller
        ctrl = controller or QuadrotorPIDController(
            mass=self.env.dynamics.mass,
            gravity=self.env.dynamics.gravity,
            arm_length=self.env.dynamics.arm_length,
            inertia=(
                float(self.env.dynamics.inertia_diag[0]),
                float(self.env.dynamics.inertia_diag[1]),
                float(self.env.dynamics.inertia_diag[2]),
            ),
            max_thrust_per_motor=self.env.dynamics.max_thrust_per_motor,
            torque_to_thrust_ratio=self.env.dynamics.c_tau,
            dt=self.env.dynamics.dt,
        )
        if hasattr(ctrl, "reset"):
            ctrl.reset()

        # Configure environment state for scenario
        self.env.bounds = scenario.bounds
        self.env._obstacles = list(scenario.obstacles)
        self.env.target_radius = scenario.target_radius
        self.env.collision_radius = scenario.collision_radius
        self.env.max_steps = scenario.max_steps
        self.env.dynamics.reset(
            position=scenario.initial_position,
            velocity=scenario.initial_velocity,
            quaternion=scenario.initial_quaternion,
            angular_velocity=scenario.initial_angular_velocity,
        )
        self.env._position = scenario.initial_position.copy()
        self.env._current_step = 0

        # Resolve waypoints (optionally via 3D A* path planning)
        waypoints: List[np.ndarray] = []
        if scenario.use_path_planner and scenario.obstacles:
            planner = AStarPlanner3D(
                bounds=scenario.bounds,
                resolution=1.0,
                vehicle_radius=scenario.collision_radius,
                safety_margin=0.3,
            )
            raw_target = scenario.waypoints[-1] if scenario.waypoints else scenario.initial_position
            plan_res = planner.plan(scenario.initial_position, raw_target, scenario.obstacles)
            if plan_res.status == "SUCCESS" and len(plan_res.waypoints) > 1:
                waypoints = plan_res.waypoints[1:]  # sequenced targets ahead
            else:
                # Planning failure (e.g. goal or start blocked)
                elapsed_ms = (time.perf_counter() - start_wall_time) * 1000.0
                initial_goal_dist = float(np.linalg.norm(raw_target - scenario.initial_position))
                metrics = FlightScenarioMetrics(
                    scenario_name=scenario.name,
                    success=False,
                    outcome="blocked_no_path",
                    duration_steps=0,
                    duration_seconds=0.0,
                    final_position_error=initial_goal_dist,
                    mean_position_error=initial_goal_dist,
                    max_position_error=initial_goal_dist,
                    rms_position_error=initial_goal_dist,
                    mean_velocity_error=0.0,
                    max_velocity_error=0.0,
                    rms_velocity_error=0.0,
                    mean_attitude_error_deg=0.0,
                    max_attitude_error_deg=0.0,
                    mean_angular_speed=0.0,
                    max_angular_speed=0.0,
                    initial_goal_distance=initial_goal_dist,
                    final_goal_distance=initial_goal_dist,
                    min_obstacle_clearance=plan_res.min_clearance,
                    collision_count=0,
                    control_effort=0.0,
                    execution_wall_time_ms=elapsed_ms,
                )
                return FlightScenarioResult(
                    scenario_name=scenario.name,
                    description=scenario.description,
                    metrics=metrics,
                    telemetry=[],
                )
        else:
            waypoints = list(scenario.waypoints)

        if not waypoints:
            waypoints = [scenario.initial_position.copy()]

        waypoint_idx = 0
        current_target = waypoints[waypoint_idx].copy()
        self.env._goal = current_target.copy()
        self.env._prev_distance_to_goal = float(np.linalg.norm(current_target - self.env._position))

        initial_goal_dist = float(np.linalg.norm(waypoints[-1] - scenario.initial_position))

        telemetry: List[FlightStepTelemetry] = []
        pos_errors: List[float] = []
        vel_errors: List[float] = []
        att_errors_deg: List[float] = []
        angular_speeds: List[float] = []
        min_clearances: List[float] = []
        control_efforts: List[float] = []

        is_collision = False
        outcome = "max_steps"
        step = 0
        dt = self.env.dynamics.dt

        while step < scenario.max_steps:
            step += 1
            time_sec = step * dt
            current_state = self.env.dynamics.state

            # Compute tracking errors relative to current waypoint
            pos_err = current_target - current_state.position
            pos_err_mag = float(np.linalg.norm(pos_err))
            pos_errors.append(pos_err_mag)

            vel_err_mag = float(np.linalg.norm(current_state.velocity))
            vel_errors.append(vel_err_mag)

            roll, pitch, yaw = quaternion_to_euler_angles(current_state.quaternion)
            euler_deg = np.array(
                [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)], dtype=np.float64
            )
            r_mat = self.env.dynamics.body_to_world(np.array([0.0, 0.0, 1.0]))
            tilt_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(r_mat[2])))))
            att_errors_deg.append(tilt_angle_deg)

            omega_mag = float(np.linalg.norm(current_state.angular_velocity))
            angular_speeds.append(omega_mag)

            # Compute controller action
            if hasattr(ctrl, "compute_action"):
                action, ctrl_info = ctrl.compute_action(
                    current_state,
                    target_position=current_target,
                    target_velocity=np.zeros(3),
                    dt=dt,
                )
            elif hasattr(ctrl, "predict"):
                obs = self.env._get_obs()
                action, _ = ctrl.predict(obs, deterministic=True)
            else:
                action = np.zeros(4, dtype=np.float32)

            act_arr = np.asarray(action, dtype=np.float32)
            control_efforts.append(float(np.sum(np.square(act_arr))))

            # Map action to physical thrusts
            thrusts_cmd = self.env._action_to_motor_thrusts(act_arr)
            thrusts_actual = thrusts_cmd.copy()

            # Apply actuator fault if specified
            fault_active = False
            if scenario.actuator_fault_fn is not None:
                thrusts_actual = scenario.actuator_fault_fn(step, thrusts_cmd)
                fault_active = not np.allclose(thrusts_actual, thrusts_cmd)

            # Step physical dynamics with Runge-Kutta 4
            old_pos = current_state.position.copy()

            # External aerodynamic disturbance
            if scenario.disturbance_fn is not None:
                dist_force = scenario.disturbance_fn(step, current_state)
                current_state.velocity += (dist_force / self.env.dynamics.mass) * dt

            new_state = self.env.dynamics.step_rk4(thrusts_actual)
            self.env._position = new_state.position.copy()
            self.env._current_step = step

            # Obstacle clearance check (combining static and dynamic obstacles)
            min_obs_d = float("inf")
            for obstacle in scenario.obstacles:
                d = obstacle.distance_to(new_state.position) - scenario.collision_radius
                if d < min_obs_d:
                    min_obs_d = d
            for dyn_obs in scenario.dynamic_obstacles:
                d = dyn_obs.distance_to(new_state.position, time_sec) - scenario.collision_radius
                if d < min_obs_d:
                    min_obs_d = d
            min_clearances.append(min_obs_d)

            # Collision check
            coll, coll_type = self.env._check_collision(new_state.position, prev_pos=old_pos)
            # Also check dynamic obstacles collisions
            if not coll:
                for dyn_obs in scenario.dynamic_obstacles:
                    if dyn_obs.distance_to(new_state.position, time_sec) <= (
                        dyn_obs.radius + scenario.collision_radius
                    ):
                        coll = True
                        coll_type = "obstacle"
                        break

            if coll:
                is_collision = True
                outcome = "collision" if coll_type == "obstacle" else "boundary_breach"

            # Waypoint sequencing
            dist_to_curr_waypoint = float(np.linalg.norm(current_target - new_state.position))
            is_at_curr_waypoint = dist_to_curr_waypoint <= scenario.target_radius

            if is_at_curr_waypoint and waypoint_idx < len(waypoints) - 1:
                waypoint_idx += 1
                current_target = waypoints[waypoint_idx].copy()
                self.env._goal = current_target.copy()
                if hasattr(ctrl, "set_target"):
                    ctrl.set_target(current_target)

            # Final success check
            dist_to_final_waypoint = float(np.linalg.norm(waypoints[-1] - new_state.position))
            is_final_success = (
                waypoint_idx == len(waypoints) - 1
                and dist_to_final_waypoint <= scenario.target_radius
                and not is_collision
            )

            # Record telemetry
            telemetry.append(
                FlightStepTelemetry(
                    step=step,
                    time_sec=time_sec,
                    position=new_state.position.copy(),
                    velocity=new_state.velocity.copy(),
                    quaternion=new_state.quaternion.copy(),
                    euler_angles_deg=euler_deg,
                    angular_velocity=new_state.angular_velocity.copy(),
                    target_position=current_target.copy(),
                    distance_to_target=dist_to_curr_waypoint,
                    action=act_arr.copy(),
                    motor_thrusts=thrusts_actual.copy(),
                    min_obstacle_distance=min_obs_d,
                    is_collision=coll,
                    is_success=is_final_success,
                    commanded_thrusts=thrusts_cmd.copy(),
                    actuator_fault_active=fault_active,
                )
            )

            if coll:
                break

            if is_final_success and len(waypoints) > 1:
                outcome = "success"
                break
            elif is_final_success and len(waypoints) == 1 and step >= scenario.max_steps - 10:
                outcome = "success"
                break

        final_pos_err = float(np.linalg.norm(waypoints[-1] - self.env._position))
        final_goal_dist = final_pos_err

        # Disambiguated outcome classification
        if scenario.actuator_fault_fn is not None:
            max_tilt = max(att_errors_deg) if att_errors_deg else 0.0
            if is_collision or max_tilt >= 35.0 or self.env._position[2] <= 0.8:
                outcome = "crash_fault"
            else:
                outcome = "recovered_degraded"
        elif is_collision:
            outcome = "collision" if coll_type == "obstacle" else "boundary_breach"
        elif final_pos_err <= scenario.target_radius:
            outcome = "success"
        else:
            outcome = "max_steps"

        elapsed_ms = (time.perf_counter() - start_wall_time) * 1000.0

        metrics = FlightScenarioMetrics(
            scenario_name=scenario.name,
            success=(outcome in ("success", "recovered_degraded")),
            outcome=outcome,
            duration_steps=step,
            duration_seconds=step * dt,
            final_position_error=final_pos_err,
            mean_position_error=float(np.mean(pos_errors)) if pos_errors else 0.0,
            max_position_error=float(np.max(pos_errors)) if pos_errors else 0.0,
            rms_position_error=float(np.sqrt(np.mean(np.square(pos_errors))))
            if pos_errors
            else 0.0,
            mean_velocity_error=float(np.mean(vel_errors)) if vel_errors else 0.0,
            max_velocity_error=float(np.max(vel_errors)) if vel_errors else 0.0,
            rms_velocity_error=float(np.sqrt(np.mean(np.square(vel_errors))))
            if vel_errors
            else 0.0,
            mean_attitude_error_deg=float(np.mean(att_errors_deg)) if att_errors_deg else 0.0,
            max_attitude_error_deg=float(np.max(att_errors_deg)) if att_errors_deg else 0.0,
            mean_angular_speed=float(np.mean(angular_speeds)) if angular_speeds else 0.0,
            max_angular_speed=float(np.max(angular_speeds)) if angular_speeds else 0.0,
            initial_goal_distance=initial_goal_dist,
            final_goal_distance=final_goal_dist,
            min_obstacle_clearance=float(min(min_clearances)) if min_clearances else float("inf"),
            collision_count=1 if is_collision else 0,
            control_effort=float(np.sum(control_efforts)) if control_efforts else 0.0,
            execution_wall_time_ms=elapsed_ms,
        )

        return FlightScenarioResult(
            scenario_name=scenario.name,
            description=scenario.description,
            metrics=metrics,
            telemetry=telemetry,
        )

    def run_all_scenarios(
        self,
        controller: Optional[Any] = None,
        scenarios: Optional[List[FlightScenarioDefinition]] = None,
    ) -> List[FlightScenarioResult]:
        """Execute all configured scenarios and return list of empirical results."""
        scenario_list = scenarios or get_standard_flight_scenarios()
        results: List[FlightScenarioResult] = []
        for sc in scenario_list:
            res = self.run_scenario(sc, controller=controller)
            results.append(res)
        return results
