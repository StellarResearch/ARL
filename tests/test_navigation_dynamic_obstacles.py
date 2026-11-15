"""Unit tests for continuous dynamic obstacle collision detection and predictive evasion (Phase III, Task 5)."""

from __future__ import annotations

import numpy as np

from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.mission import MissionManager


def test_stationary_drone_crossed_by_moving_obstacle() -> None:
    """A stationary drone crossed by a moving obstacle between endpoints must register collision."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([0.0, 10.0, 5.0]),
        radius=1.0,
        velocity=np.array([20.0, 0.0, 0.0]),  # Translates from x=0 to x=20 in 1 second
        bounds=(50.0, 50.0, 20.0),
        motion_type="bounce",
    )
    drone_pos = np.array([10.0, 10.0, 5.0])
    vehicle_radius = 0.5

    t0 = 0.0
    t1 = 1.0

    # At t0=0, obs is at x=0 (dist=10 > 1.5)
    # At t1=1, obs is at x=20 (dist=10 > 1.5)
    # Neither endpoint registers collision
    assert obs.distance_to(drone_pos, t0) > (obs.radius + vehicle_radius)
    assert obs.distance_to(drone_pos, t1) > (obs.radius + vehicle_radius)

    # Continuous swept check must detect collision at t=0.5s when obstacle passes through x=10
    coll = obs.check_continuous_collision(
        prev_drone_pos=drone_pos,
        curr_drone_pos=drone_pos,
        prev_time=t0,
        curr_time=t1,
        vehicle_radius=vehicle_radius,
    )
    assert coll is True


def test_moving_drone_crossing_stationary_obstacle() -> None:
    """A moving drone crossing a stationary obstacle between endpoints must register collision."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([10.0, 10.0, 5.0]),
        radius=1.0,
        velocity=np.zeros(3),
        bounds=(50.0, 50.0, 20.0),
    )
    vehicle_radius = 0.5

    p0 = np.array([0.0, 10.0, 5.0])
    p1 = np.array([20.0, 10.0, 5.0])
    t0 = 0.0
    t1 = 1.0

    assert obs.distance_to(p0, t0) > (obs.radius + vehicle_radius)
    assert obs.distance_to(p1, t1) > (obs.radius + vehicle_radius)

    coll = obs.check_continuous_collision(
        prev_drone_pos=p0,
        curr_drone_pos=p1,
        prev_time=t0,
        curr_time=t1,
        vehicle_radius=vehicle_radius,
    )
    assert coll is True


def test_two_moving_objects_crossing_between_step_endpoints() -> None:
    """Two moving objects crossing between step endpoints must register collision even if endpoints are clear."""
    # Drone moving along Y from (10, 0, 5) to (10, 20, 5) in 1s
    p0 = np.array([10.0, 0.0, 5.0])
    p1 = np.array([10.0, 20.0, 5.0])

    # Obstacle moving along X from (0, 10, 5) to (20, 10, 5) in 1s
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([0.0, 10.0, 5.0]),
        radius=0.8,
        velocity=np.array([20.0, 0.0, 0.0]),
        bounds=(50.0, 50.0, 20.0),
    )
    vehicle_radius = 0.5
    t0 = 0.0
    t1 = 1.0

    # At t0=0: drone=(10, 0, 5), obs=(0, 10, 5), dist = sqrt(100+100) = 14.14 > 1.3
    # At t1=1: drone=(10, 20, 5), obs=(20, 10, 5), dist = sqrt(100+100) = 14.14 > 1.3
    assert np.linalg.norm(p0 - obs.center_at(t0)) > 1.3
    assert np.linalg.norm(p1 - obs.center_at(t1)) > 1.3

    # At t=0.5: drone=(10, 10, 5), obs=(10, 10, 5), collision!
    coll = obs.check_continuous_collision(
        prev_drone_pos=p0,
        curr_drone_pos=p1,
        prev_time=t0,
        curr_time=t1,
        vehicle_radius=vehicle_radius,
    )
    assert coll is True


def test_high_speed_obstacle_swept_detection() -> None:
    """High-speed obstacle (40 m/s) over a short interval (0.05s) does not tunnel through the aircraft."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([8.0, 10.0, 5.0]),
        radius=0.6,
        velocity=np.array([40.0, 0.0, 0.0]),  # 2.0 meters displacement in 0.05s
        bounds=(50.0, 50.0, 20.0),
    )
    drone_pos = np.array([9.0, 10.0, 5.0])
    vehicle_radius = 0.3

    t0 = 0.0
    t1 = 0.05  # Obstacle moves from x=8.0 to x=10.0; drone is at x=9.0

    coll = obs.check_continuous_collision(
        prev_drone_pos=drone_pos,
        curr_drone_pos=drone_pos,
        prev_time=t0,
        curr_time=t1,
        vehicle_radius=vehicle_radius,
    )
    assert coll is True


def test_safe_parallel_trajectories_no_false_collision() -> None:
    """Safe parallel trajectories with clearance never trigger false positive collision."""
    p0 = np.array([5.0, 0.0, 5.0])
    p1 = np.array([5.0, 20.0, 5.0])

    obs = DynamicObstacleSphere3D(
        center_initial=np.array([12.0, 0.0, 5.0]),  # 7 meters lateral separation
        radius=1.0,
        velocity=np.array([0.0, 20.0, 0.0]),
        bounds=(50.0, 50.0, 20.0),
    )

    coll = obs.check_continuous_collision(
        prev_drone_pos=p0,
        curr_drone_pos=p1,
        prev_time=0.0,
        curr_time=1.0,
        vehicle_radius=0.5,
    )
    assert coll is False


def test_dynamic_obstacle_predictive_evasion_replanning() -> None:
    """MissionManager predictive trajectory checking detects collision and triggers autonomous replanning."""
    from adaptive_rl.environments.drone_6dof import DroneState6DOF
    from adaptive_rl.navigation.planner import AStarPlanner3D

    # Target path from (5, 10, 5) to (25, 10, 5)
    planner = AStarPlanner3D(bounds=(30.0, 30.0, 15.0), resolution=1.0)
    mgr = MissionManager(planner=planner)
    success = mgr.start_mission(
        start_position=np.array([5.0, 10.0, 5.0]),
        goals=[np.array([25.0, 10.0, 5.0])],
        static_obstacles=[],
        dynamic_obstacles=[],
    )
    assert success is True

    # Dynamic obstacle crossing path at x=15, y=5 moving towards y=15
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([15.0, 5.0, 5.0]),
        radius=1.5,
        velocity=np.array([0.0, 2.0, 0.0]),
        bounds=(30.0, 30.0, 15.0),
    )

    state = DroneState6DOF(
        position=np.array([5.0, 10.0, 5.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )

    # At t=2.0s, obstacle reaches y=9.0 (blocking the path at x=15, y=10)
    current_target, telem = mgr.update(
        drone_state=state,
        time_sec=2.0,
        static_obstacles=[],
        dynamic_obstacles=[dyn_obs],
    )

    # MissionManager should detect the predicted intrusion and successfully replan
    assert telem.replan_count >= 1
