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


def test_sequential_waypoints_preserved_during_dynamic_replan() -> None:
    """Dynamic replan must preserve all remaining sequential mission waypoints in order."""
    from adaptive_rl.environments.drone_6dof import DroneState6DOF
    from adaptive_rl.navigation.planner import AStarPlanner3D

    planner = AStarPlanner3D(bounds=(30.0, 30.0, 15.0), resolution=1.0)
    mgr = MissionManager(planner=planner, waypoint_acceptance_radius=1.0)

    # 3 sequential waypoints
    wp1 = np.array([5.0, 10.0, 5.0])
    wp2 = np.array([15.0, 10.0, 5.0])
    wp3 = np.array([25.0, 10.0, 5.0])

    ok = mgr.start_mission(
        start_position=np.array([2.0, 10.0, 5.0]),
        goals=[wp1, wp2, wp3],
        static_obstacles=[],
    )
    assert ok is True
    assert mgr.current_mission_waypoint_idx == 0
    assert len(mgr.mission_waypoints) == 3

    # Dynamic obstacle blocks path segment between drone and wp1
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([3.5, 10.0, 5.0]),
        radius=1.0,
        velocity=np.zeros(3),
    )

    state = DroneState6DOF(
        position=np.array([2.0, 10.0, 5.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )

    _, telem = mgr.update(
        drone_state=state,
        time_sec=0.0,
        static_obstacles=[],
        dynamic_obstacles=[dyn_obs],
    )

    # Should have replanned
    assert telem.replan_count >= 1
    # Active path must end at final waypoint wp3
    assert np.allclose(mgr.active_path[-1], wp3)
    # The replanned path MUST pass through wp1 before reaching wp2 and wp3
    dists_wp1 = [np.linalg.norm(p - wp1) for p in mgr.active_path]
    dists_wp2 = [np.linalg.norm(p - wp2) for p in mgr.active_path]
    dists_wp3 = [np.linalg.norm(p - wp3) for p in mgr.active_path]
    idx1 = int(np.argmin(dists_wp1))
    idx2 = int(np.argmin(dists_wp2))
    idx3 = int(np.argmin(dists_wp3))
    assert dists_wp1[idx1] <= 1.0, "Replan did not include wp1!"
    assert dists_wp2[idx2] <= 1.0, "Replan did not include wp2!"
    assert dists_wp3[idx3] <= 1.0, "Replan did not include wp3!"
    assert idx1 < idx2 < idx3, "Waypoints in active path are out of order!"
    assert mgr.current_mission_waypoint_idx == 0


def test_mission_cannot_arrive_skipping_intermediate_waypoints() -> None:
    """Drone reaching final waypoint coordinates without visiting intermediate waypoints cannot achieve ARRIVED."""
    from adaptive_rl.environments.drone_6dof import DroneState6DOF
    from adaptive_rl.navigation.mission import MissionState
    from adaptive_rl.navigation.planner import AStarPlanner3D

    planner = AStarPlanner3D(bounds=(30.0, 30.0, 15.0), resolution=1.0)
    mgr = MissionManager(planner=planner, waypoint_acceptance_radius=1.0)

    wp1 = np.array([5.0, 10.0, 5.0])
    wp2 = np.array([15.0, 10.0, 5.0])
    wp3 = np.array([25.0, 10.0, 5.0])

    mgr.start_mission(
        start_position=np.array([2.0, 10.0, 5.0]),
        goals=[wp1, wp2, wp3],
        static_obstacles=[],
    )

    # Teleport directly to wp3 without touching wp1 or wp2
    state_at_final = DroneState6DOF(
        position=wp3.copy(),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )

    _, telem = mgr.update(
        drone_state=state_at_final,
        time_sec=1.0,
        static_obstacles=[],
    )

    # Mission must NOT be ARRIVED because intermediate waypoints were skipped
    assert mgr.state != MissionState.ARRIVED
    assert telem.state != MissionState.ARRIVED
    assert mgr.current_mission_waypoint_idx < 3


def test_failed_replan_hovers_safely_without_advancing_mission() -> None:
    """When replanning fails (e.g. completely boxed in), drone holds position safely and does not advance."""
    from adaptive_rl.environments.drone import ObstacleSphere3D
    from adaptive_rl.environments.drone_6dof import DroneState6DOF
    from adaptive_rl.navigation.mission import MissionState
    from adaptive_rl.navigation.planner import AStarPlanner3D

    planner = AStarPlanner3D(bounds=(30.0, 30.0, 15.0), resolution=1.0)
    mgr = MissionManager(planner=planner, waypoint_acceptance_radius=1.0)

    wp1 = np.array([10.0, 10.0, 5.0])
    mgr.start_mission(
        start_position=np.array([5.0, 10.0, 5.0]),
        goals=[wp1],
        static_obstacles=[],
    )

    pos = np.array([5.0, 10.0, 5.0])
    blocking_dyn = DynamicObstacleSphere3D(
        center_initial=np.array([6.0, 10.0, 5.0]),
        radius=2.0,
        velocity=np.zeros(3),
    )
    goal_blocking = ObstacleSphere3D(center=wp1, radius=2.0)

    state = DroneState6DOF(
        position=pos.copy(),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )

    target, _ = mgr.update(
        drone_state=state,
        time_sec=0.0,
        static_obstacles=[goal_blocking],
        dynamic_obstacles=[blocking_dyn],
    )

    # Target must be safe hover at current position
    assert np.allclose(target, pos)
    assert mgr.state == MissionState.REPLANNING
    assert mgr.current_mission_waypoint_idx == 0


def test_oscillating_obstacle_speed_bound_and_collision_detection() -> None:
    """Oscillating obstacle with zero linear velocity calculates valid speed bound and detects swept collision."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([10.0, 10.0, 5.0]),
        radius=1.0,
        velocity=np.zeros(3),  # Zero linear velocity
        motion_type="oscillate",
        oscillation_period=2.0,  # omega = 2*pi / 2 = pi
        oscillation_amplitude=np.array([10.0, 0.0, 0.0]),  # Amplitude 10m
    )
    max_spd = obs.max_speed()
    assert abs(max_spd - (np.pi * 10.0)) < 1e-4

    coll = obs.check_continuous_collision(
        prev_drone_pos=np.array([19.8, 10.0, 5.0]),
        curr_drone_pos=np.array([19.8, 10.0, 5.0]),
        prev_time=0.4,
        curr_time=0.6,
        vehicle_radius=0.5,
    )
    assert coll is True


def test_predict_occupied_spheres_zero_gaps() -> None:
    """Predictive occupied spheres cover swept volume densely with spacing <= 1.2 * radius (zero gaps)."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([5.0, 5.0, 5.0]),
        radius=1.0,
        velocity=np.array([10.0, 0.0, 0.0]),  # Moves 10 m/s
        bounds=(50.0, 50.0, 20.0),
    )
    spheres = obs.predict_occupied_spheres(t_start=0.0, horizon_sec=2.0, dt=0.5)
    assert len(spheres) > 5

    for i in range(len(spheres) - 1):
        dist = float(np.linalg.norm(spheres[i + 1].center - spheres[i].center))
        assert dist <= 1.2 * obs.radius + 1e-4, (
            f"Gap detected at index {i}: dist={dist} > {1.2 * obs.radius}"
        )
