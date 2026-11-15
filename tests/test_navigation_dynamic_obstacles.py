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


def test_obstacle_parameter_validation() -> None:
    """DynamicObstacleSphere3D validates positive radius, valid bounds, positive period, and motion type."""
    import pytest

    # Non-positive radius
    with pytest.raises(ValueError, match="strictly positive"):
        DynamicObstacleSphere3D(center_initial=np.array([10.0, 10.0, 5.0]), radius=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        DynamicObstacleSphere3D(center_initial=np.array([10.0, 10.0, 5.0]), radius=-0.5)

    # Invalid bounds <= 2 * radius
    with pytest.raises(ValueError, match=r"Arena bounds must all exceed 2 \* radius"):
        DynamicObstacleSphere3D(
            center_initial=np.array([10.0, 10.0, 5.0]),
            radius=5.0,
            bounds=(10.0, 30.0, 20.0),  # bounds[0] = 10 <= 2 * 5
        )

    # Invalid motion type
    with pytest.raises(ValueError, match="Unsupported motion_type"):
        DynamicObstacleSphere3D(
            center_initial=np.array([10.0, 10.0, 5.0]),
            radius=1.0,
            motion_type="teleport",
        )

    # Non-positive oscillation period
    with pytest.raises(ValueError, match="Oscillation period must be strictly positive"):
        DynamicObstacleSphere3D(
            center_initial=np.array([10.0, 10.0, 5.0]),
            radius=1.0,
            motion_type="oscillate",
            oscillation_period=0.0,
        )


def test_predict_occupied_spheres_period_equal_sampling_interval() -> None:
    """Obstacle oscillating with period equal to sampling interval does not lose intermediate swept path."""
    T = 0.5
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([10.0, 10.0, 5.0]),
        radius=0.5,
        velocity=np.zeros(3),
        motion_type="oscillate",
        oscillation_period=T,
        oscillation_amplitude=np.array([4.0, 0.0, 0.0]),
        bounds=(30.0, 30.0, 15.0),
    )
    # Samples at t=0 and t=0.5 are both at x=10.0 (endpoint displacement is 0.0!)
    assert np.allclose(obs.center_at(0.0), obs.center_at(T))

    spheres = obs.predict_occupied_spheres(t_start=0.0, horizon_sec=T, dt=T)
    assert len(spheres) > 5

    x_coords = [s.center[0] for s in spheres]
    # At t = T/4 = 0.125s, x reaches 14.0; at t = 3T/4 = 0.375s, x reaches 6.0
    assert max(x_coords) >= 13.5, f"Peak amplitude not captured: {max(x_coords)}"
    assert min(x_coords) <= 6.5, f"Trough amplitude not captured: {min(x_coords)}"

    # Ensure no gaps between consecutive spheres
    for i in range(len(spheres) - 1):
        dist = float(np.linalg.norm(spheres[i + 1].center - spheres[i].center))
        assert dist <= 1.2 * spheres[i].radius + 1e-4


def test_predict_occupied_spheres_multiple_cycles_between_samples() -> None:
    """Multiple full oscillation cycles occurring between samples are fully captured without gaps."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([10.0, 10.0, 5.0]),
        radius=0.5,
        velocity=np.zeros(3),
        motion_type="oscillate",
        oscillation_period=0.1,  # 10 full cycles in 1.0 second!
        oscillation_amplitude=np.array([5.0, 0.0, 0.0]),
        bounds=(30.0, 30.0, 15.0),
    )
    spheres = obs.predict_occupied_spheres(t_start=0.0, horizon_sec=1.0, dt=1.0)
    assert len(spheres) > 20

    x_coords = [s.center[0] for s in spheres]
    assert max(x_coords) >= 14.5
    assert min(x_coords) <= 5.5

    for i in range(len(spheres) - 1):
        dist = float(np.linalg.norm(spheres[i + 1].center - spheres[i].center))
        assert dist <= 1.2 * spheres[i].radius + 1e-4


def test_predict_occupied_spheres_bounce_returning_to_start() -> None:
    """Bouncing obstacle reflecting and returning to its start position preserves its full swept path."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([2.0, 10.0, 5.0]),
        radius=1.0,
        velocity=np.array(
            [28.0, 0.0, 0.0]
        ),  # Travels 28m in 1s, bounces, travels back in 1s (cycle=2s)
        bounds=(30.0, 30.0, 15.0),
        motion_type="bounce",
    )
    c0 = obs.center_at(0.0)
    c2 = obs.center_at(2.0)
    assert np.allclose(c0, c2)

    spheres = obs.predict_occupied_spheres(t_start=0.0, horizon_sec=2.0, dt=2.0)
    x_coords = [s.center[0] for s in spheres]
    assert max(x_coords) >= 27.0
    assert min(x_coords) <= 3.0

    for i in range(len(spheres) - 1):
        dist = float(np.linalg.norm(spheres[i + 1].center - spheres[i].center))
        assert dist <= 1.2 * spheres[i].radius + 1e-4


def test_predict_occupied_spheres_tiny_radius_no_floor_violation() -> None:
    """Very small positive obstacle radius (0.01m) strictly satisfies <= 1.2 * radius spacing without floor violation."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([10.0, 10.0, 5.0]),
        radius=0.01,  # 1 cm radius
        velocity=np.array([0.1, 0.0, 0.0]),  # 10 cm/s
        bounds=(30.0, 30.0, 15.0),
    )
    spheres = obs.predict_occupied_spheres(t_start=0.0, horizon_sec=0.2, dt=0.2)
    assert len(spheres) >= 2
    for i in range(len(spheres) - 1):
        dist = float(np.linalg.norm(spheres[i + 1].center - spheres[i].center))
        assert dist <= 1.2 * obs.radius + 1e-6, (
            f"Spacing {dist} exceeded 1.2 * radius ({1.2 * obs.radius})"
        )


def test_predict_occupied_spheres_computational_limit_conservative() -> None:
    """When needed steps exceed max_spheres, radius is expanded conservatively so no gaps occur."""
    obs = DynamicObstacleSphere3D(
        center_initial=np.array([5.0, 5.0, 5.0]),
        radius=0.001,  # 1 mm radius
        velocity=np.array([50.0, 0.0, 0.0]),  # 50 m/s
        bounds=(100.0, 100.0, 50.0),
    )
    spheres = obs.predict_occupied_spheres(t_start=0.0, horizon_sec=1.0, dt=1.0, max_spheres=100)
    assert len(spheres) == 101

    eff_r = spheres[0].radius
    assert eff_r > obs.radius
    for i in range(len(spheres) - 1):
        dist = float(np.linalg.norm(spheres[i + 1].center - spheres[i].center))
        assert dist <= 1.2 * eff_r + 1e-4, (
            f"Gap opened under computational limit: dist={dist} > 1.2 * eff_r"
        )


def test_end_to_end_repeated_replanning_sequential_progression() -> None:
    """P2: End-to-end mission with repeated dynamic obstacle interruptions preserves sequential progression."""
    from adaptive_rl.environments.drone_6dof import DroneState6DOF
    from adaptive_rl.navigation.mission import MissionManager, MissionState
    from adaptive_rl.navigation.planner import AStarPlanner3D

    planner = AStarPlanner3D(
        bounds=(30.0, 30.0, 15.0),
        resolution=1.0,
        vehicle_radius=0.4,
        safety_margin=0.2,
    )
    mgr = MissionManager(planner=planner, waypoint_acceptance_radius=0.8)

    # 3 sequential high-level waypoints
    wp1 = np.array([10.0, 10.0, 5.0])
    wp2 = np.array([20.0, 10.0, 5.0])
    wp3 = np.array([28.0, 10.0, 5.0])

    ok = mgr.start_mission(
        start_position=np.array([2.0, 10.0, 5.0]),
        goals=[wp1, wp2, wp3],
        static_obstacles=[],
    )
    assert ok is True
    assert mgr.current_mission_waypoint_idx == 0

    # 1. Obstacle 1 appears during segment 1 (at x=6.0, y=10.0)
    obs1 = DynamicObstacleSphere3D(
        center_initial=np.array([6.0, 10.0, 5.0]),
        radius=0.6,
        velocity=np.zeros(3),
    )
    drone_state = DroneState6DOF(
        position=np.array([2.0, 10.0, 5.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    tgt, telem = mgr.update(
        drone_state, time_sec=0.1, static_obstacles=[], dynamic_obstacles=[obs1]
    )
    assert telem.replan_count == 1
    assert mgr.state == MissionState.EXECUTING
    assert mgr.current_mission_waypoint_idx == 0

    # Fly through replanned path to reach wp1
    while mgr.current_mission_waypoint_idx == 0:
        drone_state.position = mgr.active_path[mgr.active_path_idx].copy()
        tgt, telem = mgr.update(
            drone_state, time_sec=1.0, static_obstacles=[], dynamic_obstacles=[obs1]
        )
    assert mgr.current_mission_waypoint_idx == 1  # Successfully reached wp1!

    # 2. Obstacle 2 appears during segment 2 (at x=15.0, y=10.0)
    obs2 = DynamicObstacleSphere3D(
        center_initial=np.array([15.0, 10.0, 5.0]),
        radius=0.6,
        velocity=np.zeros(3),
    )
    tgt, telem = mgr.update(
        drone_state, time_sec=2.0, static_obstacles=[], dynamic_obstacles=[obs1, obs2]
    )
    assert telem.replan_count == 2
    assert mgr.state == MissionState.EXECUTING
    assert mgr.current_mission_waypoint_idx == 1  # Still on segment to wp2

    # Fly through replanned path to reach wp2
    while mgr.current_mission_waypoint_idx == 1:
        drone_state.position = mgr.active_path[mgr.active_path_idx].copy()
        tgt, telem = mgr.update(
            drone_state, time_sec=3.0, static_obstacles=[], dynamic_obstacles=[obs1, obs2]
        )
    assert mgr.current_mission_waypoint_idx == 2  # Successfully reached wp2!

    # 3. Proceed along path to wp3
    while mgr.current_mission_waypoint_idx < 3:
        drone_state.position = mgr.active_path[mgr.active_path_idx].copy()
        tgt, telem = mgr.update(
            drone_state, time_sec=4.0, static_obstacles=[], dynamic_obstacles=[obs1, obs2]
        )

    # Arrived at final waypoint wp3
    assert mgr.current_mission_waypoint_idx == 3
    assert mgr.state == MissionState.ARRIVED
    assert telem.state == MissionState.ARRIVED
    assert telem.replan_count == 2
