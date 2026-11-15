"""Comprehensive physics validation suite for 6-DOF Quadrotor Dynamics (Issue #253, #277).

Tests independently verified analytical solutions and invariants for:
1. Hover equilibrium (T = mg)
2. Free fall acceleration ([0, 0, -g])
3. Pure roll torque
4. Pure pitch torque
5. Pure yaw torque
6. Translational aerodynamic drag (exponential velocity decay)
7. Quaternion normalization during prolonged integration
8. Rotation matrix orthogonality and determinant
9. Motor thrust saturation
10. Numerical stability under high angular rates
11. Finite values under extreme motor commands
12. Parameter validation, units, dimensions, and non-finite input handling
13. Continuous swept collision detection against obstacles and boundaries
14. Reward component decomposition
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import (
    Drone6DOFEnv,
    DroneDynamics6DOF,
    body_to_world_vector,
    euler_to_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_euler_angles,
    quaternion_to_rotation_matrix,
    world_to_body_vector,
)


# ---------------------------------------------------------------------------
# 1. Hover Equilibrium
# ---------------------------------------------------------------------------
def test_hover_equilibrium_analytical() -> None:
    """Hover thrust T_total = m * g produces zero linear and angular acceleration."""
    mass = 1.25
    gravity = 9.81
    dynamics = DroneDynamics6DOF(mass=mass, gravity=gravity, dt=0.01, substeps=1)

    expected_total_thrust = mass * gravity
    expected_hover_per_motor = expected_total_thrust / 4.0
    assert math.isclose(dynamics.hover_thrust_per_motor, expected_hover_per_motor, rel_tol=1e-6)

    # Level attitude at rest
    pos_init = np.array([12.0, 12.0, 6.0])
    dynamics.reset(position=pos_init, velocity=np.zeros(3))

    thrusts = np.full(4, expected_hover_per_motor)
    total_t, torques = dynamics.compute_forces_and_torques(thrusts)
    assert math.isclose(total_t, expected_total_thrust, rel_tol=1e-6)
    assert np.allclose(torques, 0.0, atol=1e-9)

    d_pos, d_vel, d_quat, d_omega = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        total_t,
        torques,
    )
    assert np.allclose(d_pos, 0.0, atol=1e-9)
    assert np.allclose(d_vel, 0.0, atol=1e-9)
    assert np.allclose(d_quat, 0.0, atol=1e-9)
    assert np.allclose(d_omega, 0.0, atol=1e-9)

    # Step forward 100 steps; position and velocity must remain invariant
    for _ in range(100):
        state = dynamics.step_rk4(thrusts)

    assert np.allclose(state.position, pos_init, atol=1e-5)
    assert np.allclose(state.velocity, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Free Fall Acceleration
# ---------------------------------------------------------------------------
def test_free_fall_acceleration() -> None:
    """Zero thrust with zero damping yields exactly [0, 0, -g] acceleration."""
    gravity = 9.80665
    dynamics = DroneDynamics6DOF(mass=0.85, gravity=gravity, linear_damping=0.0)
    dynamics.reset(position=np.array([5.0, 5.0, 10.0]), velocity=np.zeros(3))

    zero_thrust = np.zeros(4)
    total_t, torques = dynamics.compute_forces_and_torques(zero_thrust)
    assert total_t == 0.0

    _, d_vel, _, _ = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        total_t,
        torques,
    )
    assert np.allclose(d_vel, [0.0, 0.0, -gravity], atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Pure Roll Torque
# ---------------------------------------------------------------------------
def test_pure_roll_torque_uncoupled() -> None:
    """Differential left/right thrust produces pure roll torque without pitch or yaw."""
    arm_len = 0.25
    d = arm_len / math.sqrt(2.0)
    dynamics = DroneDynamics6DOF(arm_length=arm_len, inertia=(0.02, 0.02, 0.04))
    f_h = dynamics.hover_thrust_per_motor
    delta = 0.4

    # Motors: F1 (FR), F2 (FL), F3 (RL), F4 (RR)
    # Roll torque = d * (-F1 + F2 + F3 - F4)
    # Pitch torque = d * (-F1 - F2 + F3 + F4)
    # Yaw torque = c_tau * (+F1 - F2 + F3 - F4)
    thrusts = np.array([f_h - delta, f_h + delta, f_h + delta, f_h - delta])

    total_t, torques = dynamics.compute_forces_and_torques(thrusts)
    expected_roll_torque = d * (-(f_h - delta) + (f_h + delta) + (f_h + delta) - (f_h - delta))
    assert math.isclose(total_t, 4.0 * f_h, rel_tol=1e-6)
    assert math.isclose(torques[0], expected_roll_torque, rel_tol=1e-6)
    assert math.isclose(torques[1], 0.0, abs_tol=1e-9)
    assert math.isclose(torques[2], 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 4. Pure Pitch Torque
# ---------------------------------------------------------------------------
def test_pure_pitch_torque_uncoupled() -> None:
    """Differential front/rear thrust produces pure pitch torque without roll or yaw."""
    arm_len = 0.25
    d = arm_len / math.sqrt(2.0)
    dynamics = DroneDynamics6DOF(arm_length=arm_len, inertia=(0.02, 0.02, 0.04))
    f_h = dynamics.hover_thrust_per_motor
    delta = 0.4

    thrusts = np.array([f_h - delta, f_h - delta, f_h + delta, f_h + delta])
    total_t, torques = dynamics.compute_forces_and_torques(thrusts)
    expected_pitch_torque = d * (-(f_h - delta) - (f_h - delta) + (f_h + delta) + (f_h + delta))

    assert math.isclose(total_t, 4.0 * f_h, rel_tol=1e-6)
    assert math.isclose(torques[0], 0.0, abs_tol=1e-9)
    assert math.isclose(torques[1], expected_pitch_torque, rel_tol=1e-6)
    assert math.isclose(torques[2], 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 5. Pure Yaw Torque
# ---------------------------------------------------------------------------
def test_pure_yaw_torque_uncoupled() -> None:
    """Differential CCW/CW thrust produces pure yaw torque without roll or pitch."""
    c_tau = 0.015
    dynamics = DroneDynamics6DOF(torque_to_thrust_ratio=c_tau, inertia=(0.02, 0.02, 0.04))
    f_h = dynamics.hover_thrust_per_motor
    delta = 0.4

    thrusts = np.array([f_h + delta, f_h - delta, f_h + delta, f_h - delta])
    total_t, torques = dynamics.compute_forces_and_torques(thrusts)
    expected_yaw_torque = c_tau * (+(f_h + delta) - (f_h - delta) + (f_h + delta) - (f_h - delta))

    assert math.isclose(total_t, 4.0 * f_h, rel_tol=1e-6)
    assert math.isclose(torques[0], 0.0, abs_tol=1e-9)
    assert math.isclose(torques[1], 0.0, abs_tol=1e-9)
    assert math.isclose(torques[2], expected_yaw_torque, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 6. Translational Aerodynamic Drag
# ---------------------------------------------------------------------------
def test_translational_drag_analytical_decay() -> None:
    """Horizontal velocity under linear drag matches analytical v(t) = v0 * exp(-D_v/m * t)."""
    mass = 1.0
    linear_damping = 0.2
    dt = 0.01
    dynamics = DroneDynamics6DOF(
        mass=mass,
        linear_damping=linear_damping,
        gravity=0.0,  # disable gravity to isolate horizontal drag
        dt=dt,
        substeps=1,
    )
    v0_x = 8.0
    dynamics.reset(position=np.zeros(3), velocity=np.array([v0_x, 0.0, 0.0]))

    zero_thrust = np.zeros(4)
    n_steps = 100
    for step in range(n_steps):
        state = dynamics.step_rk4(zero_thrust)
        t = (step + 1) * dt
        analytical_vx = v0_x * math.exp(-(linear_damping / mass) * t)
        assert math.isclose(state.velocity[0], analytical_vx, rel_tol=1e-3, abs_tol=1e-4)
        assert math.isclose(state.velocity[1], 0.0, abs_tol=1e-9)
        assert math.isclose(state.velocity[2], 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 7. Quaternion Normalization During Prolonged Integration
# ---------------------------------------------------------------------------
def test_quaternion_normalization_prolonged() -> None:
    """Quaternion attitude norm remains strictly 1.0 within 1e-6 over 2,000 steps."""
    dynamics = DroneDynamics6DOF(dt=0.02, substeps=2)
    rng = np.random.default_rng(9876)

    for _ in range(2000):
        thrusts = rng.uniform(0.0, 5.0, size=4)
        state = dynamics.step_rk4(thrusts)
        norm = np.linalg.norm(state.quaternion)
        assert math.isclose(norm, 1.0, abs_tol=1e-6)


@pytest.mark.slow
def test_quaternion_stability_50000_steps() -> None:
    """Verify 50,000-step numerical quaternion stability (Issue #277 requirement)."""
    dynamics = DroneDynamics6DOF(dt=0.02, substeps=1)
    rng = np.random.default_rng(42)

    for i in range(50000):
        thrusts = rng.uniform(1.0, 4.0, size=4)
        state = dynamics.step_rk4(thrusts)
        if i % 5000 == 0:
            norm = float(np.linalg.norm(state.quaternion))
            assert math.isclose(norm, 1.0, abs_tol=1e-5)
            assert np.all(np.isfinite(state.position))
            assert np.all(np.isfinite(state.velocity))
            assert np.all(np.isfinite(state.angular_velocity))

    final_norm = float(np.linalg.norm(state.quaternion))
    assert math.isclose(final_norm, 1.0, abs_tol=1e-5)


@pytest.mark.slow
def test_quaternion_stress_100000_steps() -> None:
    """Verify 100,000-step numerical stress requirement from Issue #277."""
    dynamics = DroneDynamics6DOF(dt=0.01, substeps=1)
    rng = np.random.default_rng(2024)

    for i in range(100000):
        thrusts = rng.uniform(0.5, 4.5, size=4)
        state = dynamics.step_rk4(thrusts)
        if i % 10000 == 0:
            norm = float(np.linalg.norm(state.quaternion))
            assert math.isclose(norm, 1.0, abs_tol=1e-5)
            assert np.all(np.isfinite(state.position))
            assert np.all(np.isfinite(state.velocity))
            assert np.all(np.isfinite(state.angular_velocity))

    final_norm = float(np.linalg.norm(state.quaternion))
    assert math.isclose(final_norm, 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 8. Rotation Matrix Orthogonality, Determinant, and Euler Angles
# ---------------------------------------------------------------------------
def test_rotation_matrix_and_euler_conversion() -> None:
    """Verify R * R^T = I, det(R) = +1, and roundtrip Euler angle conversion."""
    # 1. Identity quaternion
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    r_ident = quaternion_to_rotation_matrix(q_ident)
    assert np.allclose(r_ident, np.eye(3), atol=1e-9)
    roll, pitch, yaw = quaternion_to_euler_angles(q_ident)
    assert math.isclose(roll, 0.0, abs_tol=1e-9)
    assert math.isclose(pitch, 0.0, abs_tol=1e-9)
    assert math.isclose(yaw, 0.0, abs_tol=1e-9)

    # 2. Known 90-degree yaw rotation (pi/2 about Z)
    q_yaw90 = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    r_yaw90 = quaternion_to_rotation_matrix(q_yaw90)
    assert np.allclose(r_yaw90 @ r_yaw90.T, np.eye(3), atol=1e-7)
    assert math.isclose(np.linalg.det(r_yaw90), 1.0, abs_tol=1e-7)
    roll, pitch, yaw = quaternion_to_euler_angles(q_yaw90)
    assert math.isclose(roll, 0.0, abs_tol=1e-6)
    assert math.isclose(pitch, 0.0, abs_tol=1e-6)
    assert math.isclose(yaw, math.pi / 2, abs_tol=1e-6)

    # 3. Known 45-degree pitch rotation
    q_pitch45 = np.array([math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0])
    r_pitch45 = quaternion_to_rotation_matrix(q_pitch45)
    assert np.allclose(r_pitch45 @ r_pitch45.T, np.eye(3), atol=1e-7)
    assert math.isclose(np.linalg.det(r_pitch45), 1.0, abs_tol=1e-7)
    roll, pitch, yaw = quaternion_to_euler_angles(q_pitch45)
    assert math.isclose(roll, 0.0, abs_tol=1e-6)
    assert math.isclose(pitch, math.pi / 4, abs_tol=1e-6)
    assert math.isclose(yaw, 0.0, abs_tol=1e-6)

    # 4. Random attitudes
    rng = np.random.default_rng(555)
    for _ in range(50):
        q_rand = rng.normal(size=4)
        q_rand /= np.linalg.norm(q_rand)
        r_mat = quaternion_to_rotation_matrix(q_rand)
        assert np.allclose(r_mat @ r_mat.T, np.eye(3), atol=1e-6)
        assert math.isclose(np.linalg.det(r_mat), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 9. Motor Thrust Saturation
# ---------------------------------------------------------------------------
def test_motor_thrust_saturation_policy() -> None:
    """Motor thrust commands out of [0, max_thrust] are safely saturated."""
    max_thrust = 6.0
    dynamics = DroneDynamics6DOF(max_thrust_per_motor=max_thrust)

    extreme_inputs = np.array([-50.0, 100.0, 3.0, -0.001])
    total_t, _ = dynamics.compute_forces_and_torques(extreme_inputs)
    # Expected clamping: [0.0, 6.0, 3.0, 0.0] -> sum = 9.0
    assert math.isclose(total_t, 9.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 10. Numerical Stability Under High Angular Rates
# ---------------------------------------------------------------------------
def test_numerical_stability_high_angular_rates() -> None:
    """Dynamics handles extreme body angular rates (50+ rad/s) without NaN or divergence."""
    dynamics = DroneDynamics6DOF(dt=0.01, substeps=2)
    extreme_omega = np.array([45.0, -35.0, 60.0])
    dynamics.reset(position=np.array([10.0, 10.0, 10.0]), angular_velocity=extreme_omega)

    thrusts = np.full(4, dynamics.hover_thrust_per_motor)
    for _ in range(50):
        state = dynamics.step_rk4(thrusts)
        assert np.all(np.isfinite(state.position))
        assert np.all(np.isfinite(state.velocity))
        assert np.all(np.isfinite(state.quaternion))
        assert np.all(np.isfinite(state.angular_velocity))
        assert math.isclose(np.linalg.norm(state.quaternion), 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 11. Repeated Extreme Motor Commands
# ---------------------------------------------------------------------------
def test_finite_values_under_extreme_motor_commands() -> None:
    """Repeated bang-bang full/zero thrust commands maintain strictly finite values."""
    env = Drone6DOFEnv(max_steps=200)
    obs, info = env.reset(seed=123)

    for step in range(150):
        # Alternate between max thrust [1, 1, 1, 1] and zero thrust [-1, -1, -1, -1]
        action = np.ones(4, dtype=np.float32) if (step % 2 == 0) else -np.ones(4, dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert math.isfinite(reward)
        assert np.all(np.isfinite(step_info["position"]))
        assert np.all(np.isfinite(step_info["velocity"]))
        assert np.all(np.isfinite(step_info["quaternion"]))
        assert np.all(np.isfinite(step_info["angular_velocity"]))
        if terminated or truncated:
            obs, info = env.reset()

    env.close()


# ---------------------------------------------------------------------------
# 12. Physical Parameter Validation and Input Sanitation
# ---------------------------------------------------------------------------
def test_physical_parameter_validation_and_dimensions() -> None:
    """Invalid physical parameters, vector shapes, or non-finite inputs raise ValueError."""
    # Invalid mass
    with pytest.raises(ValueError, match="mass"):
        DroneDynamics6DOF(mass=-1.0)
    with pytest.raises(ValueError, match="mass"):
        DroneDynamics6DOF(mass=float("nan"))

    # Invalid arm length
    with pytest.raises(ValueError, match="arm_length"):
        DroneDynamics6DOF(arm_length=0.0)

    # Invalid inertia
    with pytest.raises(ValueError, match="Inertia"):
        DroneDynamics6DOF(inertia=(0.01, -0.01, 0.02))

    dynamics = DroneDynamics6DOF()

    # Invalid motor thrust shape
    with pytest.raises(ValueError, match="Motor thrusts must have exactly 4 elements"):
        dynamics.compute_forces_and_torques(np.array([1.0, 2.0, 3.0]))

    # Invalid state reset dimensions
    with pytest.raises(ValueError, match="Position"):
        dynamics.reset(position=np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Velocity"):
        dynamics.reset(velocity=np.array([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="Quaternion"):
        dynamics.reset(quaternion=np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="Angular velocity"):
        dynamics.reset(angular_velocity=np.array([0.0]))

    # Safe handling of non-finite motor commands
    nan_thrusts = np.array([float("nan"), float("inf"), float("-inf"), 2.0])
    total_t, torques = dynamics.compute_forces_and_torques(nan_thrusts)
    assert np.all(np.isfinite(torques))
    assert math.isfinite(total_t)

    # Environment action shape check
    env = Drone6DOFEnv()
    with pytest.raises(ValueError, match="Action must have shape"):
        env.step(np.array([0.0, 0.0, 0.0]))
    env.close()


# ---------------------------------------------------------------------------
# 13. Continuous Swept Collision Detection
# ---------------------------------------------------------------------------
def test_continuous_swept_collision_detection() -> None:
    """Continuous swept sphere check prevents tunneling through obstacles at high speeds."""
    env = Drone6DOFEnv(bounds=(50.0, 50.0, 25.0), num_obstacles=0)
    env.reset(seed=42)

    # Add a single test obstacle at [20, 20, 10] with radius 2.0
    test_obs = ObstacleSphere3D(center=np.array([20.0, 20.0, 10.0]), radius=2.0)
    env._obstacles = [test_obs]

    # Drone path passing directly through obstacle center from [10, 20, 10] to [30, 20, 10]
    # Without swept collision, if drone starts at 10 and jumps to 30, point check at 30 misses obstacle!
    p0 = np.array([10.0, 20.0, 10.0])
    p1 = np.array([30.0, 20.0, 10.0])

    is_coll_point, _ = env._check_collision(p1, prev_pos=None)
    assert not is_coll_point, "Point check at p1 should be clear of obstacle"

    is_coll_swept, coll_type = env._check_collision(p1, prev_pos=p0)
    assert is_coll_swept, "Swept check must detect collision with obstacle along trajectory"
    assert coll_type == "obstacle"

    # Swept boundary check (jumping across boundary)
    p_inside = np.array([25.0, 25.0, 12.0])
    p_outside = np.array([60.0, 25.0, 12.0])
    is_boundary_coll, b_type = env._check_collision(p_outside, prev_pos=p_inside)
    assert is_boundary_coll
    assert b_type == "boundary_x"

    env.close()


# ---------------------------------------------------------------------------
# 14. Reward Component Decomposition
# ---------------------------------------------------------------------------
def test_reward_components_decomposition() -> None:
    """Reward components (progress, step, effort, angular rate, tilt) are exposed in info."""
    env = Drone6DOFEnv(max_steps=50)
    obs, info = env.reset(seed=42)

    action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    obs, reward, terminated, truncated, step_info = env.step(action)

    assert "reward_components" in step_info
    rc = step_info["reward_components"]
    assert "progress_reward" in rc
    assert "step_penalty" in rc
    assert "effort_penalty" in rc
    assert "rate_penalty" in rc
    assert "tilt_penalty" in rc
    assert "terminal_reward" in rc

    # Sum of reward components should equal total step reward
    component_sum = sum(rc.values())
    assert math.isclose(reward, component_sum, abs_tol=1e-5)

    env.close()


# ---------------------------------------------------------------------------
# 15. Coordinate Transformations: World-to-Body and Body-to-World
# ---------------------------------------------------------------------------
def test_coordinate_transformations_and_quaternion_algebra() -> None:
    """Validate body_to_world, world_to_body, euler_to_quaternion, and quaternion algebra."""
    # 1. Identity quaternion: body and world frames are coincident
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    v_b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(body_to_world_vector(v_b, q_ident), v_b, atol=1e-9)
    assert np.allclose(world_to_body_vector(v_b, q_ident), v_b, atol=1e-9)

    # 2. 90-degree yaw rotation (pi/2 about +Z):
    # Body +X (East/Forward) rotates into World +Y (North)
    q_yaw90 = euler_to_quaternion(roll=0.0, pitch=0.0, yaw=math.pi / 2.0)
    v_forward = np.array([1.0, 0.0, 0.0])
    v_w = body_to_world_vector(v_forward, q_yaw90)
    assert np.allclose(v_w, [0.0, 1.0, 0.0], atol=1e-6)

    # Invert transformation back: world to body
    v_b_rec = world_to_body_vector(v_w, q_yaw90)
    assert np.allclose(v_b_rec, v_forward, atol=1e-6)

    # 3. Arbitrary nontrivial 3D orientation roundtrip
    rng = np.random.default_rng(777)
    for _ in range(25):
        roll_in = rng.uniform(-math.pi / 3, math.pi / 3)
        pitch_in = rng.uniform(-math.pi / 4, math.pi / 4)
        yaw_in = rng.uniform(-math.pi, math.pi)

        q = euler_to_quaternion(roll=roll_in, pitch=pitch_in, yaw=yaw_in)
        assert math.isclose(np.linalg.norm(q), 1.0, abs_tol=1e-6)

        # Roundtrip Euler angles
        r_out, p_out, y_out = quaternion_to_euler_angles(q)
        assert math.isclose(roll_in, r_out, abs_tol=1e-5)
        assert math.isclose(pitch_in, p_out, abs_tol=1e-5)
        assert math.isclose(yaw_in, y_out, abs_tol=1e-5)

        # Inversion invariant: v_B == R^T * (R * v_B)
        v_rand = rng.normal(size=3)
        v_world = body_to_world_vector(v_rand, q)
        v_body_back = world_to_body_vector(v_world, q)
        assert np.allclose(v_rand, v_body_back, atol=1e-6)

        # Quaternion conjugate property: q (x) q* == [1, 0, 0, 0]
        q_star = quaternion_conjugate(q)
        q_identity_res = quaternion_multiply(q, q_star)
        assert np.allclose(q_identity_res, [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    # 4. Method calls on DroneDynamics6DOF
    dynamics = DroneDynamics6DOF()
    dynamics.reset(quaternion=q_yaw90)
    assert np.allclose(dynamics.body_to_world(v_forward), [0.0, 1.0, 0.0], atol=1e-6)
    assert np.allclose(dynamics.world_to_body([0.0, 1.0, 0.0]), v_forward, atol=1e-6)

    # 5. Invalid vector and quaternion inputs
    with pytest.raises(ValueError, match="vector_b"):
        body_to_world_vector([1.0, 2.0], q_ident)
    with pytest.raises(ValueError, match="vector_w"):
        world_to_body_vector([1.0, 2.0, 3.0, 4.0], q_ident)
    with pytest.raises(ValueError, match="finite"):
        body_to_world_vector([float("nan"), 0.0, 0.0], q_ident)
    with pytest.raises(ValueError, match="Quaternions must have shape"):
        quaternion_multiply([1.0, 0.0], q_ident)


# ---------------------------------------------------------------------------
# 16. Axis Isolation & Rotational Coupling
# ---------------------------------------------------------------------------
def test_axis_isolation_and_rotational_cross_coupling() -> None:
    """Verify independent vertical response, torque isolation, and gyroscopic cross-coupling."""
    mass = 1.0
    gravity = 9.81
    dynamics = DroneDynamics6DOF(
        mass=mass,
        gravity=gravity,
        inertia=(0.01, 0.02, 0.03),
        linear_damping=0.0,
        angular_damping=0.0,
        dt=0.01,
        substeps=1,
    )

    # 1. Pure vertical acceleration: Collective thrust exceeding gravity
    # Hover thrust per motor = 9.81 / 4 = 2.4525 N. Command +1.0 N delta per motor.
    f_hover = dynamics.hover_thrust_per_motor
    delta_f = 0.5

    d_pos, d_vel, d_quat, d_omega = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        float(4.0 * (f_hover + delta_f)),
        np.zeros(3),
    )
    expected_az = (4.0 * delta_f) / mass  # (2.0 N) / 1.0 kg = 2.0 m/s^2
    assert math.isclose(d_vel[0], 0.0, abs_tol=1e-9)
    assert math.isclose(d_vel[1], 0.0, abs_tol=1e-9)
    assert math.isclose(d_vel[2], expected_az, rel_tol=1e-5)
    assert np.allclose(d_omega, 0.0, atol=1e-9)

    # 2. Gyroscopic cross-coupling: omega x (I * omega)
    # When rotating simultaneously about X (p) and Y (q), asymmetric inertia (Ixx != Iyy)
    # induces a non-zero gyroscopic angular acceleration along Z:
    # d_omega_z = (1/Izz) * (tau_z - (Iyy - Ixx) * p * q)
    p_rate, q_rate = 5.0, 8.0
    omega_init = np.array([p_rate, q_rate, 0.0])
    dynamics.reset(angular_velocity=omega_init)

    _, _, _, d_omega_coupled = dynamics._state_derivatives(
        dynamics.state.position,
        dynamics.state.velocity,
        dynamics.state.quaternion,
        dynamics.state.angular_velocity,
        dynamics.hover_thrust_per_motor * 4.0,
        np.zeros(3),  # zero applied torque
    )

    # Analytical Euler coupling:
    # I_omega = [Ixx*p, Iyy*q, Izz*r] = [0.01*5, 0.02*8, 0] = [0.05, 0.16, 0.0]
    # omega x (I*omega) = [0, 0, 5*0.16 - 8*0.05] = [0, 0, 0.8 - 0.4] = [0, 0, 0.4]
    # d_omega_z = -0.4 / Izz = -0.4 / 0.03 = -13.333 rad/s^2
    expected_d_omega_z = -(0.02 - 0.01) * p_rate * q_rate / 0.03
    assert math.isclose(d_omega_coupled[2], expected_d_omega_z, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# 17. Numerical Convergence Analysis Across Substeps
# ---------------------------------------------------------------------------
def test_numerical_convergence_across_time_steps() -> None:
    """Verify trajectory and attitude convergence under refined integration time steps."""
    total_time = 0.5  # 500 ms simulation
    coarse_dt = 0.05
    n_steps_coarse = int(total_time / coarse_dt)

    # Case A: coarse integration (dt = 0.05, substeps = 1)
    dyn_coarse = DroneDynamics6DOF(dt=coarse_dt, substeps=1)
    # Case B: refined integration (dt = 0.05, substeps = 5) -> dt_sub = 0.01
    dyn_medium = DroneDynamics6DOF(dt=coarse_dt, substeps=5)
    # Case C: highly refined reference (dt = 0.05, substeps = 25) -> dt_sub = 0.002
    dyn_fine = DroneDynamics6DOF(dt=coarse_dt, substeps=25)

    pos_init = np.array([5.0, 5.0, 5.0])
    dyn_coarse.reset(position=pos_init)
    dyn_medium.reset(position=pos_init)
    dyn_fine.reset(position=pos_init)

    # Asymmetric motor command creating coupled translation, roll, and climb
    thrusts = np.array([2.8, 3.2, 2.5, 2.1])

    for _ in range(n_steps_coarse):
        st_coarse = dyn_coarse.step_rk4(thrusts)
        st_medium = dyn_medium.step_rk4(thrusts)
        st_fine = dyn_fine.step_rk4(thrusts)

    # Divergence from fine reference
    divergence_coarse = float(np.linalg.norm(st_coarse.position - st_fine.position))
    divergence_medium = float(np.linalg.norm(st_medium.position - st_fine.position))

    # Refined integration must achieve strictly smaller error than coarse
    assert divergence_medium < divergence_coarse, (
        f"Convergence failure: medium error ({divergence_medium}) is not less than coarse error ({divergence_coarse})"
    )
    # Medium divergence from high-res reference must be within strict numerical tolerance (< 5e-3 m)
    assert divergence_medium < 5e-3
    assert math.isclose(np.linalg.norm(st_medium.quaternion), 1.0, abs_tol=1e-5)


# ---------------------------------------------------------------------------
# 18. Prolonged Hover Stability & Equilibrium Drift
# ---------------------------------------------------------------------------
def test_prolonged_hover_equilibrium_zero_drift() -> None:
    """At exact hover equilibrium (T = mg), position drift remains < 1e-4 m over 1,000 steps."""
    dynamics = DroneDynamics6DOF(mass=1.2, gravity=9.81, dt=0.01, substeps=2)
    pos_0 = np.array([10.0, 15.0, 8.0])
    dynamics.reset(position=pos_0)

    hover_thrusts = np.full(4, dynamics.hover_thrust_per_motor)
    for _ in range(1000):
        state = dynamics.step_rk4(hover_thrusts)

    pos_drift = float(np.linalg.norm(state.position - pos_0))
    vel_magnitude = float(np.linalg.norm(state.velocity))
    quat_drift = float(np.linalg.norm(state.quaternion - np.array([1.0, 0.0, 0.0, 0.0])))

    assert pos_drift < 1e-4, f"Hover position drift exceeded tolerance: {pos_drift} m"
    assert vel_magnitude < 1e-4, f"Hover velocity drift exceeded tolerance: {vel_magnitude} m/s"
    assert quat_drift < 1e-6, f"Hover attitude drift exceeded tolerance: {quat_drift}"


# ---------------------------------------------------------------------------
# 19. Individual Motor Contributions & Actuator Symmetry
# ---------------------------------------------------------------------------
def test_individual_motor_contributions_and_symmetry() -> None:
    """Verify physical torque signs and thrust symmetry of each motor individually."""
    dynamics = DroneDynamics6DOF(arm_length=0.2, torque_to_thrust_ratio=0.01)
    d = dynamics.d
    c_tau = dynamics.c_tau
    unit_thrust = 2.0

    # Motor 1: Front-Right (+d, -d) CCW
    t1, tau1 = dynamics.compute_forces_and_torques(np.array([unit_thrust, 0.0, 0.0, 0.0]))
    assert math.isclose(t1, unit_thrust)
    assert math.isclose(tau1[0], -d * unit_thrust)  # -roll
    assert math.isclose(tau1[1], -d * unit_thrust)  # -pitch
    assert math.isclose(tau1[2], +c_tau * unit_thrust)  # +yaw

    # Motor 2: Front-Left (+d, +d) CW
    t2, tau2 = dynamics.compute_forces_and_torques(np.array([0.0, unit_thrust, 0.0, 0.0]))
    assert math.isclose(t2, unit_thrust)
    assert math.isclose(tau2[0], +d * unit_thrust)  # +roll
    assert math.isclose(tau2[1], -d * unit_thrust)  # -pitch
    assert math.isclose(tau2[2], -c_tau * unit_thrust)  # -yaw

    # Motor 3: Rear-Left (-d, +d) CCW
    t3, tau3 = dynamics.compute_forces_and_torques(np.array([0.0, 0.0, unit_thrust, 0.0]))
    assert math.isclose(t3, unit_thrust)
    assert math.isclose(tau3[0], +d * unit_thrust)  # +roll
    assert math.isclose(tau3[1], +d * unit_thrust)  # +pitch
    assert math.isclose(tau3[2], +c_tau * unit_thrust)  # +yaw

    # Motor 4: Rear-Right (-d, -d) CW
    t4, tau4 = dynamics.compute_forces_and_torques(np.array([0.0, 0.0, 0.0, unit_thrust]))
    assert math.isclose(t4, unit_thrust)
    assert math.isclose(tau4[0], -d * unit_thrust)  # -roll
    assert math.isclose(tau4[1], +d * unit_thrust)  # +pitch
    assert math.isclose(tau4[2], -c_tau * unit_thrust)  # -yaw

    # Thrust symmetry: Diagonal pairs (1, 3) vs (2, 4) produce equal collective thrust
    t_diag13, tau_diag13 = dynamics.compute_forces_and_torques(
        np.array([unit_thrust, 0.0, unit_thrust, 0.0])
    )
    t_diag24, tau_diag24 = dynamics.compute_forces_and_torques(
        np.array([0.0, unit_thrust, 0.0, unit_thrust])
    )
    assert math.isclose(t_diag13, t_diag24)
    # Roll and pitch torques cancel for diagonal pairs:
    assert math.isclose(tau_diag13[0], 0.0, abs_tol=1e-9)
    assert math.isclose(tau_diag13[1], 0.0, abs_tol=1e-9)
    assert math.isclose(tau_diag24[0], 0.0, abs_tol=1e-9)
    assert math.isclose(tau_diag24[1], 0.0, abs_tol=1e-9)
    # Opposite yaw reaction torques:
    assert math.isclose(tau_diag13[2], -tau_diag24[2])
