"""Unit tests for Phase IV higher-fidelity actuator dynamics, aerodynamic wind disturbances, and ground effect."""

from __future__ import annotations

import math

import numpy as np

from adaptive_rl.environments.drone_6dof import DroneDynamics6DOF


def test_motor_first_order_dynamics_response_lag() -> None:
    """With enable_motor_dynamics=True, motor thrust follows first-order lag with time constant tau."""
    tau = 0.04  # 40ms motor response time constant
    dyn = DroneDynamics6DOF(
        dt=0.01,
        substeps=1,
        enable_motor_dynamics=True,
        motor_time_constant=tau,
    )
    dyn.reset()

    # Initial thrusts are at hover equilibrium (2.4525 N)
    hover_t = dyn.hover_thrust_per_motor
    np.testing.assert_allclose(dyn.actual_motor_thrusts, [hover_t] * 4)

    # Command a step to maximum thrust 5.0 N
    cmd = np.array([5.0, 5.0, 5.0, 5.0])
    dyn.step_rk4(cmd)

    # After dt=0.01s (t/tau = 0.01 / 0.04 = 0.25):
    # Expected actual thrust: T(0) + (dt / tau) * (T_cmd - T(0)) = hover_t + 0.25 * (5.0 - hover_t)
    expected_first_step = hover_t + (0.01 / tau) * (5.0 - hover_t)
    np.testing.assert_allclose(dyn.actual_motor_thrusts, [expected_first_step] * 4, rtol=1e-3)

    # Step for several time constants: actual thrust approaches 5.0 N
    for _ in range(50):  # 0.5s = 12.5 tau
        dyn.step_rk4(cmd)

    np.testing.assert_allclose(dyn.actual_motor_thrusts, [5.0] * 4, atol=1e-3)


def test_motor_dynamics_disabled_compatibility_mode() -> None:
    """When enable_motor_dynamics=False (default), thrust changes instantaneously (backwards compatible)."""
    dyn = DroneDynamics6DOF(enable_motor_dynamics=False)
    dyn.reset()

    cmd = np.array([4.5, 3.2, 2.1, 1.0])
    dyn.step_rk4(cmd)
    np.testing.assert_allclose(dyn.actual_motor_thrusts, cmd)


def test_continuous_aerodynamic_wind_force_in_rk4() -> None:
    """Wind vector enters translational equations of motion continuously as drag force F_drag = -D_v * (v - v_wind)."""
    mass = 1.0
    linear_damping = 0.2
    dt = 0.02
    substeps = 1
    wind = np.array([5.0, 0.0, 0.0])  # +5 m/s Eastward wind

    dyn = DroneDynamics6DOF(
        mass=mass,
        linear_damping=linear_damping,
        dt=dt,
        substeps=substeps,
        wind_vector=wind,
        enable_motor_dynamics=False,
    )
    # Start drone stationary at [10, 10, 10]
    dyn.reset(position=np.array([10.0, 10.0, 10.0]), velocity=np.zeros(3))

    # Apply pure hover thrust (counters gravity exactly)
    hover_thrusts = np.full(4, dyn.hover_thrust_per_motor)
    dyn.step_rk4(hover_thrusts)

    # Theoretical acceleration due to wind at v=0:
    # F_aero = -linear_damping * (v - v_wind) = -0.2 * (0 - 5.0) = +1.0 N in X
    # a_x = F_x / m = 1.0 / 1.0 = 1.0 m/s^2
    # In RK4 over dt=0.02s: dv_x approx 1.0 * 0.02 = 0.02 m/s
    assert dyn.state.velocity[0] > 0.0
    assert math.isclose(dyn.state.velocity[0], 1.0 * dt, rel_tol=0.05)
    # Y and Z velocity should remain zero
    assert math.isclose(dyn.state.velocity[1], 0.0, abs_tol=1e-5)
    assert math.isclose(dyn.state.velocity[2], 0.0, abs_tol=1e-5)


def test_ground_effect_cheeseman_bennett_model() -> None:
    """Ground effect model amplifies rotor thrust near the surface according to Cheeseman-Bennett equation."""
    dyn = DroneDynamics6DOF(
        enable_ground_effect=True,
        arm_length=0.2,
        max_thrust_per_motor=10.0,
    )
    r_blade = dyn.rotor_radius  # 0.45 * 0.2 = 0.09m
    assert math.isclose(r_blade, 0.09)

    nominal_thrusts = np.array([2.5, 2.5, 2.5, 2.5])

    # 1. High altitude (z = 10.0m >> R): ground effect is negligible (factor ~ 1.0)
    t_high, _ = dyn.compute_forces_and_torques(nominal_thrusts, altitude=10.0)
    assert math.isclose(t_high, 10.0, rel_tol=1e-4)

    # 2. Near ground (z = 0.15m):
    # Theoretical factor: 1 / (1 - (R / (4*z))^2)
    # R / (4*z) = 0.09 / (4 * 0.15) = 0.09 / 0.6 = 0.15
    # factor = 1 / (1 - 0.15^2) = 1 / (1 - 0.0225) = 1 / 0.9775 = 1.023018
    t_near, _ = dyn.compute_forces_and_torques(nominal_thrusts, altitude=0.15)
    expected_near = 10.0 * (1.0 / (1.0 - (0.09 / (4.0 * 0.15)) ** 2))
    assert math.isclose(t_near, expected_near, rel_tol=1e-3)
    assert t_near > 10.0

    # 3. Very close to ground (z = 0.01m): capped at maximum boost 1.30
    t_extreme, _ = dyn.compute_forces_and_torques(nominal_thrusts, altitude=0.01)
    assert math.isclose(t_extreme, 10.0 * 1.30, rel_tol=1e-4)


def test_ground_effect_disabled_returns_nominal_thrust() -> None:
    """When enable_ground_effect=False, thrust is unchanged regardless of altitude."""
    dyn = DroneDynamics6DOF(enable_ground_effect=False)
    nominal_thrusts = np.array([2.5, 2.5, 2.5, 2.5])
    t_zero_alt, _ = dyn.compute_forces_and_torques(nominal_thrusts, altitude=0.05)
    assert math.isclose(t_zero_alt, 10.0, rel_tol=1e-6)
