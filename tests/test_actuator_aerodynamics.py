"""Unit tests for Phase IV higher-fidelity actuator dynamics, aerodynamic wind disturbances, and ground effect."""

from __future__ import annotations

import math

import numpy as np

from adaptive_rl.environments.drone_6dof import Drone6DOFEnv, DroneDynamics6DOF


def test_motor_first_order_dynamics_response_lag() -> None:
    """With enable_motor_dynamics=True, motor thrust follows exact analytical first-order lag with time constant tau."""
    tau = 0.04  # 40ms motor response time constant
    dt = 0.01
    dyn = DroneDynamics6DOF(
        dt=dt,
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

    # Exact analytical step response: f(t+h) = f(t)*exp(-h/tau) + f_cmd*(1 - exp(-h/tau))
    alpha = math.exp(-dt / tau)
    expected_first_step = hover_t * alpha + 5.0 * (1.0 - alpha)
    np.testing.assert_allclose(dyn.actual_motor_thrusts, [expected_first_step] * 4, rtol=1e-4)

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


def test_quadratic_wind_drag_in_translation() -> None:
    """Quadratic drag force F_drag = -D_v * v_rel - D_q * |v_rel| * v_rel increases drag non-linearly with airspeed."""
    dyn_linear = DroneDynamics6DOF(
        dt=0.02,
        substeps=1,
        linear_damping=0.1,
        quadratic_drag_coeff=0.0,
        enable_motor_dynamics=False,
    )
    dyn_quadratic = DroneDynamics6DOF(
        dt=0.02,
        substeps=1,
        linear_damping=0.1,
        quadratic_drag_coeff=0.05,
        enable_motor_dynamics=False,
    )

    pos = np.array([10.0, 10.0, 10.0])
    vel = np.array([4.0, 0.0, 0.0])  # 4 m/s in X
    dyn_linear.reset(position=pos, velocity=vel)
    dyn_quadratic.reset(position=pos, velocity=vel)

    hover_thrusts = np.full(4, dyn_linear.hover_thrust_per_motor)
    dyn_linear.step_rk4(hover_thrusts)
    dyn_quadratic.step_rk4(hover_thrusts)

    # Linear drag: F_drag_x = -0.1 * 4 = -0.4 N
    # Quadratic drag: F_drag_x = -0.1 * 4 - 0.05 * 4 * 4 = -0.4 - 0.8 = -1.2 N (3x greater deceleration)
    assert dyn_quadratic.state.velocity[0] < dyn_linear.state.velocity[0]


def test_gym_env_motor_dynamics_step_response() -> None:
    """Gymnasium Drone6DOFEnv respects motor dynamics time constant and lags motor commands."""
    env_instant = Drone6DOFEnv(enable_motor_dynamics=False)
    env_lag = Drone6DOFEnv(enable_motor_dynamics=True, motor_time_constant=0.04)

    env_instant.reset(seed=42)
    env_lag.reset(seed=42)

    action = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)  # Full throttle
    env_instant.step(action)
    env_lag.step(action)

    # Without lag, actual motor thrust reaches max_thrust immediately
    np.testing.assert_allclose(env_instant.dynamics.actual_motor_thrusts, [5.0] * 4, atol=1e-3)
    # With lag, actual motor thrust transitions smoothly toward max_thrust
    assert np.all(env_lag.dynamics.actual_motor_thrusts < 5.0)
    assert np.all(env_lag.dynamics.actual_motor_thrusts > env_lag.dynamics.hover_thrust_per_motor)


def test_gym_env_wind_drag_linear_and_quadratic() -> None:
    """Gymnasium Drone6DOFEnv applies wind drag correctly for opposing and matching relative winds."""
    # Headwind opposing vehicle motion (+X motion, -X wind)
    env_headwind = Drone6DOFEnv(
        wind_vector=np.array([-5.0, 0.0, 0.0]),
        quadratic_drag_coeff=0.05,
    )
    # Tailwind matching vehicle motion (+X motion, +X wind)
    env_tailwind = Drone6DOFEnv(
        wind_vector=np.array([5.0, 0.0, 0.0]),
        quadratic_drag_coeff=0.05,
    )

    env_headwind.reset(seed=42)
    env_tailwind.reset(seed=42)

    # Hover command with zero action (hover equilibrium)
    action = np.zeros(4, dtype=np.float32)
    env_headwind.step(action)
    env_tailwind.step(action)

    # Headwind pushes drone in -X direction
    assert env_headwind.dynamics.state.velocity[0] < 0.0
    # Tailwind pushes drone in +X direction
    assert env_tailwind.dynamics.state.velocity[0] > 0.0


def test_gym_env_ground_effect_thrust_boost() -> None:
    """Gymnasium Drone6DOFEnv with enable_ground_effect=True amplifies rotor thrust near ground."""
    env_ige = Drone6DOFEnv(
        enable_ground_effect=True,
        start_pos=np.array([10.0, 10.0, 0.10]),  # Very near ground
    )
    env_oge = Drone6DOFEnv(
        enable_ground_effect=False,
        start_pos=np.array([10.0, 10.0, 0.10]),
    )

    env_ige.reset(seed=42)
    env_oge.reset(seed=42)

    # Apply nominal hover thrust
    action = np.zeros(4, dtype=np.float32)
    env_ige.step(action)
    env_oge.step(action)

    # In ground effect, upward vertical acceleration / velocity is higher than out of ground effect
    assert env_ige.dynamics.state.velocity[2] > env_oge.dynamics.state.velocity[2]


def test_gym_env_domain_randomization_on_reset() -> None:
    """Gymnasium Drone6DOFEnv applies reproducible parameter shifts on reset when randomize_domain=True."""
    env = Drone6DOFEnv(randomize_domain=True)
    _, info1 = env.reset(seed=101)
    assert "randomized_parameters" in info1
    mass1 = env.dynamics.mass
    assert info1["randomized_parameters"]["mass"] == mass1

    # Same seed reproduces exact randomized parameters
    _, info1_repeat = env.reset(seed=101)
    assert info1_repeat["randomized_parameters"]["mass"] == mass1
    assert env.dynamics.mass == mass1

    # Different seed samples different parameters
    _, info2 = env.reset(seed=202)
    mass2 = env.dynamics.mass
    assert mass1 != mass2


def test_motor_coupled_velocity_integration_fidelity() -> None:
    """Coupled stage-wise RK4 integration tracks exact analytical velocity integral across time steps and tau."""
    for tau in [0.02, 0.04, 0.08]:
        for dt in [0.005, 0.01, 0.02]:
            dyn = DroneDynamics6DOF(
                mass=1.0,
                dt=dt,
                substeps=1,
                enable_motor_dynamics=True,
                motor_time_constant=tau,
                linear_damping=0.0,
            )
            dyn.reset(position=np.array([10.0, 10.0, 10.0]), velocity=np.zeros(3))
            hover_f = dyn.hover_thrust_per_motor
            cmd_f = 4.0
            cmd = np.array([cmd_f, cmd_f, cmd_f, cmd_f])
            dyn.step_rk4(cmd)

            v_sim = dyn.state.velocity[2]
            delta_f = 4.0 * (cmd_f - hover_f)
            # Analytical v(dt) = (delta_F / m) * (dt + tau * (exp(-dt/tau) - 1))
            v_analytical = (delta_f / 1.0) * (dt + tau * (math.exp(-dt / tau) - 1.0))
            rel_err = abs(v_sim - v_analytical) / v_analytical
            assert rel_err < 1e-3
