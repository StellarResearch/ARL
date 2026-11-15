"""Unit tests for QuadrotorPIDController and reference flight control architecture."""

from __future__ import annotations

import math

import numpy as np

from adaptive_rl.controllers import (
    PIDGains,
    QuadrotorPIDController,
    ReferenceFlightController,
)
from adaptive_rl.environments.drone_6dof import (
    DroneDynamics6DOF,
    DroneState6DOF,
)


def test_pid_controller_initialization_and_gains() -> None:
    """Verify PID controller parameters, custom gains, and aliases."""
    ctrl = QuadrotorPIDController(
        mass=1.2,
        gravity=9.81,
        arm_length=0.25,
        inertia=(0.015, 0.015, 0.03),
        max_thrust_per_motor=6.0,
        torque_to_thrust_ratio=0.012,
        dt=0.02,
        pos_gains_xy=PIDGains(kp=3.0, ki=0.2, kd=3.5),
    )
    assert ctrl.mass == 1.2
    assert ctrl.arm_length == 0.25
    assert ctrl.max_thrust_per_motor == 6.0
    assert ctrl.max_total_thrust == 24.0
    assert ctrl.dt == 0.02
    assert ctrl.pos_gains_xy.kp == 3.0

    # Test ReferenceFlightController alias
    assert ReferenceFlightController is QuadrotorPIDController
    ref_inst = ReferenceFlightController()
    assert isinstance(ref_inst, QuadrotorPIDController)


def test_analytical_mixer_inversion() -> None:
    """Verify that allocate_thrusts exactly inverts the physical quadrotor X-mixer matrix."""
    mass = 1.0
    gravity = 9.81
    arm_len = 0.2
    c_tau = 0.01
    max_f = 5.0

    ctrl = QuadrotorPIDController(
        mass=mass,
        gravity=gravity,
        arm_length=arm_len,
        torque_to_thrust_ratio=c_tau,
        max_thrust_per_motor=max_f,
    )
    dynamics = DroneDynamics6DOF(
        mass=mass,
        gravity=gravity,
        arm_length=arm_len,
        torque_to_thrust_ratio=c_tau,
        max_thrust_per_motor=max_f,
    )

    # Test cases: pure collective thrust, pure roll torque, pure pitch torque, pure yaw torque
    test_demands = [
        # (total_thrust, tau_x, tau_y, tau_z)
        (9.81, 0.0, 0.0, 0.0),  # exact hover
        (12.0, 0.0, 0.0, 0.0),  # climb
        (9.81, 0.05, 0.0, 0.0),  # roll right
        (9.81, 0.0, 0.05, 0.0),  # pitch up
        (9.81, 0.0, 0.0, 0.005),  # yaw CW
        (10.5, 0.03, -0.02, 0.003),  # coupled multi-axis demand
    ]

    for total_t, tx, ty, tz in test_demands:
        thrusts = ctrl.allocate_thrusts(total_thrust=total_t, tau_x=tx, tau_y=ty, tau_z=tz)
        # All individual motor thrusts must respect actuator limits [0, max_f]
        assert np.all(thrusts >= 0.0)
        assert np.all(thrusts <= max_f)

        # Forward mapping through real physical model must reproduce requested demand exactly
        achieved_total_t, achieved_torques = dynamics.compute_forces_and_torques(thrusts)
        assert math.isclose(achieved_total_t, total_t, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(achieved_torques[0], tx, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(achieved_torques[1], ty, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(achieved_torques[2], tz, rel_tol=1e-5, abs_tol=1e-5)


def test_actuator_saturation_and_clamping() -> None:
    """Extreme demands exceeding actuator limits are safely saturated."""
    ctrl = QuadrotorPIDController(max_thrust_per_motor=5.0)

    # Demanding 100 N thrust with max_thrust_per_motor = 5.0 (max total = 20.0 N)
    thrusts = ctrl.allocate_thrusts(total_thrust=100.0, tau_x=0.0, tau_y=0.0, tau_z=0.0)
    assert np.allclose(thrusts, 5.0)

    # Negative thrust demand clamped to 0.0
    thrusts_neg = ctrl.allocate_thrusts(total_thrust=-50.0, tau_x=0.0, tau_y=0.0, tau_z=0.0)
    assert np.allclose(thrusts_neg, 0.0)


def test_controller_action_computation_and_hover() -> None:
    """At hover position, controller outputs equilibrium hover commands."""
    ctrl = QuadrotorPIDController(mass=1.0, gravity=9.81, max_thrust_per_motor=5.0)
    pos_target = np.array([10.0, 10.0, 5.0])
    ctrl.set_target(position=pos_target)

    # State exactly at target at rest
    state_hover = DroneState6DOF(
        position=pos_target.copy(),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )

    action, info = ctrl.compute_action(state_hover, dt=0.05)
    assert action.shape == (4,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)

    # Hover thrust per motor is (1.0 * 9.81) / 4 = 2.4525 N
    # Normalized action = 2 * (2.4525 / 5.0) - 1 = -0.019
    expected_hover_action = 2.0 * ((1.0 * 9.81 / 4.0) / 5.0) - 1.0
    assert np.allclose(action, expected_hover_action, atol=1e-2)

    # Check telemetry dictionary entries
    assert "target_position" in info
    assert "position_error" in info
    assert "allocated_motor_thrusts" in info
    assert np.allclose(info["position_error"], 0.0, atol=1e-9)


def test_controller_observation_unpacking_and_predict_interface() -> None:
    """Controller handles 36-D observation vector and implements BaseAlgorithm.predict."""
    ctrl = QuadrotorPIDController()
    ctrl.set_target(position=[15.0, 15.0, 7.5])

    # Construct synthetic 36-D observation
    obs = np.zeros(36, dtype=np.float32)
    obs[0:3] = [0.5, 0.5, 0.5]  # position [15, 15, 7.5]
    obs[6] = 1.0  # qw = 1.0 (level attitude)

    # predict method
    action, state_ret = ctrl.predict(obs, deterministic=True)
    assert state_ret is None
    assert action.shape == (4,)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))


def test_controller_reset_and_anti_windup() -> None:
    """Reset clears accumulators; integral terms are clamped to prevent windup."""
    ctrl = QuadrotorPIDController()
    state_far = DroneState6DOF(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    ctrl.set_target(position=[25.0, 25.0, 10.0])

    # Step repeatedly with large error to trigger anti-windup clamping
    for _ in range(100):
        ctrl.compute_action(state_far, dt=0.05)

    assert np.all(np.abs(ctrl.pos_integral[:2]) <= ctrl.pos_gains_xy.max_integral)
    assert abs(ctrl.pos_integral[2]) <= ctrl.pos_gains_z.max_integral

    # Reset
    ctrl.reset()
    assert np.allclose(ctrl.pos_integral, 0.0)
    assert np.allclose(ctrl.att_integral, 0.0)
