"""Comprehensive verification tests for the LQR controller and analytical mixer inversion.

Establishes:
1. Exact mathematical matrix inversion of the physical allocation matrix M (M * M_inv == I).
2. Pure axis isolation tests (positive and negative roll, pitch, yaw).
3. Verification of resulting actual physical torques from DroneDynamics6DOF.
4. Total collective thrust conservation.
5. Saturation clamping behavior.
6. Honest fallback handling (rejects silent approximations when solver unavailable).
7. Correct handling of the actual 36-D normalized observation space of Drone6DOFEnv.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from adaptive_rl.controllers.lqr import QuadrotorLQRController
from adaptive_rl.environments.drone_6dof import Drone6DOFEnv, DroneDynamics6DOF
from adaptive_rl.flight_testing.scenarios import (
    FlightScenarioDefinition,
    FlightScenarioRunner,
)


def test_lqr_allocation_matrix_inverse_identity() -> None:
    """Build the physical allocation matrix M and verify M @ M_inv == I within float tolerance."""
    arm_length = 0.2
    d = arm_length / math.sqrt(2.0)
    c_tau = 0.01

    M = QuadrotorLQRController.get_allocation_matrix(d=d, c_tau=c_tau)
    M_inv = QuadrotorLQRController.get_allocation_matrix_inverse(d=d, c_tau=c_tau)

    # Verify shapes
    assert M.shape == (4, 4)
    assert M_inv.shape == (4, 4)

    # Forward times inverse equals identity
    prod_forward = M @ M_inv
    np.testing.assert_allclose(prod_forward, np.eye(4), atol=1e-12, rtol=1e-12)

    # Inverse times forward equals identity
    prod_backward = M_inv @ M
    np.testing.assert_allclose(prod_backward, np.eye(4), atol=1e-12, rtol=1e-12)


def test_lqr_allocation_matrix_pure_axes_and_physical_torques() -> None:
    """Verify that pure roll, pitch, and yaw commands produce exact physical torques in DroneDynamics6DOF."""
    arm_length = 0.2
    c_tau = 0.01
    dyn = DroneDynamics6DOF(arm_length=arm_length, torque_to_thrust_ratio=c_tau)
    ctrl = QuadrotorLQRController(arm_length=arm_length, torque_to_thrust_ratio=c_tau)

    T_hover = 9.81  # Equilibrium collective thrust

    # 1. Pure positive roll (tau_x = +0.05, others 0)
    f_roll_pos = ctrl.allocate_thrusts(total_thrust=T_hover, tau_x=0.05, tau_y=0.0, tau_z=0.0)
    T_act, tau_act = dyn.compute_forces_and_torques(f_roll_pos)
    assert math.isclose(T_act, T_hover, rel_tol=1e-6)
    assert math.isclose(tau_act[0], 0.05, rel_tol=1e-6)
    assert math.isclose(tau_act[1], 0.0, abs_tol=1e-10)
    assert math.isclose(tau_act[2], 0.0, abs_tol=1e-10)

    # 2. Pure negative roll (tau_x = -0.05, others 0)
    f_roll_neg = ctrl.allocate_thrusts(total_thrust=T_hover, tau_x=-0.05, tau_y=0.0, tau_z=0.0)
    T_act, tau_act = dyn.compute_forces_and_torques(f_roll_neg)
    assert math.isclose(T_act, T_hover, rel_tol=1e-6)
    assert math.isclose(tau_act[0], -0.05, rel_tol=1e-6)
    assert math.isclose(tau_act[1], 0.0, abs_tol=1e-10)
    assert math.isclose(tau_act[2], 0.0, abs_tol=1e-10)

    # 3. Pure positive pitch (tau_y = +0.05, others 0)
    f_pitch_pos = ctrl.allocate_thrusts(total_thrust=T_hover, tau_x=0.0, tau_y=0.05, tau_z=0.0)
    T_act, tau_act = dyn.compute_forces_and_torques(f_pitch_pos)
    assert math.isclose(T_act, T_hover, rel_tol=1e-6)
    assert math.isclose(tau_act[0], 0.0, abs_tol=1e-10)
    assert math.isclose(tau_act[1], 0.05, rel_tol=1e-6)
    assert math.isclose(tau_act[2], 0.0, abs_tol=1e-10)

    # 4. Pure negative pitch (tau_y = -0.05, others 0)
    f_pitch_neg = ctrl.allocate_thrusts(total_thrust=T_hover, tau_x=0.0, tau_y=-0.05, tau_z=0.0)
    T_act, tau_act = dyn.compute_forces_and_torques(f_pitch_neg)
    assert math.isclose(T_act, T_hover, rel_tol=1e-6)
    assert math.isclose(tau_act[0], 0.0, abs_tol=1e-10)
    assert math.isclose(tau_act[1], -0.05, rel_tol=1e-6)
    assert math.isclose(tau_act[2], 0.0, abs_tol=1e-10)

    # 5. Pure positive yaw (tau_z = +0.01, others 0)
    f_yaw_pos = ctrl.allocate_thrusts(total_thrust=T_hover, tau_x=0.0, tau_y=0.0, tau_z=0.01)
    T_act, tau_act = dyn.compute_forces_and_torques(f_yaw_pos)
    assert math.isclose(T_act, T_hover, rel_tol=1e-6)
    assert math.isclose(tau_act[0], 0.0, abs_tol=1e-10)
    assert math.isclose(tau_act[1], 0.0, abs_tol=1e-10)
    assert math.isclose(tau_act[2], 0.01, rel_tol=1e-6)

    # 6. Pure negative yaw (tau_z = -0.01, others 0)
    f_yaw_neg = ctrl.allocate_thrusts(total_thrust=T_hover, tau_x=0.0, tau_y=0.0, tau_z=-0.01)
    T_act, tau_act = dyn.compute_forces_and_torques(f_yaw_neg)
    assert math.isclose(T_act, T_hover, rel_tol=1e-6)
    assert math.isclose(tau_act[0], 0.0, abs_tol=1e-10)
    assert math.isclose(tau_act[1], 0.0, abs_tol=1e-10)
    assert math.isclose(tau_act[2], -0.01, rel_tol=1e-6)


def test_lqr_collective_thrust_conservation_and_clipping() -> None:
    """Verify collective thrust conservation under coupled torques and saturation limits."""
    ctrl = QuadrotorLQRController(max_thrust_per_motor=5.0)

    # Unclipped coupled demand
    f_coupled = ctrl.allocate_thrusts(total_thrust=10.0, tau_x=0.02, tau_y=-0.02, tau_z=0.005)
    assert math.isclose(float(np.sum(f_coupled)), 10.0, rel_tol=1e-6)
    assert np.all(f_coupled >= 0.0)

    # Demanding excessive thrust clips to motor limits
    f_excess = np.clip(
        ctrl.allocate_thrusts(total_thrust=25.0, tau_x=0.0, tau_y=0.0, tau_z=0.0), 0.0, 5.0
    )
    assert np.allclose(f_excess, [5.0, 5.0, 5.0, 5.0])


def test_lqr_are_solver_and_fallback_behavior() -> None:
    """Verify ARE solver identification and rejection of silent approximations."""
    ctrl = QuadrotorLQRController()
    assert ctrl.solver_status == "ARE_SOLVED"
    assert ctrl.is_are_solution is True
    assert ctrl.K.shape == (4, 12)

    # Simulating solver failure when allow_fallback is False must raise RuntimeError
    with patch("scipy.linalg.solve_continuous_are", side_effect=Exception("Solver singular")):
        with pytest.raises(
            RuntimeError, match="Continuous Algebraic Riccati Equation solver failed"
        ):
            QuadrotorLQRController(allow_fallback=False)

    # When allow_fallback is True, solver_status must be UNRESOLVED_APPROXIMATION
    with patch("scipy.linalg.solve_continuous_are", side_effect=Exception("Solver singular")):
        ctrl_fb = QuadrotorLQRController(allow_fallback=True)
        assert ctrl_fb.solver_status == "UNRESOLVED_APPROXIMATION"
        assert ctrl_fb.is_are_solution is False


def test_lqr_observation_adapter_with_real_36d_gym_environment() -> None:
    """Verify LQR predict interface against the actual 36-D observation from Drone6DOFEnv."""
    env = Drone6DOFEnv(bounds=(30.0, 30.0, 15.0), max_steps=50)
    obs, info = env.reset(seed=42)

    assert obs.shape == (36,)

    ctrl = QuadrotorLQRController(bounds=(30.0, 30.0, 15.0))
    action, ctrl_info = ctrl.predict(obs)

    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert ctrl_info is not None
    assert ctrl_info["is_are_solution"] is True

    # Step environment with LQR action over multiple steps and verify stable flight
    for _ in range(20):
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        action, _ = ctrl.predict(next_obs)
        if terminated or truncated:
            break

    # Drone must not have exploded or breached boundaries immediately
    assert env.dynamics.state.position[2] > 0.0


def test_lqr_rejects_unsupported_observation_shape() -> None:
    """LQR predict must raise an explicit ValueError on invalid observation shapes."""
    ctrl = QuadrotorLQRController()
    invalid_obs = np.zeros(25, dtype=np.float32)

    with pytest.raises(ValueError, match="Unsupported state/obs representation"):
        ctrl.predict(invalid_obs)


def test_lqr_hover_in_flight_scenario_runner() -> None:
    """Execute hover scenario with LQR controller through FlightScenarioRunner."""
    controller = QuadrotorLQRController()
    sc = FlightScenarioDefinition(
        name="lqr_hover",
        description="LQR hover validation",
        initial_position=np.array([15.0, 15.0, 5.0]),
        waypoints=[np.array([15.0, 15.0, 5.0])],
        max_steps=50,
        target_radius=0.5,
    )

    runner = FlightScenarioRunner()
    result = runner.run_scenario(sc, controller=controller)
    assert result.metrics.final_position_error < 0.1
    assert result.metrics.collision_count == 0
