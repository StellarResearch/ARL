"""Unit tests for the optimal Linear Quadratic Regulator (LQR) flight controller."""

from __future__ import annotations

import math

import numpy as np

from adaptive_rl.controllers.lqr import QuadrotorLQRController
from adaptive_rl.environments.drone_6dof import DroneState6DOF
from adaptive_rl.flight_testing.scenarios import (
    FlightScenarioDefinition,
    FlightScenarioRunner,
)


def test_lqr_controller_gain_matrix_dimensions_and_finiteness() -> None:
    """Verify that the continuous ARE is solved producing a valid 4x12 gain matrix K."""
    controller = QuadrotorLQRController(mass=1.0, gravity=9.81)
    assert controller.K.shape == (4, 12)
    assert np.all(np.isfinite(controller.K))


def test_lqr_hover_equilibrium_action() -> None:
    """At equilibrium setpoint with zero velocity, LQR outputs hover equilibrium thrust."""
    controller = QuadrotorLQRController(mass=1.0, gravity=9.81, max_thrust_per_motor=5.0)
    hover_state = DroneState6DOF(
        position=np.array([10.0, 10.0, 5.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    action, info = controller.compute_action(
        hover_state, target_position=np.array([10.0, 10.0, 5.0])
    )

    # Virtual control inputs should be near zero at equilibrium
    assert math.isclose(info["delta_T"], 0.0, abs_tol=1e-5)
    assert math.isclose(info["tau_x"], 0.0, abs_tol=1e-5)
    assert math.isclose(info["tau_y"], 0.0, abs_tol=1e-5)
    assert math.isclose(info["tau_z"], 0.0, abs_tol=1e-5)

    # Motor thrusts must equal hover thrust per motor: 9.81 / 4 = 2.4525 N
    expected_hover_thrust = 9.81 / 4.0
    for t in info["motor_thrusts"]:
        assert math.isclose(t, expected_hover_thrust, rel_tol=1e-4)


def test_lqr_vertical_and_lateral_control_response() -> None:
    """Position displacements produce expected restoring thrust and torque directions."""
    controller = QuadrotorLQRController(mass=1.0, gravity=9.81, max_thrust_per_motor=5.0)

    # 1. Drone is below target (z=4.0, target=5.0): must command positive thrust delta
    state_low = DroneState6DOF(
        position=np.array([10.0, 10.0, 4.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    _, info_low = controller.compute_action(state_low, target_position=np.array([10.0, 10.0, 5.0]))
    assert info_low["delta_T"] > 0.0
    assert info_low["total_thrust"] > 9.81

    # 2. Drone is to the left (y=9.0, target=10.0): must command restoring roll
    state_left = DroneState6DOF(
        position=np.array([10.0, 9.0, 5.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    _, info_left = controller.compute_action(
        state_left, target_position=np.array([10.0, 10.0, 5.0])
    )
    # Restoring roll torque should be non-zero
    assert abs(info_left["tau_x"]) > 0.0


def test_lqr_predict_gym_interface() -> None:
    """Verify predict interface compatibility with Gym observation vector."""
    controller = QuadrotorLQRController()
    obs = np.zeros(18, dtype=np.float32)
    obs[0:3] = [5.0, 5.0, 5.0]  # pos
    obs[6] = 1.0  # qw
    obs[13:16] = [10.0, 10.0, 8.0]  # target

    action, info = controller.predict(obs)
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert info is not None


def test_lqr_in_flight_scenario_runner() -> None:
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
