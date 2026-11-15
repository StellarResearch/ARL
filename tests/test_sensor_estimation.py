"""Unit tests for physically consistent IMU sensor model and complementary state estimation."""

from __future__ import annotations

import math

import numpy as np

from adaptive_rl.environments.drone_6dof import DroneState6DOF, euler_to_quaternion
from adaptive_rl.sensors.estimation import ComplementaryFilter
from adaptive_rl.sensors.imu import IMUMeasurement, IMUSensor


def test_imu_specific_force_hover_equilibrium() -> None:
    """Stationary upright quadrotor measures +g (9.81 m/s^2) pointing upwards along body +Z."""
    # Zero noise for exact physical verification
    imu = IMUSensor(gravity=9.81, accel_noise_std=0.0, gyro_noise_std=0.0, bias_drift_std=0.0)
    state = DroneState6DOF(
        position=np.array([10.0, 10.0, 5.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    meas = imu.measure(state, inertial_acceleration=np.zeros(3), dt=0.01)

    assert isinstance(meas, IMUMeasurement)
    # Accelerometer: ax ~ 0, ay ~ 0, az ~ +9.81
    assert math.isclose(meas.accelerometer[0], 0.0, abs_tol=1e-5)
    assert math.isclose(meas.accelerometer[1], 0.0, abs_tol=1e-5)
    assert math.isclose(meas.accelerometer[2], 9.81, rel_tol=1e-4)

    # Gyroscope: zero angular velocity
    np.testing.assert_allclose(meas.gyroscope, np.zeros(3), atol=1e-5)


def test_imu_tilt_gravity_projection() -> None:
    """When tilted 90 degrees roll (phi = pi/2), gravity projects onto body -Y axis."""
    q_roll90 = euler_to_quaternion(roll=math.pi / 2.0, pitch=0.0, yaw=0.0)
    imu = IMUSensor(gravity=9.81, accel_noise_std=0.0, gyro_noise_std=0.0, bias_drift_std=0.0)
    state = DroneState6DOF(
        position=np.array([10.0, 10.0, 5.0]),
        velocity=np.zeros(3),
        quaternion=q_roll90,
        angular_velocity=np.zeros(3),
    )
    meas = imu.measure(state, inertial_acceleration=np.zeros(3), dt=0.01)

    # When rolled 90 deg right, world +Z points towards body +Y in R_x(pi/2)
    assert math.isclose(meas.accelerometer[0], 0.0, abs_tol=1e-4)
    assert math.isclose(meas.accelerometer[1], 9.81, rel_tol=1e-4)
    assert math.isclose(meas.accelerometer[2], 0.0, abs_tol=1e-4)


def test_imu_noise_and_drift_reproducibility() -> None:
    """IMU produces non-zero noise within expected variance and is deterministic with fixed seed."""
    imu1 = IMUSensor(accel_noise_std=0.1, gyro_noise_std=0.01, seed=12345)
    imu2 = IMUSensor(accel_noise_std=0.1, gyro_noise_std=0.01, seed=12345)

    state = DroneState6DOF(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    m1 = imu1.measure(state, inertial_acceleration=np.zeros(3), dt=0.05)
    m2 = imu2.measure(state, inertial_acceleration=np.zeros(3), dt=0.05)

    np.testing.assert_allclose(m1.accelerometer, m2.accelerometer)
    np.testing.assert_allclose(m1.gyroscope, m2.gyroscope)


def test_complementary_filter_attitude_convergence() -> None:
    """Filter corrects an initial attitude tilt error using accelerometer gravity reference."""
    filt = ComplementaryFilter(kp_accel=1.5, ki_accel=0.0)
    # Start filter initialized with a 15-degree roll error
    q_err = euler_to_quaternion(roll=math.radians(15.0), pitch=0.0, yaw=0.0)
    filt.reset(quaternion=q_err)

    # True state is perfectly level hover
    true_state = DroneState6DOF(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )

    imu = IMUSensor(accel_noise_std=0.02, gyro_noise_std=0.002, seed=42)

    # Run filter over 100 steps (5 seconds)
    for step in range(100):
        meas = imu.measure(true_state, inertial_acceleration=np.zeros(3), dt=0.05)
        est = filt.update(meas, dt=0.05, true_state=true_state)

    # After 5s of leveling, attitude error must converge from 15 deg to < 1.0 deg
    assert est.attitude_error_deg < 1.5


def test_complementary_filter_position_altitude_tracking() -> None:
    """Filter tracks altitude accurately when provided barometric/range reference."""
    filt = ComplementaryFilter()
    filt.reset(position=np.array([0.0, 0.0, 5.0]))

    true_state = DroneState6DOF(
        position=np.array([0.0, 0.0, 5.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    imu = IMUSensor(accel_noise_std=0.05, seed=100)

    for _ in range(50):
        meas = imu.measure(true_state, inertial_acceleration=np.zeros(3), dt=0.05)
        est = filt.update(meas, dt=0.05, altitude_ref=5.0, true_state=true_state)

    assert math.isclose(est.position[2], 5.0, abs_tol=0.2)
