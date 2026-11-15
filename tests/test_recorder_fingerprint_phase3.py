"""Tests for Phase III comprehensive configuration fingerprinting, record digest, and tamper invalidation."""

from __future__ import annotations

from pathlib import Path

from adaptive_rl.flight_testing.recorder import (
    ExperimentConfigurationFingerprint,
    FlightRecorder,
    PhysicsSimulationFingerprint,
    canonical_json_dumps,
    compute_config_fingerprint,
)
from adaptive_rl.flight_testing.scenarios import FlightScenarioMetrics


def test_comprehensive_config_fingerprint_coverage() -> None:
    """ExperimentConfigurationFingerprint covers physics, bounds, scenarios, controller, seed, and algorithm."""
    physics = PhysicsSimulationFingerprint(
        mass=1.0,
        gravity=9.81,
        arm_length=0.2,
        arm_span=0.4,
        rotor_offset_d=0.1414,
        rotor_radius=0.09,
        max_thrust_per_motor=5.0,
        torque_to_thrust_ratio=0.01,
        linear_damping=0.1,
        angular_damping=0.02,
        inertia=[0.01, 0.01, 0.02],
        dt=0.05,
        substeps=5,
        substep_dt=0.01,
        body_frame_lidar=False,
        num_lidar_rays=16,
        lidar_range=20.0,
    )

    exp_cfg = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="ppo",
        checkpoint_hash="a1b2c3d4",
        random_seed=42,
        scenario_names=["hover_equilibrium", "obstacle_avoidance"],
        controller_type="QuadrotorPIDController",
        controller_gains={"kp_pos": 2.5, "kd_pos": 1.8},
    )

    fp1 = compute_config_fingerprint(exp_cfg.to_dict())
    assert len(fp1) == 64

    # Changing seed invalidates fingerprint
    exp_cfg_seed = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="ppo",
        checkpoint_hash="a1b2c3d4",
        random_seed=43,  # Changed seed
        scenario_names=["hover_equilibrium", "obstacle_avoidance"],
        controller_type="QuadrotorPIDController",
        controller_gains={"kp_pos": 2.5, "kd_pos": 1.8},
    )
    fp_seed = compute_config_fingerprint(exp_cfg_seed.to_dict())
    assert fp_seed != fp1

    # Changing controller gain invalidates fingerprint
    exp_cfg_gains = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="ppo",
        checkpoint_hash="a1b2c3d4",
        random_seed=42,
        scenario_names=["hover_equilibrium", "obstacle_avoidance"],
        controller_type="QuadrotorPIDController",
        controller_gains={"kp_pos": 3.0, "kd_pos": 1.8},  # Changed gain
    )
    fp_gains = compute_config_fingerprint(exp_cfg_gains.to_dict())
    assert fp_gains != fp1


def test_dual_integrity_digests_and_record_tamper_detection(tmp_path: Path) -> None:
    """FlightRecord maintains separate config_fingerprint and record_digest; tampering with results is caught."""
    recorder = FlightRecorder(output_dir=tmp_path)
    physics_params = {"mass": 1.0, "gravity": 9.81, "dt": 0.05}

    metric = FlightScenarioMetrics(
        scenario_name="hover",
        success=True,
        outcome="success",
        duration_steps=100,
        duration_seconds=5.0,
        final_position_error=0.01,
        mean_position_error=0.02,
        max_position_error=0.05,
        rms_position_error=0.02,
        mean_velocity_error=0.01,
        max_velocity_error=0.02,
        rms_velocity_error=0.01,
        mean_attitude_error_deg=0.5,
        max_attitude_error_deg=1.2,
        mean_angular_speed=0.1,
        max_angular_speed=0.2,
        initial_goal_distance=1.0,
        final_goal_distance=0.01,
        min_obstacle_clearance=5.0,
        collision_count=0,
        control_effort=10.0,
        execution_wall_time_ms=50.0,
    )

    record = recorder.create_record(
        experiment_id="dual_digest_test",
        physics_params=physics_params,
        scenario_results=[metric],
        signature="hmac_test_signature_placeholder",
    )

    assert record.verify_config_fingerprint() is True
    assert record.verify_record_digest() is True
    assert record.signature == "hmac_test_signature_placeholder"

    # Tampering with scenario outcome in record leaves config valid but invalidates record_digest!
    record.scenario_metrics[0].outcome = "faked_success"
    assert record.verify_config_fingerprint() is True  # physics didn't change
    assert record.verify_record_digest() is False  # result digest detects tampering!


def test_small_damping_and_inertia_float_precision() -> None:
    """Very small damping or inertia terms (< 1e-5) are preserved in canonical serialization."""
    d = {"angular_damping": 1.25e-6, "mass": 1.0}
    serialized = canonical_json_dumps(d)
    assert "1.25e-06" in serialized or "0.00000125" in serialized or "1.25e-6" in serialized
