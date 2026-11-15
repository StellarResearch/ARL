"""Tests for Phase III comprehensive configuration fingerprinting, record digest, and tamper invalidation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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


def test_trajectories_preserved_and_covered_by_digest(tmp_path: Path) -> None:
    """FlightRecord preserves full trajectories, and tampering with trajectory step invalidates record_digest."""
    from adaptive_rl.flight_testing.scenarios import (
        FlightScenarioDefinition,
        FlightScenarioRunner,
    )

    runner = FlightScenarioRunner()
    sc = FlightScenarioDefinition(
        name="short_hover",
        description="Short hover for trajectory test",
        initial_position=np.array([5.0, 5.0, 5.0]),
        waypoints=[np.array([5.0, 5.0, 5.0])],
        max_steps=10,
    )
    result = runner.run_scenario(sc)
    assert len(result.telemetry) > 0

    recorder = FlightRecorder(output_dir=tmp_path)
    record = recorder.create_record(
        experiment_id="trajectory_record_test",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[result],
    )

    assert "short_hover" in record.trajectories
    assert len(record.trajectories["short_hover"]) == len(result.telemetry)
    assert record.verify_config_fingerprint() is True
    assert record.verify_record_digest() is True

    # Tamper with telemetry in trajectory: record_digest MUST be invalidated!
    record.trajectories["short_hover"][0]["position"] = [999.0, 999.0, 999.0]
    assert record.verify_record_digest() is False


def test_signature_distinction_from_digest(tmp_path: Path) -> None:
    """Digital signature is an authenticated cryptographic signature, distinct from unkeyed digests."""
    recorder = FlightRecorder(output_dir=tmp_path)
    record = recorder.create_record(
        experiment_id="sig_test",
        physics_params={"mass": 1.0},
        scenario_results=[],
        signature="ed25519_sig_abc123",
    )
    assert record.signature == "ed25519_sig_abc123"
    assert record.config_fingerprint != record.signature
    assert record.record_digest != record.signature
    assert record.verify_record_digest() is True

    # Tampering with signature invalidates record_digest
    record.signature = "tampered_signature"
    assert record.verify_record_digest() is False


def test_small_damping_and_inertia_float_precision() -> None:
    """Very small damping or inertia terms (< 1e-5) are preserved in canonical serialization."""
    d = {"angular_damping": 1.25e-6, "mass": 1.0}
    serialized = canonical_json_dumps(d)
    assert "1.25e-06" in serialized or "0.00000125" in serialized or "1.25e-6" in serialized


def test_sub_microgram_mass_distinct_fingerprints() -> None:
    """Distinct floating-point mass differences at sub-microgram scale (1e-9 kg) produce distinct digests."""
    d1 = {"mass": 1.0}
    d2 = {"mass": 1.000000001}
    fp1 = compute_config_fingerprint(d1)
    fp2 = compute_config_fingerprint(d2)
    assert fp1 != fp2

    # Multiple serialization passes remain strictly deterministic
    assert compute_config_fingerprint(d1) == fp1
    assert compute_config_fingerprint(d2) == fp2


def test_sub_millimeter_obstacle_position_distinct_fingerprints() -> None:
    """Sub-millimeter geometry shifts produce distinct cryptographic fingerprints."""
    from adaptive_rl.environments.drone import ObstacleSphere3D
    from adaptive_rl.flight_testing.scenarios import FlightScenarioDefinition

    sc1 = FlightScenarioDefinition(
        name="obs_sc",
        description="Obstacle test",
        initial_position=np.array([0.0, 0.0, 0.0]),
        waypoints=[np.array([10.0, 10.0, 5.0])],
        obstacles=[ObstacleSphere3D(center=np.array([5.0, 5.0, 5.0]), radius=2.0)],
    )

    sc2 = FlightScenarioDefinition(
        name="obs_sc",
        description="Obstacle test",
        initial_position=np.array([0.0, 0.0, 0.0]),
        waypoints=[np.array([10.0, 10.0, 5.0])],
        obstacles=[
            ObstacleSphere3D(center=np.array([5.00005, 5.0, 5.0]), radius=2.0)
        ],  # 50 microns shift
    )

    fp1 = compute_config_fingerprint(sc1.to_dict())
    fp2 = compute_config_fingerprint(sc2.to_dict())
    assert fp1 != fp2


def test_nan_inf_deterministic_handling() -> None:
    """Non-finite floats (NaN, Inf, -Inf) are deterministically serialized and hashed."""
    d_nan = {"val": float("nan")}
    d_inf = {"val": float("inf")}
    d_neginf = {"val": float("-inf")}

    s_nan1 = canonical_json_dumps(d_nan)
    s_nan2 = canonical_json_dumps(d_nan)
    assert s_nan1 == s_nan2
    assert '"val":"NaN"' in s_nan1

    s_inf = canonical_json_dumps(d_inf)
    assert '"val":"Infinity"' in s_inf

    s_neginf = canonical_json_dumps(d_neginf)
    assert '"val":"-Infinity"' in s_neginf


def test_scenario_provenance_invalidation() -> None:
    """Changing waypoint, obstacle, or controller parameter invalidates scenario fingerprint."""
    from adaptive_rl.environments.drone import ObstacleSphere3D
    from adaptive_rl.flight_testing.scenarios import FlightScenarioDefinition

    base_sc = FlightScenarioDefinition(
        name="base_sc",
        description="Base scenario",
        initial_position=np.array([1.0, 2.0, 3.0]),
        waypoints=[np.array([10.0, 10.0, 5.0]), np.array([20.0, 20.0, 8.0])],
        obstacles=[ObstacleSphere3D(center=np.array([5.0, 5.0, 4.0]), radius=1.5)],
    )
    fp_base = compute_config_fingerprint(base_sc.to_dict())

    # 1. Waypoint shift
    alt_wp = FlightScenarioDefinition(
        name="base_sc",
        description="Base scenario",
        initial_position=np.array([1.0, 2.0, 3.0]),
        waypoints=[np.array([10.0, 10.0, 5.0]), np.array([20.0, 20.0, 8.0001])],
        obstacles=[ObstacleSphere3D(center=np.array([5.0, 5.0, 4.0]), radius=1.5)],
    )
    assert compute_config_fingerprint(alt_wp.to_dict()) != fp_base

    # 2. Obstacle shift
    alt_obs = FlightScenarioDefinition(
        name="base_sc",
        description="Base scenario",
        initial_position=np.array([1.0, 2.0, 3.0]),
        waypoints=[np.array([10.0, 10.0, 5.0]), np.array([20.0, 20.0, 8.0])],
        obstacles=[ObstacleSphere3D(center=np.array([5.0, 5.0, 4.0001]), radius=1.5)],
    )
    assert compute_config_fingerprint(alt_obs.to_dict()) != fp_base


def test_create_record_infers_scenario_provenance_automatically(tmp_path: Path) -> None:
    """FlightRecorder.create_record automatically infers scenario provenance from scenario_results."""
    from adaptive_rl.flight_testing.scenarios import (
        FlightScenarioDefinition,
        FlightScenarioRunner,
    )

    runner = FlightScenarioRunner()
    sc = FlightScenarioDefinition(
        name="test_inferred_scenario",
        description="Test scenario for provenance inference",
        initial_position=np.array([1.0, 2.0, 3.0]),
        waypoints=[np.array([1.0, 2.0, 3.0])],
        max_steps=5,
    )
    res = runner.run_scenario(sc)
    assert res.scenario_definition is not None

    recorder = FlightRecorder(output_dir=tmp_path)
    record = recorder.create_record(
        experiment_id="test_infer_exp",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[res],  # No scenarios explicitly passed!
    )

    assert record.configuration is not None
    assert len(record.configuration.scenarios) == 1
    assert record.configuration.scenarios[0]["name"] == "test_inferred_scenario"
    assert record.configuration.provenance_complete is True
    assert record.verify_config_fingerprint() is True
    assert record.verify_record_digest() is True


def test_incomplete_scenario_provenance_flagged(tmp_path: Path) -> None:
    """When scenario results cannot be resolved to definitions, provenance_complete is False."""
    recorder = FlightRecorder(output_dir=tmp_path)
    unknown_metric = FlightScenarioMetrics(
        scenario_name="unknown_custom_scenario_xyz",
        success=True,
        outcome="success",
        duration_steps=10,
        duration_seconds=0.5,
        final_position_error=0.01,
        mean_position_error=0.02,
        max_position_error=0.05,
        rms_position_error=0.02,
        mean_velocity_error=0.01,
        max_velocity_error=0.02,
        rms_velocity_error=0.01,
        mean_attitude_error_deg=0.5,
        max_attitude_error_deg=1.0,
        mean_angular_speed=0.1,
        max_angular_speed=0.2,
        initial_goal_distance=1.0,
        final_goal_distance=0.01,
        min_obstacle_clearance=-1.0,
        collision_count=0,
        control_effort=10.0,
        execution_wall_time_ms=5.0,
    )

    record = recorder.create_record(
        experiment_id="test_incomplete_exp",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[unknown_metric],
    )

    assert record.configuration is not None
    assert record.configuration.provenance_complete is False
    assert record.provenance_complete is False


def test_callable_provenance_contract_closure_and_code() -> None:
    """Callable provenance captures closure variables, bytecode hash, and explicit metadata."""
    from adaptive_rl.flight_testing.scenarios import (
        FlightScenarioDefinition,
        extract_callable_provenance,
    )

    def make_thrust_fault(factor: float, motor_idx: int):
        def fault(step: int, thrusts: np.ndarray) -> np.ndarray:
            t = thrusts.copy()
            t[motor_idx] *= factor
            return t

        return fault

    fault_50 = make_thrust_fault(0.5, 0)
    fault_70 = make_thrust_fault(0.7, 0)

    prov_50 = extract_callable_provenance(fault_50)
    prov_70 = extract_callable_provenance(fault_70)

    # Different closure values must yield different provenance
    assert prov_50["closure"]["factor"] == 0.5
    assert prov_70["closure"]["factor"] == 0.7
    assert prov_50 != prov_70

    # Scenarios using these distinct faults must yield distinct cryptographic fingerprints
    sc1 = FlightScenarioDefinition(
        name="sc1",
        description="Sc 1",
        initial_position=np.array([0.0, 0.0, 1.0]),
        actuator_fault_fn=fault_50,
    )
    sc2 = FlightScenarioDefinition(
        name="sc1",
        description="Sc 1",
        initial_position=np.array([0.0, 0.0, 1.0]),
        actuator_fault_fn=fault_70,
    )
    assert compute_config_fingerprint(sc1.to_dict()) != compute_config_fingerprint(sc2.to_dict())

    # Metadata inclusion
    fault_50.metadata = {"subsystem": "avionics_esc_1"}  # type: ignore[attr-defined]
    prov_meta = extract_callable_provenance(fault_50)
    assert prov_meta.get("metadata") == {"subsystem": "avionics_esc_1"}
    assert prov_meta != prov_50
