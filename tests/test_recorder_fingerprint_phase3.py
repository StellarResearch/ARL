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


GLOBAL_SCALING_FACTOR = 1.0
_test_dummy_pkg = None
_test_opaque_val = None
_test_factorial_global = None


def test_callable_provenance_defaults_and_kwdefaults() -> None:
    """Functions with identical code but different default or kwdefault arguments produce distinct provenance."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    # 1. Positional defaults
    def fault_a(step: int, thrusts: np.ndarray, factor: float = 0.5) -> np.ndarray:
        return thrusts * factor

    def fault_b(step: int, thrusts: np.ndarray, factor: float = 0.8) -> np.ndarray:
        return thrusts * factor

    prov_a = extract_callable_provenance(fault_a)
    prov_b = extract_callable_provenance(fault_b)
    assert prov_a["defaults"] == [0.5]
    assert prov_b["defaults"] == [0.8]
    assert prov_a != prov_b

    # 2. Keyword-only defaults
    def fault_kw1(step: int, thrusts: np.ndarray, *, scale: float = 1.5) -> np.ndarray:
        return thrusts * scale

    def fault_kw2(step: int, thrusts: np.ndarray, *, scale: float = 2.5) -> np.ndarray:
        return thrusts * scale

    prov_kw1 = extract_callable_provenance(fault_kw1)
    prov_kw2 = extract_callable_provenance(fault_kw2)
    assert prov_kw1["kwdefaults"] == {"scale": 1.5}
    assert prov_kw2["kwdefaults"] == {"scale": 2.5}
    assert prov_kw1 != prov_kw2


def test_callable_provenance_referenced_globals() -> None:
    """Functions referencing different global values produce distinct provenance."""
    global GLOBAL_SCALING_FACTOR
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    def global_scaled_fault(step: int, thrusts: np.ndarray) -> np.ndarray:
        return thrusts * GLOBAL_SCALING_FACTOR

    GLOBAL_SCALING_FACTOR = 1.0
    prov_g1 = extract_callable_provenance(global_scaled_fault)
    assert prov_g1["referenced_globals"]["GLOBAL_SCALING_FACTOR"] == 1.0

    GLOBAL_SCALING_FACTOR = 2.5
    prov_g2 = extract_callable_provenance(global_scaled_fault)
    assert prov_g2["referenced_globals"]["GLOBAL_SCALING_FACTOR"] == 2.5
    assert prov_g1 != prov_g2


def test_callable_objects_provenance() -> None:
    """Callable class instances capture class identity, instance configuration, and code metadata."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    class ActuatorLossModel:
        def __init__(self, motor_id: int, efficiency: float) -> None:
            self.motor_id = motor_id
            self.efficiency = efficiency

        def __call__(self, step: int, thrusts: np.ndarray) -> np.ndarray:
            t = thrusts.copy()
            t[self.motor_id] *= self.efficiency
            return t

    model1 = ActuatorLossModel(motor_id=0, efficiency=0.4)
    model2 = ActuatorLossModel(motor_id=0, efficiency=0.9)

    prov1 = extract_callable_provenance(model1)
    prov2 = extract_callable_provenance(model2)

    assert "ActuatorLossModel" in prov1["callable_class"]
    assert prov1["attributes"]["efficiency"] == 0.4
    assert prov2["attributes"]["efficiency"] == 0.9
    assert prov1["provenance_complete"] is True
    assert prov2["provenance_complete"] is True
    assert prov1 != prov2


def test_unsupported_non_serializable_callable_state_marked_incomplete() -> None:
    """Callable with non-serializable state is marked incomplete without leaking memory addresses."""
    import threading

    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    class ComplexFault:
        def __init__(self) -> None:
            self.lock = threading.Lock()  # Non-serializable

        def __call__(self, step: int, thrusts: np.ndarray) -> np.ndarray:
            return thrusts

    fault = ComplexFault()
    prov = extract_callable_provenance(fault)

    assert prov["provenance_complete"] is False
    # Verify no memory address leaks
    serialized = canonical_json_dumps(prov)
    assert "0x" not in serialized


def test_deterministic_repeated_callable_serialization() -> None:
    """Repeated calls to extract_callable_provenance produce identical deterministic digests."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    def sample_fault(step: int, thrusts: np.ndarray, gain: float = 0.5) -> np.ndarray:
        return thrusts * gain

    p1 = extract_callable_provenance(sample_fault)
    p2 = extract_callable_provenance(sample_fault)
    assert canonical_json_dumps(p1) == canonical_json_dumps(p2)


def test_explicit_configuration_missing_scenario_definitions_marked_incomplete(
    tmp_path: Path,
) -> None:
    """Explicit configuration missing scenario definitions is marked incomplete."""
    from adaptive_rl.flight_testing.scenarios import (
        FlightScenarioDefinition,
        FlightScenarioRunner,
    )

    runner = FlightScenarioRunner()
    sc1 = FlightScenarioDefinition(
        name="sc1",
        description="1",
        initial_position=np.array([1.0, 1.0, 1.0]),
        max_steps=2,
    )
    sc2 = FlightScenarioDefinition(
        name="sc2",
        description="2",
        initial_position=np.array([2.0, 2.0, 2.0]),
        max_steps=2,
    )
    res1 = runner.run_scenario(sc1)
    res2 = runner.run_scenario(sc2)

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

    # Supply explicit configuration that only covers sc1, but scenario_results has [res1, res2]
    cfg = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="test",
        checkpoint_hash=None,
        random_seed=42,
        scenario_names=["sc1"],  # Missing sc2!
        scenarios=[sc1.to_dict()],
        provenance_complete=True,  # Caller falsely claims complete!
    )

    recorder = FlightRecorder(output_dir=tmp_path)
    record = recorder.create_record(
        experiment_id="missing_sc_test",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[res1, res2],
        configuration=cfg,
    )

    # Must be marked incomplete!
    assert record.provenance_complete is False
    assert record.configuration.provenance_complete is False
    assert record.verify_config_fingerprint() is True


def test_explicit_configuration_conflicting_completeness_detected(tmp_path: Path) -> None:
    """Record tampering with conflicting top-level and configuration completeness fails verification."""
    from adaptive_rl.flight_testing.scenarios import (
        FlightScenarioDefinition,
        FlightScenarioRunner,
    )

    runner = FlightScenarioRunner()
    sc = FlightScenarioDefinition(
        name="sc_ok",
        description="ok",
        initial_position=np.array([1.0, 1.0, 1.0]),
        max_steps=2,
    )
    res = runner.run_scenario(sc)

    recorder = FlightRecorder(output_dir=tmp_path)
    record = recorder.create_record(
        experiment_id="tamper_test",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[res],
    )
    assert record.verify_config_fingerprint() is True
    assert record.verify_record_digest() is True

    # Tamper with top-level flag to disagree with configuration flag
    record.provenance_complete = not record.configuration.provenance_complete

    # Both verifications must catch the inconsistency!
    assert record.verify_config_fingerprint() is False
    assert record.verify_record_digest() is False


# Global helpers for referenced callable dependency testing
def _global_helper_v1(x: float) -> float:
    return x * 2.0


def _global_helper_v2(x: float) -> float:
    return x * 10.0


def test_referenced_global_helper_code_change_alters_provenance() -> None:
    """Changing a referenced global helper's code changes the parent callable's provenance digest."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    # Function referencing global helper
    def parent_fn(x: float) -> float:
        return _global_helper_v1(x)

    def parent_fn_v2(x: float) -> float:
        return _global_helper_v2(x)

    prov1 = extract_callable_provenance(parent_fn)
    prov2 = extract_callable_provenance(parent_fn_v2)

    # Both parent functions have identical names, signatures, but different referenced helpers
    assert canonical_json_dumps(prov1) != canonical_json_dumps(prov2)
    assert (
        prov1["referenced_globals"]["_global_helper_v1"]["code_hash"]
        != (prov2["referenced_globals"]["_global_helper_v2"]["code_hash"])
    )


def test_referenced_global_helper_defaults_and_closures_alter_provenance() -> None:
    """Changing helper default arguments or closure variables alters the parent callback's provenance."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    def make_parent(gain: float):
        def inner_helper(val: float, scale: float = 1.0) -> float:
            return val * scale * gain

        def parent_cb(x: float) -> float:
            return inner_helper(x)

        return parent_cb

    cb1 = make_parent(gain=2.0)
    cb2 = make_parent(gain=5.0)

    p1 = extract_callable_provenance(cb1)
    p2 = extract_callable_provenance(cb2)

    assert canonical_json_dumps(p1) != canonical_json_dumps(p2)


def _module_recursive_factorial(n: int) -> int:
    return 1 if n <= 1 else n * _module_recursive_factorial(n - 1)


def test_recursive_callable_dependencies_terminate_safely() -> None:
    """Self-recursive and mutually recursive callable dependencies terminate deterministically without error."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    prov_fact = extract_callable_provenance(_module_recursive_factorial)
    assert prov_fact["provenance_complete"] is True
    # The referenced global '_module_recursive_factorial' contains a recursion reference
    assert "_module_recursive_factorial" in prov_fact["referenced_globals"]
    assert "recursion_ref" in prov_fact["referenced_globals"]["_module_recursive_factorial"]


def test_explicit_configuration_content_mismatch_marked_incomplete(tmp_path: Path) -> None:
    """Explicit configuration with matching scenario names but differing scenario definitions is marked incomplete."""
    from adaptive_rl.flight_testing.scenarios import (
        FlightScenarioDefinition,
        FlightScenarioRunner,
    )

    runner = FlightScenarioRunner()
    sc_real = FlightScenarioDefinition(
        name="hover_scenario",
        description="real execution",
        initial_position=np.array([1.0, 1.0, 1.0]),
        max_steps=5,
    )
    res = runner.run_scenario(sc_real)

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

    # Construct false configuration: same name "hover_scenario", but different initial_position [9.0, 9.0, 9.0]!
    sc_fake = FlightScenarioDefinition(
        name="hover_scenario",
        description="falsified execution",
        initial_position=np.array([9.0, 9.0, 9.0]),
        max_steps=5,
    )

    cfg = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="test",
        checkpoint_hash=None,
        random_seed=42,
        scenario_names=["hover_scenario"],
        scenarios=[sc_fake.to_dict()],
        provenance_complete=True,
    )

    recorder = FlightRecorder(output_dir=tmp_path)
    record = recorder.create_record(
        experiment_id="content_mismatch_test",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[res],
        configuration=cfg,
    )

    # Content mismatch MUST result in provenance_complete = False!
    assert record.provenance_complete is False
    assert record.configuration.provenance_complete is False
    assert record.verify_config_fingerprint() is True


def test_explicit_configuration_fully_equivalent_preserved(tmp_path: Path) -> None:
    """Explicit configuration with matching scenario content preserves provenance_complete = True."""
    from adaptive_rl.flight_testing.scenarios import (
        FlightScenarioDefinition,
        FlightScenarioRunner,
    )

    runner = FlightScenarioRunner()
    sc = FlightScenarioDefinition(
        name="hover_scenario_equiv",
        description="legitimate execution",
        initial_position=np.array([2.0, 2.0, 2.0]),
        max_steps=5,
    )
    res = runner.run_scenario(sc)

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

    # Legitimate matching configuration
    cfg = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="test",
        checkpoint_hash=None,
        random_seed=42,
        scenario_names=["hover_scenario_equiv"],
        scenarios=[sc.to_dict()],
        provenance_complete=True,
    )

    recorder = FlightRecorder(output_dir=tmp_path)
    record = recorder.create_record(
        experiment_id="content_match_test",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[res],
        configuration=cfg,
    )

    # Legitimate match MUST preserve provenance_complete = True!
    assert record.provenance_complete is True
    assert record.configuration.provenance_complete is True
    assert record.verify_config_fingerprint() is True
    assert record.verify_record_digest() is True


def test_provenance_captures_module_helper_changes() -> None:
    """extract_callable_provenance fingerprints module-qualified helper functions referenced in co_names."""
    import types

    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    mod1 = types.ModuleType("dummy_helper_mod")
    mod1.transform = lambda x: x * 2  # type: ignore[attr-defined]

    mod2 = types.ModuleType("dummy_helper_mod")
    mod2.transform = lambda x: x * 999  # type: ignore[attr-defined]

    def caller_fn(val: float) -> float:
        return dummy_mod.transform(val)  # type: ignore[name-defined] # noqa: F821

    # Caller with mod1
    caller_fn.__globals__["dummy_mod"] = mod1
    prov1 = extract_callable_provenance(caller_fn)

    # Caller with mod2
    caller_fn.__globals__["dummy_mod"] = mod2
    prov2 = extract_callable_provenance(caller_fn)

    # Different module helper implementation MUST yield different referenced_globals provenance!
    assert prov1["referenced_globals"] != prov2["referenced_globals"]


def test_provenance_captures_inherited_class_methods() -> None:
    """Custom callable class instances capture MRO hierarchy and inherited methods."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    class BaseA:
        def compute(self, x: float) -> float:
            return x + 1.0

    class DerivedA(BaseA):
        def __call__(self, x: float) -> float:
            return self.compute(x)

    class BaseB:
        def compute(self, x: float) -> float:
            return x + 100.0

    class DerivedB(BaseB):
        def __call__(self, x: float) -> float:
            return self.compute(x)

    inst_a = DerivedA()
    inst_b = DerivedB()

    prov_a = extract_callable_provenance(inst_a)
    prov_b = extract_callable_provenance(inst_b)

    assert prov_a["callable_class"] != prov_b["callable_class"]
    assert "class_methods" in prov_a
    assert "class_methods" in prov_b
    assert prov_a["class_methods"] != prov_b["class_methods"]


def test_explicit_configuration_duplicate_scenario_names_1to1_matching(tmp_path: Path) -> None:
    """1-to-1 canonical matching correctly handles duplicate scenario names without collision."""
    from adaptive_rl.flight_testing.scenarios import (
        FlightScenarioDefinition,
        FlightScenarioRunner,
    )

    runner = FlightScenarioRunner()
    sc1 = FlightScenarioDefinition(
        name="repeat_scenario",
        description="trial 1",
        initial_position=np.array([1.0, 1.0, 1.0]),
        max_steps=5,
    )
    sc2 = FlightScenarioDefinition(
        name="repeat_scenario",
        description="trial 2 with different initial position",
        initial_position=np.array([4.0, 4.0, 2.0]),
        max_steps=5,
    )
    res1 = runner.run_scenario(sc1)
    res2 = runner.run_scenario(sc2)

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

    # 1. Matching 1-to-1 duplicate configuration
    cfg_matching = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="test",
        checkpoint_hash=None,
        random_seed=42,
        scenario_names=["repeat_scenario", "repeat_scenario"],
        scenarios=[sc1.to_dict(), sc2.to_dict()],
        provenance_complete=True,
    )

    recorder = FlightRecorder(output_dir=tmp_path)
    record_ok = recorder.create_record(
        experiment_id="dup_sc_match",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[res1, res2],
        configuration=cfg_matching,
    )
    assert record_ok.provenance_complete is True
    assert record_ok.configuration.provenance_complete is True

    # 2. Missing one of the duplicate runs (length mismatch)
    cfg_single = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="test",
        checkpoint_hash=None,
        random_seed=42,
        scenario_names=["repeat_scenario"],
        scenarios=[sc1.to_dict()],
        provenance_complete=True,
    )
    record_missing = recorder.create_record(
        experiment_id="dup_sc_missing",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[res1, res2],
        configuration=cfg_single,
    )
    assert record_missing.provenance_complete is False
    assert record_missing.configuration.provenance_complete is False

    # 3. Differing duplicate scenario configuration (e.g. sc1 and sc1 instead of sc1 and sc2)
    cfg_mismatch = ExperimentConfigurationFingerprint(
        physics=physics,
        environment_bounds=[30.0, 30.0, 15.0],
        algorithm_name="test",
        checkpoint_hash=None,
        random_seed=42,
        scenario_names=["repeat_scenario", "repeat_scenario"],
        scenarios=[sc1.to_dict(), sc1.to_dict()],  # Second sc2 replaced by duplicate of sc1!
        provenance_complete=True,
    )
    record_mismatch = recorder.create_record(
        experiment_id="dup_sc_mismatch",
        physics_params={"mass": 1.0, "gravity": 9.81},
        scenario_results=[res1, res2],
        configuration=cfg_mismatch,
    )
    assert record_mismatch.provenance_complete is False
    assert record_mismatch.configuration.provenance_complete is False


def test_provenance_captures_descriptors_staticmethod_classmethod_property() -> None:
    """Class methods, static methods, and properties are unwrapped and their changes alter fingerprints."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    class ControllerA:
        @staticmethod
        def static_gain() -> float:
            return 1.5

        @classmethod
        def class_limit(cls) -> float:
            return 10.0

        @property
        def prop_val(self) -> float:
            return 42.0

        def __call__(self, x: float) -> float:
            return x + self.static_gain() + self.class_limit() + self.prop_val

    class ControllerB:
        @staticmethod
        def static_gain() -> float:
            return 2.5  # Changed static method

        @classmethod
        def class_limit(cls) -> float:
            return 10.0

        @property
        def prop_val(self) -> float:
            return 42.0

        def __call__(self, x: float) -> float:
            return x + self.static_gain() + self.class_limit() + self.prop_val

    class ControllerC:
        @staticmethod
        def static_gain() -> float:
            return 1.5

        @classmethod
        def class_limit(cls) -> float:
            return 20.0  # Changed class method

        @property
        def prop_val(self) -> float:
            return 42.0

        def __call__(self, x: float) -> float:
            return x + self.static_gain() + self.class_limit() + self.prop_val

    class ControllerD:
        @staticmethod
        def static_gain() -> float:
            return 1.5

        @classmethod
        def class_limit(cls) -> float:
            return 10.0

        @property
        def prop_val(self) -> float:
            return 99.0  # Changed property

        def __call__(self, x: float) -> float:
            return x + self.static_gain() + self.class_limit() + self.prop_val

    prov_a = extract_callable_provenance(ControllerA())
    prov_b = extract_callable_provenance(ControllerB())
    prov_c = extract_callable_provenance(ControllerC())
    prov_d = extract_callable_provenance(ControllerD())

    fp_a = compute_config_fingerprint(prov_a)
    fp_b = compute_config_fingerprint(prov_b)
    fp_c = compute_config_fingerprint(prov_c)
    fp_d = compute_config_fingerprint(prov_d)

    # All fingerprints must differ and provenance must be complete
    assert fp_a != fp_b
    assert fp_a != fp_c
    assert fp_a != fp_d
    assert fp_b != fp_c
    assert fp_c != fp_d

    assert prov_a.get("provenance_complete") is True
    assert "class_methods" in prov_a
    assert "static_gain" in prov_a["class_methods"]
    assert "class_limit" in prov_a["class_methods"]
    assert "prop_val" in prov_a["class_methods"]
    assert prov_a["class_methods"]["static_gain"]["descriptor"] == "staticmethod"
    assert prov_a["class_methods"]["class_limit"]["descriptor"] == "classmethod"
    assert prov_a["class_methods"]["prop_val"]["descriptor"] == "property"


def test_provenance_nested_module_dependency_resolution() -> None:
    """Nested module attributes accessed via co_names are recursively resolved and fingerprinted."""
    import types

    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    dummy_pkg = types.ModuleType("dummy_pkg")
    dummy_sub = types.ModuleType("dummy_sub")
    dummy_pkg.sub = dummy_sub

    def helper_v1(x: float) -> float:
        return x + 1.0

    def helper_v2(x: float) -> float:
        return x + 2.0

    dummy_sub.helper = helper_v1

    globals()["_test_dummy_pkg"] = dummy_pkg
    try:

        def compute(val: float) -> float:
            return _test_dummy_pkg.sub.helper(val)

        prov1 = extract_callable_provenance(compute)
        assert prov1.get("provenance_complete") is True
        assert "referenced_globals" in prov1
        assert "_test_dummy_pkg" in prov1["referenced_globals"]
        assert "_test_dummy_pkg.attributes" in prov1["referenced_globals"]
        assert "sub" in prov1["referenced_globals"]["_test_dummy_pkg.attributes"]
        sub_info = prov1["referenced_globals"]["_test_dummy_pkg.attributes"]["sub"]
        assert "helper" in sub_info.get("attributes", {})

        dummy_sub.helper = helper_v2
        prov2 = extract_callable_provenance(compute)
        fp1 = compute_config_fingerprint(prov1)
        fp2 = compute_config_fingerprint(prov2)
        assert fp1 != fp2
    finally:
        globals().pop("_test_dummy_pkg", None)


def test_provenance_unsupported_dynamic_dependency_records_incomplete() -> None:
    """Unsupported dynamic objects mark provenance as incomplete with explanatory reason."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    globals()["_test_opaque_val"] = object()
    try:

        def dynamic_func(x: float) -> float:
            return x + float(id(_test_opaque_val))

        prov = extract_callable_provenance(dynamic_func)
        assert prov.get("provenance_complete") is False
        assert "referenced_globals" in prov
        assert "_test_opaque_val" in prov["referenced_globals"]
        assert prov["referenced_globals"]["_test_opaque_val"]["unsupported_dependency"] == "object"
        assert "reason" in prov["referenced_globals"]["_test_opaque_val"]
    finally:
        globals().pop("_test_opaque_val", None)


def test_provenance_self_recursive_function_cycle_handling() -> None:
    """Self-recursive functions terminate cycle tracking cleanly without infinite recursion."""
    from adaptive_rl.flight_testing.scenarios import extract_callable_provenance

    # 1. Closure-based self-recursion
    def factorial_closure(n: int) -> int:
        if n <= 1:
            return 1
        return n * factorial_closure(n - 1)

    prov_c = extract_callable_provenance(factorial_closure)
    assert prov_c.get("provenance_complete") is True
    fp_c = compute_config_fingerprint(prov_c)
    assert len(fp_c) == 64
    assert "closure" in prov_c
    assert "factorial_closure" in prov_c["closure"]
    assert "recursion_ref" in prov_c["closure"]["factorial_closure"]

    # 2. Global-based self-recursion
    globals()["_test_factorial_global"] = lambda n: 1
    try:

        def factorial_global(n: int) -> int:
            global _test_factorial_global
            if n <= 1:
                return 1
            return n * _test_factorial_global(n - 1)

        globals()["_test_factorial_global"] = factorial_global
        prov_g = extract_callable_provenance(factorial_global)
        assert prov_g.get("provenance_complete") is True
        fp_g = compute_config_fingerprint(prov_g)
        assert len(fp_g) == 64
        assert "referenced_globals" in prov_g
        assert "_test_factorial_global" in prov_g["referenced_globals"]
        assert "recursion_ref" in prov_g["referenced_globals"]["_test_factorial_global"]
    finally:
        globals().pop("_test_factorial_global", None)
