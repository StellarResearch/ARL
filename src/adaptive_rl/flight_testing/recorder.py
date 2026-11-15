"""Reproducible Flight Recorder, run fingerprinting, and canonical serialization (Issue #253, #277).

Maintains tamper-evident cryptographic provenance for flight validation runs,
capturing repository state, pinned dependencies, simulation physics parameters,
random seeds, numerical integrator configuration, and empirical flight metrics.

SECURITY AND CRYPTOGRAPHIC PROVENANCE NOTE:
------------------------------------------
The SHA-256 digests generated here (config_fingerprint and record_digest) provide
mathematical integrity comparison and tamper detection. They detect accidental corruption
or unauthorized modification of simulation parameters between execution and analysis.
A SHA-256 digest is an unkeyed cryptographic checksum that verifies data integrity,
NOT identity or origin authenticity. It does NOT provide non-repudiation or authenticated
identity against an active attacker who possesses write permissions to both the record data
and digest calculation logic.

In contrast, a digital signature (or HMAC) requires an authenticated cryptographic key
(such as an asymmetric private key or pre-shared HMAC secret). The optional `signature`
field in FlightRecord is reserved strictly for authenticated cryptographic signatures
(e.g., ECDSA, Ed25519, or HMAC-SHA256) and must never be conflated with plain SHA-256 digests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from adaptive_rl.flight_testing.scenarios import (
    FlightScenarioMetrics,
    FlightScenarioResult,
)
from adaptive_rl.manifest import (
    collect_git_metadata,
    collect_hardware_telemetry,
    collect_host_system,
    collect_software_packages,
)

logger = logging.getLogger(__name__)


def canonical_json_dumps(data: Any) -> str:
    """Serialize dictionary or primitive to canonical JSON with sorted keys and exact IEEE 754 float formatting.

    Preserves full finite floating-point precision without lossy rounding, ensuring distinct
    scientific configurations (such as sub-microgram mass or sub-millimeter geometry differences)
    produce distinct cryptographic digests.
    """

    def _sanitize(val: Any) -> Any:
        if isinstance(val, bool) or val is None:
            return val
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            f_val = float(val)
            if not math.isfinite(f_val):
                if math.isnan(f_val):
                    return "NaN"
                return "Infinity" if f_val > 0 else "-Infinity"
            # Normalize -0.0 to 0.0 to guarantee deterministic hashing
            return 0.0 if f_val == 0.0 else f_val
        if isinstance(val, str):
            return val
        if isinstance(val, np.ndarray):
            return [_sanitize(x) for x in val.tolist()]
        if isinstance(val, (list, tuple)):
            return [_sanitize(x) for x in val]
        if hasattr(val, "to_dict"):
            return _sanitize(val.to_dict())
        if isinstance(val, dict):
            return {
                str(k): _sanitize(v) for k, v in sorted(val.items(), key=lambda item: str(item[0]))
            }
        return str(val)

    sanitized = _sanitize(data)
    return json.dumps(sanitized, sort_keys=True, separators=(",", ":"))


def compute_config_fingerprint(config_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint of behavior-affecting configuration parameters."""
    canonical_str = canonical_json_dumps(config_dict)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def compute_record_digest(record_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 digest of a completed experiment record."""
    canonical_str = canonical_json_dumps(record_dict)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


@dataclass
class PhysicsSimulationFingerprint:
    """Behavior-affecting physical parameters defining the numerical flight experiment."""

    mass: float
    gravity: float
    arm_length: float
    arm_span: float
    rotor_offset_d: float
    rotor_radius: float
    max_thrust_per_motor: float
    torque_to_thrust_ratio: float
    linear_damping: float
    angular_damping: float
    inertia: List[float]
    dt: float
    substeps: int
    substep_dt: float
    body_frame_lidar: bool
    num_lidar_rays: int
    lidar_range: float
    enable_motor_dynamics: bool = False
    motor_time_constant: float = 0.0
    wind_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    enable_ground_effect: bool = False
    quadratic_drag_coeff: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfigurationFingerprint:
    """Comprehensive specification covering all behavior-affecting experiment inputs."""

    physics: PhysicsSimulationFingerprint
    environment_bounds: List[float]
    algorithm_name: str
    checkpoint_hash: Optional[str]
    random_seed: Optional[int]
    scenario_names: List[str]
    controller_type: str = "QuadrotorPIDController"
    controller_gains: Dict[str, float] = field(default_factory=dict)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    extra_parameters: Dict[str, Any] = field(default_factory=dict)
    provenance_complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "physics": self.physics.to_dict(),
            "environment_bounds": self.environment_bounds,
            "algorithm_name": self.algorithm_name,
            "checkpoint_hash": self.checkpoint_hash,
            "random_seed": self.random_seed,
            "scenario_names": sorted(self.scenario_names),
            "scenarios": self.scenarios,
            "controller_type": self.controller_type,
            "controller_gains": self.controller_gains,
            "extra_parameters": self.extra_parameters,
            "provenance_complete": self.provenance_complete,
        }


@dataclass
class ExperimentMetadata:
    """Environmental, software, and hardware provenance record."""

    git_commit: str
    git_branch: str
    git_dirty: bool
    os_name: str
    os_version: str
    python_version: str
    software_packages: Dict[str, str]
    device: str
    cpu_count: int
    gpu_count: int
    gpu_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlightRecord:
    """Complete, self-contained reproducible experiment record with dual integrity digests."""

    experiment_id: str
    timestamp_utc: str
    config_fingerprint: str
    record_digest: str
    physics: PhysicsSimulationFingerprint
    provenance: ExperimentMetadata
    algorithm_name: str
    checkpoint_hash: Optional[str]
    random_seed: Optional[int]
    scenario_metrics: List[FlightScenarioMetrics]
    trajectories: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    configuration: Optional[ExperimentConfigurationFingerprint] = None
    aggregate_summary: Dict[str, Any] = field(default_factory=dict)
    provenance_complete: bool = True
    signature: Optional[str] = None  # Optional asymmetric HMAC / digital signature

    def to_dict(self, include_digest: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "timestamp_utc": self.timestamp_utc,
            "config_fingerprint": self.config_fingerprint,
            "physics": self.physics.to_dict(),
            "provenance": self.provenance.to_dict(),
            "algorithm_name": self.algorithm_name,
            "checkpoint_hash": self.checkpoint_hash,
            "random_seed": self.random_seed,
            "scenario_metrics": [m.to_dict() for m in self.scenario_metrics],
            "trajectories": self.trajectories,
            "aggregate_summary": self.aggregate_summary,
            "provenance_complete": self.provenance_complete,
            "signature": self.signature,
        }
        if self.configuration is not None:
            data["configuration"] = self.configuration.to_dict()
        if include_digest:
            data["record_digest"] = self.record_digest
        return data

    def verify_config_fingerprint(self) -> bool:
        """Verify that the stored config_fingerprint matches the configuration."""
        if self.configuration is not None:
            # Inconsistent provenance check: top-level and configuration flags must agree
            if self.provenance_complete != self.configuration.provenance_complete:
                return False
            expected = compute_config_fingerprint(self.configuration.to_dict())
        else:
            expected = compute_config_fingerprint(self.physics.to_dict())
        return self.config_fingerprint == expected

    def verify_fingerprint(self) -> bool:
        """Backwards compatibility alias for verify_config_fingerprint."""
        return self.verify_config_fingerprint()

    def verify_record_digest(self) -> bool:
        """Verify that the stored record_digest matches the SHA-256 digest of record contents."""
        if self.configuration is not None:
            if self.provenance_complete != self.configuration.provenance_complete:
                return False
        payload = self.to_dict(include_digest=False)
        expected = compute_record_digest(payload)
        return self.record_digest == expected


class FlightRecorder:
    """Manages creation, canonical serialization, export, and verification of flight experiment records."""

    def __init__(self, output_dir: Path | str = "artifacts/flight_records") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_record(
        self,
        experiment_id: str,
        physics_params: Dict[str, Any],
        scenario_results: Sequence[FlightScenarioResult | FlightScenarioMetrics],
        algorithm_name: str = "pid_reference",
        checkpoint_path: Optional[Path | str] = None,
        random_seed: Optional[int] = None,
        signature: Optional[str] = None,
        configuration: Optional[ExperimentConfigurationFingerprint | Dict[str, Any]] = None,
        trajectories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        scenarios: Optional[Sequence[Any]] = None,
    ) -> FlightRecord:
        """Create a cryptographically fingerprinted FlightRecord from execution data."""
        from datetime import datetime, timezone

        now_utc = datetime.now(timezone.utc).isoformat()

        # Build physics fingerprint
        physics = PhysicsSimulationFingerprint(
            mass=float(physics_params.get("mass", 1.0)),
            gravity=float(physics_params.get("gravity", 9.81)),
            arm_length=float(physics_params.get("arm_length", 0.2)),
            arm_span=float(physics_params.get("arm_span", 0.4)),
            rotor_offset_d=float(physics_params.get("rotor_offset_d", 0.2 / math.sqrt(2.0))),
            rotor_radius=float(physics_params.get("rotor_radius", 0.09)),
            max_thrust_per_motor=float(physics_params.get("max_thrust_per_motor", 5.0)),
            torque_to_thrust_ratio=float(physics_params.get("torque_to_thrust_ratio", 0.01)),
            linear_damping=float(physics_params.get("linear_damping", 0.1)),
            angular_damping=float(physics_params.get("angular_damping", 0.02)),
            inertia=[float(x) for x in physics_params.get("inertia", [0.01, 0.01, 0.02])],
            dt=float(physics_params.get("dt", 0.05)),
            substeps=int(physics_params.get("substeps", 5)),
            substep_dt=float(physics_params.get("substep_dt", 0.01)),
            body_frame_lidar=bool(physics_params.get("body_frame_lidar", False)),
            num_lidar_rays=int(physics_params.get("num_lidar_rays", 16)),
            lidar_range=float(physics_params.get("lidar_range", 20.0)),
            enable_motor_dynamics=bool(physics_params.get("enable_motor_dynamics", False)),
            motor_time_constant=float(physics_params.get("motor_time_constant", 0.0)),
            wind_vector=[float(x) for x in physics_params.get("wind_vector", [0.0, 0.0, 0.0])],
            enable_ground_effect=bool(physics_params.get("enable_ground_effect", False)),
            quadratic_drag_coeff=float(physics_params.get("quadratic_drag_coeff", 0.0)),
        )

        # Extract metrics and trajectories
        metrics_list: List[FlightScenarioMetrics] = []
        collected_trajectories: Dict[str, List[Dict[str, Any]]] = {}
        for r in scenario_results:
            if isinstance(r, FlightScenarioResult):
                metrics_list.append(r.metrics)
                collected_trajectories[r.scenario_name] = [step.to_dict() for step in r.telemetry]
            elif isinstance(r, FlightScenarioMetrics):
                metrics_list.append(r)
                collected_trajectories[r.scenario_name] = []
            else:
                raise ValueError(f"Unsupported result type: {type(r)}")

        if trajectories is not None:
            collected_trajectories.update(trajectories)

        # Checkpoint hash
        chk_hash: Optional[str] = None
        if checkpoint_path is not None:
            c_path = Path(checkpoint_path)
            if c_path.is_file():
                with open(c_path, "rb") as f:
                    chk_hash = hashlib.sha256(f.read()).hexdigest()
            else:
                chk_hash = "not_found"

        # Serialize scenario provenance definitions
        scenario_provenance: List[Dict[str, Any]] = []
        provenance_complete: bool = True

        raw_scenarios = (
            scenarios if scenarios is not None else physics_params.get("scenarios", None)
        )

        if raw_scenarios is not None:
            for sc in raw_scenarios:
                if hasattr(sc, "to_dict"):
                    d = sc.to_dict()
                    scenario_provenance.append(d)
                    if (
                        getattr(sc, "provenance_complete", True) is False
                        or d.get("provenance_complete") is False
                    ):
                        provenance_complete = False
                elif isinstance(sc, dict):
                    scenario_provenance.append(sc)
                    if sc.get("provenance_complete") is False:
                        provenance_complete = False
        else:
            # Automatically infer scenario definitions from scenario_results
            for r in scenario_results:
                sc_def = getattr(r, "scenario_definition", None)
                if sc_def is not None and hasattr(sc_def, "to_dict"):
                    d = sc_def.to_dict()
                    scenario_provenance.append(d)
                    if (
                        getattr(sc_def, "provenance_complete", True) is False
                        or d.get("provenance_complete") is False
                    ):
                        provenance_complete = False
                else:
                    name = getattr(r, "scenario_name", None)
                    if name:
                        try:
                            from adaptive_rl.flight_testing.scenarios import get_scenario

                            sc_obj = get_scenario(name)
                            d = sc_obj.to_dict()
                            scenario_provenance.append(d)
                            if (
                                getattr(sc_obj, "provenance_complete", True) is False
                                or d.get("provenance_complete") is False
                            ):
                                provenance_complete = False
                        except Exception:
                            provenance_complete = False
                    else:
                        provenance_complete = False

        scenario_provenance.sort(key=lambda s: str(s.get("name", "")))

        # Verify whether all scenario_names have corresponding provenance
        expected_scenario_names = set(m.scenario_name for m in metrics_list)
        found_scenario_names = set(s.get("name") for s in scenario_provenance if "name" in s)
        if expected_scenario_names and not expected_scenario_names.issubset(found_scenario_names):
            provenance_complete = False
        if not expected_scenario_names and not scenario_provenance:
            provenance_complete = True

        # Build comprehensive ExperimentConfigurationFingerprint
        if configuration is not None:
            if isinstance(configuration, ExperimentConfigurationFingerprint):
                exp_config = configuration
            elif isinstance(configuration, dict):
                exp_config = ExperimentConfigurationFingerprint(**configuration)
            else:
                raise ValueError(f"Invalid configuration type: {type(configuration)}")

            # Validate explicitly supplied configuration against executed scenarios:
            cfg_valid = True
            if expected_scenario_names:
                cfg_names = set(exp_config.scenario_names)
                if not expected_scenario_names.issubset(cfg_names):
                    cfg_valid = False

                cfg_sc_names = set(
                    s.get("name")
                    for s in exp_config.scenarios
                    if isinstance(s, dict) and "name" in s
                )
                if not expected_scenario_names.issubset(cfg_sc_names):
                    cfg_valid = False

                for sc in exp_config.scenarios:
                    if isinstance(sc, dict) and sc.get("provenance_complete") is False:
                        cfg_valid = False

            # Define ONE authoritative provenance completeness result
            authoritative_complete = (
                provenance_complete and cfg_valid and bool(exp_config.provenance_complete)
            )
            # Synchronize configuration flag so top-level and configuration always agree
            exp_config.provenance_complete = authoritative_complete
        else:
            authoritative_complete = provenance_complete
            exp_config = ExperimentConfigurationFingerprint(
                physics=physics,
                environment_bounds=list(
                    physics_params.get("environment_bounds", [30.0, 30.0, 15.0])
                ),
                algorithm_name=algorithm_name,
                checkpoint_hash=chk_hash,
                random_seed=random_seed,
                scenario_names=sorted([m.scenario_name for m in metrics_list]),
                controller_type=str(
                    physics_params.get("controller_type", "QuadrotorPIDController")
                ),
                controller_gains=dict(physics_params.get("controller_gains", {})),
                scenarios=scenario_provenance,
                extra_parameters=dict(physics_params.get("extra_parameters", {})),
                provenance_complete=authoritative_complete,
            )

        config_hash = compute_config_fingerprint(exp_config.to_dict())

        # Software and host provenance
        git_meta = collect_git_metadata()
        host_sys = collect_host_system()
        pkgs = collect_software_packages().model_dump()
        hw = collect_hardware_telemetry()

        provenance = ExperimentMetadata(
            git_commit=git_meta.git_commit,
            git_branch=git_meta.git_branch,
            git_dirty=git_meta.git_dirty,
            os_name=host_sys.os_name,
            os_version=host_sys.os_version,
            python_version=host_sys.python_version,
            software_packages=pkgs,
            device=hw.device,
            cpu_count=hw.cpu_count,
            gpu_count=hw.gpu_count,
            gpu_name=hw.gpu_name,
        )

        # Aggregate summary
        total_scenarios = len(metrics_list)
        success_count = sum(1 for m in metrics_list if m.success)
        success_rate = (success_count / total_scenarios) if total_scenarios > 0 else 0.0
        collision_count = sum(m.collision_count for m in metrics_list)

        agg_summary = {
            "total_scenarios": total_scenarios,
            "successful_scenarios": success_count,
            "success_rate": round(success_rate, 4),
            "total_collisions": collision_count,
            "mean_duration_seconds": (
                round(float(np.mean([m.duration_seconds for m in metrics_list])), 4)
                if metrics_list
                else 0.0
            ),
            "mean_rms_position_error": (
                round(float(np.mean([m.rms_position_error for m in metrics_list])), 4)
                if metrics_list
                else 0.0
            ),
            "mean_control_effort": (
                round(float(np.mean([m.control_effort for m in metrics_list])), 4)
                if metrics_list
                else 0.0
            ),
        }

        # Pre-build record dict without record_digest to calculate exact record digest
        record_pre = {
            "experiment_id": experiment_id,
            "timestamp_utc": now_utc,
            "config_fingerprint": config_hash,
            "physics": physics.to_dict(),
            "provenance": provenance.to_dict(),
            "algorithm_name": algorithm_name,
            "checkpoint_hash": chk_hash,
            "random_seed": random_seed,
            "scenario_metrics": [m.to_dict() for m in metrics_list],
            "trajectories": collected_trajectories,
            "aggregate_summary": agg_summary,
            "provenance_complete": authoritative_complete,
            "signature": signature,
            "configuration": exp_config.to_dict(),
        }
        rec_digest = compute_record_digest(record_pre)

        return FlightRecord(
            experiment_id=experiment_id,
            timestamp_utc=now_utc,
            config_fingerprint=config_hash,
            record_digest=rec_digest,
            physics=physics,
            provenance=provenance,
            algorithm_name=algorithm_name,
            checkpoint_hash=chk_hash,
            random_seed=random_seed,
            scenario_metrics=metrics_list,
            trajectories=collected_trajectories,
            configuration=exp_config,
            aggregate_summary=agg_summary,
            provenance_complete=authoritative_complete,
            signature=signature,
        )

    def save_record_json(self, record: FlightRecord, filename: Optional[str] = None) -> Path:
        """Save a FlightRecord to disk in structured canonical JSON."""
        name = filename or f"{record.experiment_id}_record.json"
        target_path = self.output_dir / name
        serialized = canonical_json_dumps(record.to_dict())
        target_path.write_text(serialized, encoding="utf-8")
        return target_path

    def save_summary_csv(self, record: FlightRecord, filename: Optional[str] = None) -> Path:
        """Export scenario performance metrics from a FlightRecord to CSV."""
        name = filename or f"{record.experiment_id}_summary.csv"
        target_path = self.output_dir / name

        fieldnames = [
            "experiment_id",
            "scenario_name",
            "success",
            "outcome",
            "duration_steps",
            "duration_seconds",
            "final_position_error",
            "rms_position_error",
            "rms_velocity_error",
            "mean_attitude_error_deg",
            "min_obstacle_clearance",
            "collision_count",
            "control_effort",
            "config_fingerprint",
        ]

        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in record.scenario_metrics:
                writer.writerow(
                    {
                        "experiment_id": record.experiment_id,
                        "scenario_name": m.scenario_name,
                        "success": int(m.success),
                        "outcome": m.outcome,
                        "duration_steps": m.duration_steps,
                        "duration_seconds": round(m.duration_seconds, 4),
                        "final_position_error": round(m.final_position_error, 4),
                        "rms_position_error": round(m.rms_position_error, 4),
                        "rms_velocity_error": round(m.rms_velocity_error, 4),
                        "mean_attitude_error_deg": round(m.mean_attitude_error_deg, 2),
                        "min_obstacle_clearance": (
                            round(m.min_obstacle_clearance, 4)
                            if math.isfinite(m.min_obstacle_clearance)
                            else -1.0
                        ),
                        "collision_count": m.collision_count,
                        "control_effort": round(m.control_effort, 4),
                        "config_fingerprint": record.config_fingerprint,
                    }
                )
        return target_path

    @staticmethod
    def load_record_json(filepath: Path | str) -> FlightRecord:
        """Load and cryptographically validate a serialized FlightRecord."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Flight record file not found: {path}")

        raw_str = path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw_str)
        except Exception as exc:
            raise ValueError(f"Corrupt or non-JSON content in flight record {path}: {exc}") from exc

        # Validate required top-level schema fields
        required_fields = [
            "experiment_id",
            "timestamp_utc",
            "config_fingerprint",
            "physics",
            "provenance",
            "scenario_metrics",
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(
                f"Incomplete flight record schema; missing required schema keys: {missing}"
            )

        physics_data = data["physics"]
        physics = PhysicsSimulationFingerprint(**physics_data)

        prov_data = data["provenance"]
        provenance = ExperimentMetadata(**prov_data)

        metrics_list: List[FlightScenarioMetrics] = []
        for m in data["scenario_metrics"]:
            metrics_list.append(FlightScenarioMetrics(**m))

        trajectories = data.get("trajectories", {})

        cfg_data = data.get("configuration")
        if cfg_data:
            # Synchronize cfg_physics with physics from top-level to detect if either was tampered with
            configuration = ExperimentConfigurationFingerprint(
                physics=physics,
                environment_bounds=list(cfg_data.get("environment_bounds", [30.0, 30.0, 15.0])),
                algorithm_name=str(cfg_data.get("algorithm_name", "pid_reference")),
                checkpoint_hash=cfg_data.get("checkpoint_hash"),
                random_seed=cfg_data.get("random_seed"),
                scenario_names=list(cfg_data.get("scenario_names", [])),
                controller_type=str(cfg_data.get("controller_type", "QuadrotorPIDController")),
                controller_gains=dict(cfg_data.get("controller_gains", {})),
                scenarios=list(cfg_data.get("scenarios", [])),
                extra_parameters=dict(cfg_data.get("extra_parameters", {})),
                provenance_complete=bool(cfg_data.get("provenance_complete", True)),
            )
        else:
            configuration = ExperimentConfigurationFingerprint(
                physics=physics,
                environment_bounds=list(data.get("environment_bounds", [30.0, 30.0, 15.0])),
                algorithm_name=str(data.get("algorithm_name", "pid_reference")),
                checkpoint_hash=data.get("checkpoint_hash"),
                random_seed=data.get("random_seed"),
                scenario_names=sorted([m.scenario_name for m in metrics_list]),
                provenance_complete=bool(data.get("provenance_complete", True)),
            )

        record = FlightRecord(
            experiment_id=data["experiment_id"],
            timestamp_utc=data["timestamp_utc"],
            config_fingerprint=data["config_fingerprint"],
            record_digest=data.get("record_digest", ""),
            physics=physics,
            provenance=provenance,
            algorithm_name=data.get("algorithm_name", "pid_reference"),
            checkpoint_hash=data.get("checkpoint_hash"),
            random_seed=data.get("random_seed"),
            scenario_metrics=metrics_list,
            trajectories=trajectories,
            configuration=configuration,
            aggregate_summary=data.get("aggregate_summary", {}),
            provenance_complete=bool(data.get("provenance_complete", True)),
            signature=data.get("signature"),
        )

        if not record.verify_config_fingerprint():
            raise ValueError(
                f"Config fingerprint mismatch in '{path.name}'. Stored fingerprint '{record.config_fingerprint}' "
                f"does not match recomputed configuration fingerprint."
            )

        return record
