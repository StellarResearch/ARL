"""Reproducible Flight Recorder, run fingerprinting, and canonical serialization (Issue #253, #277).

Maintains tamper-evident cryptographic provenance for flight validation runs,
capturing repository state, pinned dependencies, simulation physics parameters,
random seeds, numerical integrator configuration, and empirical flight metrics.

SECURITY AND CRYPTOGRAPHIC PROVENANCE NOTE:
------------------------------------------
The SHA-256 digests generated here (config_fingerprint and record_digest) provide
mathematical integrity comparison and tamper detection. They detect accidental corruption
or unauthorized modification of simulation parameters between execution and analysis.
They do NOT provide non-repudiation or authenticated identity against an active attacker
who possesses write permissions to both the record data and digest calculation logic.
For strict adversarial non-repudiation, an optional cryptographic HMAC/digital signature
field is provided to attach an asymmetric key signature.
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
    """Serialize dictionary or primitive to canonical JSON with sorted keys and stable float formatting.

    Preserves 6 decimal places for standard values while preserving 12 significant digits for small
    physical parameters (< 1e-5, such as damping or inertia terms) to prevent loss of precision.
    """

    def _sanitize(val: Any) -> Any:
        if isinstance(val, (int, bool, str)) or val is None:
            return val
        if isinstance(val, float):
            if not math.isfinite(val):
                return str(val)
            if abs(val) < 1e-5 and val != 0.0:
                return round(val, 12)
            return round(val, 6)
        if isinstance(val, (list, tuple)):
            return [_sanitize(x) for x in val]
        if isinstance(val, dict):
            return {
                str(k): _sanitize(v) for k, v in sorted(val.items(), key=lambda item: str(item[0]))
            }
        if hasattr(val, "to_dict"):
            return _sanitize(val.to_dict())
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
    extra_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "physics": self.physics.to_dict(),
            "environment_bounds": self.environment_bounds,
            "algorithm_name": self.algorithm_name,
            "checkpoint_hash": self.checkpoint_hash,
            "random_seed": self.random_seed,
            "scenario_names": sorted(self.scenario_names),
            "controller_type": self.controller_type,
            "controller_gains": self.controller_gains,
            "extra_parameters": self.extra_parameters,
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
    aggregate_summary: Dict[str, Any] = field(default_factory=dict)
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
            "aggregate_summary": self.aggregate_summary,
            "signature": self.signature,
        }
        if include_digest:
            data["record_digest"] = self.record_digest
        return data

    def verify_config_fingerprint(self) -> bool:
        """Verify that the stored config_fingerprint matches the actual physics configuration."""
        expected = compute_config_fingerprint(self.physics.to_dict())
        return self.config_fingerprint == expected

    def verify_fingerprint(self) -> bool:
        """Backwards compatibility alias for verify_config_fingerprint."""
        return self.verify_config_fingerprint()

    def verify_record_digest(self) -> bool:
        """Verify that the stored record_digest matches the SHA-256 digest of record contents."""
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
        )

        config_hash = compute_config_fingerprint(physics.to_dict())

        # Checkpoint hash
        chk_hash: Optional[str] = None
        if checkpoint_path is not None:
            c_path = Path(checkpoint_path)
            if c_path.is_file():
                with open(c_path, "rb") as f:
                    chk_hash = hashlib.sha256(f.read()).hexdigest()
            else:
                chk_hash = "not_found"

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

        # Extract metrics
        metrics_list: List[FlightScenarioMetrics] = []
        for r in scenario_results:
            if isinstance(r, FlightScenarioResult):
                metrics_list.append(r.metrics)
            elif isinstance(r, FlightScenarioMetrics):
                metrics_list.append(r)
            else:
                raise ValueError(f"Unsupported result type: {type(r)}")

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
            "aggregate_summary": agg_summary,
            "signature": signature,
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
            aggregate_summary=agg_summary,
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
            aggregate_summary=data.get("aggregate_summary", {}),
            signature=data.get("signature"),
        )

        if not record.verify_config_fingerprint():
            raise ValueError(
                f"Config fingerprint mismatch in '{path.name}'. Stored fingerprint '{record.config_fingerprint}' "
                f"does not match recomputed physics fingerprint '{compute_config_fingerprint(record.physics.to_dict())}'."
            )

        return record
