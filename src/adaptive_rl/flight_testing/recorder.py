"""Reproducible Flight Recorder, run fingerprinting, and canonical serialization (Issue #253, #277).

Maintains tamper-evident cryptographic provenance for flight validation runs,
capturing repository state, pinned dependencies, simulation physics parameters,
random seeds, numerical integrator configuration, and empirical flight metrics.
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
    """Serialize dictionary or primitive to canonical JSON with sorted keys and stable float formatting."""

    def _sanitize(val: Any) -> Any:
        if isinstance(val, (int, bool, str)) or val is None:
            return val
        if isinstance(val, float):
            if not math.isfinite(val):
                return str(val)
            # 6 decimal places for stable floating point hashing
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
    """Complete, self-contained reproducible experiment record."""

    experiment_id: str
    timestamp_utc: str
    config_fingerprint: str
    physics: PhysicsSimulationFingerprint
    provenance: ExperimentMetadata
    algorithm_name: str
    checkpoint_hash: Optional[str]
    random_seed: Optional[int]
    scenario_metrics: List[FlightScenarioMetrics]
    aggregate_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
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
        }

    def verify_fingerprint(self) -> bool:
        """Verify that the stored config_fingerprint matches the actual physics configuration."""
        expected = compute_config_fingerprint(self.physics.to_dict())
        return self.config_fingerprint == expected


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

        return FlightRecord(
            experiment_id=experiment_id,
            timestamp_utc=now_utc,
            config_fingerprint=config_hash,
            physics=physics,
            provenance=provenance,
            algorithm_name=algorithm_name,
            checkpoint_hash=chk_hash,
            random_seed=random_seed,
            scenario_metrics=metrics_list,
            aggregate_summary=agg_summary,
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
            raise ValueError(f"Corrupt or non-JSON flight record file: {exc}")

        # Validate required root keys
        required_keys = [
            "experiment_id",
            "config_fingerprint",
            "physics",
            "provenance",
            "scenario_metrics",
        ]
        for k in required_keys:
            if k not in data:
                raise ValueError(f"Flight record missing required schema key: '{k}'")

        # Parse physics
        p_data = data["physics"]
        physics = PhysicsSimulationFingerprint(
            mass=float(p_data["mass"]),
            gravity=float(p_data["gravity"]),
            arm_length=float(p_data["arm_length"]),
            arm_span=float(p_data["arm_span"]),
            rotor_offset_d=float(p_data["rotor_offset_d"]),
            rotor_radius=float(p_data["rotor_radius"]),
            max_thrust_per_motor=float(p_data["max_thrust_per_motor"]),
            torque_to_thrust_ratio=float(p_data["torque_to_thrust_ratio"]),
            linear_damping=float(p_data["linear_damping"]),
            angular_damping=float(p_data["angular_damping"]),
            inertia=[float(x) for x in p_data["inertia"]],
            dt=float(p_data["dt"]),
            substeps=int(p_data["substeps"]),
            substep_dt=float(p_data["substep_dt"]),
            body_frame_lidar=bool(p_data.get("body_frame_lidar", False)),
            num_lidar_rays=int(p_data.get("num_lidar_rays", 16)),
            lidar_range=float(p_data.get("lidar_range", 20.0)),
        )

        # Validate fingerprint integrity
        expected_hash = compute_config_fingerprint(physics.to_dict())
        if data["config_fingerprint"] != expected_hash:
            raise ValueError(
                f"Config fingerprint mismatch in record '{data['experiment_id']}': "
                f"stored {data['config_fingerprint']}, computed {expected_hash}. "
                "Behavior-affecting parameters have been altered or corrupted."
            )

        # Parse provenance
        prov_data = data["provenance"]
        provenance = ExperimentMetadata(
            git_commit=str(prov_data.get("git_commit", "unknown")),
            git_branch=str(prov_data.get("git_branch", "unknown")),
            git_dirty=bool(prov_data.get("git_dirty", False)),
            os_name=str(prov_data.get("os_name", "")),
            os_version=str(prov_data.get("os_version", "")),
            python_version=str(prov_data.get("python_version", "")),
            software_packages=dict(prov_data.get("software_packages", {})),
            device=str(prov_data.get("device", "cpu")),
            cpu_count=int(prov_data.get("cpu_count", 1)),
            gpu_count=int(prov_data.get("gpu_count", 0)),
            gpu_name=prov_data.get("gpu_name"),
        )

        # Parse metrics
        metrics: List[FlightScenarioMetrics] = []
        for m in data.get("scenario_metrics", []):
            metrics.append(FlightScenarioMetrics(**m))

        return FlightRecord(
            experiment_id=str(data["experiment_id"]),
            timestamp_utc=str(data.get("timestamp_utc", "")),
            config_fingerprint=str(data["config_fingerprint"]),
            physics=physics,
            provenance=provenance,
            algorithm_name=str(data.get("algorithm_name", "unknown")),
            checkpoint_hash=data.get("checkpoint_hash"),
            random_seed=data.get("random_seed"),
            scenario_metrics=metrics,
            aggregate_summary=dict(data.get("aggregate_summary", {})),
        )
