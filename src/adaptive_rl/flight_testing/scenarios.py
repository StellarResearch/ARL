"""Autonomous flight validation scenarios and metric evaluation suite for 6-DOF quadrotors.

Implements reproducible benchmark scenarios covering:
1. Stable hovering
2. Straight-line waypoint navigation
3. Vertical ascent and descent
4. Horizontal movement
5. Position holding under aerodynamic disturbance
6. 3D Obstacle avoidance (A* collision-free path planning)
7. Boundary avoidance
8. Target approach and arrival
9. Recovery from moderate disturbances
10. Actuator fault degradation (canonical 50% motor loss)

Advanced Phase III 3D Navigation Scenarios:
11. Multi-obstacle navigation
12. Narrow passage navigation
13. Dynamic obstacle crossing
14. Moving target tracking
15. Multiple sequential 3D waypoints
16. Dead-end / blocked-route handling
17. Dynamic replanning after obstacle shift
18. Return-to-Home (RTH)

Computes measurable, independently verifiable aerospace flight metrics:
- Position tracking error (mean, max, RMS)
- Velocity tracking error (mean, max, RMS)
- Attitude tracking error (mean, max)
- Angular velocity magnitude (mean, max)
- Goal distance and settling time
- Minimum obstacle clearance
- Actuator control effort
- Flight duration and wall-clock execution time
"""

from __future__ import annotations

import builtins
import csv
import functools
import hashlib
import inspect
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptive_rl.controllers.pid import QuadrotorPIDController
from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import (
    Drone6DOFEnv,
    DroneState6DOF,
    euler_to_quaternion,
    quaternion_to_euler_angles,
)
from adaptive_rl.navigation.dynamic_obstacles import (
    DivergentDynamicObstacleSphere3D,
    DynamicObstacleSphere3D,
)
from adaptive_rl.navigation.mission import MissionManager, MissionState
from adaptive_rl.navigation.planner import AStarPlanner3D

logger = logging.getLogger(__name__)


@dataclass
class FlightStepTelemetry:
    """Telemetry recorded at each physical simulation step."""

    step: int
    time_sec: float
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    euler_angles_deg: np.ndarray
    angular_velocity: np.ndarray
    target_position: np.ndarray
    distance_to_target: float
    action: np.ndarray
    motor_thrusts: np.ndarray
    min_obstacle_distance: float
    is_collision: bool
    is_success: bool
    commanded_thrusts: Optional[np.ndarray] = None
    actuator_fault_active: bool = False
    estimated_position: Optional[np.ndarray] = None
    estimated_velocity: Optional[np.ndarray] = None
    position_estimation_error: Optional[float] = None
    attitude_estimation_error_deg: Optional[float] = None
    mission_state: Optional[str] = None
    replan_reason: Optional[str] = None
    waiting_duration: Optional[float] = None
    min_predicted_clearance: Optional[float] = None
    planned_vs_actual_deviation: Optional[float] = None
    active_route_cost: Optional[float] = None
    state_source: Optional[str] = None
    obstacle_prediction_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        cmd_thrusts = (
            [round(float(x), 4) for x in self.commanded_thrusts]
            if self.commanded_thrusts is not None
            else [round(float(x), 4) for x in self.motor_thrusts]
        )
        d: Dict[str, Any] = {
            "step": self.step,
            "time_sec": round(self.time_sec, 4),
            "position": [round(float(x), 4) for x in self.position],
            "velocity": [round(float(x), 4) for x in self.velocity],
            "speed": round(float(np.linalg.norm(self.velocity)), 4),
            "quaternion": [round(float(x), 5) for x in self.quaternion],
            "euler_angles_deg": [round(float(x), 2) for x in self.euler_angles_deg],
            "angular_velocity": [round(float(x), 4) for x in self.angular_velocity],
            "target_position": [round(float(x), 4) for x in self.target_position],
            "distance_to_target": round(float(self.distance_to_target), 4),
            "action": [round(float(x), 4) for x in self.action],
            "motor_thrusts": [round(float(x), 4) for x in self.motor_thrusts],
            "commanded_thrusts": cmd_thrusts,
            "actuator_fault_active": self.actuator_fault_active,
            "min_obstacle_distance": (
                round(float(self.min_obstacle_distance), 4)
                if math.isfinite(self.min_obstacle_distance)
                else None
            ),
            "is_collision": self.is_collision,
            "is_success": self.is_success,
        }
        if self.estimated_position is not None:
            d["estimated_position"] = [round(float(x), 4) for x in self.estimated_position]
        if self.estimated_velocity is not None:
            d["estimated_velocity"] = [round(float(x), 4) for x in self.estimated_velocity]
        if self.position_estimation_error is not None:
            d["position_estimation_error"] = round(float(self.position_estimation_error), 4)
        if self.attitude_estimation_error_deg is not None:
            d["attitude_estimation_error_deg"] = round(float(self.attitude_estimation_error_deg), 3)
        if self.mission_state is not None:
            d["mission_state"] = self.mission_state
        if self.replan_reason is not None:
            d["replan_reason"] = self.replan_reason
        if self.waiting_duration is not None:
            d["waiting_duration"] = round(float(self.waiting_duration), 4)
        if self.min_predicted_clearance is not None:
            d["min_predicted_clearance"] = (
                round(float(self.min_predicted_clearance), 4)
                if math.isfinite(self.min_predicted_clearance)
                else -1.0
            )
        if self.planned_vs_actual_deviation is not None:
            d["planned_vs_actual_deviation"] = round(float(self.planned_vs_actual_deviation), 4)
        if self.active_route_cost is not None:
            d["active_route_cost"] = round(float(self.active_route_cost), 4)
        if self.state_source is not None:
            d["state_source"] = self.state_source
        if self.obstacle_prediction_error is not None:
            d["obstacle_prediction_error"] = round(float(self.obstacle_prediction_error), 4)
        return d


@dataclass
class FlightScenarioMetrics:
    """Summary flight performance metrics computed from empirical simulation data."""

    scenario_name: str
    success: bool
    outcome: str  # "success", "collision", "boundary_breach", "max_steps", "timeout", "recovered_degraded", "crash_fault", "blocked_no_path"
    duration_steps: int
    duration_seconds: float
    final_position_error: float
    mean_position_error: float
    max_position_error: float
    rms_position_error: float
    mean_velocity_error: float
    max_velocity_error: float
    rms_velocity_error: float
    mean_attitude_error_deg: float
    max_attitude_error_deg: float
    mean_angular_speed: float
    max_angular_speed: float
    initial_goal_distance: float
    final_goal_distance: float
    min_obstacle_clearance: float
    collision_count: int
    control_effort: float
    execution_wall_time_ms: float
    replan_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FlightScenarioResult:
    """Full scenario execution record containing metrics and step telemetry."""

    scenario_name: str
    description: str
    metrics: FlightScenarioMetrics
    telemetry: List[FlightStepTelemetry] = field(default_factory=list)
    scenario_definition: Optional[FlightScenarioDefinition] = None

    def to_dict(self, include_telemetry: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "metrics": self.metrics.to_dict(),
        }
        if self.scenario_definition is not None:
            data["scenario_definition"] = self.scenario_definition.to_dict()
        if include_telemetry:
            data["telemetry"] = [step.to_dict() for step in self.telemetry]
        return data

    def export_json(self, filepath: Path | str, include_telemetry: bool = True) -> Path:
        """Export scenario result to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(include_telemetry=include_telemetry), f, indent=2)
        return path

    def export_telemetry_csv(self, filepath: Path | str) -> Path:
        """Export time-series telemetry to CSV file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.telemetry:
            path.write_text("no telemetry recorded\n", encoding="utf-8")
            return path

        fieldnames = [
            "step",
            "time_sec",
            "pos_x",
            "pos_y",
            "pos_z",
            "vel_x",
            "vel_y",
            "vel_z",
            "speed",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "omega_p",
            "omega_q",
            "omega_r",
            "target_x",
            "target_y",
            "target_z",
            "distance_to_target",
            "act_m1",
            "act_m2",
            "act_m3",
            "act_m4",
            "thrust_m1",
            "thrust_m2",
            "thrust_m3",
            "thrust_m4",
            "fault_active",
            "min_obstacle_distance",
            "is_collision",
            "is_success",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in self.telemetry:
                writer.writerow(
                    {
                        "step": t.step,
                        "time_sec": round(t.time_sec, 4),
                        "pos_x": round(float(t.position[0]), 4),
                        "pos_y": round(float(t.position[1]), 4),
                        "pos_z": round(float(t.position[2]), 4),
                        "vel_x": round(float(t.velocity[0]), 4),
                        "vel_y": round(float(t.velocity[1]), 4),
                        "vel_z": round(float(t.velocity[2]), 4),
                        "speed": round(float(np.linalg.norm(t.velocity)), 4),
                        "roll_deg": round(float(t.euler_angles_deg[0]), 2),
                        "pitch_deg": round(float(t.euler_angles_deg[1]), 2),
                        "yaw_deg": round(float(t.euler_angles_deg[2]), 2),
                        "omega_p": round(float(t.angular_velocity[0]), 4),
                        "omega_q": round(float(t.angular_velocity[1]), 4),
                        "omega_r": round(float(t.angular_velocity[2]), 4),
                        "target_x": round(float(t.target_position[0]), 4),
                        "target_y": round(float(t.target_position[1]), 4),
                        "target_z": round(float(t.target_position[2]), 4),
                        "distance_to_target": round(float(t.distance_to_target), 4),
                        "act_m1": round(float(t.action[0]), 4),
                        "act_m2": round(float(t.action[1]), 4),
                        "act_m3": round(float(t.action[2]), 4),
                        "act_m4": round(float(t.action[3]), 4),
                        "thrust_m1": round(float(t.motor_thrusts[0]), 4),
                        "thrust_m2": round(float(t.motor_thrusts[1]), 4),
                        "thrust_m3": round(float(t.motor_thrusts[2]), 4),
                        "thrust_m4": round(float(t.motor_thrusts[3]), 4),
                        "fault_active": int(t.actuator_fault_active),
                        "min_obstacle_distance": (
                            round(float(t.min_obstacle_distance), 4)
                            if math.isfinite(t.min_obstacle_distance)
                            else -1.0
                        ),
                        "is_collision": int(t.is_collision),
                        "is_success": int(t.is_success),
                    }
                )
        return path


def _safe_provenance_value(val: Any, _visited: Optional[set[int]] = None) -> Tuple[Any, bool]:
    """Deterministically serialize a value for provenance hashing without leaking memory addresses.

    Returns (serialized_value, is_complete).
    """
    if _visited is None:
        _visited = set()

    if val is None or isinstance(val, bool):
        return val, True
    if isinstance(val, (int, np.integer)):
        return int(val), True
    if isinstance(val, (float, np.floating)):
        f_val = float(val)
        if not math.isfinite(f_val):
            return str(f_val), True
        return 0.0 if f_val == 0.0 else f_val, True
    if isinstance(val, str):
        # Sanitize any hex memory addresses (e.g., '0x7f...', '<... at 0x...>')
        if re.search(r"\bat 0x[0-9a-fA-F]+\b", val) or re.search(r"\b0x[0-9a-fA-F]{6,16}\b", val):
            cleaned = re.sub(r"\bat 0x[0-9a-fA-F]+\b", "", val)
            cleaned = re.sub(r"\b0x[0-9a-fA-F]{6,16}\b", "<addr>", cleaned).strip()
            return cleaned, False
        return val, True
    if isinstance(val, np.ndarray):
        items, ok = _safe_provenance_value(val.tolist(), _visited=_visited)
        return items, ok
    if isinstance(val, (list, tuple)):
        list_results = [_safe_provenance_value(x, _visited=_visited) for x in val]
        return [r[0] for r in list_results], all(r[1] for r in list_results)
    if isinstance(val, dict):
        dict_results = {
            str(k): _safe_provenance_value(v, _visited=_visited)
            for k, v in sorted(val.items(), key=lambda item: str(item[0]))
        }
        return {k: r[0] for k, r in dict_results.items()}, all(r[1] for r in dict_results.values())
    if hasattr(val, "to_dict") and callable(getattr(val, "to_dict")):
        try:
            return _safe_provenance_value(val.to_dict(), _visited=_visited)
        except Exception:
            return f"unserializable:{type(val).__name__}", False
    if hasattr(val, "metadata"):
        meta = getattr(val, "metadata")
        if meta is not None:
            return _safe_provenance_value(meta, _visited=_visited)
    if isinstance(val, (staticmethod, classmethod)):
        fn_val = getattr(val, "__func__", None)
        fn_prov, ok = _safe_provenance_value(fn_val, _visited=_visited)
        return {"descriptor": type(val).__name__, "func": fn_prov}, ok

    if isinstance(val, property):
        prop_info: Dict[str, Any] = {"descriptor": "property"}
        all_ok = True
        for part_name, fn_part in [("fget", val.fget), ("fset", val.fset), ("fdel", val.fdel)]:
            if fn_part is not None:
                p_prov, p_ok = _safe_provenance_value(fn_part, _visited=_visited)
                prop_info[part_name] = p_prov
                if not p_ok:
                    all_ok = False
        return prop_info, all_ok

    if inspect.ismodule(val):
        return f"module:{val.__name__}", True

    # Referenced helper function or method
    if inspect.isfunction(val) or inspect.ismethod(val):
        if id(val) in _visited:
            qual = getattr(val, "__qualname__", getattr(val, "__name__", str(val)))
            return {"recursion_ref": qual}, True
        fn_prov = extract_callable_provenance(val, _visited=set(_visited))
        return fn_prov, fn_prov.get("provenance_complete", True)

    # Class dependency
    if inspect.isclass(val):
        if id(val) in _visited:
            qual = getattr(val, "__qualname__", getattr(val, "__name__", str(val)))
            return {"recursion_ref": qual}, True
        _visited.add(id(val))
        mod = getattr(val, "__module__", "")
        qual = getattr(val, "__qualname__", getattr(val, "__name__", str(val)))
        if mod == "builtins" or mod.startswith("numpy") or mod.startswith("scipy"):
            return f"class:{mod}.{qual}", True

        # Custom class: inspect MRO for base classes and inherited methods
        mro_bases = [
            f"{b.__module__}.{b.__qualname__}"
            for b in inspect.getmro(val)
            if b not in (object,) and getattr(b, "__module__", "") != "builtins"
        ]
        cls_attrs: Dict[str, Any] = {}
        cls_complete = True

        for base in reversed(inspect.getmro(val)):
            if base is object or getattr(base, "__module__", "") == "builtins":
                continue
            for k, v in sorted(base.__dict__.items(), key=lambda item: str(item[0])):
                if not k.startswith("__") or k in ("__call__", "__init__"):
                    ser_v, ok_v = _safe_provenance_value(v, _visited=_visited)
                    cls_attrs[k] = ser_v
                    if not ok_v:
                        cls_complete = False

        return {
            "class": f"{mod}.{qual}",
            "mro": mro_bases,
            "attributes": cls_attrs,
        }, cls_complete

    # Callable class instance
    if callable(val) and hasattr(val, "__call__"):
        if id(val) in _visited:
            return {"recursion_ref": type(val).__qualname__}, True
        inst_prov = extract_callable_provenance(val, _visited=set(_visited))
        return inst_prov, inst_prov.get("provenance_complete", True)

    if hasattr(val, "__dict__"):
        if id(val) in _visited:
            return {"recursion_ref": type(val).__qualname__}, True
        _visited.add(id(val))
        try:
            custom_attrs = {k: v for k, v in val.__dict__.items() if not k.startswith("__")}
            if custom_attrs:
                return _safe_provenance_value(custom_attrs, _visited=_visited)
        except Exception:
            pass
        return f"object:{type(val).__qualname__}", False

    s = str(val)
    if re.search(r"\bat 0x[0-9a-fA-F]+\b", s) or re.search(r"\b0x[0-9a-fA-F]{6,16}\b", s):
        cleaned = re.sub(r"\bat 0x[0-9a-fA-F]+\b", "", s)
        cleaned = re.sub(r"\b0x[0-9a-fA-F]{6,16}\b", "<addr>", cleaned).strip()
        return {
            "unsupported_dependency": type(val).__name__,
            "reason": "non_deterministic_address_or_unsupported_object",
            "repr": cleaned,
        }, False
    return {
        "unsupported_dependency": type(val).__name__,
        "reason": "unserializable_opaque_value",
        "repr": s,
    }, False


def _hash_code_recursive(code_obj: Any, hasher: Any) -> None:
    """Recursively hash code object bytecode, referenced names, and nested code constants."""
    hasher.update(getattr(code_obj, "co_code", b""))
    for name in getattr(code_obj, "co_names", ()):
        hasher.update(name.encode("utf-8", errors="replace"))
    for vname in getattr(code_obj, "co_varnames", ()):
        hasher.update(vname.encode("utf-8", errors="replace"))
    for const in getattr(code_obj, "co_consts", ()):
        if inspect.iscode(const) or hasattr(const, "co_code"):
            _hash_code_recursive(const, hasher)
        else:
            hasher.update(str(const).encode("utf-8", errors="replace"))


def _resolve_module_dependencies(
    mod: Any,
    co_names: Sequence[str],
    visited_modules: set[int],
    _visited: set[int],
) -> Tuple[Dict[str, Any], bool]:
    """Recursively resolve module and submodule attributes referenced in co_names."""
    if id(mod) in visited_modules:
        return {"recursion_ref": getattr(mod, "__name__", str(mod))}, True
    visited_modules.add(id(mod))

    builtin_names = set(dir(builtins))
    attrs: Dict[str, Any] = {}
    all_ok = True

    for attr_name in sorted(set(co_names)):
        if hasattr(mod, attr_name) and attr_name not in builtin_names:
            try:
                attr_obj = getattr(mod, attr_name)
                if inspect.ismodule(attr_obj):
                    sub_attrs, sub_ok = _resolve_module_dependencies(
                        attr_obj, co_names, set(visited_modules), _visited
                    )
                    attrs[attr_name] = {
                        "module": getattr(attr_obj, "__name__", attr_name),
                        "attributes": sub_attrs,
                    }
                    if not sub_ok:
                        all_ok = False
                else:
                    ser_attr, ok_attr = _safe_provenance_value(attr_obj, _visited=_visited)
                    attrs[attr_name] = ser_attr
                    if not ok_attr:
                        all_ok = False
            except Exception as exc:
                attrs[attr_name] = {
                    "unresolved_attribute": attr_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                all_ok = False

    return attrs, all_ok


def extract_callable_provenance(fn: Any, _visited: Optional[set[int]] = None) -> Dict[str, Any]:
    """Extract a stable, deterministic provenance record for a callable.

    Captures function identity, module, qualified name, function defaults, keyword defaults,
    referenced global dependencies (recursively), closure variables, custom attributes,
    explicit metadata, and a SHA-256 digest of code object bytecode, constants, and referenced names.
    Guarantees different callback configurations generate different digests.
    """
    if fn is None:
        return {}

    if _visited is None:
        _visited = set()

    if id(fn) in _visited:
        return {
            "recursion_ref": getattr(
                fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__)
            ),
            "provenance_complete": True,
        }
    _visited.add(id(fn))

    provenance_complete = True

    # Unwrap functools.partial
    partial_args: List[Any] = []
    partial_keywords: Dict[str, Any] = {}
    if isinstance(fn, functools.partial):
        partial_args, ok_args = _safe_provenance_value(fn.args, _visited=set(_visited))
        partial_keywords, ok_kw = _safe_provenance_value(fn.keywords or {}, _visited=set(_visited))
        if not (ok_args and ok_kw):
            provenance_complete = False
        fn = fn.func

    # Determine underlying code object and callable type
    is_callable_instance = False
    if inspect.isfunction(fn) or inspect.ismethod(fn):
        code = getattr(fn, "__code__", None)
        fn_globals = getattr(fn, "__globals__", {})
        target_fn = fn
    elif hasattr(fn, "__call__"):
        is_callable_instance = True
        call_method = fn.__call__
        code = getattr(call_method, "__code__", None)
        fn_globals = getattr(call_method, "__globals__", {})
        target_fn = call_method
    else:
        code = getattr(fn, "__code__", None)
        fn_globals = getattr(fn, "__globals__", {})
        target_fn = fn

    info: Dict[str, Any] = {
        "name": getattr(fn, "__name__", type(fn).__name__),
        "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", str(type(fn)))),
        "callable_type": type(fn).__name__,
    }
    if hasattr(fn, "__module__") and fn.__module__:
        info["module"] = str(fn.__module__)

    if is_callable_instance:
        cls = type(fn)
        info["callable_class"] = f"{cls.__module__}.{cls.__qualname__}"
        info["class_mro"] = [
            f"{b.__module__}.{b.__qualname__}"
            for b in inspect.getmro(cls)
            if b not in (object,) and getattr(b, "__module__", "") != "builtins"
        ]
        # Inspect inherited methods across MRO
        inherited_methods: Dict[str, Any] = {}
        for base in reversed(inspect.getmro(cls)):
            if base is object or getattr(base, "__module__", "") == "builtins":
                continue
            for k, v in base.__dict__.items():
                if not k.startswith("__") or k in ("__call__", "__init__"):
                    if (
                        inspect.isfunction(v)
                        or inspect.ismethod(v)
                        or isinstance(v, (staticmethod, classmethod, property))
                    ):
                        m_prov, m_ok = _safe_provenance_value(v, _visited=set(_visited))
                        inherited_methods[k] = m_prov
                        if not m_ok:
                            provenance_complete = False
        if inherited_methods:
            info["class_methods"] = inherited_methods

    if partial_args:
        info["partial_args"] = partial_args
    if partial_keywords:
        info["partial_keywords"] = partial_keywords

    # Function positional defaults and keyword-only defaults
    defaults = getattr(target_fn, "__defaults__", None)
    if defaults is not None:
        ser_defaults, ok_def = _safe_provenance_value(defaults, _visited=set(_visited))
        info["defaults"] = ser_defaults
        if not ok_def:
            provenance_complete = False

    kwdefaults = getattr(target_fn, "__kwdefaults__", None)
    if kwdefaults is not None:
        ser_kwdefaults, ok_kwdef = _safe_provenance_value(kwdefaults, _visited=set(_visited))
        info["kwdefaults"] = ser_kwdefaults
        if not ok_kwdef:
            provenance_complete = False

    # Explicit metadata contract
    if hasattr(fn, "metadata"):
        meta = getattr(fn, "metadata")
        ser_meta, ok_meta = _safe_provenance_value(meta, _visited=set(_visited))
        info["metadata"] = ser_meta
        if isinstance(meta, dict) and meta.get("provenance_complete") is False:
            provenance_complete = False
        elif not ok_meta:
            provenance_complete = False
    elif getattr(fn, "__doc__", None):
        info["doc"] = fn.__doc__.strip()

    # Custom attributes on the callable instance or function
    custom_obj = fn if is_callable_instance else target_fn
    if hasattr(custom_obj, "__dict__"):
        custom_attrs = {
            k: v
            for k, v in custom_obj.__dict__.items()
            if not k.startswith("__") and k != "metadata"
        }
        if custom_attrs:
            ser_attrs, ok_attrs = _safe_provenance_value(custom_attrs, _visited=set(_visited))
            info["attributes"] = ser_attrs
            if not ok_attrs:
                provenance_complete = False

    # Closure variables
    closure_vars: Dict[str, Any] = {}
    if hasattr(target_fn, "__closure__") and target_fn.__closure__ is not None and code is not None:
        co_freevars = getattr(code, "co_freevars", ())
        for var_name, cell in zip(co_freevars, target_fn.__closure__):
            try:
                contents = cell.cell_contents
                ser_val, ok_val = _safe_provenance_value(contents, _visited=set(_visited))
                closure_vars[var_name] = ser_val
                if not ok_val:
                    provenance_complete = False
            except ValueError:
                closure_vars[var_name] = "<empty_cell>"
    if closure_vars:
        info["closure"] = {k: closure_vars[k] for k in sorted(closure_vars.keys())}

    # Global dependencies referenced in code.co_names (recursively fingerprinted)
    if code is not None and fn_globals:
        co_names = getattr(code, "co_names", ())
        builtin_names = set(dir(builtins))
        ref_globals: Dict[str, Any] = {}
        for name in sorted(set(co_names)):
            if name in fn_globals and name not in builtin_names:
                g_val = fn_globals[name]
                if inspect.ismodule(g_val):
                    # Recursively resolve module-qualified dependencies referenced in co_names
                    mod_attrs, mod_ok = _resolve_module_dependencies(
                        g_val, co_names, set(), set(_visited)
                    )
                    if mod_attrs:
                        ref_globals[f"{name}.attributes"] = mod_attrs
                    ref_globals[name] = f"module:{g_val.__name__}"
                    if not mod_ok:
                        provenance_complete = False
                else:
                    ser_g, ok_g = _safe_provenance_value(g_val, _visited=set(_visited))
                    ref_globals[name] = ser_g
                    if not ok_g:
                        provenance_complete = False
        if ref_globals:
            info["referenced_globals"] = ref_globals

    # Code object metadata and digest
    if code is not None:
        info["code_metadata"] = {
            "co_argcount": code.co_argcount,
            "co_kwonlyargcount": getattr(code, "co_kwonlyargcount", 0),
            "co_flags": code.co_flags,
            "co_names": list(code.co_names),
            "co_varnames": list(code.co_varnames),
        }

        hasher = hashlib.sha256()
        _hash_code_recursive(code, hasher)
        info["code_hash"] = hasher.hexdigest()

    info["provenance_complete"] = provenance_complete
    return info


@dataclass
class FlightScenarioDefinition:
    """Specification of an autonomous flight validation scenario."""

    name: str
    description: str
    initial_position: np.ndarray
    initial_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    initial_quaternion: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    initial_angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    waypoints: List[np.ndarray] = field(default_factory=list)
    obstacles: List[ObstacleSphere3D] = field(default_factory=list)
    dynamic_obstacles: List[DynamicObstacleSphere3D] = field(default_factory=list)
    bounds: Tuple[float, float, float] = (30.0, 30.0, 15.0)
    max_steps: int = 300
    target_radius: float = 1.2
    collision_radius: float = 0.8
    step_duration_sec: float = 0.05
    use_path_planner: bool = False
    use_time_aware_planner: bool = False
    settling_velocity_threshold: float = 0.5
    max_recovery_tilt_deg: float = 35.0
    min_recovery_altitude: float = 0.8
    enable_sensor_estimation: bool = False
    sensor_seed: Optional[int] = None
    sensor_in_the_loop: bool = False
    max_duration_seconds: Optional[float] = None
    disturbance_fn: Optional[Callable[[int, DroneState6DOF], np.ndarray]] = None
    actuator_fault_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None

    @property
    def provenance_complete(self) -> bool:
        """Indicates whether all behavior-affecting components have complete provenance."""
        if self.actuator_fault_fn is not None:
            fault_info = extract_callable_provenance(self.actuator_fault_fn)
            if fault_info.get("provenance_complete") is False:
                return False
        if self.disturbance_fn is not None:
            dist_info = extract_callable_provenance(self.disturbance_fn)
            if dist_info.get("provenance_complete") is False:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic, fully serializable dictionary representation for cryptographic hashing."""
        obs_dicts = [
            {"center": [float(x) for x in obs.center], "radius": float(obs.radius)}
            for obs in sorted(
                self.obstacles,
                key=lambda o: (
                    float(o.center[0]),
                    float(o.center[1]),
                    float(o.center[2]),
                    float(o.radius),
                ),
            )
        ]
        dyn_obs_dicts = [
            d.to_dict()
            if hasattr(d, "to_dict")
            else {
                "center_initial": [float(x) for x in d.center_initial],
                "radius": float(d.radius),
                "velocity": [float(x) for x in d.velocity],
                "bounds": [float(x) for x in d.bounds],
                "motion_type": str(d.motion_type),
                "oscillation_period": float(d.oscillation_period),
                "oscillation_amplitude": [float(x) for x in d.oscillation_amplitude],
            }
            for d in self.dynamic_obstacles
        ]

        fault_info: Dict[str, Any] = {}
        if self.actuator_fault_fn is not None:
            fault_info = extract_callable_provenance(self.actuator_fault_fn)

        dist_info: Dict[str, Any] = {}
        if self.disturbance_fn is not None:
            dist_info = extract_callable_provenance(self.disturbance_fn)

        return {
            "name": self.name,
            "description": self.description,
            "initial_position": [float(x) for x in self.initial_position],
            "initial_velocity": [float(x) for x in self.initial_velocity],
            "initial_quaternion": [float(x) for x in self.initial_quaternion],
            "initial_angular_velocity": [float(x) for x in self.initial_angular_velocity],
            "waypoints": [[float(x) for x in w] for w in self.waypoints],
            "obstacles": obs_dicts,
            "dynamic_obstacles": dyn_obs_dicts,
            "bounds": [float(x) for x in self.bounds],
            "max_steps": self.max_steps,
            "target_radius": float(self.target_radius),
            "collision_radius": float(self.collision_radius),
            "step_duration_sec": float(self.step_duration_sec),
            "use_path_planner": self.use_path_planner,
            "use_time_aware_planner": self.use_time_aware_planner,
            "settling_velocity_threshold": float(self.settling_velocity_threshold),
            "max_recovery_tilt_deg": float(self.max_recovery_tilt_deg),
            "min_recovery_altitude": float(self.min_recovery_altitude),
            "enable_sensor_estimation": self.enable_sensor_estimation,
            "sensor_seed": self.sensor_seed,
            "sensor_in_the_loop": self.sensor_in_the_loop,
            "max_duration_seconds": self.max_duration_seconds,
            "actuator_fault": fault_info,
            "disturbance": dist_info,
            "provenance_complete": self.provenance_complete,
        }


def get_standard_flight_scenarios() -> List[FlightScenarioDefinition]:
    """Instantiate the 10 standard autonomous flight scenarios."""
    scenarios: List[FlightScenarioDefinition] = []

    # 1. Stable Hovering
    scenarios.append(
        FlightScenarioDefinition(
            name="hover_equilibrium",
            description="Hold steady hover position at [15.0, 15.0, 7.5] for 200 simulation steps.",
            initial_position=np.array([15.0, 15.0, 7.5]),
            waypoints=[np.array([15.0, 15.0, 7.5])],
            obstacles=[],
            max_steps=200,
            target_radius=0.5,
        )
    )

    # 2. Straight-Line Waypoint Navigation
    scenarios.append(
        FlightScenarioDefinition(
            name="straight_line_navigation",
            description="Navigate straight line from [5.0, 5.0, 3.0] to diagonal waypoint [25.0, 25.0, 12.0].",
            initial_position=np.array([5.0, 5.0, 3.0]),
            waypoints=[np.array([25.0, 25.0, 12.0])],
            obstacles=[],
            max_steps=350,
            target_radius=1.2,
        )
    )

    # 3. Vertical Ascent and Descent
    scenarios.append(
        FlightScenarioDefinition(
            name="vertical_ascent_descent",
            description="Ascend vertically from z=2.0m to z=12.0m, then descend to z=5.0m.",
            initial_position=np.array([15.0, 15.0, 2.0]),
            waypoints=[np.array([15.0, 15.0, 12.0]), np.array([15.0, 15.0, 5.0])],
            obstacles=[],
            max_steps=350,
            target_radius=1.0,
        )
    )

    # 4. Horizontal Movement
    scenarios.append(
        FlightScenarioDefinition(
            name="horizontal_translation",
            description="Execute horizontal square pattern at constant altitude z=6.0m.",
            initial_position=np.array([8.0, 8.0, 6.0]),
            waypoints=[
                np.array([22.0, 8.0, 6.0]),
                np.array([22.0, 22.0, 6.0]),
                np.array([8.0, 22.0, 6.0]),
                np.array([8.0, 8.0, 6.0]),
            ],
            obstacles=[],
            max_steps=500,
            target_radius=1.2,
        )
    )

    # 5. Position Holding Under Wind-Like Force
    def wind_disturbance(step: int, state: DroneState6DOF) -> np.ndarray:
        if 50 <= step <= 150:
            return np.array([1.2, 0.8, 0.0])  # +1.2 N in X, +0.8 N in Y
        return np.zeros(3)

    scenarios.append(
        FlightScenarioDefinition(
            name="position_holding_disturbance",
            description="Hold position at [12.0, 12.0, 6.0] while subjected to lateral step disturbances.",
            initial_position=np.array([12.0, 12.0, 6.0]),
            waypoints=[np.array([12.0, 12.0, 6.0])],
            obstacles=[],
            max_steps=250,
            target_radius=1.0,
            disturbance_fn=wind_disturbance,
        )
    )

    # 6. Obstacle Avoidance (Using 3D A* Collision-Free Path Planning)
    obs1 = ObstacleSphere3D(center=np.array([15.0, 15.0, 7.0]), radius=2.2)
    obs2 = ObstacleSphere3D(center=np.array([10.0, 18.0, 6.0]), radius=1.8)
    scenarios.append(
        FlightScenarioDefinition(
            name="obstacle_avoidance",
            description="Navigate from [5.0, 5.0, 7.0] to [25.0, 25.0, 7.0] through spherical obstacle field using 3D A* collision-free planning.",
            initial_position=np.array([5.0, 5.0, 7.0]),
            waypoints=[np.array([25.0, 25.0, 7.0])],
            obstacles=[obs1, obs2],
            max_steps=400,
            target_radius=1.5,
            use_path_planner=True,
        )
    )

    # 7. Boundary Avoidance
    scenarios.append(
        FlightScenarioDefinition(
            name="boundary_avoidance",
            description="Fly toward arena boundary corner [28.0, 28.0, 13.5] and arrest motion before collision.",
            initial_position=np.array([15.0, 15.0, 8.0]),
            waypoints=[np.array([28.0, 28.0, 13.5])],
            bounds=(30.0, 30.0, 15.0),
            obstacles=[],
            max_steps=300,
            target_radius=1.5,
        )
    )

    # 8. Target Approach and Arrival (Precision Settling)
    scenarios.append(
        FlightScenarioDefinition(
            name="target_approach_arrival",
            description="Approach target waypoint [20.0, 15.0, 6.0] and achieve stable terminal hover (< 0.2 m/s).",
            initial_position=np.array([6.0, 15.0, 6.0]),
            waypoints=[np.array([20.0, 15.0, 6.0])],
            obstacles=[],
            max_steps=300,
            target_radius=0.8,
        )
    )

    # 9. Recovery from Moderate Disturbances
    q_init_pert = euler_to_quaternion(roll=math.radians(25.0), pitch=math.radians(-15.0), yaw=0.0)
    omega_init_pert = np.array([1.5, -1.2, 0.5])
    scenarios.append(
        FlightScenarioDefinition(
            name="attitude_disturbance_recovery",
            description="Recover to level hover from an initial 25-degree attitude tilt and angular rate.",
            initial_position=np.array([15.0, 15.0, 8.0]),
            initial_quaternion=q_init_pert,
            initial_angular_velocity=omega_init_pert,
            waypoints=[np.array([15.0, 15.0, 8.0])],
            obstacles=[],
            max_steps=250,
            target_radius=0.8,
        )
    )

    # 10. Failure Verification Scenario (Canonical 50% thrust loss on Motor 1)
    def actuator_canonical_fault(step: int, thrusts: np.ndarray) -> np.ndarray:
        if step >= 40:
            degraded = thrusts.copy()
            degraded[0] *= 0.5  # Canonical 50% thrust reduction on front-right motor
            return degraded
        return thrusts

    scenarios.append(
        FlightScenarioDefinition(
            name="actuator_fault_degradation",
            description="Evaluate flight control stability under canonical 50% partial thrust loss on Motor 1 at step 40.",
            initial_position=np.array([15.0, 15.0, 7.0]),
            waypoints=[np.array([15.0, 15.0, 7.0])],
            obstacles=[],
            max_steps=250,
            target_radius=1.5,
            actuator_fault_fn=actuator_canonical_fault,
        )
    )

    return scenarios


def get_advanced_navigation_scenarios() -> List[FlightScenarioDefinition]:
    """Instantiate the 8 advanced 3D autonomous navigation scenarios (Phase III)."""
    scenarios: List[FlightScenarioDefinition] = []

    # 11. Multi-Obstacle Field Navigation
    obs_field = [
        ObstacleSphere3D(center=np.array([10.0, 10.0, 6.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([15.0, 18.0, 7.0]), radius=2.2),
        ObstacleSphere3D(center=np.array([20.0, 12.0, 8.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([22.0, 22.0, 6.0]), radius=1.8),
    ]
    scenarios.append(
        FlightScenarioDefinition(
            name="multi_obstacle_navigation",
            description="Navigate through a dense 4-sphere obstacle field using 3D A* global path planning.",
            initial_position=np.array([4.0, 4.0, 5.0]),
            waypoints=[np.array([26.0, 26.0, 9.0])],
            obstacles=obs_field,
            max_steps=500,
            target_radius=1.5,
            use_path_planner=True,
        )
    )

    # 12. Narrow Passage Navigation
    # Two large obstacles forming a narrow corridor
    obs_gap_1 = ObstacleSphere3D(center=np.array([15.0, 12.0, 7.0]), radius=2.0)
    obs_gap_2 = ObstacleSphere3D(center=np.array([15.0, 18.0, 7.0]), radius=2.0)
    scenarios.append(
        FlightScenarioDefinition(
            name="narrow_passage_navigation",
            description="Navigate safely through a narrow 2.0m corridor between two large obstacles.",
            initial_position=np.array([5.0, 15.0, 7.0]),
            waypoints=[np.array([25.0, 15.0, 7.0])],
            obstacles=[obs_gap_1, obs_gap_2],
            max_steps=400,
            target_radius=1.2,
            use_path_planner=True,
        )
    )

    # 13. Dynamic Obstacle Crossing
    dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([15.0, 5.0, 7.0]),
        radius=1.6,
        velocity=np.array([0.0, 1.2, 0.0]),  # Translating across the flight path
        bounds=(30.0, 30.0, 15.0),
    )
    scenarios.append(
        FlightScenarioDefinition(
            name="dynamic_obstacle_crossing",
            description="Cross flight path with a moving obstacle translating deterministically across trajectory.",
            initial_position=np.array([5.0, 15.0, 7.0]),
            waypoints=[np.array([25.0, 15.0, 7.0])],
            obstacles=[],
            dynamic_obstacles=[dyn_obs],
            max_steps=400,
            target_radius=1.5,
            use_path_planner=True,
        )
    )

    # 14. Multiple Sequential 3D Waypoints (Slalom)
    scenarios.append(
        FlightScenarioDefinition(
            name="multiple_sequential_waypoints",
            description="Follow a 5-waypoint 3D spatial slalom course.",
            initial_position=np.array([5.0, 5.0, 3.0]),
            waypoints=[
                np.array([10.0, 20.0, 5.0]),
                np.array([15.0, 8.0, 7.0]),
                np.array([20.0, 22.0, 9.0]),
                np.array([25.0, 10.0, 11.0]),
                np.array([27.0, 25.0, 8.0]),
            ],
            obstacles=[],
            max_steps=600,
            target_radius=1.2,
        )
    )

    # 15. Blocked Route / Dead-End Handling
    # Impassable barrier completely surrounding goal
    barrier = [
        ObstacleSphere3D(center=np.array([25.0, 25.0, 7.0]), radius=3.0),
    ]
    scenarios.append(
        FlightScenarioDefinition(
            name="blocked_route_dead_end",
            description="Attempt navigation to an obstructed goal inside an obstacle, verifying clean NO_PATH handling.",
            initial_position=np.array([5.0, 5.0, 5.0]),
            waypoints=[np.array([25.0, 25.0, 7.0])],  # Goal inside obstacle
            obstacles=barrier,
            max_steps=100,
            target_radius=1.0,
            use_path_planner=True,
        )
    )

    # 16. Return-to-Home (RTH) Navigation
    scenarios.append(
        FlightScenarioDefinition(
            name="return_to_home_mission",
            description="Fly outbound to waypoint [22.0, 22.0, 10.0] and execute autonomous Return-to-Home.",
            initial_position=np.array([5.0, 5.0, 3.0]),
            waypoints=[np.array([22.0, 22.0, 10.0]), np.array([5.0, 5.0, 3.0])],
            obstacles=[],
            max_steps=500,
            target_radius=1.2,
        )
    )

    # 17. Turbulent Canyon Navigation
    canyon_obs = [
        ObstacleSphere3D(center=np.array([10.0, 8.0, 5.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([10.0, 12.0, 5.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([15.0, 8.0, 5.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([15.0, 12.0, 5.0]), radius=2.0),
    ]
    scenarios.append(
        FlightScenarioDefinition(
            name="turbulent_canyon_wind",
            description="Navigate narrow passage under constant 3.0 m/s lateral wind disturbance.",
            initial_position=np.array([5.0, 10.0, 5.0]),
            waypoints=[np.array([20.0, 10.0, 5.0])],
            obstacles=canyon_obs,
            disturbance_fn=lambda step, state: np.array([0.0, 3.0, 0.0]),
            max_steps=300,
            target_radius=1.0,
            use_path_planner=True,
        )
    )

    # 18. Degraded Actuator Loiter
    scenarios.append(
        FlightScenarioDefinition(
            name="degraded_actuator_loiter",
            description="Maintain hover loiter under 10% motor thrust loss (Motor 1).",
            initial_position=np.array([10.0, 10.0, 5.0]),
            waypoints=[np.array([10.0, 10.0, 5.0])],
            obstacles=[],
            actuator_fault_fn=lambda step, thrusts: np.array(
                [
                    thrusts[0] * 0.90 if step >= 20 else thrusts[0],
                    thrusts[1],
                    thrusts[2],
                    thrusts[3],
                ]
            ),
            max_steps=200,
            target_radius=1.0,
        )
    )

    # 19. State Estimation In-the-Loop Flight
    scenarios.append(
        FlightScenarioDefinition(
            name="sensor_estimated_navigation",
            description="Precision hover loiter driven entirely by sensor-fusion estimated state under IMU noise and drift.",
            initial_position=np.array([10.0, 10.0, 5.0]),
            waypoints=[np.array([10.0, 10.0, 5.0])],
            obstacles=[],
            max_steps=200,
            target_radius=1.0,
            enable_sensor_estimation=True,
            sensor_in_the_loop=True,
            sensor_seed=101,
        )
    )

    return scenarios


def get_phase4_predictive_scenarios() -> List[FlightScenarioDefinition]:
    """Phase IV: Predictive Autonomous Mission Planning and Intelligent Navigation Scenarios."""
    scenarios: List[FlightScenarioDefinition] = []

    # 20. Dynamic Obstacle Crossing with Time-Aware Planning
    crossing_obs = DynamicObstacleSphere3D(
        center_initial=np.array([10.0, 8.0, 5.0]),
        radius=1.2,
        velocity=np.array([0.0, 1.5, 0.0]),
        bounds=(30.0, 30.0, 15.0),
        motion_type="bounce",
    )
    scenarios.append(
        FlightScenarioDefinition(
            name="dynamic_obstacle_crossing_time_aware",
            description="Time-aware 4D planning anticipates a moving obstacle crossing perpendicular to flight path.",
            initial_position=np.array([3.0, 15.0, 5.0]),
            waypoints=[np.array([18.0, 15.0, 5.0])],
            obstacles=[],
            dynamic_obstacles=[crossing_obs],
            max_steps=400,
            target_radius=1.5,
            collision_radius=0.3,
            settling_velocity_threshold=1.5,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    # 21. Multi-Dynamic Obstacle Corridor
    multi_obs = [
        DynamicObstacleSphere3D(
            center_initial=np.array([9.0, 8.0, 5.0]),
            radius=1.2,
            velocity=np.array([0.0, 1.5, 0.0]),
            bounds=(30.0, 30.0, 15.0),
            motion_type="bounce",
        ),
        DynamicObstacleSphere3D(
            center_initial=np.array([15.0, 22.0, 5.0]),
            radius=1.2,
            velocity=np.array([0.0, -1.5, 0.0]),
            bounds=(30.0, 30.0, 15.0),
            motion_type="bounce",
        ),
    ]
    scenarios.append(
        FlightScenarioDefinition(
            name="multi_dynamic_obstacle_corridor",
            description="Navigate a corridor populated by multiple crossing dynamic obstacles.",
            initial_position=np.array([3.0, 15.0, 5.0]),
            waypoints=[np.array([20.0, 15.0, 5.0])],
            obstacles=[],
            dynamic_obstacles=multi_obs,
            max_steps=450,
            target_radius=1.5,
            collision_radius=0.3,
            settling_velocity_threshold=2.0,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    # 22. Temporary Corridor Blockage & Wait
    slot_static_obs = [
        ObstacleSphere3D(center=np.array([15.0, 10.0, 5.0]), radius=2.5),
        ObstacleSphere3D(center=np.array([15.0, 20.0, 5.0]), radius=2.5),
    ]
    slot_dyn_obs = DynamicObstacleSphere3D(
        center_initial=np.array([15.0, 12.0, 5.0]),
        radius=1.2,
        velocity=np.array([0.0, 2.0, 0.0]),
        bounds=(30.0, 30.0, 15.0),
        motion_type="bounce",
    )
    scenarios.append(
        FlightScenarioDefinition(
            name="temporary_corridor_blockage_wait",
            description="Drone waits for a temporary dynamic blockage to clear a narrow passage.",
            initial_position=np.array([5.0, 15.0, 5.0]),
            waypoints=[np.array([25.0, 15.0, 5.0])],
            obstacles=slot_static_obs,
            dynamic_obstacles=[slot_dyn_obs],
            max_steps=450,
            target_radius=1.5,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    # 23. Sequential Multi-Waypoint Adaptive Mission
    scenarios.append(
        FlightScenarioDefinition(
            name="sequential_multi_waypoint_adaptive",
            description="Execute 4-waypoint sequential mission adapting to dynamic moving obstacles.",
            initial_position=np.array([3.0, 8.0, 4.0]),
            waypoints=[
                np.array([8.0, 14.0, 5.0]),
                np.array([13.0, 8.0, 5.0]),
                np.array([18.0, 14.0, 6.0]),
                np.array([23.0, 8.0, 5.0]),
            ],
            obstacles=[],
            dynamic_obstacles=[
                DynamicObstacleSphere3D(
                    center_initial=np.array([15.0, 18.0, 5.5]),
                    radius=1.2,
                    velocity=np.array([0.0, -1.0, 0.0]),
                    bounds=(30.0, 30.0, 15.0),
                    motion_type="bounce",
                )
            ],
            max_steps=600,
            target_radius=1.5,
            collision_radius=0.3,
            settling_velocity_threshold=2.0,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    # 24a. Repeated Dynamic Replanning (Stress Evaluation)
    scenarios.append(
        FlightScenarioDefinition(
            name="repeated_dynamic_replanning_stress",
            description="Stress test: evaluate sustained collision-free dynamic replanning stability under continuous obstacle interference.",
            initial_position=np.array([3.0, 15.0, 5.0]),
            waypoints=[np.array([20.0, 15.0, 5.0])],
            obstacles=[],
            dynamic_obstacles=[
                DynamicObstacleSphere3D(
                    center_initial=np.array([10.0, 15.0, 5.0]),
                    radius=0.9,
                    velocity=np.zeros(3),
                    bounds=(30.0, 30.0, 15.0),
                    motion_type="oscillate",
                    oscillation_period=4.0,
                    oscillation_amplitude=np.array([0.0, 2.5, 0.0]),
                ),
                DynamicObstacleSphere3D(
                    center_initial=np.array([15.0, 15.0, 5.0]),
                    radius=0.9,
                    velocity=np.zeros(3),
                    bounds=(30.0, 30.0, 15.0),
                    motion_type="oscillate",
                    oscillation_period=4.5,
                    oscillation_amplitude=np.array([0.0, 0.0, 2.0]),
                ),
            ],
            max_steps=500,
            target_radius=1.5,
            collision_radius=0.3,
            settling_velocity_threshold=2.0,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    # 24b. Repeated Dynamic Replanning (Mission Completion)
    scenarios.append(
        FlightScenarioDefinition(
            name="repeated_dynamic_replanning_completed",
            description="Evaluate dynamic replanning under oscillating obstacles with verified arrival and settling at destination.",
            initial_position=np.array([3.0, 15.0, 5.0]),
            waypoints=[np.array([18.0, 15.0, 5.0])],
            obstacles=[],
            dynamic_obstacles=[
                DynamicObstacleSphere3D(
                    center_initial=np.array([10.0, 15.0, 5.0]),
                    radius=0.9,
                    velocity=np.zeros(3),
                    bounds=(30.0, 30.0, 15.0),
                    motion_type="oscillate",
                    oscillation_period=6.0,
                    oscillation_amplitude=np.array([0.0, 3.0, 0.0]),
                ),
            ],
            max_steps=400,
            target_radius=1.5,
            collision_radius=0.3,
            settling_velocity_threshold=2.0,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    # 25. Sensor-in-the-Loop Predictive Navigation
    scenarios.append(
        FlightScenarioDefinition(
            name="sensor_in_the_loop_navigation",
            description="Autonomous navigation driven by state-estimation avionics under IMU noise and drift.",
            initial_position=np.array([5.0, 15.0, 5.0]),
            waypoints=[np.array([20.0, 15.0, 5.0])],
            obstacles=[ObstacleSphere3D(center=np.array([12.0, 15.0, 5.0]), radius=1.5)],
            dynamic_obstacles=[],
            max_steps=400,
            target_radius=1.5,
            settling_velocity_threshold=1.5,
            enable_sensor_estimation=True,
            sensor_in_the_loop=True,
            sensor_seed=202,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    # 26. Unreachable Mission Safe Blocked
    scenarios.append(
        FlightScenarioDefinition(
            name="unreachable_mission_safe_blocked",
            description="Identify unfeasible route when goal is completely enclosed, commanding safe blocked hover.",
            initial_position=np.array([5.0, 15.0, 5.0]),
            waypoints=[np.array([25.0, 15.0, 5.0])],
            obstacles=[ObstacleSphere3D(center=np.array([25.0, 15.0, 5.0]), radius=3.0)],
            dynamic_obstacles=[],
            max_steps=100,
            target_radius=1.5,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    # 27. Prediction Divergence Recovery
    scenarios.append(
        FlightScenarioDefinition(
            name="prediction_divergence_recovery",
            description="Receding-horizon replanning recovers safely when moving obstacle trajectory changes.",
            initial_position=np.array([3.0, 15.0, 5.0]),
            waypoints=[np.array([18.0, 15.0, 5.0])],
            obstacles=[],
            dynamic_obstacles=[
                DivergentDynamicObstacleSphere3D(
                    center_initial=np.array([10.0, 8.0, 5.0]),
                    radius=1.0,
                    velocity=np.array([0.0, 0.4, 0.0]),
                    bounds=(30.0, 30.0, 15.0),
                    motion_type="bounce",
                    divergence_time=2.0,
                    divergence_velocity=np.array([0.0, 1.8, 0.0]),
                )
            ],
            max_steps=400,
            target_radius=1.5,
            collision_radius=0.3,
            settling_velocity_threshold=1.5,
            use_path_planner=True,
            use_time_aware_planner=True,
        )
    )

    return scenarios


def get_all_flight_scenarios() -> List[FlightScenarioDefinition]:
    """Return all standard, advanced, and Phase IV predictive flight validation scenarios."""
    return (
        get_standard_flight_scenarios()
        + get_advanced_navigation_scenarios()
        + get_phase4_predictive_scenarios()
    )


def get_scenario(name: str) -> FlightScenarioDefinition:
    """Retrieve a scenario definition by name."""
    clean = name.strip().lower()
    if clean == "repeated_dynamic_replanning":
        clean = "repeated_dynamic_replanning_stress"
    for sc in get_all_flight_scenarios():
        if sc.name.lower() == clean:
            return sc
    raise KeyError(f"Flight scenario '{name}' not found.")


class FlightScenarioRunner:
    """Executes flight scenarios against real 6-DOF simulation models and records telemetry."""

    def __init__(self, env: Optional[Drone6DOFEnv] = None) -> None:
        self.env = env or Drone6DOFEnv()

    def run_scenario(
        self,
        scenario: FlightScenarioDefinition,
        controller: Optional[Any] = None,
    ) -> FlightScenarioResult:
        """Execute a flight scenario and compute full empirical aerospace flight metrics.

        Args:
            scenario: FlightScenarioDefinition specifying geometry, bounds, and waypoints.
            controller: Controller instance (defaults to QuadrotorPIDController with scenario parameters).
        """
        start_wall_time = time.perf_counter()

        # Instantiate or reset controller
        ctrl = controller or QuadrotorPIDController(
            mass=self.env.dynamics.mass,
            gravity=self.env.dynamics.gravity,
            arm_length=self.env.dynamics.arm_length,
            inertia=(
                float(self.env.dynamics.inertia_diag[0]),
                float(self.env.dynamics.inertia_diag[1]),
                float(self.env.dynamics.inertia_diag[2]),
            ),
            max_thrust_per_motor=self.env.dynamics.max_thrust_per_motor,
            torque_to_thrust_ratio=self.env.dynamics.c_tau,
            dt=self.env.dynamics.dt,
        )
        if hasattr(ctrl, "reset"):
            ctrl.reset()

        # Configure environment state for scenario
        self.env.bounds = scenario.bounds
        self.env._obstacles = list(scenario.obstacles)
        self.env.target_radius = scenario.target_radius
        self.env.collision_radius = scenario.collision_radius
        self.env.max_steps = scenario.max_steps
        self.env.dynamics.reset(
            position=scenario.initial_position,
            velocity=scenario.initial_velocity,
            quaternion=scenario.initial_quaternion,
            angular_velocity=scenario.initial_angular_velocity,
        )
        self.env._position = scenario.initial_position.copy()
        self.env._current_step = 0

        # Resolve waypoints (optionally via 3D A* path planning or MissionManager)
        waypoints: List[np.ndarray] = []
        mission_manager: Optional[MissionManager] = None
        if scenario.use_path_planner and (scenario.obstacles or scenario.dynamic_obstacles):
            planner = AStarPlanner3D(
                bounds=scenario.bounds,
                resolution=1.0,
                vehicle_radius=scenario.collision_radius,
                safety_margin=0.3,
            )
            time_aware_planner = None
            if getattr(scenario, "use_time_aware_planner", False):
                from adaptive_rl.navigation.time_aware_planner import TimeAwareAStarPlanner3D

                time_aware_planner = TimeAwareAStarPlanner3D(
                    bounds=scenario.bounds,
                    spatial_resolution=1.0,
                    time_resolution=0.5,
                    vehicle_radius=scenario.collision_radius,
                    safety_margin=0.3,
                )

            raw_target = scenario.waypoints[-1] if scenario.waypoints else scenario.initial_position
            mission_manager = MissionManager(
                planner=planner,
                time_aware_planner=time_aware_planner,
                waypoint_acceptance_radius=scenario.target_radius,
                replan_clearance_threshold=scenario.collision_radius + 0.5,
                state_source="estimated" if scenario.sensor_in_the_loop else "ground_truth",
            )
            targets = scenario.waypoints if scenario.waypoints else [scenario.initial_position]
            ok = mission_manager.start_mission(
                start_position=scenario.initial_position,
                goals=targets,
                static_obstacles=scenario.obstacles,
                dynamic_obstacles=scenario.dynamic_obstacles,
                time_sec=0.0,
            )
            if not ok or mission_manager.state == MissionState.FAILED_BLOCKED:
                # Planning failure (e.g. goal or start blocked)
                elapsed_ms = (time.perf_counter() - start_wall_time) * 1000.0
                initial_goal_dist = float(np.linalg.norm(raw_target - scenario.initial_position))
                metrics = FlightScenarioMetrics(
                    scenario_name=scenario.name,
                    success=False,
                    outcome="blocked_no_path",
                    duration_steps=0,
                    duration_seconds=0.0,
                    final_position_error=initial_goal_dist,
                    mean_position_error=initial_goal_dist,
                    max_position_error=initial_goal_dist,
                    rms_position_error=initial_goal_dist,
                    mean_velocity_error=0.0,
                    max_velocity_error=0.0,
                    rms_velocity_error=0.0,
                    mean_attitude_error_deg=0.0,
                    max_attitude_error_deg=0.0,
                    mean_angular_speed=0.0,
                    max_angular_speed=0.0,
                    initial_goal_distance=initial_goal_dist,
                    final_goal_distance=initial_goal_dist,
                    min_obstacle_clearance=-1.0,
                    collision_count=0,
                    control_effort=0.0,
                    execution_wall_time_ms=elapsed_ms,
                )
                return FlightScenarioResult(
                    scenario_name=scenario.name,
                    description=scenario.description,
                    metrics=metrics,
                    telemetry=[],
                    scenario_definition=scenario,
                )
            waypoints = [w.copy() for w in mission_manager.active_path]
        else:
            waypoints = [w.copy() for w in scenario.waypoints]

        if not waypoints:
            waypoints = [scenario.initial_position.copy()]

        waypoint_idx = 0
        current_target = waypoints[waypoint_idx].copy()
        self.env._goal = current_target.copy()
        self.env._prev_distance_to_goal = float(np.linalg.norm(current_target - self.env._position))

        initial_goal_dist = float(np.linalg.norm(waypoints[-1] - scenario.initial_position))

        # Sensor estimation avionics in-the-loop if enabled
        imu_sensor = None
        state_estimator = None
        sensor_rng = None
        est_drone_state: Optional[DroneState6DOF] = None
        if scenario.enable_sensor_estimation or scenario.sensor_in_the_loop:
            from adaptive_rl.sensors.estimation import ComplementaryFilter
            from adaptive_rl.sensors.imu import IMUSensor

            sensor_rng = np.random.default_rng(scenario.sensor_seed or 42)
            imu_sensor = IMUSensor(
                gravity=self.env.dynamics.gravity, seed=scenario.sensor_seed or 42
            )
            state_estimator = ComplementaryFilter(
                gravity=self.env.dynamics.gravity,
                kp_accel=0.2,
                ki_accel=0.001,
                kp_pos=5.0,
                kp_vel=3.0,
            )
            state_estimator.reset(
                position=scenario.initial_position,
                velocity=scenario.initial_velocity,
                quaternion=scenario.initial_quaternion,
            )
            est_drone_state = DroneState6DOF(
                position=scenario.initial_position.copy(),
                velocity=scenario.initial_velocity.copy(),
                quaternion=scenario.initial_quaternion.copy(),
                angular_velocity=scenario.initial_angular_velocity.copy(),
            )

        telemetry: List[FlightStepTelemetry] = []
        pos_errors: List[float] = []
        vel_errors: List[float] = []
        att_errors_deg: List[float] = []
        angular_speeds: List[float] = []
        min_clearances: List[float] = []
        control_efforts: List[float] = []

        is_collision = False
        coll_type = "none"
        outcome = "max_steps"
        step = 0
        dt = self.env.dynamics.dt
        fault_ever_active = False
        all_waypoints_completed = False

        while step < scenario.max_steps:
            step += 1
            time_sec = step * dt
            current_state = self.env.dynamics.state

            # Determine feedback state for closed-loop control (sensor-in-the-loop if active)
            if scenario.sensor_in_the_loop:
                if (
                    est_drone_state is None
                    or not np.all(np.isfinite(est_drone_state.position))
                    or not np.all(np.isfinite(est_drone_state.velocity))
                    or not np.all(np.isfinite(est_drone_state.quaternion))
                    or not np.all(np.isfinite(est_drone_state.angular_velocity))
                    or abs(float(np.linalg.norm(est_drone_state.quaternion)) - 1.0) >= 0.05
                ):
                    # Fail closed immediately: never silently substitute ground truth
                    if mission_manager is not None:
                        mission_manager.state = MissionState.ABORTED
                        mission_manager.replan_reason = "sensor_estimation_invalid_fail_closed"
                    outcome = "sensor_estimation_failure"
                    break
                feedback_state: DroneState6DOF = est_drone_state
            else:
                feedback_state = current_state

            # Predictive dynamic obstacle avoidance via MissionManager
            mission_telemetry = None
            if mission_manager is not None:
                current_target, mission_telemetry = mission_manager.update(
                    drone_state=feedback_state,
                    time_sec=time_sec,
                    static_obstacles=scenario.obstacles,
                    dynamic_obstacles=scenario.dynamic_obstacles,
                )
                self.env._goal = current_target.copy()
                if hasattr(ctrl, "set_target"):
                    ctrl.set_target(current_target)

            # Compute tracking errors relative to current waypoint
            pos_err = current_target - current_state.position
            pos_err_mag = float(np.linalg.norm(pos_err))
            pos_errors.append(pos_err_mag)

            vel_err_mag = float(np.linalg.norm(current_state.velocity))
            vel_errors.append(vel_err_mag)

            roll, pitch, yaw = quaternion_to_euler_angles(current_state.quaternion)
            euler_deg = np.array(
                [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)], dtype=np.float64
            )
            r_mat = self.env.dynamics.body_to_world(np.array([0.0, 0.0, 1.0]))
            tilt_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(r_mat[2])))))
            att_errors_deg.append(tilt_angle_deg)

            omega_mag = float(np.linalg.norm(current_state.angular_velocity))
            angular_speeds.append(omega_mag)

            # Compute controller action using feedback_state (sensor-in-the-loop if active)
            target_vel = (
                mission_manager.current_target_velocity
                if (
                    mission_manager is not None
                    and getattr(scenario, "use_time_aware_planner", False)
                )
                else np.zeros(3, dtype=np.float64)
            )
            if hasattr(ctrl, "compute_action"):
                action, ctrl_info = ctrl.compute_action(
                    feedback_state,
                    target_position=current_target,
                    target_velocity=target_vel,
                    dt=dt,
                )
            elif hasattr(ctrl, "predict"):
                obs = self.env._get_obs(custom_state=feedback_state)
                action, _ = ctrl.predict(obs, deterministic=True)
            else:
                action = np.zeros(4, dtype=np.float32)

            act_arr = np.asarray(action, dtype=np.float32)
            control_efforts.append(float(np.sum(np.square(act_arr))))

            # Map action to physical thrusts
            thrusts_cmd = self.env._action_to_motor_thrusts(act_arr)
            thrusts_actual = thrusts_cmd.copy()

            # Apply actuator fault if specified
            fault_active = False
            if scenario.actuator_fault_fn is not None:
                thrusts_actual = scenario.actuator_fault_fn(step, thrusts_cmd)
                fault_active = not np.allclose(thrusts_actual, thrusts_cmd)
            if fault_active:
                fault_ever_active = True

            # Snapshot pre-integration physical state to prevent object aliasing
            pre_state = self.env.dynamics.state.copy()
            pre_velocity = pre_state.velocity.copy()
            old_pos = pre_state.position.copy()

            # External aerodynamic disturbance applied to velocity
            if scenario.disturbance_fn is not None:
                dist_force = scenario.disturbance_fn(step, pre_state)
                self.env.dynamics.state.velocity += (dist_force / self.env.dynamics.mass) * dt

            # Step physical dynamics with Runge-Kutta 4
            new_state = self.env.dynamics.step_rk4(thrusts_actual)
            self.env._position = new_state.position.copy()
            self.env._current_step = step

            # Sensor in-the-loop estimation if enabled
            est_pos = None
            est_vel = None
            pos_est_err = None
            att_est_err = None
            if imu_sensor is not None and state_estimator is not None and sensor_rng is not None:
                # Calculate true net inertial acceleration between distinct pre and post states
                inertial_acc = (new_state.velocity - pre_velocity) / dt
                meas = imu_sensor.measure(
                    new_state,
                    inertial_acceleration=inertial_acc,
                    dt=dt,
                    timestamp=time_sec,
                )
                # Avionics sensor fusion: IMU fused with noisy position and altitude observations
                pos_obs_noise = sensor_rng.normal(0.0, 0.02, size=3)
                noisy_pos_ref = new_state.position + pos_obs_noise
                est_state = state_estimator.update(
                    meas,
                    dt=dt,
                    position_ref=noisy_pos_ref,
                    altitude_ref=float(noisy_pos_ref[2]),
                    true_state=new_state,
                )
                est_pos = est_state.position.copy()
                est_vel = est_state.velocity.copy()
                pos_est_err = est_state.position_error
                att_est_err = est_state.attitude_error_deg
                est_drone_state = DroneState6DOF(
                    position=est_state.position.copy(),
                    velocity=est_state.velocity.copy(),
                    quaternion=est_state.quaternion.copy(),
                    angular_velocity=est_state.angular_velocity.copy(),
                )

            # Obstacle clearance check (combining static and dynamic obstacles)
            min_obs_d = float("inf")
            for obstacle in scenario.obstacles:
                d = obstacle.distance_to(new_state.position) - scenario.collision_radius
                if d < min_obs_d:
                    min_obs_d = d
            for dyn_obs in scenario.dynamic_obstacles:
                d = dyn_obs.distance_to(new_state.position, time_sec) - scenario.collision_radius
                if d < min_obs_d:
                    min_obs_d = d
            min_clearances.append(min_obs_d)

            # Collision check with analytical curvature error bound
            if hasattr(self.env.dynamics, "max_translational_acceleration"):
                a_max = self.env.dynamics.max_translational_acceleration(
                    max_speed_bound=getattr(scenario, "max_speed", 25.0)
                )
            else:
                raise ValueError(
                    "Safety certification failure: environment dynamics does not provide "
                    "a physically defensible max_translational_acceleration() calculation."
                )
            substep_h = getattr(self.env.dynamics, "substep_dt", dt)
            eps_curv = 0.125 * a_max * (substep_h**2)
            intermediate_positions = getattr(self.env.dynamics, "last_substep_positions", None)

            coll, coll_type = self.env._check_collision(
                new_state.position,
                prev_pos=old_pos,
                intermediate_positions=intermediate_positions,
                curvature_error_bound=eps_curv,
            )
            # Continuous swept dynamic obstacle collision check
            if not coll:
                for dyn_obs in scenario.dynamic_obstacles:
                    if dyn_obs.check_continuous_collision(
                        prev_drone_pos=old_pos,
                        curr_drone_pos=new_state.position,
                        prev_time=time_sec - dt,
                        curr_time=time_sec,
                        vehicle_radius=scenario.collision_radius,
                        intermediate_drone_positions=intermediate_positions,
                        veh_acc_max=a_max,
                    ):
                        coll = True
                        coll_type = "obstacle"
                        break

            if coll:
                is_collision = True
                outcome = "collision" if coll_type == "obstacle" else "boundary_breach"

            # Waypoint sequencing for static waypoint list
            dist_to_curr_waypoint = float(np.linalg.norm(current_target - new_state.position))
            is_at_curr_waypoint = dist_to_curr_waypoint <= scenario.target_radius

            if (
                mission_manager is None
                and is_at_curr_waypoint
                and waypoint_idx < len(waypoints) - 1
            ):
                waypoint_idx += 1
                current_target = waypoints[waypoint_idx].copy()
                self.env._goal = current_target.copy()
                if hasattr(ctrl, "set_target"):
                    ctrl.set_target(current_target)

            # Final arrival and settling check
            final_coord = (
                scenario.waypoints[-1] if scenario.waypoints else scenario.initial_position
            )
            dist_to_final = float(np.linalg.norm(final_coord - new_state.position))
            curr_speed = float(np.linalg.norm(new_state.velocity))
            is_pos_settled = dist_to_final <= scenario.target_radius
            is_vel_settled = curr_speed <= scenario.settling_velocity_threshold

            all_waypoints_completed = (
                (
                    mission_manager.state == MissionState.ARRIVED
                    and mission_manager.current_mission_waypoint_idx
                    >= len(mission_manager.mission_waypoints)
                )
                if mission_manager is not None
                else (waypoint_idx == len(waypoints) - 1 and is_at_curr_waypoint)
            )

            is_mission_completed = (
                all_waypoints_completed and is_pos_settled and is_vel_settled and not is_collision
            )

            is_final_success = is_mission_completed

            # Record telemetry
            telemetry.append(
                FlightStepTelemetry(
                    step=step,
                    time_sec=time_sec,
                    position=new_state.position.copy(),
                    velocity=new_state.velocity.copy(),
                    quaternion=new_state.quaternion.copy(),
                    euler_angles_deg=euler_deg,
                    angular_velocity=new_state.angular_velocity.copy(),
                    target_position=current_target.copy(),
                    distance_to_target=dist_to_curr_waypoint,
                    action=act_arr.copy(),
                    motor_thrusts=thrusts_actual.copy(),
                    min_obstacle_distance=min_obs_d,
                    is_collision=coll,
                    is_success=is_final_success,
                    commanded_thrusts=thrusts_cmd.copy(),
                    actuator_fault_active=fault_active,
                    estimated_position=est_pos,
                    estimated_velocity=est_vel,
                    position_estimation_error=pos_est_err,
                    attitude_estimation_error_deg=att_est_err,
                    mission_state=mission_telemetry.state.value if mission_telemetry else None,
                    replan_reason=mission_telemetry.replan_reason if mission_telemetry else None,
                    waiting_duration=mission_telemetry.waiting_duration
                    if mission_telemetry
                    else None,
                    min_predicted_clearance=(
                        mission_telemetry.min_predicted_clearance if mission_telemetry else None
                    ),
                    planned_vs_actual_deviation=(
                        mission_telemetry.planned_vs_actual_deviation if mission_telemetry else None
                    ),
                    active_route_cost=(
                        mission_telemetry.active_route_cost if mission_telemetry else None
                    ),
                    state_source=(
                        mission_telemetry.state_source
                        if mission_telemetry
                        else ("estimated" if scenario.sensor_in_the_loop else "ground_truth")
                    ),
                    obstacle_prediction_error=(
                        mission_telemetry.obstacle_prediction_error if mission_telemetry else None
                    ),
                )
            )

            if coll:
                break

            if (
                scenario.max_duration_seconds is not None
                and time_sec >= scenario.max_duration_seconds
            ):
                break

            if is_final_success and (
                len(waypoints) > 1
                or (mission_manager is not None and mission_manager.state == MissionState.ARRIVED)
                or (
                    scenario.use_path_planner
                    and float(np.linalg.norm(scenario.initial_position - waypoints[-1]))
                    > scenario.target_radius
                )
            ):
                outcome = "success"
                break
            elif is_final_success and step >= scenario.max_steps - 10:
                outcome = "success"
                break

        final_pos_err = float(
            np.linalg.norm(
                (scenario.waypoints[-1] if scenario.waypoints else scenario.initial_position)
                - self.env._position
            )
        )
        final_speed = float(np.linalg.norm(self.env.dynamics.state.velocity))
        max_tilt = max(att_errors_deg) if att_errors_deg else 0.0
        min_alt = (
            float(np.min([t.position[2] for t in telemetry]))
            if telemetry
            else float(self.env._position[2])
        )

        is_pos_settled = final_pos_err <= scenario.target_radius
        is_vel_settled = final_speed <= scenario.settling_velocity_threshold
        is_mission_completed = (
            all_waypoints_completed and is_pos_settled and is_vel_settled and not is_collision
        )

        # Authoritative, mutually exclusive outcome classification
        if outcome == "sensor_estimation_failure":
            pass
        elif fault_ever_active and (
            is_collision
            or max_tilt >= scenario.max_recovery_tilt_deg
            or min_alt <= scenario.min_recovery_altitude
        ):
            outcome = "crash_fault"
        elif is_collision:
            outcome = "collision" if coll_type == "obstacle" else "boundary_breach"
        elif fault_ever_active:
            # Active actuator fault was injected:
            # Requires authoritative mission completion, stable flight, altitude maintenance, and velocity settling
            if (
                is_mission_completed
                and max_tilt < scenario.max_recovery_tilt_deg
                and min_alt > scenario.min_recovery_altitude
            ):
                outcome = "recovered_degraded"
            elif (
                scenario.max_duration_seconds is not None
                and (step * dt) >= scenario.max_duration_seconds
            ):
                outcome = "timeout"
            else:
                outcome = "max_steps"
        elif is_mission_completed:
            outcome = "success"
        elif (
            scenario.max_duration_seconds is not None
            and (step * dt) >= scenario.max_duration_seconds
        ):
            outcome = "timeout"
        else:
            outcome = "max_steps"

        elapsed_ms = (time.perf_counter() - start_wall_time) * 1000.0

        metrics = FlightScenarioMetrics(
            scenario_name=scenario.name,
            success=(outcome in ("success", "recovered_degraded")),
            outcome=outcome,
            duration_steps=step,
            duration_seconds=step * dt,
            final_position_error=final_pos_err,
            mean_position_error=float(np.mean(pos_errors)) if pos_errors else 0.0,
            max_position_error=float(np.max(pos_errors)) if pos_errors else 0.0,
            rms_position_error=float(np.sqrt(np.mean(np.square(pos_errors))))
            if pos_errors
            else 0.0,
            mean_velocity_error=float(np.mean(vel_errors)) if vel_errors else 0.0,
            max_velocity_error=float(np.max(vel_errors)) if vel_errors else 0.0,
            rms_velocity_error=float(np.sqrt(np.mean(np.square(vel_errors))))
            if vel_errors
            else 0.0,
            mean_attitude_error_deg=float(np.mean(att_errors_deg)) if att_errors_deg else 0.0,
            max_attitude_error_deg=float(np.max(att_errors_deg)) if att_errors_deg else 0.0,
            mean_angular_speed=float(np.mean(angular_speeds)) if angular_speeds else 0.0,
            max_angular_speed=float(np.max(angular_speeds)) if angular_speeds else 0.0,
            initial_goal_distance=initial_goal_dist,
            final_goal_distance=final_pos_err,
            min_obstacle_clearance=float(min(min_clearances)) if min_clearances else float("inf"),
            collision_count=1 if is_collision else 0,
            control_effort=float(np.sum(control_efforts)) if control_efforts else 0.0,
            execution_wall_time_ms=elapsed_ms,
            replan_count=mission_manager.replan_count if mission_manager is not None else 0,
        )

        return FlightScenarioResult(
            scenario_name=scenario.name,
            description=scenario.description,
            metrics=metrics,
            telemetry=telemetry,
            scenario_definition=scenario,
        )

    def run_all_scenarios(
        self,
        controller: Optional[Any] = None,
        scenarios: Optional[List[FlightScenarioDefinition]] = None,
    ) -> List[FlightScenarioResult]:
        """Execute all configured scenarios and return list of empirical results."""
        scenario_list = scenarios or get_standard_flight_scenarios()
        results: List[FlightScenarioResult] = []
        for sc in scenario_list:
            res = self.run_scenario(sc, controller=controller)
            results.append(res)
        return results
