"""Export a deterministic rollout from a trained PPO or SAC checkpoint.

The script loads a real trained checkpoint, inspects environment compatibility
(3-DOF or 6-DOF), runs a single deterministic episode with a fixed seed, records
per-step simulation telemetry and precise timing (separating policy inference and
environment stepping), and exports a validated JSON replay artifact to `docs/data/flight.json`.

Usage:
    python scripts/export_flight.py --checkpoint artifacts/models/drone_6dof_ppo_gui_final.zip --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from adaptive_rl.algorithms.registry import (
    inspect_model_checkpoint,
    load_algorithm_from_pretrained,
)
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.environments.drone_6dof import Drone6DOFEnv


def _to_list(arr: Any) -> Any:
    """Recursively convert numpy arrays and floats/ints to JSON-safe Python types."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, (np.floating, float)):
        val = float(arr)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return round(val, 6)
    if isinstance(arr, (np.integer, int)):
        return int(arr)
    if isinstance(arr, (list, tuple)):
        return [_to_list(x) for x in arr]
    if isinstance(arr, dict):
        return {k: _to_list(v) for k, v in arr.items()}
    return arr


def run_deterministic_rollout(
    checkpoint_path: pathlib.Path,
    seed: int = 42,
    max_steps: int = 500,
) -> Dict[str, Any]:
    """Execute a genuine deterministic episode using the trained checkpoint."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    meta = inspect_model_checkpoint(checkpoint_path)
    if not meta.get("valid", False):
        raise ValueError(f"Invalid checkpoint metadata: {meta.get('error')}")

    compatible_env = meta.get("compatible_env")
    algorithm_name = (meta.get("algorithm") or "PPO").upper()

    if compatible_env == "6-DOF":
        env = Drone6DOFEnv()
        dynamics_mode = "6-DOF"
    elif compatible_env == "3-DOF":
        env = DroneNavigation3DEnv()
        dynamics_mode = "3-DOF"
    else:
        # Fallback check based on observation space shape
        obs_shape = meta.get("obs_shape")
        if obs_shape == (36,):
            env = Drone6DOFEnv()
            dynamics_mode = "6-DOF"
        else:
            env = DroneNavigation3DEnv()
            dynamics_mode = "3-DOF"

    model = load_algorithm_from_pretrained(checkpoint_path)

    # Reset environment deterministically
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)

    dt = float(getattr(env, "dt", 0.05 if dynamics_mode == "6-DOF" else 0.1))
    bounds = _to_list(getattr(env, "bounds", [30.0, 30.0, 15.0]))
    target_pos = _to_list(getattr(env, "_goal", getattr(env, "goal", [25.0, 25.0, 10.0])))
    target_radius = float(getattr(env, "target_radius", 1.5))
    collision_radius = float(getattr(env, "collision_radius", 0.8))

    raw_obstacles = getattr(env, "_obstacles", getattr(env, "obstacles", []))
    obstacles = []
    for obs_item in raw_obstacles:
        center = getattr(obs_item, "center", None)
        radius = getattr(obs_item, "radius", 2.0)
        if center is not None:
            obstacles.append({"position": _to_list(center), "radius": float(radius)})

    start_pos = _to_list(info["position"])

    num_rays = int(getattr(env, "num_lidar_rays", 16))
    lidar_range = float(getattr(env, "lidar_range", 20.0))

    trajectory: List[Dict[str, Any]] = []
    policy_times_ms: List[float] = []
    env_step_times_ms: List[float] = []

    # Initial frame at t=0
    init_lidar_raw = obs[-num_rays:] if len(obs) >= num_rays else np.zeros(num_rays)
    init_lidar_meters = _to_list(init_lidar_raw * lidar_range)
    init_quat = _to_list(info.get("quaternion")) if "quaternion" in info else None
    init_omega = _to_list(info.get("angular_velocity")) if "angular_velocity" in info else None

    step_0_data = {
        "step": 0,
        "time": 0.0,
        "position": _to_list(info["position"]),
        "velocity": _to_list(info["velocity"]),
        "speed": float(info.get("speed", 0.0)),
        "quaternion": init_quat,
        "angular_velocity": init_omega,
        "distance_to_goal": float(info.get("distance_to_goal", 0.0)),
        "min_obstacle_distance": (
            float(info.get("min_obstacle_distance", 999.0))
            if not math.isinf(float(info.get("min_obstacle_distance", 999.0)))
            else 999.0
        ),
        "reward": 0.0,
        "cumulative_reward": 0.0,
        "collision": False,
        "success": False,
        "terminated": False,
        "truncated": False,
        "lidar": init_lidar_meters,
        "policy_time_ms": 0.0,
        "env_step_time_ms": 0.0,
    }
    trajectory.append(step_0_data)

    total_reward = 0.0
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < max_steps:
        step_count += 1

        # 1. Policy inference timing
        t_policy_start = time.perf_counter()
        pred = model.predict(obs, deterministic=True)
        t_policy_end = time.perf_counter()
        action = pred[0] if isinstance(pred, tuple) else pred
        policy_ms = (t_policy_end - t_policy_start) * 1000.0
        policy_times_ms.append(policy_ms)

        # 2. Environment step timing
        t_step_start = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        t_step_end = time.perf_counter()
        step_ms = (t_step_end - t_step_start) * 1000.0
        env_step_times_ms.append(step_ms)

        total_reward += float(reward)

        pos = _to_list(info["position"])
        vel = _to_list(info["velocity"])
        speed = float(info.get("speed", np.linalg.norm(vel)))
        quat = _to_list(info.get("quaternion")) if "quaternion" in info else None
        omega = _to_list(info.get("angular_velocity")) if "angular_velocity" in info else None
        dist_goal = float(info.get("distance_to_goal", 0.0))
        min_obs = float(info.get("min_obstacle_distance", 999.0))
        if math.isinf(min_obs):
            min_obs = 999.0

        is_coll = bool(info.get("collision", False))
        is_succ = bool(info.get("is_success", False) or info.get("success", False))

        raw_lidar = obs[-num_rays:] if len(obs) >= num_rays else np.zeros(num_rays)
        lidar_meters = _to_list(raw_lidar * lidar_range)

        step_record = {
            "step": step_count,
            "time": round(step_count * dt, 4),
            "position": pos,
            "velocity": vel,
            "speed": round(speed, 4),
            "quaternion": quat,
            "angular_velocity": omega,
            "distance_to_goal": round(dist_goal, 4),
            "min_obstacle_distance": round(min_obs, 4),
            "reward": round(float(reward), 4),
            "cumulative_reward": round(total_reward, 4),
            "collision": is_coll,
            "success": is_succ,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "lidar": lidar_meters,
            "policy_time_ms": round(policy_ms, 4),
            "env_step_time_ms": round(step_ms, 4),
        }
        trajectory.append(step_record)

    last_step = trajectory[-1]
    if last_step["success"]:
        outcome = "SUCCESS"
    elif last_step["collision"]:
        outcome = "COLLISION"
    elif truncated or step_count >= max_steps:
        outcome = "MAX_STEPS"
    else:
        outcome = "TERMINATED"

    replay = {
        "schema_version": "1.0.0",
        "description": "Recorded trained-policy rollout",
        "algorithm": algorithm_name,
        "checkpoint": checkpoint_path.name,
        "environment": env.__class__.__name__,
        "dynamics_mode": dynamics_mode,
        "seed": seed,
        "deterministic": True,
        "dt": dt,
        "num_steps": len(trajectory),
        "outcome": outcome,
        "total_reward": round(total_reward, 4),
        "arena_size": bounds,
        "collision_radius": collision_radius,
        "target": {
            "position": target_pos,
            "radius": target_radius,
        },
        "start_pos": start_pos,
        "obstacles": obstacles,
        "avg_policy_time_ms": (
            round(sum(policy_times_ms) / len(policy_times_ms), 4) if policy_times_ms else 0.0
        ),
        "avg_env_step_time_ms": (
            round(sum(env_step_times_ms) / len(env_step_times_ms), 4)
            if env_step_times_ms
            else 0.0
        ),
        "trajectory": trajectory,
    }
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a genuine deterministic trained-policy flight rollout."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/models/drone_6dof_ppo_gui_final.zip",
        help="Path to trained model checkpoint zip file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for reproducibility.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="docs/data/flight.json",
        help="Output JSON destination path.",
    )
    args = parser.parse_args()

    checkpoint_path = pathlib.Path(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    replay = run_deterministic_rollout(checkpoint_path=checkpoint_path, seed=args.seed)

    output_path = pathlib.Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(replay, f, ensure_ascii=False, indent=2)

    print(f"Successfully exported genuine rollout to: {output_path}")
    print(f"Algorithm: {replay['algorithm']} | Environment: {replay['environment']}")
    print(
        f"Steps: {replay['num_steps']} | Outcome: {replay['outcome']} | Total Reward: {replay['total_reward']}"
    )
    print(
        f"Avg Policy Inference Time: {replay['avg_policy_time_ms']} ms | Avg Env Step Time: {replay['avg_env_step_time_ms']} ms"
    )


if __name__ == "__main__":
    main()

