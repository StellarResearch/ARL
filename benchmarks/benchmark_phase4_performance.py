"""Empirical performance benchmark for Phase IV predictive autonomous planning components.

Measures:
1. 4D Time-Aware A* search throughput (planning latency, node expansions/sec, generated nodes/sec).
2. Continuous trajectory risk evaluator throughput (evaluations/sec for static and moving obstacles).
3. MissionManager closed-loop update and dynamic replanning latency (ms per cycle).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from adaptive_rl.environments.drone import ObstacleSphere3D
from adaptive_rl.environments.drone_6dof import DroneState6DOF
from adaptive_rl.navigation.dynamic_obstacles import DynamicObstacleSphere3D
from adaptive_rl.navigation.mission import MissionManager
from adaptive_rl.navigation.time_aware_planner import (
    RouteEvaluator,
    TimeAwareAStarPlanner3D,
    TimedWaypoint,
)


def benchmark_4d_astar_planning(num_trials: int = 10) -> Dict[str, Any]:
    """Measure 4D Time-Aware A* planning throughput across clear and cluttered scenarios."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(30.0, 30.0, 15.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
    )
    start = np.array([2.0, 2.0, 2.0])
    goal = np.array([25.0, 25.0, 10.0])

    # 1. Clear environment
    latencies_clear = []
    expansions_clear = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        res = planner.plan(start, goal, static_obstacles=[], dynamic_obstacles=[])
        latencies_clear.append((time.perf_counter() - t0) * 1000.0)
        expansions_clear.append(res.nodes_expanded)

    # 2. Cluttered environment with moving obstacles
    static_obs = [
        ObstacleSphere3D(center=np.array([10.0, 10.0, 5.0]), radius=2.0),
        ObstacleSphere3D(center=np.array([15.0, 18.0, 7.0]), radius=2.5),
        ObstacleSphere3D(center=np.array([20.0, 12.0, 6.0]), radius=2.0),
    ]
    dyn_obs = [
        DynamicObstacleSphere3D(
            center_initial=np.array([12.0, 5.0, 5.0]),
            radius=1.2,
            velocity=np.array([0.0, 1.5, 0.0]),
            bounds=(30.0, 30.0, 15.0),
            motion_type="bounce",
        ),
        DynamicObstacleSphere3D(
            center_initial=np.array([18.0, 25.0, 8.0]),
            radius=1.2,
            velocity=np.array([0.0, -1.5, 0.0]),
            bounds=(30.0, 30.0, 15.0),
            motion_type="bounce",
        ),
    ]

    latencies_cluttered = []
    expansions_cluttered = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        res = planner.plan(start, goal, static_obstacles=static_obs, dynamic_obstacles=dyn_obs)
        latencies_cluttered.append((time.perf_counter() - t0) * 1000.0)
        expansions_cluttered.append(res.nodes_expanded)

    return {
        "clear_mean_ms": float(np.mean(latencies_clear)),
        "clear_std_ms": float(np.std(latencies_clear)),
        "clear_mean_expansions": float(np.mean(expansions_clear)),
        "cluttered_mean_ms": float(np.mean(latencies_cluttered)),
        "cluttered_std_ms": float(np.std(latencies_cluttered)),
        "cluttered_mean_expansions": float(np.mean(expansions_cluttered)),
    }


def benchmark_continuous_risk_evaluator(num_evals: int = 500) -> Dict[str, Any]:
    """Measure RouteEvaluator throughput for continuous chord and swept-sphere collision checks."""
    evaluator = RouteEvaluator(nominal_speed=2.0)
    timed_waypoints = [
        TimedWaypoint(position=np.array([float(x), float(x), 5.0]), time_sec=float(x) * 0.5)
        for x in range(20)
    ]
    static_obs = [
        ObstacleSphere3D(center=np.array([5.0, 5.0, 5.0]), radius=1.0),
        ObstacleSphere3D(center=np.array([10.0, 10.0, 5.0]), radius=1.0),
        ObstacleSphere3D(center=np.array([15.0, 15.0, 5.0]), radius=1.0),
    ]
    dyn_obs = [
        DynamicObstacleSphere3D(
            center_initial=np.array([10.0, 0.0, 5.0]),
            radius=1.0,
            velocity=np.array([0.0, 2.0, 0.0]),
            bounds=(30.0, 30.0, 15.0),
            motion_type="bounce",
        )
    ]

    t0 = time.perf_counter()
    for _ in range(num_evals):
        evaluator.evaluate(
            timed_waypoints=timed_waypoints,
            static_obstacles=static_obs,
            dynamic_obstacles=dyn_obs,
        )
    elapsed = time.perf_counter() - t0

    evals_per_sec = num_evals / elapsed
    mean_eval_ms = (elapsed / num_evals) * 1000.0

    return {
        "num_evaluations": num_evals,
        "elapsed_seconds": elapsed,
        "evaluations_per_second": evals_per_sec,
        "mean_latency_ms": mean_eval_ms,
    }


def benchmark_mission_manager_update_latency(num_steps: int = 500) -> Dict[str, Any]:
    """Measure MissionManager real-time update and replanning cycle latency."""
    planner = TimeAwareAStarPlanner3D(
        bounds=(30.0, 30.0, 15.0),
        spatial_resolution=1.0,
        time_resolution=0.5,
        nominal_speed=2.0,
    )
    mgr = MissionManager(
        time_aware_planner=planner,
        waypoint_acceptance_radius=1.5,
    )
    start = np.array([2.0, 15.0, 5.0])
    goal = np.array([25.0, 15.0, 5.0])
    dyn_obs = [
        DynamicObstacleSphere3D(
            center_initial=np.array([12.0, 10.0, 5.0]),
            radius=1.2,
            velocity=np.array([0.0, 1.5, 0.0]),
            bounds=(30.0, 30.0, 15.0),
            motion_type="bounce",
        )
    ]
    mgr.start_mission(start, [goal], static_obstacles=[], dynamic_obstacles=dyn_obs, time_sec=0.0)

    drone_state = DroneState6DOF(
        position=start.copy(),
        velocity=np.array([1.5, 0.0, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )

    latencies = []
    for step in range(num_steps):
        t_sec = step * 0.05
        # Simulate gentle forward progress
        drone_state.position[0] = min(25.0, start[0] + 1.5 * t_sec)

        t0 = time.perf_counter()
        mgr.update(
            drone_state=drone_state,
            time_sec=t_sec,
            static_obstacles=[],
            dynamic_obstacles=dyn_obs,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "total_steps": num_steps,
        "mean_update_ms": float(np.mean(latencies)),
        "max_update_ms": float(np.max(latencies)),
        "p95_update_ms": float(np.percentile(latencies, 95)),
        "updates_per_second": 1000.0 / float(np.mean(latencies)),
    }


def run_all_benchmarks() -> Dict[str, Any]:
    print("=" * 60)
    print("Phase IV Autonomous Navigation Performance Benchmarks")
    print("=" * 60)

    print("\n1. Running 4D Time-Aware A* Planning Benchmark...")
    bench_astar = benchmark_4d_astar_planning(num_trials=5)
    print(f"   Clear Environment Mean Latency:     {bench_astar['clear_mean_ms']:.2f} ms")
    print(f"   Clear Environment Mean Expansions:  {bench_astar['clear_mean_expansions']:.1f}")
    print(f"   Cluttered Mean Latency:             {bench_astar['cluttered_mean_ms']:.2f} ms")
    print(f"   Cluttered Mean Expansions:          {bench_astar['cluttered_mean_expansions']:.1f}")

    print("\n2. Running Continuous Trajectory Risk Evaluator Benchmark...")
    bench_risk = benchmark_continuous_risk_evaluator(num_evals=500)
    print(f"   Throughput:   {bench_risk['evaluations_per_second']:.1f} evaluations/sec")
    print(f"   Mean Latency: {bench_risk['mean_latency_ms']:.3f} ms/eval")

    print("\n3. Running MissionManager Update Cycle Benchmark...")
    bench_mgr = benchmark_mission_manager_update_latency(num_steps=500)
    print(f"   Mean Latency: {bench_mgr['mean_update_ms']:.3f} ms/update")
    print(f"   P95 Latency:  {bench_mgr['p95_update_ms']:.3f} ms/update")
    print(f"   Max Latency:  {bench_mgr['max_update_ms']:.3f} ms/update")
    print(f"   Throughput:   {bench_mgr['updates_per_second']:.1f} updates/sec")

    print("\n" + "=" * 60)
    print("All benchmarks completed successfully.")
    print("=" * 60)

    return {
        "astar_planning": bench_astar,
        "risk_evaluator": bench_risk,
        "mission_manager": bench_mgr,
    }


if __name__ == "__main__":
    run_all_benchmarks()
