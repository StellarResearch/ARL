"""LiDAR sensor robustness benchmark for 3D drone navigation."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.environments.drone import DroneNavigation3DEnv
from adaptive_rl.evaluation.evaluator import EpisodeEvaluationRecord, Evaluator


@dataclass(frozen=True)
class LidarCondition:
    name: str
    noise_std: float
    dropout_prob: float
    min_range: float = 0.0

    def environment_kwargs(self) -> dict[str, float]:
        return {
            "lidar_noise_std": self.noise_std,
            "lidar_dropout_prob": self.dropout_prob,
            "lidar_min_range": self.min_range,
        }


LIDAR_CONDITIONS: tuple[LidarCondition, ...] = (
    LidarCondition("LIDAR_IDEAL", 0.0, 0.0),
    LidarCondition("LIDAR_NOISE_MODERATE", 0.15, 0.0),
    LidarCondition("LIDAR_DROPOUT_MODERATE", 0.0, 0.20),
    LidarCondition("LIDAR_COMBINED", 0.15, 0.20),
    LidarCondition("LIDAR_SEVERE", 0.35, 0.40, 0.5),
)


class _TracedDroneEnv(DroneNavigation3DEnv):
    """Retain episode positions without changing drone dynamics."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trajectories: list[list[np.ndarray]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        self.trajectories.append([np.asarray(info["position"], dtype=np.float64).copy()])
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        self.trajectories[-1].append(np.asarray(info["position"], dtype=np.float64).copy())
        return observation, reward, terminated, truncated, info


def classify_timeout_trajectory(
    positions: Sequence[np.ndarray],
    *,
    window_steps: int = 20,
) -> str:
    """Classify timeout movement using the final 20 trajectory steps."""
    if window_steps < 1:
        raise ValueError("window_steps must be positive.")
    if len(positions) < window_steps + 1:
        return "unclassified_timeout"

    segment = np.asarray(positions[-(window_steps + 1) :], dtype=np.float64)
    if segment.shape != (window_steps + 1, 3):
        return "unclassified_timeout"
    if not np.isfinite(segment).all():
        return "unclassified_timeout"

    total_path = float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum())
    net_displacement = float(np.linalg.norm(segment[-1] - segment[0]))

    if total_path < 0.5:
        return "freeze_timeout"
    if total_path >= 2.0 and net_displacement < 1.0:
        return "oscillation_timeout"
    return "unclassified_timeout"


def _classify_failure(
    record: EpisodeEvaluationRecord,
    positions: Sequence[np.ndarray],
) -> str | None:
    """Classify observed failures; successful episodes have no failure mode."""
    if record.collision:
        if record.collision_type == "obstacle":
            return "near_obstacle_collision"
        if (record.collision_type or "").startswith("boundary"):
            return "boundary_collision"
        return "other_collision"
    if record.timeout:
        return classify_timeout_trajectory(positions)
    return None


def run_lidar_benchmark(
    algorithm: BaseAlgorithm,
    *,
    num_episodes: int = 30,
    base_seed: int = 1000,
    seeds: Sequence[int] | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one unchanged policy across matched LiDAR conditions."""
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")

    episode_seeds = (
        list(seeds) if seeds is not None else list(range(base_seed, base_seed + num_episodes))
    )
    if not episode_seeds:
        raise ValueError("At least one evaluation seed is required.")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in episode_seeds
    ):
        raise ValueError("Evaluation seeds must be non-negative integers.")
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("Evaluation seeds must be unique.")

    environment_options = dict(env_kwargs or {})
    sensor_keys = set(LIDAR_CONDITIONS[0].environment_kwargs())
    if sensor_keys.intersection(environment_options):
        raise ValueError("Sensor parameters are controlled by benchmark conditions.")

    summaries: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []

    for condition in LIDAR_CONDITIONS:
        env = _TracedDroneEnv(
            **environment_options,
            **condition.environment_kwargs(),
        )
        try:
            evaluator = Evaluator(algorithm=algorithm, env=env)
            metrics = evaluator.evaluate(
                seeds=episode_seeds,
                deterministic=True,
            )
            records = evaluator.last_episode_records
            if len(records) != len(env.trajectories):
                raise RuntimeError("Episode trajectories and records do not match.")
            failure_modes = [
                _classify_failure(record, trajectory)
                for record, trajectory in zip(records, env.trajectories, strict=True)
            ]

            summaries.append(
                {
                    "condition": condition.name,
                    "noise_std_m": condition.noise_std,
                    "dropout_prob": condition.dropout_prob,
                    "blind_range_m": condition.min_range,
                    "episodes": metrics.episodes,
                    "success_rate": metrics.success_rate,
                    "collision_rate": metrics.collision_rate,
                    "timeout_rate": metrics.timeout_rate,
                    "mean_clearance_m": metrics.mean_min_obstacle_clearance,
                    "mean_path_length_m": metrics.mean_path_length,
                    "mean_reward": metrics.mean_reward,
                    "near_obstacle_collisions": failure_modes.count("near_obstacle_collision"),
                    "boundary_collisions": failure_modes.count("boundary_collision"),
                    "other_collisions": failure_modes.count("other_collision"),
                    "freeze_timeouts": failure_modes.count("freeze_timeout"),
                    "oscillation_timeouts": failure_modes.count("oscillation_timeout"),
                    "unclassified_timeouts": failure_modes.count("unclassified_timeout"),
                }
            )

            for record, failure_mode in zip(records, failure_modes, strict=True):
                episodes.append(
                    {
                        "condition": condition.name,
                        "failure_mode": failure_mode,
                        **record.to_dict(),
                    }
                )
        finally:
            env.close()

    return {
        "metadata": {
            "seeds": episode_seeds,
            "episodes_per_condition": len(episode_seeds),
            "deterministic_actions": True,
            "policy_retrained_between_conditions": False,
            "timeout_taxonomy": {
                "window_steps": 20,
                "freeze_path_threshold_m": 0.5,
                "oscillation_path_threshold_m": 2.0,
                "oscillation_net_displacement_threshold_m": 1.0,
                "insufficient_or_ambiguous_trajectory": "unclassified_timeout",
            },
        },
        "conditions": summaries,
        "episodes": episodes,
    }


def export_lidar_benchmark(
    result: dict[str, Any],
    output_dir: str = "artifacts/benchmarks",
    *,
    plot: bool = True,
) -> dict[str, str]:
    """Write summary, episode-level results, and robustness curves."""
    import csv
    import json
    from pathlib import Path

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    summaries = result["conditions"]
    episodes = result["episodes"]

    paths = {
        "json": directory / "lidar_robustness.json",
        "csv": directory / "lidar_robustness.csv",
        "episodes_csv": directory / "lidar_robustness_episodes.csv",
    }

    with paths["json"].open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")

    for key, rows in (
        ("csv", summaries),
        ("episodes_csv", episodes),
    ):
        if not rows:
            raise ValueError(f"No rows available for {key} export.")

        fieldnames = list(dict.fromkeys(field for row in rows for field in row))

        with paths[key].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = list(range(len(summaries)))
        labels = [row["condition"].removeprefix("LIDAR_") for row in summaries]

        fig, ax = plt.subplots(figsize=(10, 5))
        try:
            for key, label in (
                ("success_rate", "Success rate"),
                ("collision_rate", "Collision rate"),
                ("timeout_rate", "Timeout rate"),
            ):
                ax.plot(
                    x,
                    [row[key] for row in summaries],
                    marker="o",
                    label=label,
                )

            ax.set_xticks(x, labels, rotation=25, ha="right")
            ax.set_ylabel("Episode rate")
            ax.set_ylim(0, 1.05)
            ax.set_title("LiDAR robustness across sensor conditions")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()

            paths["plot"] = directory / "lidar_robustness.png"
            fig.savefig(paths["plot"], dpi=160)
        finally:
            plt.close(fig)

    return {key: str(path) for key, path in paths.items()}
