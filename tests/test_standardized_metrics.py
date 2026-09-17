"""Tests for Phase 16 — Standardized Result Metrics and Serialization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from adaptive_rl.evaluation.metrics import (
    EvaluationMetrics,
    StandardizedExperimentMetrics,
)
from adaptive_rl.experiments.manager import ExperimentManager
from adaptive_rl.planners.adapter import PlannerEvaluationMetrics


def test_standardized_metrics_defaults_to_none() -> None:
    """Unavailable metrics must default to None and serialize to JSON as null, not 0."""
    metrics = StandardizedExperimentMetrics(episodes=10)
    data = json.loads(metrics.model_dump_json())

    assert data["episodes"] == 10
    assert data["episode_return"] is None
    assert data["success_rate"] is None
    assert data["collision_rate"] is None
    assert data["episode_length"] is None
    assert data["path_length"] is None
    assert data["path_efficiency"] is None
    assert data["planning_time"] is None
    assert data["generalization_gap"] is None
    assert data["battery_remaining"] is None
    assert data["battery_used"] is None
    assert data["dynamic_collision_count"] is None


def test_standardized_metrics_from_rl() -> None:
    """RL EvaluationMetrics conversion populates episode_return and RL-specific metrics."""
    rl_metrics = EvaluationMetrics(
        episodes=20,
        mean_reward=88.5,
        std_reward=5.2,
        min_reward=70.0,
        max_reward=95.0,
        success_rate=0.95,
        collision_rate=0.05,
        mean_episode_length=12.4,
        std_episode_length=2.1,
        additional_metrics={
            "mean_path_length": 12.0,
            "path_efficiency": 0.85,
            "mean_battery_remaining": 72.0,
            "battery_used": 28.0,
            "dynamic_collision_count": 1,
        },
    )

    std = StandardizedExperimentMetrics.from_rl_metrics(rl_metrics, generalization_gap=0.08)

    assert std.episodes == 20
    assert std.episode_return == 88.5
    assert std.success_rate == 0.95
    assert std.collision_rate == 0.05
    assert std.episode_length == 12.4
    assert std.path_length == 12.0
    assert std.path_efficiency == 0.85
    assert std.generalization_gap == 0.08
    assert std.battery_remaining == 72.0
    assert std.battery_used == 28.0
    assert std.dynamic_collision_count == 1


def test_standardized_metrics_from_planner() -> None:
    """Planner metrics conversion sets episode_return to None while preserving path metrics."""
    planner_metrics = PlannerEvaluationMetrics(
        episodes=15,
        success_rate=1.0,
        mean_path_length=9.2,
        std_path_length=0.8,
        min_path_length=8.0,
        max_path_length=10.0,
        mean_planning_time=0.0035,
        std_planning_time=0.0004,
        collision_rate=0.0,
        all_path_lengths=[9.0] * 15,
        all_planning_times=[0.0035] * 15,
        additional_metrics={"path_efficiency": 0.92},
    )

    std = StandardizedExperimentMetrics.from_planner_metrics(planner_metrics)

    assert std.episodes == 15
    assert std.episode_return is None  # Planners do not produce RL reward return
    assert std.success_rate == 1.0
    assert std.collision_rate == 0.0
    assert std.path_length == 9.2
    assert std.planning_time == 0.0035
    assert std.path_efficiency == 0.92
    assert std.battery_remaining is None


def test_standardized_metrics_csv_serialization(tmp_path: Path) -> None:
    """Flat CSV serialization correctly writes headers and handles None values."""
    csv_path = tmp_path / "metrics.csv"
    raw_dict = {
        "episodes": 10,
        "episode_return": None,
        "success_rate": 0.9,
        "collision_rate": 0.0,
        "mean_path_length": 8.5,
        "all_path_lengths": [8, 9, 8],
    }

    ExperimentManager._save_metrics_csv(raw_dict, csv_path)
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["episodes"] == "10"
    assert row["episode_return"] == ""
    assert row["success_rate"] == "0.9"
    assert "all_path_lengths" not in row  # Lists must be excluded from scalar CSV


def test_standardized_metrics_preserves_zero_values() -> None:
    """StandardizedExperimentMetrics.from_rl_metrics must preserve measured 0.0 values."""
    rl_metrics = EvaluationMetrics(
        episodes=10,
        mean_reward=0.0,
        std_reward=0.0,
        min_reward=0.0,
        max_reward=0.0,
        success_rate=0.0,
        collision_rate=0.0,
        mean_episode_length=0.0,
        std_episode_length=0.0,
        additional_metrics={
            "mean_path_length": 0.0,
            "path_length": 15.0,  # Should not fall through
            "mean_planning_time": 0.0,
            "planning_time": 1.2,  # Should not fall through
            "mean_battery_remaining": 0.0,
            "battery_remaining": 50.0,  # Should not fall through
            "mean_battery_used": 0.0,
            "battery_used": 10.0,  # Should not fall through
            "generalization_gap": 0.0,
        },
    )

    std = StandardizedExperimentMetrics.from_rl_metrics(rl_metrics, generalization_gap=0.0)

    assert std.path_length == 0.0
    assert std.planning_time == 0.0
    assert std.generalization_gap == 0.0
    assert std.battery_remaining == 0.0
    assert std.battery_used == 0.0
