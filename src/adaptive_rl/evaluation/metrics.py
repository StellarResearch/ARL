"""Evaluation metrics data structures and standardization schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class EvaluationMetrics(BaseModel):
    """Container for reinforcement learning evaluation results."""

    model_config = ConfigDict(extra="ignore")

    episodes: int = Field(..., gt=0, description="Total evaluation episodes executed")
    mean_reward: float = Field(..., description="Mean cumulative episodic reward")
    std_reward: float = Field(0.0, description="Standard deviation of episodic reward")
    min_reward: float = Field(0.0, description="Minimum episodic reward observed")
    max_reward: float = Field(0.0, description="Maximum episodic reward observed")
    success_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of episodes reaching target"
    )
    collision_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of episodes ending in collision"
    )
    mean_episode_length: float = Field(..., ge=0.0, description="Mean step count per episode")
    std_episode_length: float = Field(
        0.0, ge=0.0, description="Standard deviation of episode length"
    )
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment-specific metrics (e.g. energy consumption, path length)",
    )


class StandardizedExperimentMetrics(BaseModel):
    """Standardized cross-paradigm evaluation metrics schema for AdaptiveRL.

    Provides a stable, uniform schema across reinforcement learning policies
    and classical deterministic/sampling planners. Metrics unavailable for a
    particular algorithm or environment are explicitly set to None (JSON null),
    never silently defaulted to 0.
    """

    model_config = ConfigDict(extra="ignore")

    episodes: int = Field(..., gt=0, description="Total evaluation episodes executed")
    episode_return: Optional[float] = Field(
        None, description="Mean cumulative episodic return (RL only; null for classical planners)"
    )
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes reaching goal"
    )
    collision_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes ending in collision"
    )
    episode_length: Optional[float] = Field(None, ge=0.0, description="Mean step count per episode")
    path_length: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean geometric path length (grid steps or Euclidean distance)",
    )
    path_efficiency: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Ratio of optimal/straight-line distance to actual path length",
    )
    planning_time: Optional[float] = Field(
        None, ge=0.0, description="Mean planning/search or inference wall-clock time in seconds"
    )
    generalization_gap: Optional[float] = Field(
        None, description="Performance gap between training and unseen test distributions"
    )
    battery_remaining: Optional[float] = Field(
        None, ge=0.0, description="Mean remaining battery level (drone environments)"
    )
    battery_used: Optional[float] = Field(
        None, ge=0.0, description="Mean battery energy consumed (drone environments)"
    )
    dynamic_collision_count: Optional[int] = Field(
        None, ge=0, description="Total dynamic obstacle collision events"
    )
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary domain-specific metric dictionary"
    )

    @classmethod
    def from_rl_metrics(
        cls,
        eval_metrics: EvaluationMetrics,
        generalization_gap: Optional[float] = None,
    ) -> StandardizedExperimentMetrics:
        """Construct standardized metrics from RL EvaluationMetrics."""
        extra = eval_metrics.additional_metrics

        return cls(
            episodes=eval_metrics.episodes,
            episode_return=eval_metrics.mean_reward,
            success_rate=eval_metrics.success_rate,
            collision_rate=eval_metrics.collision_rate,
            episode_length=eval_metrics.mean_episode_length,
            path_length=extra.get("mean_path_length") or extra.get("path_length"),
            path_efficiency=extra.get("path_efficiency"),
            planning_time=extra.get("mean_planning_time") or extra.get("planning_time"),
            generalization_gap=generalization_gap or extra.get("generalization_gap"),
            battery_remaining=extra.get("mean_battery_remaining") or extra.get("battery_remaining"),
            battery_used=extra.get("mean_battery_used") or extra.get("battery_used"),
            dynamic_collision_count=extra.get("dynamic_collision_count"),
            additional_metrics={
                k: v
                for k, v in extra.items()
                if k
                not in (
                    "mean_path_length",
                    "path_length",
                    "path_efficiency",
                    "mean_planning_time",
                    "planning_time",
                    "generalization_gap",
                    "mean_battery_remaining",
                    "battery_remaining",
                    "mean_battery_used",
                    "battery_used",
                    "dynamic_collision_count",
                )
            },
        )

    @classmethod
    def from_planner_metrics(
        cls,
        planner_metrics: Any,
    ) -> StandardizedExperimentMetrics:
        """Construct standardized metrics from PlannerEvaluationMetrics."""
        extra = getattr(planner_metrics, "additional_metrics", {})
        mean_path = getattr(planner_metrics, "mean_path_length", None)
        mean_time = getattr(planner_metrics, "mean_planning_time", None)

        return cls(
            episodes=getattr(planner_metrics, "episodes", 1),
            episode_return=None,  # Classical planners do not accumulate RL reward returns
            success_rate=getattr(planner_metrics, "success_rate", None),
            collision_rate=getattr(planner_metrics, "collision_rate", 0.0),
            episode_length=mean_path if getattr(planner_metrics, "is_discrete", False) else None,
            path_length=mean_path,
            path_efficiency=extra.get("path_efficiency"),
            planning_time=mean_time,
            generalization_gap=None,
            battery_remaining=None,
            battery_used=None,
            dynamic_collision_count=None,
            additional_metrics=extra,
        )

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert flat scalar fields to a dictionary suitable for CSV serialization."""
        data = self.model_dump(exclude={"additional_metrics"})
        flat: Dict[str, Any] = {}
        for k, v in data.items():
            if v is not None:
                flat[k] = v
            else:
                flat[k] = ""
        return flat
