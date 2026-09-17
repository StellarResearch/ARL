"""Configuration system and schemas for AdaptiveRL experiments.

Provides schema validation, YAML loading, and deterministic configuration
management for environments, algorithms (RL policies and classical planners),
training, and evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# Canonical set of classical planner algorithm names
PLANNER_ALGORITHMS: Set[str] = {"astar", "rrt_star", "rrt*"}


class ConfigError(Exception):
    """Exception raised for configuration parsing or validation failures."""

    pass


class AStarParametersConfig(BaseModel):
    """Configuration parameters for A* planner."""

    model_config = ConfigDict(extra="forbid")

    heuristic: Literal["manhattan", "euclidean", "chebyshev"] = Field(
        "manhattan",
        description="Heuristic function to use ('manhattan', 'euclidean', 'chebyshev')",
    )
    seed: Optional[int] = Field(
        None, description="Optional random seed (accepted for interface uniformity)"
    )


class RRTStarParametersConfig(BaseModel):
    """Configuration parameters for RRT* planner."""

    model_config = ConfigDict(extra="forbid")

    step_size: float = Field(0.5, gt=0.0, description="Maximum extension distance per tree step")
    max_iterations: int = Field(
        1500, ge=1, le=100_000, description="Maximum random samples to expand (max 100,000)"
    )
    goal_bias: float = Field(
        0.1, ge=0.0, le=1.0, description="Probability of sampling goal directly"
    )
    search_radius: float = Field(1.5, gt=0.0, description="Radius for rewiring near neighbors")
    collision_resolution: float = Field(
        0.05, ge=1e-4, description="Step size for collision checking (minimum 1e-4)"
    )
    seed: Optional[int] = Field(None, description="Optional fixed random seed for planner")

    @model_validator(mode="after")
    def _validate_relational_constraints(self) -> RRTStarParametersConfig:
        if self.collision_resolution > self.step_size:
            raise ValueError(
                f"collision_resolution ({self.collision_resolution}) cannot be greater than "
                f"step_size ({self.step_size}) to prevent tunneling through obstacles."
            )
        if self.search_radius < 0.5 * self.step_size:
            raise ValueError(
                f"search_radius ({self.search_radius}) must be >= 0.5 * step_size ({0.5 * self.step_size}) "
                "to allow effective near-neighbor rewiring."
            )
        return self


class RLAlgorithmConfig(BaseModel):
    """Configuration parameters for a reinforcement learning algorithm."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., description="Algorithm name, e.g. 'ppo' or 'sac'")
    learning_rate: float = Field(3e-4, gt=0.0, description="Optimizer learning rate")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor")
    batch_size: int = Field(64, gt=0, description="Minibatch size")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional algorithm-specific hyperparameters"
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_params(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" in data and "parameters" not in data:
            data["parameters"] = data.pop("params")
        return data


class PlannerAlgorithmConfig(BaseModel):
    """Configuration parameters for a classical deterministic/sampling planner."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., description="Planner name, e.g. 'astar' or 'rrt_star'")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Planner-specific hyperparameters (e.g. heuristic, step_size)",
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_params(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            raw_name = str(data.get("name", "")).strip().lower()
            if raw_name == "rrt":
                raise ValueError(
                    "Planner name 'rrt' is not supported. Did you mean 'rrt_star'? "
                    "Standard RRT does not perform tree rewiring and is not implemented."
                )
            if raw_name == "rrt*":
                data["name"] = "rrt_star"
                raw_name = "rrt_star"
            raw_params = data.get("parameters", {})
            if raw_name == "astar":
                data["parameters"] = AStarParametersConfig.model_validate(raw_params).model_dump(
                    exclude_none=True
                )
            elif raw_name in ("rrt_star", "rrt*"):
                data["parameters"] = RRTStarParametersConfig.model_validate(raw_params).model_dump(
                    exclude_none=True
                )
        return data


class AlgorithmConfig(BaseModel):
    """Polymorphic configuration for RL algorithms or classical planners.

    Distinguishes between trainable RL algorithms (PPO, SAC) and deterministic/sampling
    planners (A*, RRT*). Planners strictly reject RL-only hyperparameters such as
    `learning_rate`, `gamma`, or `batch_size`.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(
        ..., description="Algorithm or planner name, e.g. 'ppo', 'sac', 'astar', 'rrt_star'"
    )
    learning_rate: Optional[float] = Field(
        None, gt=0.0, description="Optimizer learning rate (RL only)"
    )
    gamma: Optional[float] = Field(None, ge=0.0, le=1.0, description="Discount factor (RL only)")
    batch_size: Optional[int] = Field(None, gt=0, description="Minibatch size (RL only)")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Algorithm or planner specific hyperparameters"
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_and_normalize(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize params alias
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")

            raw_name = str(data.get("name", "")).strip().lower()
            if raw_name == "rrt":
                raise ValueError(
                    "Algorithm name 'rrt' is not supported. Did you mean 'rrt_star'? "
                    "Standard RRT does not perform tree rewiring and is not implemented."
                )
            if raw_name == "rrt*":
                raw_name = "rrt_star"
            data["name"] = raw_name
            is_planner = raw_name in PLANNER_ALGORITHMS

            rl_fields = ["learning_rate", "gamma", "batch_size"]
            present_rl = [f for f in rl_fields if f in data and data[f] is not None]

            if is_planner:
                if present_rl:
                    raise ValueError(
                        f"Classical planner '{raw_name}' does not accept RL hyperparameter(s): {', '.join(present_rl)}. "
                        "Classical planners evaluate paths directly and do not use learning rate, discount factor, or batch size. "
                        "Configure planner parameters under 'parameters' (or 'params')."
                    )
                raw_params = data.get("parameters", {})
                if raw_name == "astar":
                    data["parameters"] = AStarParametersConfig.model_validate(
                        raw_params
                    ).model_dump(exclude_none=True)
                elif raw_name in ("rrt_star", "rrt*"):
                    data["parameters"] = RRTStarParametersConfig.model_validate(
                        raw_params
                    ).model_dump(exclude_none=True)
            else:
                # Supply default RL hyperparameter values if not specified
                if "learning_rate" not in data:
                    data["learning_rate"] = 3e-4
                if "gamma" not in data:
                    data["gamma"] = 0.99
                if "batch_size" not in data:
                    data["batch_size"] = 64

        return data

    @property
    def is_planner(self) -> bool:
        """Return True if this configuration is for a classical planner."""
        return self.name.lower() in PLANNER_ALGORITHMS


class EnvironmentConfig(BaseModel):
    """Configuration parameters for the Gymnasium environment."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = Field(..., description="Registered environment name, e.g. 'gridworld'")
    max_steps: int = Field(100, gt=0, description="Maximum steps per episode")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment-specific parameters (e.g. grid size, obstacle count)",
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_params(cls, data: Any) -> Any:
        if isinstance(data, dict) and "params" in data and "parameters" not in data:
            data["parameters"] = data.pop("params")
        return data


class TrainingConfig(BaseModel):
    """Configuration parameters for the training loop."""

    model_config = ConfigDict(extra="forbid")

    total_timesteps: int = Field(10000, gt=0, description="Total environment steps to train")
    checkpoint_freq: int = Field(
        2000, ge=0, description="Frequency of saving model checkpoints (0 = disabled)"
    )
    log_interval: int = Field(10, gt=0, description="Frequency of logging metrics")


class EvaluationConfig(BaseModel):
    """Configuration parameters for evaluation and benchmarking."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    eval_episodes: int = Field(10, gt=0, description="Number of evaluation episodes")
    deterministic: bool = Field(
        True, description="Whether to use deterministic actions in evaluation"
    )

    @model_validator(mode="before")
    @classmethod
    def _alias_episodes(cls, data: Any) -> Any:
        if isinstance(data, dict) and "episodes" in data and "eval_episodes" not in data:
            data["eval_episodes"] = data.pop("episodes")
        return data


class CurriculumStageConfig(BaseModel):
    """Configuration for a single curriculum progression stage."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Descriptive stage name")
    environment_parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Environment parameter overrides for this stage"
    )
    success_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum success rate to advance to next stage"
    )
    mean_reward_threshold: Optional[float] = Field(
        None, description="Minimum mean reward to advance to next stage"
    )
    max_timesteps: Optional[int] = Field(
        None, gt=0, description="Max timesteps in stage before automatic advance"
    )
    min_episodes: int = Field(10, ge=1, description="Minimum episodes before advancing")
    description: str = Field("", description="Optional stage description")


class CurriculumConfig(BaseModel):
    """Configuration for curriculum learning staged progression."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Whether curriculum learning is enabled")
    preset: Optional[str] = Field(
        None, description="Optional preset name ('navigation' or 'gridworld')"
    )
    stages: list[CurriculumStageConfig] = Field(
        default_factory=list, description="Explicit sequence of curriculum stages"
    )
    eval_window: int = Field(
        20, gt=0, description="Rolling window size of recent episodes used for advancement"
    )


class ExperimentConfig(BaseModel):
    """Top-level configuration schema for an AdaptiveRL experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique experiment identifier")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    algorithm: AlgorithmConfig
    environment: EnvironmentConfig
    training: Optional[TrainingConfig] = Field(
        None,
        description="Training configuration (required for RL algorithms, omitted for planners)",
    )
    evaluation: EvaluationConfig = Field(
        default_factory=lambda: EvaluationConfig(eval_episodes=10, deterministic=True)
    )
    curriculum: Optional[CurriculumConfig] = Field(
        None, description="Optional curriculum learning configuration"
    )
    output_dir: Path = Field(
        default_factory=lambda: Path("experiments/results"),
        description="Directory for saving models and evaluations",
    )
    log_dir: Path = Field(
        default_factory=lambda: Path("experiments/logs"),
        description="Directory for logging and tensorboard metrics",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_experiment_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Unpack optional 'experiment' block:
            # experiment:
            #   name: ...
            #   seed: ...
            if "experiment" in data and isinstance(data["experiment"], dict):
                exp_dict = data.pop("experiment")
                if "name" in exp_dict and "name" not in data:
                    data["name"] = exp_dict["name"]
                if "seed" in exp_dict and "seed" not in data:
                    data["seed"] = exp_dict["seed"]
        return data

    @model_validator(mode="after")
    def _validate_algorithm_training_compatibility(self) -> ExperimentConfig:
        algo_name = self.algorithm.name.lower()
        is_planner = algo_name in PLANNER_ALGORITHMS
        if not is_planner and self.training is None:
            raise ValueError(
                f"Training configuration ('training') is required for RL algorithm '{self.algorithm.name}'. "
                "Specify 'training.total_timesteps' for RL experiments."
            )
        if is_planner and self.training is not None:
            raise ValueError(
                f"Classical planner '{self.algorithm.name}' does not support a 'training' configuration block. "
                "Planners execute direct path search without training. Remove the 'training' section."
            )
        return self


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate an AdaptiveRL experiment configuration from a YAML file.

    Args:
        config_path: Filepath to the YAML configuration file.

    Returns:
        ExperimentConfig: Validated typed configuration instance.

    Raises:
        ConfigError: If the file is missing, contains invalid YAML, or fails schema validation.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML file at {path}: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Configuration file {path} must contain a YAML mapping/dictionary, got {type(raw_data).__name__}"
        )

    try:
        return ExperimentConfig.model_validate(raw_data)
    except ValidationError as exc:
        formatted_errors = []
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            formatted_errors.append(f"  - [{loc}]: {msg}")
        errors_str = "\n".join(formatted_errors)
        raise ConfigError(f"Configuration validation failed for {path}:\n{errors_str}") from exc


def save_config(config: ExperimentConfig, target_path: str | Path) -> None:
    """Save an experiment configuration to a YAML file.

    Args:
        config: The ExperimentConfig instance to serialize.
        target_path: Destination filepath for the YAML configuration.
    """
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="python", exclude_none=True)
    # Convert Path objects to string for clean YAML representation
    data["output_dir"] = str(data["output_dir"])
    data["log_dir"] = str(data["log_dir"])

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def compute_config_sha256(config: ExperimentConfig) -> str:
    """Compute deterministic SHA-256 hash of the canonical experiment configuration.

    Excludes runtime destination paths ('output_dir', 'log_dir') so that identical
    hyperparameters always yield the exact same cryptographic hash regardless of execution directory.

    Args:
        config: Validated ExperimentConfig instance.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    import hashlib
    import json

    canonical_dict = config.model_dump(mode="json", exclude={"output_dir", "log_dir"})
    serialized = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
