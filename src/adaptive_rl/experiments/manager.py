"""Reproducible experiment orchestration manager for AdaptiveRL.

Provides ExperimentManager: a structured orchestration layer that records
provenance metadata, manages output directories, runs training/planning,
evaluates performance, and serializes results to a machine-readable manifest.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import yaml

from adaptive_rl.config import ExperimentConfig, load_config

# ---------------------------------------------------------------------------
# Result data models
# ---------------------------------------------------------------------------


@dataclass
class ExperimentManifest:
    """Machine-readable experiment provenance record.

    Written to ``<output_dir>/manifest.json`` on experiment completion.

    Attributes:
        experiment_id: Unique filesystem-safe experiment identifier.
        created_at: ISO 8601 UTC timestamp of experiment creation.
        algorithm: Algorithm name used in this experiment.
        environment: Environment name used in this experiment.
        seed: Random seed.
        config_path: Relative or absolute path to the source YAML config.
        training_timesteps: Total training timesteps (None for planners).
        git_commit: Short git commit hash at experiment time, if available.
        python_version: Python version string.
        platform_info: OS and CPU information.
        package_versions: Key package versions (adaptive-rl, gymnasium, etc.).
        artifact_paths: Dictionary mapping artifact names to their paths.
        evaluation_status: 'completed', 'failed', or 'skipped'.
        notes: Optional free-text notes.
    """

    experiment_id: str
    created_at: str
    algorithm: str
    environment: str
    seed: int
    config_path: str
    training_timesteps: Optional[int]
    git_commit: str
    python_version: str
    platform_info: str
    package_versions: Dict[str, str]
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    evaluation_status: str = "pending"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to a plain dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write manifest as formatted JSON.

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class ExperimentResult:
    """Structured result returned by :class:`ExperimentManager`.

    Attributes:
        experiment_id: Unique identifier for this experiment.
        output_dir: Root directory containing all artifacts.
        manifest: Experiment provenance manifest.
        metrics: Evaluation metrics dictionary (JSON-serializable).
        training_result: TrainingResult dataclass (None for planners).
        success: Whether the experiment completed without fatal errors.
        error_message: Error description if success is False.
    """

    experiment_id: str
    output_dir: Path
    manifest: ExperimentManifest
    metrics: Dict[str, Any] = field(default_factory=dict)
    training_result: Any = None
    success: bool = True
    error_message: str = ""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_git_commit() -> str:
    """Return the short git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_package_version(package: str) -> str:
    """Return installed version of a package, or 'not_installed'."""
    try:
        import importlib.metadata

        return importlib.metadata.version(package)
    except Exception:
        return "not_installed"


def _make_experiment_id(config: ExperimentConfig) -> str:
    """Generate a deterministic, filesystem-safe experiment identifier.

    Format: ``YYYY-MM-DD_<env>_<algo>_seed<seed>``

    Args:
        config: Experiment configuration.

    Returns:
        Experiment identifier string safe for use as a directory name.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    env = config.environment.name.lower().replace(" ", "_")
    algo = config.algorithm.name.lower().replace(" ", "_")
    seed = config.seed
    return f"{date_str}_{env}_{algo}_seed{seed}"


# ---------------------------------------------------------------------------
# ExperimentManager
# ---------------------------------------------------------------------------


class ExperimentManager:
    """Orchestrates reproducible end-to-end AdaptiveRL experiments.

    Responsibilities:
    - Load and validate configuration.
    - Generate a unique experiment ID.
    - Record provenance metadata (git commit, Python version, packages).
    - Create a structured output directory hierarchy.
    - Launch training (RL algorithms) or planning (classical planners).
    - Evaluate the resulting agent/planner.
    - Save all artifacts (model, metrics JSON, metrics CSV, manifest).
    - Return a structured :class:`ExperimentResult`.

    Output directory structure::

        experiments/results/<experiment_id>/
        ├── config.yaml          — copy of the config used
        ├── manifest.json        — provenance and artifact paths
        ├── metrics.json         — evaluation metrics
        ├── metrics.csv          — tabular metrics (optional)
        ├── evaluation.json      — full evaluation report
        ├── model/               — saved model weights (RL only)
        └── logs/                — training logs (RL only)

    Example::

        from adaptive_rl.experiments.manager import ExperimentManager

        manager = ExperimentManager(base_output_dir=Path("experiments/results"))
        result = manager.run_from_config(Path("configs/gridworld_ppo.yaml"))
        print(result.experiment_id)
    """

    def __init__(
        self,
        base_output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the experiment manager.

        Args:
            base_output_dir: Root directory for all experiment artifacts.
                Defaults to ``experiments/results`` relative to cwd.
        """
        self.base_output_dir = base_output_dir or Path("experiments/results")

    def run_from_config(
        self,
        config_path: Path,
        timesteps_override: Optional[int] = None,
        seed_override: Optional[int] = None,
    ) -> ExperimentResult:
        """Run a complete experiment from a YAML configuration file.

        Args:
            config_path: Path to the experiment YAML configuration.
            timesteps_override: Override training timesteps from config.
            seed_override: Override random seed from config.

        Returns:
            ExperimentResult containing all metadata, metrics, and artifacts.
        """

        try:
            config = load_config(config_path)
        except Exception as exc:
            experiment_id = f"failed_{int(time.time())}"
            manifest = self._make_manifest(
                experiment_id=experiment_id,
                config=None,
                config_path=str(config_path),
            )
            manifest.evaluation_status = "failed"
            return ExperimentResult(
                experiment_id=experiment_id,
                output_dir=self.base_output_dir / experiment_id,
                manifest=manifest,
                success=False,
                error_message=f"Config loading failed: {exc}",
            )

        if timesteps_override is not None and config.training is not None:
            config.training.total_timesteps = timesteps_override
        if seed_override is not None:
            config.seed = seed_override

        return self.run(config=config, config_path=config_path)

    def run(
        self,
        config: ExperimentConfig,
        config_path: Optional[Path] = None,
    ) -> ExperimentResult:
        """Run a complete experiment from an already-loaded ExperimentConfig.

        Args:
            config: Validated experiment configuration.
            config_path: Optional original config file path (for manifest).

        Returns:
            ExperimentResult containing all metadata, metrics, and artifacts.
        """
        experiment_id = _make_experiment_id(config)
        output_dir = self.base_output_dir / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "model").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "plots").mkdir(exist_ok=True)

        # Save config copy
        config_copy_path = output_dir / "config.yaml"
        self._save_config_copy(config, config_copy_path)

        # Build manifest
        manifest = self._make_manifest(
            experiment_id=experiment_id,
            config=config,
            config_path=str(config_path or config_copy_path),
        )
        manifest.artifact_paths["config"] = str(config_copy_path)

        algo_name = config.algorithm.name.lower()
        is_planner = algo_name in ("astar", "rrt_star", "rrt")

        if is_planner:
            result = self._run_planner_experiment(config, output_dir, manifest)
        else:
            result = self._run_rl_experiment(config, output_dir, manifest)

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        manifest.save(manifest_path)
        result.manifest = manifest

        return result

    # ------------------------------------------------------------------
    # RL experiment
    # ------------------------------------------------------------------

    def _run_rl_experiment(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        manifest: ExperimentManifest,
    ) -> ExperimentResult:
        """Run training + evaluation for an RL algorithm.

        Args:
            config: Experiment configuration.
            output_dir: Root output directory for this experiment.
            manifest: Manifest to populate with artifact paths.

        Returns:
            ExperimentResult with training and evaluation outputs.
        """
        from adaptive_rl.training import get_trainer

        # Override output paths to the experiment directory
        config.output_dir = output_dir
        config.log_dir = output_dir / "logs"

        training_result = None
        metrics: Dict[str, Any] = {}
        error_msg = ""
        success = True

        try:
            trainer = get_trainer(config=config)
            training_result = trainer.fit()

            model_path = training_result.final_model_path
            manifest.artifact_paths["model"] = str(model_path)
            manifest.training_timesteps = training_result.total_timesteps

            # Evaluate
            from adaptive_rl.algorithms.base import BaseAlgorithm
            from adaptive_rl.algorithms.ppo import PPOAlgorithm
            from adaptive_rl.algorithms.sac import SACAlgorithm
            from adaptive_rl.environments.registry import make_env
            from adaptive_rl.evaluation.evaluator import Evaluator

            env = make_env(config.environment.name, **config.environment.parameters)

            algo_name = config.algorithm.name.lower()
            algo: BaseAlgorithm
            if algo_name == "ppo":
                algo = PPOAlgorithm.from_pretrained(model_path, env=env)
            elif algo_name == "sac":
                algo = SACAlgorithm.from_pretrained(model_path, env=env)
            else:
                algo = PPOAlgorithm.from_pretrained(model_path, env=env)

            evaluator = Evaluator(algorithm=algo, env=env)
            eval_metrics = evaluator.evaluate(
                num_episodes=config.evaluation.eval_episodes,
                deterministic=config.evaluation.deterministic,
                base_seed=config.seed + 10000,  # Separate evaluation seeds
            )

            from adaptive_rl.evaluation.metrics import StandardizedExperimentMetrics

            std_metrics = StandardizedExperimentMetrics.from_rl_metrics(eval_metrics)
            metrics = eval_metrics.model_dump()
            metrics["standardized"] = std_metrics.model_dump()
            for k, v in std_metrics.model_dump().items():
                if k != "additional_metrics" and k not in metrics:
                    metrics[k] = v

            manifest.evaluation_status = "completed"

            # Save metrics
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            manifest.artifact_paths["metrics"] = str(metrics_path)

            # Save CSV
            csv_path = output_dir / "metrics.csv"
            self._save_metrics_csv(metrics, csv_path)
            manifest.artifact_paths["metrics_csv"] = str(csv_path)

            # Full evaluation report
            eval_path = output_dir / "evaluation.json"
            evaluator.save_report(eval_metrics, eval_path)
            manifest.artifact_paths["evaluation"] = str(eval_path)

            env.close()

        except Exception as exc:
            success = False
            error_msg = str(exc)
            manifest.evaluation_status = "failed"
            manifest.notes = f"Error: {exc}"

        return ExperimentResult(
            experiment_id=manifest.experiment_id,
            output_dir=output_dir,
            manifest=manifest,
            metrics=metrics,
            training_result=training_result,
            success=success,
            error_message=error_msg,
        )

    # ------------------------------------------------------------------
    # Planner experiment
    # ------------------------------------------------------------------

    def _run_planner_experiment(
        self,
        config: ExperimentConfig,
        output_dir: Path,
        manifest: ExperimentManifest,
    ) -> ExperimentResult:
        """Run evaluation for a classical planner (no training phase).

        Args:
            config: Experiment configuration.
            output_dir: Root output directory for this experiment.
            manifest: Manifest to populate with artifact paths.

        Returns:
            ExperimentResult with planner evaluation outputs.
        """
        from adaptive_rl.environments.registry import make_env
        from adaptive_rl.planners.adapter import PlannerAdapter
        from adaptive_rl.planners.astar import AStarPlanner

        metrics: Dict[str, Any] = {}
        error_msg = ""
        success = True

        try:
            env = make_env(config.environment.name, **config.environment.parameters)

            from adaptive_rl.planners.astar import AStarPlanner
            from adaptive_rl.planners.rrt_star import RRTStarPlanner

            planner: Union[AStarPlanner, RRTStarPlanner]
            algo_name = config.algorithm.name.lower()
            if algo_name == "astar":
                planner = AStarPlanner()
            elif algo_name in ("rrt_star", "rrt*", "rrt"):
                planner = RRTStarPlanner(seed=config.seed, **config.algorithm.parameters)
            else:
                raise ValueError(f"Unknown planner algorithm: '{algo_name}'")

            adapter = PlannerAdapter(planner=planner, env=env)  # type: ignore[arg-type]
            planner_metrics = adapter.evaluate(
                num_episodes=config.evaluation.eval_episodes,
                base_seed=config.seed,
            )

            from adaptive_rl.evaluation.metrics import StandardizedExperimentMetrics

            std_metrics = StandardizedExperimentMetrics.from_planner_metrics(planner_metrics)
            metrics = planner_metrics.to_dict()
            metrics["standardized"] = std_metrics.model_dump()
            for k, v in std_metrics.model_dump().items():
                if k != "additional_metrics" and k not in metrics:
                    metrics[k] = v

            manifest.evaluation_status = "completed"
            manifest.training_timesteps = None

            # Save metrics
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            manifest.artifact_paths["metrics"] = str(metrics_path)

            # Save CSV
            csv_path = output_dir / "metrics.csv"
            self._save_metrics_csv(metrics, csv_path)
            manifest.artifact_paths["metrics_csv"] = str(csv_path)

            env.close()

        except Exception as exc:
            success = False
            error_msg = str(exc)
            manifest.evaluation_status = "failed"
            manifest.notes = f"Error: {exc}"

        return ExperimentResult(
            experiment_id=manifest.experiment_id,
            output_dir=output_dir,
            manifest=manifest,
            metrics=metrics,
            training_result=None,
            success=success,
            error_message=error_msg,
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _make_manifest(
        self,
        experiment_id: str,
        config: Optional[ExperimentConfig],
        config_path: str,
    ) -> ExperimentManifest:
        """Build an experiment manifest from current runtime information."""
        try:
            rel_config_path = str(Path(config_path).resolve().relative_to(Path.cwd().resolve()))
        except Exception:
            rel_config_path = str(config_path)

        return ExperimentManifest(
            experiment_id=experiment_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            algorithm=config.algorithm.name if config else "unknown",
            environment=config.environment.name if config else "unknown",
            seed=config.seed if config else -1,
            config_path=rel_config_path,
            training_timesteps=(
                config.training.total_timesteps if config and config.training is not None else None
            ),
            git_commit=_get_git_commit(),
            python_version=sys.version,
            platform_info=f"{platform.system()} {platform.release()} {platform.machine()}",
            package_versions={
                "adaptive-rl": _get_package_version("adaptive-rl"),
                "gymnasium": _get_package_version("gymnasium"),
                "stable-baselines3": _get_package_version("stable-baselines3"),
                "torch": _get_package_version("torch"),
                "pydantic": _get_package_version("pydantic"),
            },
        )

    @staticmethod
    def _save_config_copy(config: ExperimentConfig, path: Path) -> None:
        """Save a copy of the experiment config as YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = config.model_dump(mode="python")
        data["output_dir"] = str(data["output_dir"])
        data["log_dir"] = str(data["log_dir"])
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @staticmethod
    def _save_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
        """Save flat scalar metrics as a single-row CSV file.

        List values are excluded from the CSV; only scalar values are written.

        Args:
            metrics: Metrics dictionary.
            path: Destination CSV path.
        """
        import csv

        scalar_metrics: Dict[str, Any] = {}
        for k, v in metrics.items():
            if k in (
                "standardized",
                "all_path_lengths",
                "all_planning_times",
                "additional_metrics",
            ):
                continue
            if isinstance(v, bool):
                scalar_metrics[k] = int(v)
            elif isinstance(v, (int, float, str)):
                scalar_metrics[k] = v
            elif v is None:
                scalar_metrics[k] = ""

        if not scalar_metrics:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(scalar_metrics.keys()))
            writer.writeheader()
            writer.writerow({k: scalar_metrics[k] for k in sorted(scalar_metrics.keys())})

    # ------------------------------------------------------------------
    # Experiment listing / inspection utilities
    # ------------------------------------------------------------------

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all completed experiments in the base output directory.

        Returns:
            List of manifest dictionaries, sorted by creation time descending.
            Empty list if no experiments have been run.
        """
        experiments: List[Dict[str, Any]] = []
        if not self.base_output_dir.exists():
            return experiments

        for exp_dir in sorted(self.base_output_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            manifest_path = exp_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, encoding="utf-8") as f:
                        data = json.load(f)
                    experiments.append(data)
                except Exception:
                    experiments.append(
                        {"experiment_id": exp_dir.name, "error": "manifest unreadable"}
                    )

        # Sort newest first
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load and return the manifest for a specific experiment.

        Args:
            experiment_id: Experiment identifier string.

        Returns:
            Manifest dictionary, or None if not found.
        """
        manifest_path = self.base_output_dir / experiment_id / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, encoding="utf-8") as f:
                return cast(Optional[Dict[str, Any]], json.load(f))
        except Exception:
            return None

    def get_metrics(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load and return metrics for a specific experiment.

        Args:
            experiment_id: Experiment identifier string.

        Returns:
            Metrics dictionary, or None if metrics file not found.
        """
        metrics_path = self.base_output_dir / experiment_id / "metrics.json"
        if not metrics_path.exists():
            return None
        try:
            with open(metrics_path, encoding="utf-8") as f:
                return cast(Optional[Dict[str, Any]], json.load(f))
        except Exception:
            return None
