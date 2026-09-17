"""Experiment runner orchestrating training and generalization evaluation on unseen layouts."""

from __future__ import annotations

from typing import Any, Dict, Optional

from adaptive_rl.config import ExperimentConfig
from adaptive_rl.environments.registry import make_env
from adaptive_rl.environments.seeded_wrapper import TrainingDistributionWrapper
from adaptive_rl.evaluation.generalization import (
    GeneralizationDistribution,
    GeneralizationEvaluator,
    GeneralizationReport,
)
from adaptive_rl.experiments.runner import BaseExperimentRunner
from adaptive_rl.training.trainer import PPOTrainer, SACTrainer


class GeneralizationExperimentRunner(BaseExperimentRunner):
    """Orchestrates end-to-end training on a restricted seed distribution and evaluates generalization."""

    def __init__(
        self,
        distribution: Optional[GeneralizationDistribution] = None,
        train_seeds_range: tuple[int, int] = (100, 150),
        test_seeds_range: tuple[int, int] = (200, 250),
    ) -> None:
        """Initialize generalization experiment runner.

        Args:
            distribution: Optional preconfigured GeneralizationDistribution.
            train_seeds_range: Range of seeds used to construct training distribution if distribution not provided.
            test_seeds_range: Range of unseen seeds used for testing if distribution not provided.
        """
        if distribution is not None:
            self.distribution = distribution
        else:
            self.distribution = GeneralizationDistribution.from_ranges(
                train_range=train_seeds_range,
                test_range=test_seeds_range,
                description="Standard disjoint generalization train/test split",
            )

    def run(self, config: ExperimentConfig) -> GeneralizationReport:
        """Execute training strictly on training seed distribution and evaluate on unseen test distribution.

        Args:
            config: Validated experiment configuration.

        Returns:
            GeneralizationReport: Comparative metrics on train vs unseen environments.
        """
        # 1. Instantiate base environment and wrap in TrainingDistributionWrapper
        env_params = dict(config.environment.parameters)
        if "max_steps" not in env_params:
            env_params["max_steps"] = config.environment.max_steps

        raw_train_env = make_env(config.environment.name, **env_params)
        train_env = TrainingDistributionWrapper(
            env=raw_train_env,
            seeds=self.distribution.train_seeds,
            shuffle=True,
            rng_seed=config.seed,
        )

        # 2. Select trainer based on algorithm
        algo_name = config.algorithm.name.lower()
        trainer: PPOTrainer | SACTrainer
        if algo_name == "ppo":
            trainer = PPOTrainer(config=config, env=train_env)
        elif algo_name == "sac":
            trainer = SACTrainer(config=config, env=train_env)
        else:
            raise ValueError(f"Unsupported algorithm for generalization runner: {algo_name}")

        # 3. Train agent strictly within the training seed distribution
        trainer.fit()

        # 4. Instantiate generalization evaluator
        eval_env = make_env(config.environment.name, **env_params)
        evaluator = GeneralizationEvaluator(
            algorithm=trainer.algorithm,
            env=eval_env,
            env_name=config.environment.name,
            env_kwargs=env_params,
        )

        # 5. Evaluate on train distribution and unseen test distribution
        metadata: Dict[str, Any] = {
            "total_training_timesteps": config.training.total_timesteps
            if config.training is not None
            else 0,
            "algorithm": config.algorithm.name,
            "seed": config.seed,
            "unique_training_seeds_sampled": len(set(train_env.sampled_seeds_history)),
        }
        report = evaluator.evaluate_generalization(
            distribution=self.distribution,
            experiment_name=config.name,
            deterministic=config.evaluation.deterministic,
            metadata=metadata,
        )

        # 6. Save report artifact
        report_path = config.output_dir / "generalization" / f"{config.name}_report.json"
        report.save_json(report_path)

        train_env.close()
        eval_env.close()

        return report
