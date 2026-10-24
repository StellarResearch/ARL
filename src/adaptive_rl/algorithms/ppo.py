"""Stable-Baselines3 PPO algorithm wrapper for AdaptiveRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import gymnasium as gym
try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore


from adaptive_rl.algorithms.base import BaseAlgorithm


class PPOAlgorithm(BaseAlgorithm):
    """Proximal Policy Optimization (PPO) algorithm wrapper.

    Integrates Stable-Baselines3 PPO with AdaptiveRL's unified BaseAlgorithm interface,
    providing standardized model initialization, training, action prediction, and persistence.
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seed: Optional[int] = None,
        verbose: int = 0,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize PPO algorithm with environment and hyperparameters.

        Args:
            env: Gymnasium environment to train on (optional if loading a model).
            policy: Policy network architecture string (default: 'MlpPolicy').
            learning_rate: Optimizer learning rate.
            n_steps: Number of steps to run for each environment per update.
            batch_size: Minibatch size for gradient updates.
            n_epochs: Number of epochs when optimizing the surrogate loss.
            gamma: Discount factor.
            gae_lambda: Factor for trade-off of bias vs variance for GAE.
            clip_range: Clipping parameter for surrogate objective.
            ent_coef: Entropy coefficient for exploration encouragement.
            vf_coef: Value function coefficient for loss calculation.
            max_grad_norm: Maximum value for the gradient clipping.
            seed: Seed for pseudo-random generators.
            verbose: Verbosity level (0: no output, 1: info, 2: debug).
            device: Device (cpu, cuda, auto) to run computation on.
            **kwargs: Additional hyperparameters passed to Stable-Baselines3 PPO.
        """
        self.env = env
        self.policy_name = policy
        self.hyperparameters: dict[str, Any] = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "verbose": verbose,
            "device": device,
            **kwargs,
        }

        self.model: Optional[PPO] = None
        if self.env is not None:
            self._init_model()

    def _init_model(self) -> None:
        """Instantiate the underlying Stable-Baselines3 PPO model."""
        if self.env is None:
            raise ValueError("Cannot initialize PPO model without a valid environment.")
        if PPO is None:
            raise ImportError(
                "Stable-Baselines3 is required to initialize PPOAlgorithm. "
                "Install with: pip install 'adaptive-rl[rl]'"
            )
        self.model = PPO(
            policy=self.policy_name,
            env=self.env,
            **self.hyperparameters,
        )

    def train(self, total_timesteps: int, callback: Any = None) -> None:
        """Train the algorithm for the specified number of environment steps.

        Args:
            total_timesteps: Number of timesteps to train.
            callback: Optional callback or list of callbacks for logging and checkpoints.
        """
        if self.model is None:
            if self.env is not None:
                self._init_model()
            else:
                raise RuntimeError(
                    "Cannot train: model is not initialized and no environment was provided."
                )

        if self.model is None:
            raise RuntimeError("Failed to initialize model.")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[Any, Any]:
        """Generate an action given an environment observation.

        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic action selection.

        Returns:
            Tuple[Any, Any]: (action, internal_state)
        """
        if self.model is None:
            raise RuntimeError("Cannot predict: model is not initialized or loaded.")
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str | Path) -> None:
        """Save algorithm weights and hyperparameters to disk.

        Args:
            path: Destination file path (e.g. 'experiments/models/ppo_agent.zip').
        """
        if self.model is None:
            raise RuntimeError("Cannot save: model is not initialized.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))

    def load(self, path: str | Path, env: Optional[gym.Env] = None) -> None:
        """Load algorithm weights from disk into this instance.

        Args:
            path: Path to saved model file.
            env: Optional environment to associate with loaded model.
        """
        load_path = Path(path)
        if not load_path.exists():
            # Check if .zip extension was omitted
            zip_path = load_path.with_suffix(".zip")
            if zip_path.exists():
                load_path = zip_path
            else:
                raise FileNotFoundError(f"Model file not found at: {path}")

        if PPO is None:
            raise ImportError(
                "Stable-Baselines3 is required to load PPOAlgorithm. "
                "Install with: pip install 'adaptive-rl[rl]'"
            )
        target_env = env if env is not None else self.env
        self.model = PPO.load(str(load_path), env=target_env)
        if env is not None:
            self.env = env

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        env: Optional[gym.Env] = None,
        **kwargs: Any,
    ) -> PPOAlgorithm:
        """Create a new PPOAlgorithm instance loaded from a saved model.

        Args:
            path: Path to saved model file.
            env: Optional Gymnasium environment.
            **kwargs: Overrides for loaded model configuration.

        Returns:
            PPOAlgorithm: Loaded algorithm instance.
        """
        instance = cls(env=env)
        instance.load(path, env=env)
        return instance

    @property
    def num_timesteps(self) -> int:
        """Return total timesteps trained so far."""
        if self.model is None:
            return 0
        return int(self.model.num_timesteps)
