"""Stable-Baselines3 SAC algorithm wrapper for AdaptiveRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import gymnasium as gym
from stable_baselines3 import SAC

from adaptive_rl.algorithms.base import BaseAlgorithm


class SACAlgorithm(BaseAlgorithm):
    """Soft Actor-Critic (SAC) algorithm wrapper for continuous control.

    Integrates Stable-Baselines3 SAC with AdaptiveRL's unified BaseAlgorithm interface,
    providing standardized model initialization, training, action prediction, and persistence.
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        seed: Optional[int] = None,
        verbose: int = 0,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        """Initialize SAC algorithm with environment and hyperparameters.

        Args:
            env: Gymnasium environment with continuous action space.
            policy: Policy architecture string (default: 'MlpPolicy').
            learning_rate: Learning rate for actor and critic networks.
            buffer_size: Replay buffer capacity in transitions.
            learning_starts: Steps to collect before starting gradient updates.
            batch_size: Minibatch size sampled from replay buffer.
            tau: Target network soft update smoothing coefficient.
            gamma: Discount factor.
            train_freq: Frequency (in environment steps) of gradient updates.
            gradient_steps: Number of gradient steps per update.
            ent_coef: Entropy regularization coefficient ('auto' or fixed float).
            seed: Pseudo-random generator seed.
            verbose: Verbosity level (0: silent, 1: info, 2: debug).
            device: Computation device ('cpu', 'cuda', 'auto').
            **kwargs: Additional hyperparameters passed to Stable-Baselines3 SAC.
        """
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be a positive integer, got {buffer_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if learning_starts < 0:
            raise ValueError(f"learning_starts must be non-negative, got {learning_starts}")
        if not (0.0 < tau <= 1.0):
            raise ValueError(f"tau must be in range (0.0, 1.0], got {tau}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be between 0.0 and 1.0, got {gamma}")
        if train_freq < 1:
            raise ValueError(f"train_freq must be a positive integer, got {train_freq}")
        if gradient_steps < 1 and gradient_steps != -1:
            raise ValueError(f"gradient_steps must be >= 1 or -1, got {gradient_steps}")
        if isinstance(ent_coef, (int, float)):
            if ent_coef < 0.0:
                raise ValueError(f"ent_coef must be non-negative, got {ent_coef}")
        elif isinstance(ent_coef, str):
            if not (ent_coef == "auto" or ent_coef.startswith("auto_")):
                raise ValueError(
                    f"Invalid ent_coef string '{ent_coef}'. Expected 'auto' or 'auto_<target_entropy>'"
                )
        else:
            raise ValueError(
                f"ent_coef must be a float or string ('auto'), got {type(ent_coef).__name__}"
            )

        supported_kwargs = {
            "policy_kwargs",
            "tensorboard_log",
            "stats_window_size",
            "action_noise",
            "replay_buffer_class",
            "replay_buffer_kwargs",
            "optimize_memory_usage",
            "target_update_interval",
            "target_entropy",
            "use_sde",
            "sde_sample_freq",
            "use_sde_at_warmup",
        }
        unsupported = set(kwargs.keys()) - supported_kwargs
        if unsupported:
            raise ValueError(f"Unsupported hyperparameter(s) for SAC: {sorted(unsupported)}")

        self.env = env
        self.policy_name = policy
        self.hyperparameters: dict[str, Any] = {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "tau": tau,
            "gamma": gamma,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "ent_coef": ent_coef,
            "seed": seed,
            "verbose": verbose,
            "device": device,
            **kwargs,
        }

        self.model: Optional[SAC] = None
        if self.env is not None:
            self._init_model()

    def _init_model(self) -> None:
        """Instantiate the underlying Stable-Baselines3 SAC model."""
        if self.env is None:
            raise ValueError("Cannot initialize SAC model without a valid environment.")
        self.model = SAC(
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
        """Generate a continuous action given an environment observation.

        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic mode (actor mean).

        Returns:
            Tuple[Any, Any]: (action, internal_state)
        """
        if self.model is None:
            raise RuntimeError("Cannot predict: model is not initialized or loaded.")
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str | Path) -> None:
        """Save algorithm weights and hyperparameters to disk.

        Args:
            path: Destination file path (e.g. 'experiments/models/sac_agent.zip').
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
            zip_path = load_path.with_suffix(".zip")
            if zip_path.exists():
                load_path = zip_path
            else:
                raise FileNotFoundError(f"Model file not found at: {path}")

        target_env = env if env is not None else self.env
        self.model = SAC.load(str(load_path), env=target_env)
        if env is not None:
            self.env = env

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        env: Optional[gym.Env] = None,
        **kwargs: Any,
    ) -> SACAlgorithm:
        """Create a new SACAlgorithm instance loaded from a saved model.

        Args:
            path: Path to saved model file.
            env: Optional Gymnasium environment.
            **kwargs: Overrides for loaded model configuration.

        Returns:
            SACAlgorithm: Loaded algorithm instance.
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
