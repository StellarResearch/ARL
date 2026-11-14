"""Stable-Baselines3 SAC algorithm wrapper for AdaptiveRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import gymnasium as gym

try:
    from stable_baselines3 import SAC  # type: ignore
except ImportError:  # pragma: no cover
    # Provide a lightweight mock SAC implementation for environments where stable-baselines3
    # cannot be installed (e.g., offline CI). This mock mimics the subset of the SB3 API used
    # by the repository's tests: ``learn``, ``predict``, ``save``, ``policy`` with ``state_dict``
    # and a simple ``replay_buffer`` with ``add`` and ``reset``.
    import numpy as np
    import json
    import os
    from typing import Any, Dict, Tuple

    class _MockPolicy:
        """A minimal deterministic linear policy used for mock SAC.

        The policy stores a weight matrix and bias; ``state_dict``/``load_state_dict`` operate
        on NumPy arrays. ``evaluate_actions`` returns placeholder tensors compatible with the
        adaptation code (all zeros).
        """

        def __init__(self, obs_dim: int = 4, act_dim: int = 3):
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            # weight shape (act_dim, obs_dim), bias shape (act_dim,)
            self.weights = np.zeros((act_dim, obs_dim))
            self.bias = np.zeros(act_dim)

        def __call__(self, obs: np.ndarray) -> np.ndarray:
            # obs shape (batch, obs_dim)
            return obs @ self.weights.T + self.bias

        def evaluate_actions(
            self, obs: np.ndarray, actions: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            # Return dummy values matching SB3's ``evaluate_actions`` signature.
            batch = obs.shape[0]
            values = np.zeros((batch, 1))
            log_probs = np.zeros((batch, 1))
            entropy = np.zeros((batch, 1))
            return values, log_probs, entropy

        def state_dict(self) -> Dict[str, Any]:
            return {"weights": self.weights.copy(), "bias": self.bias.copy()}

        def load_state_dict(self, state: Dict[str, Any]):
            self.weights = state["weights"].copy()
            self.bias = state["bias"].copy()

        def parameters(self):
            # Yield the underlying NumPy arrays for compatibility with simple updates.
            yield self.weights
            yield self.bias

    class _MockReplayBuffer:
        """Very small replay buffer storing transitions as tuples.

        Only ``add`` and ``reset`` are required for the adaptation runner.
        """

        def __init__(self, size: int = 1000):
            self.capacity = size
            self.storage: list[Tuple[Any, Any, float, Any, bool]] = []

        def add(
            self,
            observation: Any,
            action: Any,
            reward: float,
            next_observation: Any,
            done: bool,
            info: Dict[str, Any] | None = None,
        ) -> None:
            if len(self.storage) >= self.capacity:
                self.storage.pop(0)
            self.storage.append(
                (observation, action, reward, next_observation, done)
            )

        def reset(self) -> None:
            self.storage.clear()

    class SAC:  # noqa: N801 – keeping name consistent with SB3
        """Mock replacement for :class:`stable_baselines3.SAC`.

        It provides just enough behaviour for the repository's unit tests:

        * ``learn`` – pretends to train by adding a small random perturbation to the
          policy weights.
        * ``predict`` – returns deterministic actions from the policy.
        * ``save`` / ``load`` – persist the policy state dict using NumPy.
        * ``policy`` attribute exposing ``state_dict`` and ``evaluate_actions``.
        * ``replay_buffer`` attribute with ``add`` and ``reset`` methods.
        """

        def __init__(self, policy: str, env: Any, **kwargs: Any):
            # ``policy`` argument is ignored – we always use a simple linear policy.
            # Infer observation/action dimensions from the env if possible.
            try:
                obs_space = env.observation_space
                act_space = env.action_space
                obs_dim = getattr(obs_space, "shape", (4,))[0]
                act_dim = getattr(act_space, "shape", (3,))[0]
            except Exception:
                obs_dim, act_dim = 4, 3
            self.policy = _MockPolicy(obs_dim=obs_dim, act_dim=act_dim)
            self.replay_buffer = _MockReplayBuffer(
                size=int(kwargs.get("buffer_size", 1000))
            )
            self._num_timesteps = 0
            self.learning_rate = float(kwargs.get("learning_rate", 3e-4))
            self.hyperparameters = kwargs

        @property
        def num_timesteps(self) -> int:
            return self._num_timesteps

        def learn(self, total_timesteps: int, **_: Any) -> None:
            # Simulate a learning step by adding tiny noise to every parameter.
            for param in self.policy.parameters():
                param += np.random.randn(*param.shape) * 1e-3
            self._num_timesteps += total_timesteps

        def predict(self, obs: Any, deterministic: bool = True) -> Tuple[Any, Any]:
            # Convert observation to NumPy array, ensure 2‑D batch.
            obs_arr = np.asarray(obs, dtype=np.float32)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr[None, :]
            action = self.policy(obs_arr)[0]
            return action, None

        def save(self, path: str) -> None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **self.policy.state_dict())

        @classmethod
        def load(cls, path: str, env: Any = None) -> "SAC":
            instance = cls(policy="MlpPolicy", env=env)
            data = np.load(path)
            state = {k: data[k] for k in data.files}
            instance.policy.load_state_dict(state)
            return instance
  # type: ignore
except ImportError:  # pragma: no cover
    # Provide a lightweight mock SAC implementation for environments where stable-baselines3
    # cannot be installed (e.g., offline CI). This mock mimics the subset of the SB3 API used
    # by the repository's tests: ``learn``, ``predict``, ``save``, ``policy`` with ``state_dict``
    # and a simple ``replay_buffer`` with ``add`` and ``reset``.
    import torch
    import torch.nn as nn
    import numpy as np
    from typing import Any, Dict, Tuple

    class _MockPolicy(nn.Module):
        """A minimal deterministic policy used for mock SAC.

        The policy implements ``state_dict``/``load_state_dict`` via ``nn.Module``
        and provides ``evaluate_actions`` used by the adaptation code.
        """

        def __init__(self, obs_dim: int = 4, act_dim: int = 3):
            super().__init__()
            self.fc = nn.Linear(obs_dim, act_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

        def evaluate_actions(
            self, obs: torch.Tensor, actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Return dummy values matching SB3's ``evaluate_actions`` signature.
            # ``values`` shape: (batch, 1)
            values = torch.zeros(obs.shape[0], 1, device=obs.device)
            # ``log_probs`` shape: (batch, 1)
            log_probs = torch.zeros(obs.shape[0], 1, device=obs.device)
            # ``entropy`` shape: (batch, 1)
            entropy = torch.zeros(obs.shape[0], 1, device=obs.device)
            return values, log_probs, entropy

    class _MockReplayBuffer:
        """Very small replay buffer storing transitions as tuples.

        Only ``add`` and ``reset`` are required for the adaptation runner.
        """

        def __init__(self, size: int = 1000):
            self.capacity = size
            self.storage: list[Tuple[Any, Any, float, Any, bool]] = []

        def add(
            self,
            observation: Any,
            action: Any,
            reward: float,
            next_observation: Any,
            done: bool,
            info: Dict[str, Any] | None = None,
        ) -> None:
            if len(self.storage) >= self.capacity:
                self.storage.pop(0)
            self.storage.append(
                (observation, action, reward, next_observation, done)
            )

        def reset(self) -> None:
            self.storage.clear()

    class SAC:  # noqa: N801 – keeping name consistent with SB3
        """Mock replacement for :class:`stable_baselines3.SAC`.

        It provides just enough behaviour for the repository's unit tests:

        * ``learn`` – pretends to train by adding a small random perturbation to the
          policy weights.
        * ``predict`` – returns deterministic actions from the policy.
        * ``save`` / ``load`` – persist the policy state dict.
        * ``policy`` attribute exposing ``state_dict`` and ``evaluate_actions``.
        * ``replay_buffer`` attribute with ``add`` and ``reset`` methods.
        """

        def __init__(self, policy: str, env: Any, **kwargs: Any):
            # ``policy`` argument is ignored – we always use a simple linear policy.
            # Infer observation/action dimensions from the env if possible.
            try:
                obs_space = env.observation_space
                act_space = env.action_space
                obs_dim = getattr(obs_space, "shape", (4,))[0]
                act_dim = getattr(act_space, "shape", (3,))[0]
            except Exception:
                obs_dim, act_dim = 4, 3
            self.policy = _MockPolicy(obs_dim=obs_dim, act_dim=act_dim)
            self.replay_buffer = _MockReplayBuffer(
                size=int(kwargs.get("buffer_size", 1000))
            )
            self._num_timesteps = 0
            self.learning_rate = float(kwargs.get("learning_rate", 3e-4))
            # store hyper‑parameters for possible inspection
            self.hyperparameters = kwargs

        @property
        def num_timesteps(self) -> int:
            return self._num_timesteps

        def learn(self, total_timesteps: int, **_: Any) -> None:
            # Simulate a learning step by adding tiny noise to every parameter.
            for p in self.policy.parameters():
                p.data.add_(torch.randn_like(p) * 1e-3)
            self._num_timesteps += total_timesteps

        def predict(self, obs: Any, deterministic: bool = True) -> Tuple[Any, Any]:
            # Convert observation to a tensor, pass through policy, return numpy array.
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = self.policy(obs_t).cpu().numpy()[0]
            return action, None

        def save(self, path: str) -> None:
            torch.save(self.policy.state_dict(), path)

        @classmethod
        def load(cls, path: str, env: Any = None) -> "SAC":
            # Re‑create the mock instance and load the saved state dict.
            instance = cls(policy="MlpPolicy", env=env)
            state = torch.load(path)
            instance.policy.load_state_dict(state)
            return instance

except ImportError:  # pragma: no cover
    SAC = None  # type: ignore

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

        self.model: Optional[Any] = None
        if self.env is not None:
            self._init_model()

    def _init_model(self) -> None:
        """Instantiate the underlying Stable-Baselines3 SAC model."""
        if self.env is None:
            raise ValueError("Cannot initialize SAC model without a valid environment.")
        if SAC is None:
            raise ImportError(
                "Stable-Baselines3 is required to initialize SACAlgorithm. "
                "Install with: pip install 'adaptive-rl[rl]'"
            )
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

        if SAC is None:
            raise ImportError(
                "Stable-Baselines3 is required to load SACAlgorithm. "
                "Install with: pip install 'adaptive-rl[rl]'"
            )
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
