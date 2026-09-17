"""Algorithm registry and factory system for AdaptiveRL.

Provides a unified registry capable of resolving and inspecting both trainable
RL algorithms (PPO, SAC) and deterministic planners (A*), distinguishing
between algorithm types via capability metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class AlgorithmKind(str, Enum):
    """Classification of algorithm types in the registry.

    Values:
        RL_POLICY: Stochastic policy gradient or actor-critic RL algorithm
            (trainable via SB3, produces a learned policy).
        PLANNER: Deterministic classical planner (A*, RRT*, etc.).
            Not trained; computes paths using world-model knowledge.
    """

    RL_POLICY = "rl_policy"
    PLANNER = "planner"


@dataclass
class AlgorithmMetadata:
    """Metadata record for a registered algorithm or planner.

    Attributes:
        name: Canonical lowercase identifier used for registry lookup.
        kind: Whether this is an RL policy or deterministic planner.
        description: Human-readable description.
        action_space: Type of action space supported (e.g. 'discrete', 'continuous', 'any').
        trainable: True if the algorithm supports a training loop.
        class_name: Fully qualified or simple class name.
        hyperparameters: Default hyperparameter documentation.
        tags: Searchable tags.
    """

    name: str
    kind: AlgorithmKind
    description: str = ""
    action_space: str = "any"
    trainable: bool = True
    class_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class AlgorithmRegistryError(Exception):
    """Exception raised for algorithm registry operation failures."""

    pass


class AlgorithmRegistry:
    """Registry maintaining available algorithms, factories, and associated metadata.

    Mirrors the design of :class:`EnvironmentRegistry` for consistency.
    Distinguishes between trainable RL algorithms and deterministic planners via
    :class:`AlgorithmKind` capability metadata.

    Example::

        from adaptive_rl.algorithms.registry import algorithm_registry

        # List all registered algorithms
        names = algorithm_registry.list_algorithms()

        # Retrieve metadata
        meta = algorithm_registry.get_metadata("ppo")

        # Instantiate an algorithm
        PPOAlgorithm = algorithm_registry.get_factory("ppo")
        algo = PPOAlgorithm(env=env)
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[..., Any]] = {}
        self._metadata: Dict[str, AlgorithmMetadata] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        metadata: Optional[AlgorithmMetadata] = None,
    ) -> None:
        """Register an algorithm factory and its metadata.

        Args:
            name: Unique lowercase identifier (e.g. 'ppo', 'sac', 'astar').
            factory: Callable returning an algorithm or planner instance.
            metadata: Optional metadata describing the algorithm.

        Raises:
            AlgorithmRegistryError: If name is empty, already registered,
                or factory is not callable.
        """
        if not name or not isinstance(name, str):
            raise AlgorithmRegistryError("Algorithm name must be a non-empty string.")

        clean = name.strip().lower()
        if clean in self._factories:
            raise AlgorithmRegistryError(
                f"Algorithm '{clean}' is already registered. "
                "Use a unique name or clear the registry first."
            )
        if not callable(factory):
            raise AlgorithmRegistryError(f"Factory for algorithm '{clean}' must be callable.")

        self._factories[clean] = factory
        if metadata is not None:
            self._metadata[clean] = metadata
        else:
            self._metadata[clean] = AlgorithmMetadata(
                name=clean,
                kind=AlgorithmKind.RL_POLICY,
                description=f"Algorithm '{clean}' (no metadata provided).",
            )

    def get_factory(self, name: str) -> Callable[..., Any]:
        """Retrieve the factory callable for a registered algorithm.

        Args:
            name: Algorithm identifier.

        Returns:
            Callable factory for the algorithm.

        Raises:
            AlgorithmRegistryError: If name is not registered.
        """
        clean = name.strip().lower()
        if clean not in self._factories:
            available = ", ".join(sorted(self._factories.keys())) or "none"
            raise AlgorithmRegistryError(
                f"Unknown algorithm '{clean}'. Available registered algorithms: {available}"
            )
        return self._factories[clean]

    def get_metadata(self, name: str) -> AlgorithmMetadata:
        """Retrieve metadata for a registered algorithm.

        Args:
            name: Algorithm identifier.

        Returns:
            AlgorithmMetadata for the algorithm.

        Raises:
            AlgorithmRegistryError: If name is not registered.
        """
        clean = name.strip().lower()
        if clean not in self._metadata:
            available = ", ".join(sorted(self._metadata.keys())) or "none"
            raise AlgorithmRegistryError(
                f"Unknown algorithm '{clean}'. Available registered algorithms: {available}"
            )
        return self._metadata[clean]

    def list_algorithms(self) -> List[str]:
        """Return sorted list of registered algorithm names."""
        return sorted(self._factories.keys())

    def list_by_kind(self, kind: AlgorithmKind) -> List[str]:
        """Return sorted list of registered algorithm names matching a kind.

        Args:
            kind: AlgorithmKind to filter by.

        Returns:
            Sorted list of matching algorithm names.
        """
        return sorted(name for name, meta in self._metadata.items() if meta.kind == kind)

    def list_all_metadata(self) -> Dict[str, AlgorithmMetadata]:
        """Return mapping of all registered algorithm names to metadata."""
        return {k: self._metadata[k] for k in sorted(self._metadata.keys())}

    def is_trainable(self, name: str) -> bool:
        """Return True if the named algorithm is a trainable RL policy.

        Args:
            name: Algorithm identifier.

        Returns:
            True if trainable, False for deterministic planners.
        """
        return self.get_metadata(name).trainable

    def clear(self) -> None:
        """Clear all registered algorithms (primarily for test isolation)."""
        self._factories.clear()
        self._metadata.clear()


# ---------------------------------------------------------------------------
# Global default registry instance
# ---------------------------------------------------------------------------
algorithm_registry = AlgorithmRegistry()


def _register_defaults() -> None:
    """Register the built-in algorithms into the global registry."""
    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.algorithms.sac import SACAlgorithm
    from adaptive_rl.planners.astar import AStarPlanner

    if "ppo" not in algorithm_registry.list_algorithms():
        algorithm_registry.register(
            "ppo",
            PPOAlgorithm,
            AlgorithmMetadata(
                name="ppo",
                kind=AlgorithmKind.RL_POLICY,
                description=(
                    "Proximal Policy Optimization (PPO) — on-policy actor-critic algorithm. "
                    "Supports discrete and continuous action spaces. Wraps Stable-Baselines3 PPO."
                ),
                action_space="any",
                trainable=True,
                class_name="PPOAlgorithm",
                hyperparameters={
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.0,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
                tags=["on-policy", "actor-critic", "sb3", "discrete", "continuous"],
            ),
        )

    if "sac" not in algorithm_registry.list_algorithms():
        algorithm_registry.register(
            "sac",
            SACAlgorithm,
            AlgorithmMetadata(
                name="sac",
                kind=AlgorithmKind.RL_POLICY,
                description=(
                    "Soft Actor-Critic (SAC) — off-policy maximum entropy actor-critic algorithm. "
                    "Supports continuous action spaces. Wraps Stable-Baselines3 SAC."
                ),
                action_space="continuous",
                trainable=True,
                class_name="SACAlgorithm",
                hyperparameters={
                    "learning_rate": 3e-4,
                    "buffer_size": 100_000,
                    "learning_starts": 100,
                    "batch_size": 256,
                    "tau": 0.005,
                    "gamma": 0.99,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "ent_coef": "auto",
                },
                tags=["off-policy", "actor-critic", "sb3", "continuous", "maximum-entropy"],
            ),
        )

    if "astar" not in algorithm_registry.list_algorithms():
        algorithm_registry.register(
            "astar",
            AStarPlanner,
            AlgorithmMetadata(
                name="astar",
                kind=AlgorithmKind.PLANNER,
                description=(
                    "A* grid navigation planner. Deterministic shortest-path algorithm using "
                    "Manhattan distance heuristic. Supports discrete GridWorld environments only."
                ),
                action_space="discrete",
                trainable=False,
                class_name="AStarPlanner",
                hyperparameters={"heuristic": "manhattan"},
                tags=["planner", "deterministic", "shortest-path", "gridworld", "classical"],
            ),
        )

    if "rrt_star" not in algorithm_registry.list_algorithms():
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        algorithm_registry.register(
            "rrt_star",
            RRTStarPlanner,
            AlgorithmMetadata(
                name="rrt_star",
                kind=AlgorithmKind.PLANNER,
                description=(
                    "RRT* continuous 2D motion planner. Sampling-based kinodynamic-free planner "
                    "with tree rewiring for optimal paths in continuous spaces. "
                    "Supports ContinuousNavigation2D environments."
                ),
                action_space="continuous",
                trainable=False,
                class_name="RRTStarPlanner",
                hyperparameters={
                    "step_size": 0.5,
                    "max_iterations": 1500,
                    "goal_bias": 0.1,
                    "search_radius": 1.5,
                },
                tags=["planner", "sampling-based", "continuous", "rrt-star", "navigation-2d"],
            ),
        )


_register_defaults()

# Public convenience API (mirrors environment registry pattern)
register_algorithm = algorithm_registry.register
get_algorithm_factory = algorithm_registry.get_factory
get_algorithm_metadata = algorithm_registry.get_metadata
list_algorithms = algorithm_registry.list_algorithms
list_algorithms_by_kind = algorithm_registry.list_by_kind
list_all_algorithm_metadata = algorithm_registry.list_all_metadata
