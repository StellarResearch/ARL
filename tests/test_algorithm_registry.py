"""Tests for the algorithm registry."""

from __future__ import annotations

import pytest

from adaptive_rl.algorithms.registry import (
    AlgorithmKind,
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmRegistryError,
    get_algorithm_factory,
    get_algorithm_metadata,
    list_algorithms,
    list_algorithms_by_kind,
    list_all_algorithm_metadata,
)

# ---------------------------------------------------------------------------
# Registry unit tests (isolated registry instances)
# ---------------------------------------------------------------------------


class TestAlgorithmRegistry:
    """Tests for the AlgorithmRegistry class using isolated instances."""

    def setup_method(self) -> None:
        """Create a fresh registry for each test."""
        self.registry = AlgorithmRegistry()

    def _dummy_factory(self, **kwargs):
        return object()

    def test_register_and_lookup(self) -> None:
        """Basic registration and factory retrieval."""
        self.registry.register(
            "my_algo",
            self._dummy_factory,
            AlgorithmMetadata(name="my_algo", kind=AlgorithmKind.RL_POLICY),
        )
        factory = self.registry.get_factory("my_algo")
        assert callable(factory)

    def test_list_algorithms(self) -> None:
        """list_algorithms returns sorted list of registered names."""
        self.registry.register("b_algo", self._dummy_factory)
        self.registry.register("a_algo", self._dummy_factory)
        names = self.registry.list_algorithms()
        assert names == ["a_algo", "b_algo"]

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice raises AlgorithmRegistryError."""
        self.registry.register("my_algo", self._dummy_factory)
        with pytest.raises(AlgorithmRegistryError, match="already registered"):
            self.registry.register("my_algo", self._dummy_factory)

    def test_empty_name_raises(self) -> None:
        """Empty algorithm name raises AlgorithmRegistryError."""
        with pytest.raises(AlgorithmRegistryError, match="non-empty string"):
            self.registry.register("", self._dummy_factory)

    def test_non_callable_factory_raises(self) -> None:
        """Non-callable factory raises AlgorithmRegistryError."""
        with pytest.raises(AlgorithmRegistryError, match="callable"):
            self.registry.register("my_algo", "not_callable")  # type: ignore

    def test_unknown_algorithm_raises(self) -> None:
        """Lookup of unregistered algorithm raises AlgorithmRegistryError."""
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm"):
            self.registry.get_factory("nonexistent")

    def test_get_metadata(self) -> None:
        """Metadata retrieval returns the registered AlgorithmMetadata."""
        meta = AlgorithmMetadata(
            name="my_algo",
            kind=AlgorithmKind.RL_POLICY,
            description="Test algorithm",
            trainable=True,
        )
        self.registry.register("my_algo", self._dummy_factory, meta)
        retrieved = self.registry.get_metadata("my_algo")
        assert retrieved.description == "Test algorithm"
        assert retrieved.kind == AlgorithmKind.RL_POLICY
        assert retrieved.trainable is True

    def test_default_metadata_created(self) -> None:
        """When no metadata provided, defaults are created."""
        self.registry.register("no_meta", self._dummy_factory)
        meta = self.registry.get_metadata("no_meta")
        assert meta.name == "no_meta"
        assert meta.kind == AlgorithmKind.RL_POLICY

    def test_list_by_kind_rl(self) -> None:
        """list_by_kind filters to RL policy algorithms only."""
        self.registry.register(
            "rl_algo",
            self._dummy_factory,
            AlgorithmMetadata(name="rl_algo", kind=AlgorithmKind.RL_POLICY, trainable=True),
        )
        self.registry.register(
            "plan_algo",
            self._dummy_factory,
            AlgorithmMetadata(name="plan_algo", kind=AlgorithmKind.PLANNER, trainable=False),
        )
        rl_names = self.registry.list_by_kind(AlgorithmKind.RL_POLICY)
        planner_names = self.registry.list_by_kind(AlgorithmKind.PLANNER)
        assert "rl_algo" in rl_names
        assert "plan_algo" not in rl_names
        assert "plan_algo" in planner_names
        assert "rl_algo" not in planner_names

    def test_is_trainable(self) -> None:
        """is_trainable distinguishes RL algorithms from planners."""
        self.registry.register(
            "rl",
            self._dummy_factory,
            AlgorithmMetadata(name="rl", kind=AlgorithmKind.RL_POLICY, trainable=True),
        )
        self.registry.register(
            "plan",
            self._dummy_factory,
            AlgorithmMetadata(name="plan", kind=AlgorithmKind.PLANNER, trainable=False),
        )
        assert self.registry.is_trainable("rl") is True
        assert self.registry.is_trainable("plan") is False

    def test_list_all_metadata(self) -> None:
        """list_all_metadata returns complete mapping."""
        self.registry.register("algo_a", self._dummy_factory)
        self.registry.register("algo_b", self._dummy_factory)
        meta_map = self.registry.list_all_metadata()
        assert set(meta_map.keys()) == {"algo_a", "algo_b"}

    def test_clear(self) -> None:
        """clear() removes all registered algorithms."""
        self.registry.register("my_algo", self._dummy_factory)
        self.registry.clear()
        assert self.registry.list_algorithms() == []

    def test_case_normalization(self) -> None:
        """Algorithm names are normalized to lowercase."""
        self.registry.register("MyAlgo", self._dummy_factory)
        factory = self.registry.get_factory("myalgo")
        assert callable(factory)
        # Lookup with original case also works
        factory2 = self.registry.get_factory("MyAlgo")
        assert callable(factory2)


# ---------------------------------------------------------------------------
# Global registry integration tests
# ---------------------------------------------------------------------------


class TestGlobalAlgorithmRegistry:
    """Tests for the global default algorithm_registry with PPO, SAC, A*."""

    def test_ppo_registered(self) -> None:
        """PPO is registered in the global registry."""
        assert "ppo" in list_algorithms()

    def test_sac_registered(self) -> None:
        """SAC is registered in the global registry."""
        assert "sac" in list_algorithms()

    def test_astar_registered(self) -> None:
        """A* planner is registered in the global registry."""
        assert "astar" in list_algorithms()

    def test_ppo_is_trainable(self) -> None:
        """PPO metadata marks it as trainable."""
        meta = get_algorithm_metadata("ppo")
        assert meta.trainable is True
        assert meta.kind == AlgorithmKind.RL_POLICY

    def test_sac_is_trainable(self) -> None:
        """SAC metadata marks it as trainable."""
        meta = get_algorithm_metadata("sac")
        assert meta.trainable is True
        assert meta.kind == AlgorithmKind.RL_POLICY

    def test_astar_is_not_trainable(self) -> None:
        """A* metadata marks it as non-trainable planner."""
        meta = get_algorithm_metadata("astar")
        assert meta.trainable is False
        assert meta.kind == AlgorithmKind.PLANNER

    def test_ppo_factory_produces_ppo_algorithm(self) -> None:
        """PPO factory callable returns PPOAlgorithm class."""
        from adaptive_rl.algorithms.ppo import PPOAlgorithm

        factory = get_algorithm_factory("ppo")
        assert factory is PPOAlgorithm

    def test_sac_factory_produces_sac_algorithm(self) -> None:
        """SAC factory callable returns SACAlgorithm class."""
        from adaptive_rl.algorithms.sac import SACAlgorithm

        factory = get_algorithm_factory("sac")
        assert factory is SACAlgorithm

    def test_astar_factory_produces_astar_planner(self) -> None:
        """A* factory callable returns AStarPlanner class."""
        from adaptive_rl.planners.astar import AStarPlanner

        factory = get_algorithm_factory("astar")
        assert factory is AStarPlanner

    def test_rl_algorithms_list(self) -> None:
        """RL algorithms are listed separately from planners."""
        rl_list = list_algorithms_by_kind(AlgorithmKind.RL_POLICY)
        assert "ppo" in rl_list
        assert "sac" in rl_list
        assert "astar" not in rl_list

    def test_planners_list(self) -> None:
        """Planners are listed separately from RL algorithms."""
        planner_list = list_algorithms_by_kind(AlgorithmKind.PLANNER)
        assert "astar" in planner_list
        assert "ppo" not in planner_list
        assert "sac" not in planner_list

    def test_all_metadata_complete(self) -> None:
        """All registered algorithms have complete metadata."""
        meta_map = list_all_algorithm_metadata()
        for name, meta in meta_map.items():
            assert meta.name == name
            assert meta.description != ""
            assert meta.class_name != ""
            assert meta.action_space != ""

    def test_sac_hyperparameters(self) -> None:
        """SAC metadata includes expected hyperparameter documentation."""
        meta = get_algorithm_metadata("sac")
        hp = meta.hyperparameters
        assert "learning_rate" in hp
        assert "buffer_size" in hp
        assert "gamma" in hp

    def test_ppo_tags_include_on_policy(self) -> None:
        """PPO tags include 'on-policy' indicating training style."""
        meta = get_algorithm_metadata("ppo")
        assert "on-policy" in meta.tags

    def test_astar_tags_include_classical(self) -> None:
        """A* tags include 'classical' indicating non-RL nature."""
        meta = get_algorithm_metadata("astar")
        assert "classical" in meta.tags

    def test_unknown_algorithm_raises_error(self) -> None:
        """Looking up an unregistered name raises AlgorithmRegistryError."""
        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm"):
            get_algorithm_metadata("rrt_star_unknown")

    def test_resolve_algorithm_success(self) -> None:
        """resolve and resolve_algorithm correctly return factory callables."""
        from adaptive_rl.algorithms.registry import algorithm_registry, resolve_algorithm
        from adaptive_rl.planners.astar import AStarPlanner

        factory = resolve_algorithm("astar")
        assert factory is AStarPlanner

        # Case-insensitive resolution
        assert algorithm_registry.resolve("PPO") is not None
        assert algorithm_registry.resolve("  SaC  ") is not None

    def test_resolve_algorithm_unknown_raises(self) -> None:
        """resolve raises AlgorithmRegistryError for unknown algorithm."""
        from adaptive_rl.algorithms.registry import resolve_algorithm

        with pytest.raises(AlgorithmRegistryError, match="Unknown algorithm 'nonexistent'"):
            resolve_algorithm("nonexistent")

    def test_algorithm_aliases(self) -> None:
        """Registered aliases resolve to target algorithm, and deprecated plain 'rrt' fails cleanly."""
        from adaptive_rl.algorithms.registry import AlgorithmRegistryError, algorithm_registry
        from adaptive_rl.planners.rrt_star import RRTStarPlanner

        # Plain 'rrt' is rejected with descriptive error
        with pytest.raises(AlgorithmRegistryError, match="Plain un-rewired RRT is not implemented"):
            algorithm_registry.resolve("rrt")

        # 'rrt*' and case-insensitive aliases resolve to RRTStarPlanner
        assert algorithm_registry.resolve("rrt*") is RRTStarPlanner
        assert algorithm_registry.resolve("RRT*") is RRTStarPlanner

        meta_star = algorithm_registry.get_metadata("rrt*")
        assert meta_star.name == "rrt_star"

    def test_metadata_defensive_copy(self) -> None:
        """Modifying retrieved metadata does not mutate registry state."""
        from adaptive_rl.algorithms.registry import algorithm_registry

        meta = algorithm_registry.get_metadata("astar")
        meta.tags.append("mutated_tag")
        meta.hyperparameters["fake_param"] = "fake_val"

        fresh_meta = algorithm_registry.get_metadata("astar")
        assert "mutated_tag" not in fresh_meta.tags
        assert "fake_param" not in fresh_meta.hyperparameters

    def test_registry_restore_defaults(self) -> None:
        """restore_defaults cleanly repopulates canonical algorithms and aliases."""
        from adaptive_rl.algorithms.registry import algorithm_registry

        algorithm_registry.register("custom_temp", lambda **kw: object())
        assert "custom_temp" in algorithm_registry.list_algorithms()

        algorithm_registry.restore_defaults()
        assert "astar" in algorithm_registry.list_algorithms()
        assert "rrt_star" in algorithm_registry.list_algorithms()

    def test_planner_consistency_across_layers(self) -> None:
        """For every supported planner, list/inspect/resolve/config/execution layers agree."""
        from adaptive_rl.algorithms.registry import algorithm_registry
        from adaptive_rl.config import PLANNER_ALGORITHMS, AlgorithmConfig

        for planner_name in ("astar", "rrt_star", "rrt*"):
            # 1. Resolve works
            factory = algorithm_registry.resolve(planner_name)
            assert callable(factory)

            # 2. Inspect metadata works
            meta = algorithm_registry.get_metadata(planner_name)
            assert not meta.trainable
            assert meta.kind == AlgorithmKind.PLANNER

            # 3. Configuration validation recognizes planner
            cfg = AlgorithmConfig(name=planner_name)
            assert cfg.is_planner
            assert cfg.name in PLANNER_ALGORITHMS

    def test_rrt_star_metadata_claims(self) -> None:
        """RRT* metadata accurately claims heuristic tree rewiring without claiming formal asymptotic optimality."""
        from adaptive_rl.algorithms.registry import algorithm_registry

        meta = algorithm_registry.get_metadata("rrt_star")
        assert "optimal paths" not in meta.description
        assert "heuristic tree rewiring" in meta.description

    def test_lazy_factory_raises_descriptive_importerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling lazy factory without RL dependencies raises descriptive ImportError."""
        import sys

        from adaptive_rl.algorithms.registry import _make_ppo, _make_sac

        monkeypatch.setitem(sys.modules, "adaptive_rl.algorithms.ppo", None)
        with pytest.raises(ImportError, match="requires optional 'rl' dependencies"):
            _make_ppo()

        monkeypatch.setitem(sys.modules, "adaptive_rl.algorithms.sac", None)
        with pytest.raises(ImportError, match="requires optional 'rl' dependencies"):
            _make_sac()
