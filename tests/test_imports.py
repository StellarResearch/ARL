"""Tests verifying package and module importability."""

import importlib

import pytest

import adaptive_rl


def test_package_import() -> None:
    """Verify top-level package imports and exposes version string."""
    assert hasattr(adaptive_rl, "__version__")
    assert isinstance(adaptive_rl.__version__, str)
    assert len(adaptive_rl.__version__.split(".")) >= 3


@pytest.mark.parametrize(
    "module_name",
    [
        "adaptive_rl.algorithms",
        "adaptive_rl.algorithms.base",
        "adaptive_rl.environments",
        "adaptive_rl.environments.registry",
        "adaptive_rl.rewards",
        "adaptive_rl.rewards.base",
        "adaptive_rl.training",
        "adaptive_rl.training.trainer",
        "adaptive_rl.training.callbacks",
        "adaptive_rl.training.checkpointing",
        "adaptive_rl.evaluation",
        "adaptive_rl.evaluation.evaluator",
        "adaptive_rl.evaluation.metrics",
        "adaptive_rl.evaluation.scenarios",
        "adaptive_rl.models",
        "adaptive_rl.models.manager",
        "adaptive_rl.visualization",
        "adaptive_rl.visualization.plots",
        "adaptive_rl.visualization.renderer",
        "adaptive_rl.experiments",
        "adaptive_rl.experiments.runner",
        "adaptive_rl.planning",
        "adaptive_rl.planning.base",
        "adaptive_rl.planning.astar",
        "adaptive_rl.planning.rrt",
        "adaptive_rl.planning.benchmark",
        "adaptive_rl.config",
        "adaptive_rl.cli",
    ],
)
def test_submodule_imports(module_name: str) -> None:
    """Verify all submodules and packages can be imported without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None
