"""Tests verifying target repository structure and essential file existence."""

from pathlib import Path


def test_target_repository_structure() -> None:
    """Verify all target directories, documentation, and configuration files exist."""
    repo_root = Path(__file__).resolve().parent.parent

    # Essential root files
    assert (repo_root / "README.md").is_file()
    assert (repo_root / "LICENSE").is_file()
    assert (repo_root / "CONTRIBUTING.md").is_file()
    assert (repo_root / "pyproject.toml").is_file()
    assert (repo_root / "Makefile").is_file()
    assert (repo_root / ".gitignore").is_file()

    # Configs
    assert (repo_root / "configs" / "ppo.yaml").is_file()
    assert (repo_root / "configs" / "sac.yaml").is_file()
    assert (repo_root / "configs" / "navigation.yaml").is_file()
    assert (repo_root / "configs" / "drone.yaml").is_file()

    # Documentation
    assert (repo_root / "docs" / "ARCHITECTURE.md").is_file()
    assert (repo_root / "docs" / "RESEARCH.md").is_file()
    assert (repo_root / "docs" / "EXPERIMENTS.md").is_file()

    # Experiment placeholder directories
    assert (repo_root / "experiments" / "results" / ".gitkeep").is_file()
    assert (repo_root / "experiments" / "logs" / ".gitkeep").is_file()

    # Core source packages
    src = repo_root / "src" / "adaptive_rl"
    assert (src / "__init__.py").is_file()
    assert (src / "cli.py").is_file()
    assert (src / "config.py").is_file()

    # Subpackages
    subpackages = [
        "algorithms",
        "environments",
        "rewards",
        "training",
        "evaluation",
        "models",
        "visualization",
        "experiments",
        "planning",
    ]
    for sp in subpackages:
        sp_dir = src / sp
        assert sp_dir.is_dir(), f"Expected directory missing: {sp_dir}"
        assert (sp_dir / "__init__.py").is_file(), f"Missing __init__.py in {sp_dir}"
