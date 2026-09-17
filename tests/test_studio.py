"""Headless Qt interaction tests for AdaptiveRL Studio."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from adaptive_rl.studio.app import StudioWindow, TrainingWorker


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """Provide one QApplication for the module."""
    application = QApplication.instance() or QApplication([])
    yield application
    application.quit()


@pytest.fixture
def window(qt_app: QApplication) -> StudioWindow:
    """Create and show a Studio window for one test."""
    studio_window = StudioWindow(Path("experiments/results"))
    studio_window.show()
    qt_app.processEvents()
    yield studio_window
    studio_window.close()
    qt_app.processEvents()


def test_studio_window_startup_and_shutdown(window: StudioWindow) -> None:
    """Studio creates a visible, non-zero-sized window with the right title."""
    assert window.isVisible()
    assert window.size().width() > 0
    assert window.size().height() > 0
    assert window.windowTitle() == "AdaptiveRL Studio"


def test_studio_navigation_and_environment_controls(window: StudioWindow) -> None:
    """Navigation, reset, and one real environment step update the UI."""
    for index in range(window.navigation.count()):
        window.navigation.setCurrentRow(index)
        window.parentWidget()
        QApplication.processEvents()

    assert window.page_title.text() == "Benchmarks"
    window.navigation.setCurrentRow(1)
    window._reset_environment()
    window._step_environment()
    assert "Observation:" in window.environment_observation_label.text()
    assert window.environment is not None


def test_studio_loads_experiment_artifacts(window: StudioWindow) -> None:
    """The overview and experiment pages load existing repository artifacts."""
    assert window.recent_table.rowCount() == 7
    window.navigation.setCurrentRow(3)
    window._load_selected_metrics()
    assert window.evaluation_select.count() == 7


def test_training_worker_uses_existing_training_entrypoint() -> None:
    """TrainingWorker stores the selected config without duplicating RL logic."""
    worker = TrainingWorker("configs/smoke_gridworld_ppo.yaml", 8, 42)
    assert worker.config_path.endswith("smoke_gridworld_ppo.yaml")
    assert worker.timesteps == 8
    assert worker.seed == 42
