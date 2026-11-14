"""Model and experiment checkpoint management for AdaptiveRL."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from adaptive_rl.algorithms.base import BaseAlgorithm


class CheckpointManager:
    """Manages periodic, best-metric, and final checkpoint creation and recovery."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        """Initialize checkpoint manager with target directory.

        Args:
            checkpoint_dir: Directory where model checkpoints will be stored.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict[str, Any]] = []

    def record_checkpoint(self, path: str | Path, step: int, metric_value: float) -> None:
        """Track saved checkpoint metadata in manager history."""
        self._history.append(
            {
                "path": str(path),
                "step": step,
                "metric_value": metric_value,
                "timestamp": time.time(),
            }
        )

    def save_checkpoint(
        self,
        model: BaseAlgorithm,
        step: int,
        metric_value: float = 0.0,
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a model checkpoint to disk and register in manager history.

        Args:
            model: Algorithm instance implementing BaseAlgorithm.
            step: Timestep at which the checkpoint is taken.
            metric_value: Performance metric (e.g. mean reward) associated with checkpoint.
            filename: Custom filename (defaults to 'checkpoint_step_{step}.zip').
            metadata: Optional dictionary with provenance metadata to embed as arl_metadata.json.

        Returns:
            Path: Path where the checkpoint was written.
        """
        name = filename or f"checkpoint_step_{step}.zip"
        target_path = self.checkpoint_dir / name
        model.save(target_path)

        # Confirm exact file on disk (some libraries append .zip if omitted)
        final_path = target_path if target_path.exists() else target_path.with_suffix(".zip")

        if metadata:
            try:
                import json
                import zipfile

                with zipfile.ZipFile(final_path, "a") as z:
                    z.writestr("arl_metadata.json", json.dumps(metadata, indent=2))
            except Exception:
                pass

        self.record_checkpoint(path=final_path, step=step, metric_value=metric_value)
        return final_path

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Return recorded checkpoint history."""
        return list(self._history)

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Return the most recently saved checkpoint metadata, or None if none exist."""
        if not self._history:
            return None
        return self._history[-1]

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Return the checkpoint with the highest metric value, or None if none exist."""
        if not self._history:
            return None
        return max(self._history, key=lambda cp: cp.get("metric_value", float("-inf")))

    def clear(self) -> None:
        """Clear recorded checkpoint history."""
        self._history.clear()
