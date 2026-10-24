"""Tests for SAC online adaptation experiment runner.

The tests verify that the SAC adaptation runner follows the protocol:
* Uses only post‑shift transitions for adaptation (buffer isolation).
* Executes the correct number of adaptation blocks (B5‑B14).
* Adaptive arm performs genuine parameter updates (delta_norm > 0).
* Fixed arm never updates – its post‑shift returns are identical to the
  shared shock window returns.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from adaptive_rl.experiments.sac_shift_runner import run_sac_adaptive_vs_fixed_replicate
from adaptive_rl.protocol.constants import N_POST, WINDOW

@pytest.mark.parametrize("quick", [True])
def test_sac_adaptation_protocol(quick: bool):
    """Run a lightweight SAC adaptation replicate and check invariants.

    The quick test mode reduces training timesteps and adaptation steps so the
    test finishes within a few seconds.
    """
    # Use a deterministic training seed from the predefined list
    training_seed = 31001  # first entry from TRAINING_SEEDS
    rep = run_sac_adaptive_vs_fixed_replicate(
        replicate_index=1,
        training_seed=training_seed,
        training_timesteps=200,  # small budget for quick test
        quick_test_mode=quick,
    )

    # 1. Ensure the expected number of adaptation blocks (B5‑B14)
    assert len(rep["block_logs"]) == N_POST - WINDOW, "Expected one block per post‑shift episode after the shock window"

    # 2. Buffer isolation – the first block should have exactly WINDOW transitions
    first_block = rep["block_logs"][0]
    assert first_block["num_transitions"] == WINDOW, "B5 should use only the 5 shock‑window transitions"

    # 3. Adaptive updates must change parameters (delta_norm > 0)
    assert first_block["parameter_delta_norm"] > 0.0, "Adaptive SAC must perform a non‑zero parameter update"

    # 4. Fixed arm should never update – its post‑shift returns must equal the shock window returns
    assert rep["fixed_post_returns"][:WINDOW] == rep["shock_returns"], "Fixed arm must start with identical shock returns"
    # The rest of the fixed returns should be present (length N_POST)
    assert len(rep["fixed_post_returns"]) == N_POST, "Fixed arm must have a return for every post‑shift episode"
    assert len(rep["adaptive_post_returns"]) == N_POST, "Adaptive arm must have a return for every post‑shift episode"

    # 5. The adaptive arm should have at least one update block with a non‑zero delta
    deltas = [b["parameter_delta_norm"] for b in rep["block_logs"]]
    assert any(d > 0.0 for d in deltas), "Adaptive SAC should show parameter changes in at least one block"

    # 6. Verify that the recovery analysis runs without error and produces numeric values
    assert isinstance(rep["d_i"], float), "Difference metric d_i must be a float"
    assert rep["fixed_recovery"]["truncated_recovery_time"] >= 0, "Recovery time should be non‑negative"
    assert rep["adaptive_recovery"]["truncated_recovery_time"] >= 0, "Recovery time should be non‑negative"

