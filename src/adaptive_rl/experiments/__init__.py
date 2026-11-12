"""Experiments module for AdaptiveRL."""

from __future__ import annotations

# Import PPO experiment symbols
from adaptive_rl.experiments.shift_runner import (
    AdaptiveShiftRunner,
    ReplicateReport,
    Transition,
    TransitionBuffer,
    execute_ppo_adaptation_block,
    parameter_delta_norm,
    policy_fingerprint,
    run_adaptive_vs_fixed_experiment,
    run_adaptive_vs_fixed_replicate,
)

# Import SAC experiment symbols
from adaptive_rl.experiments.sac_shift_runner import (
    AdaptiveShiftRunner as SACAdaptiveShiftRunner,
    ReplicateReport as SACReplicateReport,
    Transition as SACTransition,
    TransitionBuffer as SACTransitionBuffer,
    execute_sac_adaptation_block,
    parameter_delta_norm as sac_parameter_delta_norm,
    policy_fingerprint as sac_policy_fingerprint,
    run_adaptive_vs_fixed_experiment as sac_run_adaptive_vs_fixed_experiment,
    run_adaptive_vs_fixed_replicate as sac_run_adaptive_vs_fixed_replicate,
)

__all__ = [
    "AdaptiveShiftRunner",
    "ReplicateReport",
    "Transition",
    "TransitionBuffer",
    "execute_ppo_adaptation_block",
    "parameter_delta_norm",
    "policy_fingerprint",
    "run_adaptive_vs_fixed_experiment",
    "run_adaptive_vs_fixed_replicate",
    # SAC exports
    "SACAdaptiveShiftRunner",
    "SACReplicateReport",
    "SACTransition",
    "SACTransitionBuffer",
    "execute_sac_adaptation_block",
    "sac_parameter_delta_norm",
    "sac_policy_fingerprint",
    "sac_run_adaptive_vs_fixed_experiment",
    "sac_run_adaptive_vs_fixed_replicate",
]
