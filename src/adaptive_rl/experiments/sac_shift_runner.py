# SAC online adaptation runner
"""Implements the SAC online adaptation experiment following the same protocol as
the PPO shift runner. The implementation mirrors the steps described in Issue #273.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional torch import – only needed for fingerprinting when torch tensors are present.
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.config import ExperimentConfig
from adaptive_rl.environments.disturbed_drone import DroneDisturbed3DEnv
from adaptive_rl.manifest import create_manifest
from adaptive_rl.protocol.constants import (
    ALPHA,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    K_PRE,
    MIN_VALID_N,
    N_POST,
    N_UPDATE,
    PROTOCOL_VERSION,
    TRAINING_SEEDS,
    WINDOW,
)
from adaptive_rl.protocol.recovery import compute_recovery
from adaptive_rl.protocol.seeds import derive_seed
from adaptive_rl.protocol.statistics import (
    bootstrap_percentile_ci,
    cohen_dz,
    exact_sign_test,
    exact_wilcoxon_signed_rank,
    paired_t_interval,
    paired_t_test,
)

# -----------------------------------------------------------------------------
# Helper data structures
# -----------------------------------------------------------------------------

@dataclass
class Transition:
    """A single environment transition."""

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    episode_index: int  # 1‑based index for post‑shift episodes


class TransitionBuffer:
    """Collects transitions after the distribution shift.

    The buffer stores all transitions from the shock window and later episodes.
    It can return the subset belonging to the first *max_episode* episodes.
    """

    def __init__(self) -> None:
        self.transitions: List[Transition] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        episode_index: int,
    ) -> None:
        self.transitions.append(
            Transition(
                obs=np.asarray(obs, dtype=np.float32).copy(),
                action=np.asarray(action, dtype=np.float32).copy(),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32).copy(),
                done=bool(done),
                episode_index=int(episode_index),
            )
        )

    def get_transitions(self, max_episode: int) -> List[Transition]:
        """Return transitions up to ``max_episode`` (inclusive)."""
        return [t for t in self.transitions if t.episode_index <= max_episode]

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def policy_fingerprint(algorithm_or_model: Any) -> str:
    """Return a SHA‑256 hash of the policy parameters.

    Works with both torch ``state_dict`` objects and simple NumPy dictionaries.
    """
    model = getattr(algorithm_or_model, "model", algorithm_or_model)
    policy = getattr(model, "policy", model)
    if not hasattr(policy, "state_dict"):
        raise RuntimeError("Target has no 'state_dict' for fingerprinting.")
    state = policy.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state.keys()):
        tensor = state[name]
        digest.update(name.encode("utf-8"))
        if hasattr(tensor, "detach"):
            raw = tensor.detach().cpu().numpy().tobytes()
        else:
            raw = np.asarray(tensor).tobytes()
        digest.update(raw)
    return digest.hexdigest()


def parameter_delta_norm(before: Dict[str, Any], after: Dict[str, Any]) -> float:
    """L2 norm of the difference between two parameter dictionaries.

    Only floating‑point entries are considered. Works with torch tensors.
    """
    total_sq = 0.0
    for k, v1 in before.items():
        if k not in after:
            continue
        v2 = after[k]
        if torch is not None and hasattr(v1, "dtype") and hasattr(v2, "dtype"):
            if v1.dtype.is_floating_point and v2.dtype.is_floating_point:
                diff = (v2.float() - v1.float()).cpu()
                total_sq += float(torch.sum(diff ** 2).item())
        else:
            diff = np.asarray(v2, dtype=np.float64) - np.asarray(v1, dtype=np.float64)
            total_sq += float(np.sum(diff ** 2))
    return math.sqrt(total_sq)

# -----------------------------------------------------------------------------
# Core experiment functions (unchanged apart from statistical handling)
# -----------------------------------------------------------------------------

def run_adaptive_vs_fixed_replicate(
    replicate_index: int,
    training_seed: int,
    training_timesteps: int = 60000,
    quick_test_mode: bool = False,
    config: Optional[ExperimentConfig] = None,
) -> ReplicateReport:
    """Execute one replicate of the SAC online adaptation protocol.

    ``quick_test_mode`` reduces episode lengths and buffer sizes to keep unit tests fast.
    """
    # Seed schedules
    pre_seeds = [derive_seed(training_seed, "pre", i) for i in range(1, K_PRE + 1)]
    post_seeds = [derive_seed(training_seed, "post", i) for i in range(1, N_POST + 1)]
    update_seeds = [derive_seed(training_seed, "update", i) for i in range(N_UPDATE)]

    # Step 1 – nominal SAC training
    steps_per_episode = 50 if quick_test_mode else 300
    train_env = DroneDisturbed3DEnv(
        num_obstacles=8,
        num_dynamic_obstacles=0,
        wind_speed=0.5,
        gust_sigma=0.15,
        disturbance_strength=0.0,
        max_steps=steps_per_episode,
    )
    sac_algo = SACAlgorithm(
        env=train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256 if not quick_test_mode else 32,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        seed=int(training_seed),
    )
    sac_algo.train(total_timesteps=training_timesteps)
    train_env.close()
    assert sac_algo.model is not None, "SAC model was not initialized"

    # Step 2 – freeze checkpoint
    frozen_fingerprint = policy_fingerprint(sac_algo)
    frozen_weights = copy.deepcopy(sac_algo.model.policy.state_dict())

    # Step 3 – pre‑shift evaluation
    eval_env = DroneDisturbed3DEnv(
        num_obstacles=8,
        num_dynamic_obstacles=0,
        wind_speed=0.5,
        gust_sigma=0.15,
        disturbance_strength=0.0,
        max_steps=steps_per_episode,
    )
    pre_shift_returns: List[float] = []
    for s in pre_seeds:
        ret, _, _, _ = evaluate_episode(eval_env, sac_algo.model, seed=s)
        pre_shift_returns.append(ret)
    eval_env.close()

    # Step 4‑5 – shift introduction & shared shock window
    shifted_env = DroneDisturbed3DEnv(
        num_obstacles=12,
        num_dynamic_obstacles=0,
        wind_speed=4.0,
        gust_sigma=0.6,
        disturbance_strength=0.0,
        max_steps=steps_per_episode,
    )
    shock_returns: List[float] = []
    adaptive_buffer = TransitionBuffer()
    for ep in range(1, WINDOW + 1):
        seed_ep = post_seeds[ep - 1]
        ret, _, _, _ = evaluate_episode(
            shifted_env,
            sac_algo.model,
            seed=seed_ep,
            collect_transitions=True,
            buffer=adaptive_buffer,
            episode_index=ep,
        )
        shock_returns.append(ret)

    # Step 6 – fork into fixed and adaptive arms
    fixed_model = sac_algo.model
    fixed_model.policy.load_state_dict(frozen_weights)

    adaptive_algo = SACAlgorithm(
        env=shifted_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256 if not quick_test_mode else 32,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        seed=int(training_seed),
        learning_starts=0,
    )
    assert adaptive_algo.model is not None, "Adaptive SAC model not initialized"
    adaptive_algo.model.policy.load_state_dict(copy.deepcopy(frozen_weights))
    adaptive_model = adaptive_algo.model

    fixed_post_returns: List[float] = list(shock_returns)
    adaptive_post_returns: List[float] = list(shock_returns)

    block_logs: List[Dict[str, Any]] = []

    # Block B5 (episodes 1‑5)
    b5 = execute_sac_adaptation_block(
        model=adaptive_model,
        buffer=adaptive_buffer,
        max_episode=5,
        derived_seed=update_seeds[0],
    )
    b5.update({"block_id": "B5", "episode_k": 5})
    block_logs.append(b5)

    # Remaining post‑shift episodes and adaptation blocks B6‑B14
    for k in range(6, N_POST + 1):
        seed_k = post_seeds[k - 1]
        fixed_ret, _, _, _ = evaluate_episode(shifted_env, fixed_model, seed=seed_k)
        fixed_post_returns.append(fixed_ret)
        adapt_ret, _, _, _ = evaluate_episode(
            shifted_env,
            adaptive_model,
            seed=seed_k,
            collect_transitions=True,
            buffer=adaptive_buffer,
            episode_index=k,
        )
        adaptive_post_returns.append(adapt_ret)
        if k < N_POST:
            block_idx = k - 5
            blk = execute_sac_adaptation_block(
                model=adaptive_model,
                buffer=adaptive_buffer,
                max_episode=k,
                derived_seed=update_seeds[block_idx],
            )
            blk.update({"block_id": f"B{k}", "episode_k": k})
            block_logs.append(blk)

    shifted_env.close()

    # Step 8 – recovery analysis
    rec_fixed = compute_recovery(pre_shift_returns, fixed_post_returns)
    rec_adaptive = compute_recovery(pre_shift_returns, adaptive_post_returns)
    d_i = float(rec_adaptive.truncated_recovery_time - rec_fixed.truncated_recovery_time)

    return ReplicateReport(
        replicate_index=int(replicate_index),
        training_seed=int(training_seed),
        pre_shift_seeds=pre_seeds,
        post_shift_seeds=post_seeds,
        update_seeds=update_seeds,
        frozen_fingerprint=frozen_fingerprint,
        pre_shift_returns=pre_shift_returns,
        shock_returns=shock_returns,
        fixed_post_returns=fixed_post_returns,
        adaptive_post_returns=adaptive_post_returns,
        fixed_recovery=asdict(rec_fixed),
        adaptive_recovery=asdict(rec_adaptive),
        d_i=d_i,
        block_logs=block_logs,
    )

# -----------------------------------------------------------------------------
# Runner class (CLI friendly)
# -----------------------------------------------------------------------------

class AdaptiveShiftRunner:
    """Runs multiple replicates and writes CSV/JSON reports.

    Mirrors the PPO ``AdaptiveShiftRunner`` but uses the SAC workflow.
    """

    def __init__(
        self,
        training_seeds: Sequence[int] = TRAINING_SEEDS,
        training_timesteps: int = 60000,
        output_dir: str = "artifacts/benchmarks",
        quick_test_mode: bool = False,
    ) -> None:
        self.training_seeds = [int(s) for s in training_seeds]
        self.training_timesteps = int(training_timesteps)
        self.output_dir = Path(output_dir)
        self.quick_test_mode = bool(quick_test_mode)

    def run(self) -> Dict[str, Any]:
        start = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        reports: List[ReplicateReport] = []
        diffs: List[float] = []
        for idx, seed in enumerate(self.training_seeds, start=1):
            timesteps = self.training_timesteps
            if self.quick_test_mode:
                timesteps = 1000
            rep = run_adaptive_vs_fixed_replicate(
                replicate_index=idx,
                training_seed=seed,
                training_timesteps=timesteps,
                quick_test_mode=self.quick_test_mode,
            )
            reports.append(rep)
            diffs.append(rep.d_i)

        n_replicates = len(diffs)
        stats_placeholder = {
            "mean_d": None,
            "std_d": None,
            "se_d": None,
            "t_statistic": None,
            "p_value_onesided": None,
            "confidence_interval_95": None,
            "cohens_dz": None,
            "bootstrap_ci_95": None,
            "sign_test_p": None,
            "wilcoxon_p": None,
            "status": None,
        }
        if n_replicates == 0:
            stats = {**stats_placeholder, "status": "insufficient_data"}
        elif n_replicates == 1:
            stats = {
                "mean_d": float(diffs[0]),
                "std_d": None,
                "se_d": None,
                "t_statistic": None,
                "p_value_onesided": None,
                "confidence_interval_95": None,
                "cohens_dz": None,
                "bootstrap_ci_95": None,
                "sign_test_p": None,
                "wilcoxon_p": None,
                "status": "not_estimable",
            }
        elif n_replicates < MIN_VALID_N:
            stats = {**stats_placeholder, "status": "insufficient_replicates"}
        else:
            mean_d = float(np.mean(diffs))
            std_d = float(np.std(diffs, ddof=1)) if n_replicates > 1 else None
            se_d = std_d / np.sqrt(n_replicates) if std_d is not None else None
            t_statistic, p_value_onesided = paired_t_test(diffs, mu=0)
            ci_low, ci_high = paired_t_interval(diffs, mu=0, confidence=0.95)
            confidence_interval_95 = (ci_low, ci_high)
            cohens_dz = cohen_dz(diffs)
            bootstrap_ci_95 = bootstrap_percentile_ci(diffs, n_reps=BOOTSTRAP_REPS, confidence=0.95, seed=BOOTSTRAP_SEED)
            sign_test_p = exact_sign_test(diffs)
            wilcoxon_p = exact_wilcoxon_signed_rank(diffs)
            stats = {
                "mean_d": mean_d,
                "std_d": std_d,
                "se_d": se_d,
                "t_statistic": t_statistic,
                "p_value_onesided": p_value_onesided,
                "confidence_interval_95": confidence_interval_95,
                "cohens_dz": cohens_dz,
                "bootstrap_ci_95": bootstrap_ci_95,
                "sign_test_p": sign_test_p,
                "wilcoxon_p": wilcoxon_p,
                "status": "evaluated",
            }

        finish = time.time()
        csv_path = self.output_dir / "adaptive_vs_fixed.csv"
        json_path = self.output_dir / "adaptive_vs_fixed.json"
        # Write CSV summary
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "replicate",
                "training_seed",
                "p_pre_mean",
                "p0_mean",
                "fixed_t_h",
                "fixed_status",
                "adaptive_t_h",
                "adaptive_status",
                "d_i",
            ])
            for r in reports:
                w.writerow([
                    r.replicate_index,
                    r.training_seed,
                    f"{r.fixed_recovery['p_pre']:.4f}",
                    f"{r.fixed_recovery['p0']:.4f}",
                    r.fixed_recovery["truncated_recovery_time"],
                    r.fixed_recovery["status"],
                    r.adaptive_recovery["truncated_recovery_time"],
                    r.adaptive_recovery["status"],
                    r.d_i,
                ])
        # Manifest and final JSON
        manifest = create_manifest(
            experiment_name="adaptive_vs_fixed_shift",
            config_dict={
                "environment": "drone_disturbed",
                "condition": "TEST-B",
                "algorithm": "sac",
                "training_seeds": self.training_seeds,
                "protocol_version": PROTOCOL_VERSION,
            },
            started_at=start,
            finished_at=finish,
            training_time_seconds=finish - start,
            artifacts={"csv_summary": csv_path},
        )
        report_dict: Dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "experiment": "online_adaptation_vs_fixed_shift",
            "condition": "TEST-B",
            "num_replicates": len(reports),
            "differences": diffs,
            "statistics": stats,
            "replicates": [asdict(r) for r in reports],
            "manifest": manifest.model_dump(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        return report_dict

# Compatibility alias expected by tests
SACAdaptiveShiftRunner = AdaptiveShiftRunner

def sac_run_adaptive_vs_fixed_replicate(
    replicate_index: int,
    training_seed: int,
    config: Optional[ExperimentConfig] = None,
    training_timesteps: int = 60000,
):
    return run_adaptive_vs_fixed_replicate(
        replicate_index=replicate_index,
        training_seed=training_seed,
        training_timesteps=training_timesteps,
        config=config,
    )
