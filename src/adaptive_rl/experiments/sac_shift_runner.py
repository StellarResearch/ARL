"""SAC online adaptation experiment runner.

Implements the complete research workflow of docs/research/TREATMENT_CARD.md
for Soft Actor-Critic (SAC) online adaptation under distribution shift:
1. Nominal training with SACAlgorithm
2. Checkpoint freeze and SHA-256 fingerprint verification
3. Shared pre-shift evaluation (K_pre = 15 episodes)
4. Distribution shift (TEST-B) and shared shock window (episodes 1-5)
5. Fork into Fixed SAC (zero updates) and Adaptive SAC
6. Replay buffer isolation (post-shift transitions only, no nominal transitions)
7. Online update blocks B5-B14 with zero-based derived seeds
8. Recovery analysis and statistical reporting matching the research protocol
"""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:
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
from adaptive_rl.experiments.shift_runner import (
    Transition,
    TransitionBuffer,
    evaluate_episode,
    parameter_delta_norm,
    policy_fingerprint,
)


def execute_sac_adaptation_block(
    algo: SACAlgorithm,
    buffer: TransitionBuffer,
    max_episode: int,
    derived_seed: int,
    train_steps: int = 10,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Execute an online SAC adaptation block B_k according to TREATMENT_CARD.md.

    Args:
        algo: Adaptive SACAlgorithm instance.
        buffer: TransitionBuffer containing post-shift transitions.
        max_episode: Maximum post-shift episode included in this update.
        derived_seed: Preregistered deterministic seed for this block.
        train_steps: Number of gradient steps to execute.
        batch_size: Minibatch size for training updates.

    Returns:
        Audit dictionary with block_id, episode_k, derived_seed, transition count,
        parameter delta norm, and SHA-256 fingerprints before and after.
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required to execute SAC adaptation blocks. "
            "Install with: pip install 'adaptive-rl[rl]'"
        )

    # Seed RNGs deterministically
    torch.manual_seed(derived_seed)
    np.random.seed(derived_seed)

    if algo.model is None or not hasattr(algo.model, "policy"):
        raise RuntimeError("Adaptive SAC model is not initialized.")

    policy_before = getattr(algo.model, "policy", algo.model)
    fingerprint_before = policy_fingerprint(policy_before)
    state_before = {k: v.clone() for k, v in policy_before.state_dict().items()}

    # Populate SAC replay buffer strictly with post-shift transitions up to max_episode
    transitions = buffer.get_transitions(max_episode)
    if not transitions:
        raise ValueError(f"No post-shift transitions available up to episode {max_episode}.")

    if hasattr(algo.model, "replay_buffer") and algo.model.replay_buffer is not None:
        algo.model.replay_buffer.reset()
        for t in transitions:
            algo.model.replay_buffer.add(
                obs=t.obs,
                next_obs=t.next_obs,
                action=t.action,
                reward=np.array([t.reward], dtype=np.float32),
                done=np.array([t.done], dtype=np.float32),
                infos=[{}],
            )
        actual_batch = min(batch_size, len(transitions))
        algo.model.train(gradient_steps=train_steps, batch_size=actual_batch)

    policy_after = getattr(algo.model, "policy", algo.model)
    fingerprint_after = policy_fingerprint(policy_after)
    state_after = policy_after.state_dict()
    delta_norm = parameter_delta_norm(state_before, state_after)

    return {
        "block_id": f"B{max_episode}",
        "episode_k": max_episode,
        "derived_seed": int(derived_seed),
        "num_transitions": len(transitions),
        "data_episodes": [1, max_episode],
        "parameter_delta_norm": float(delta_norm),
        "fingerprint_before": fingerprint_before,
        "fingerprint_after": fingerprint_after,
    }


def run_sac_adaptive_vs_fixed_replicate(
    replicate_index: int,
    training_seed: int,
    config: Optional[ExperimentConfig] = None,
    training_timesteps: int = 60000,
    quick_test_mode: bool = False,
) -> Dict[str, Any]:
    """Execute Step 1 through Step 8 of the research protocol for one SAC replicate.

    Args:
        replicate_index: 1-based replicate index (1..10).
        training_seed: Pinned training seed integer from TRAINING_SEEDS.
        config: Optional base ExperimentConfig.
        training_timesteps: Total timesteps for nominal policy training.
        quick_test_mode: If True, uses small training steps and short episodes for fast testing.

    Returns:
        Dictionary containing all pre/post returns, fingerprints, block logs, and recovery metrics.
    """
    if torch is None:
        raise ImportError(
            "PyTorch is required to execute SAC adaptation experiments. "
            "Install with: pip install 'adaptive-rl[rl]'"
        )

    # 1. Deterministic seed schedules
    pre_seeds = [derive_seed(training_seed, "pre", j) for j in range(1, K_PRE + 1)]
    post_seeds = [derive_seed(training_seed, "post", j) for j in range(1, N_POST + 1)]
    update_seeds = [derive_seed(training_seed, "update", b) for b in range(N_UPDATE)]

    # 2. Step 1: Nominal SAC training on nominal environment
    nominal_max_steps = 50 if quick_test_mode else 300
    steps_budget = 200 if quick_test_mode else training_timesteps

    train_env = DroneDisturbed3DEnv(
        num_obstacles=8,
        num_dynamic_obstacles=0,
        wind_speed=0.5,
        gust_sigma=0.15,
        disturbance_strength=0.0,
        max_steps=nominal_max_steps,
    )
    sac_algo = SACAlgorithm(
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1000 if quick_test_mode else 100000,
        batch_size=32 if quick_test_mode else 256,
        seed=int(training_seed),
    )
    sac_algo.train(total_timesteps=steps_budget)
    train_env.close()
    assert sac_algo.model is not None, "Nominal SAC model was not initialized"

    # Step 2: Freeze checkpoint and record SHA-256 fingerprint
    frozen_fingerprint = policy_fingerprint(sac_algo)
    frozen_weights = copy.deepcopy(sac_algo.model.policy.state_dict())

    # 3. Step 3: Shared pre-shift evaluation (K_pre = 15 episodes)
    eval_nominal_env = DroneDisturbed3DEnv(
        num_obstacles=8,
        num_dynamic_obstacles=0,
        wind_speed=0.5,
        gust_sigma=0.15,
        disturbance_strength=0.0,
        max_steps=nominal_max_steps,
    )
    pre_shift_returns: List[float] = []
    for s in pre_seeds:
        ret, _, _, _ = evaluate_episode(eval_nominal_env, sac_algo.model, seed=s)
        pre_shift_returns.append(ret)
    eval_nominal_env.close()

    # 4. Step 4 & 5: Shift introduction (TEST-B) and shared shock window (episodes 1-5)
    shifted_env = DroneDisturbed3DEnv(
        num_obstacles=12,
        num_dynamic_obstacles=0,
        wind_speed=4.0,
        gust_sigma=0.6,
        disturbance_strength=0.0,
        max_steps=nominal_max_steps,
    )

    shock_returns: List[float] = []
    adaptive_buffer = TransitionBuffer()

    for j in range(1, WINDOW + 1):
        s = post_seeds[j - 1]
        ret, _, _, _ = evaluate_episode(
            shifted_env,
            sac_algo.model,
            seed=s,
            collect_transitions=True,
            buffer=adaptive_buffer,
            episode_index=j,
        )
        shock_returns.append(ret)

    # 5. Step 6: Fork into Fixed and Adaptive arms
    fixed_post_returns: List[float] = list(shock_returns)
    adaptive_post_returns: List[float] = list(shock_returns)

    # Fixed arm: starts from frozen checkpoint and remains completely frozen (no gradient steps)
    fixed_algo = SACAlgorithm(
        env=shifted_env,
        learning_rate=3e-4,
        seed=int(training_seed),
    )
    assert fixed_algo.model is not None, "Fixed SAC model was not initialized"
    fixed_algo.model.policy.load_state_dict(copy.deepcopy(frozen_weights))

    # Adaptive arm: starts from frozen checkpoint with fresh replay buffer for post-shift transitions
    adaptive_algo = SACAlgorithm(
        env=shifted_env,
        learning_rate=1e-4,
        buffer_size=1000 if quick_test_mode else 100000,
        batch_size=32,
        seed=int(training_seed),
    )
    assert adaptive_algo.model is not None, "Adaptive SAC model was not initialized"
    adaptive_algo.model.policy.load_state_dict(copy.deepcopy(frozen_weights))

    block_logs: List[Dict[str, Any]] = []

    # Execute Block B5 (episodes 1-5 shock transitions) strictly before episode 6 reset
    b5_log = execute_sac_adaptation_block(
        algo=adaptive_algo,
        buffer=adaptive_buffer,
        max_episode=5,
        derived_seed=update_seeds[0],
        train_steps=2 if quick_test_mode else 10,
    )
    block_logs.append(b5_log)

    # 6. Step 7: Post-shift episodes 6..15
    for k in range(6, N_POST + 1):
        s_eval = post_seeds[k - 1]

        # Fixed arm evaluation (frozen weights, identical seed, zero gradient updates)
        fixed_ret, _, _, _ = evaluate_episode(shifted_env, fixed_algo.model, seed=s_eval)
        fixed_post_returns.append(fixed_ret)

        # Adaptive arm evaluation (adapted policy, identical seed, collect transitions)
        adapt_ret, _, _, _ = evaluate_episode(
            shifted_env,
            adaptive_algo.model,
            seed=s_eval,
            collect_transitions=True,
            buffer=adaptive_buffer,
            episode_index=k,
        )
        adaptive_post_returns.append(adapt_ret)

        # Execute Block B_k if k < N_POST
        if k < N_POST:
            block_idx = k - 5
            b_log = execute_sac_adaptation_block(
                algo=adaptive_algo,
                buffer=adaptive_buffer,
                max_episode=k,
                derived_seed=update_seeds[block_idx],
                train_steps=2 if quick_test_mode else 10,
            )
            block_logs.append(b_log)

    shifted_env.close()

    # 7. Step 8: Causal Recovery Analysis
    rec_fixed = compute_recovery(pre_shift_returns, fixed_post_returns)
    rec_adaptive = compute_recovery(pre_shift_returns, adaptive_post_returns)
    d_i = float(rec_adaptive.truncated_recovery_time - rec_fixed.truncated_recovery_time)

    return {
        "replicate_index": int(replicate_index),
        "training_seed": int(training_seed),
        "pre_shift_seeds": pre_seeds,
        "post_shift_seeds": post_seeds,
        "update_seeds": update_seeds,
        "frozen_fingerprint": frozen_fingerprint,
        "pre_shift_returns": pre_shift_returns,
        "shock_returns": shock_returns,
        "fixed_post_returns": fixed_post_returns,
        "adaptive_post_returns": adaptive_post_returns,
        "fixed_recovery": asdict(rec_fixed),
        "adaptive_recovery": asdict(rec_adaptive),
        "d_i": d_i,
        "block_logs": block_logs,
    }


class AdaptiveSACShiftRunner:
    """Orchestrates multi-replicate online adaptation vs fixed benchmark experiments using SAC."""

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
        """Execute full benchmark across all specified training seeds."""
        start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        replicate_reports: List[Dict[str, Any]] = []
        differences: List[float] = []

        for idx, seed in enumerate(self.training_seeds, start=1):
            rep = run_sac_adaptive_vs_fixed_replicate(
                replicate_index=idx,
                training_seed=seed,
                training_timesteps=self.training_timesteps,
                quick_test_mode=self.quick_test_mode,
            )
            replicate_reports.append(rep)
            differences.append(rep["d_i"])

        min_n = min(2, len(differences)) if self.quick_test_mode else MIN_VALID_N
        t_res = paired_t_test(differences, min_valid_n=min_n)
        ci_res = paired_t_interval(differences, confidence=1.0 - ALPHA, min_valid_n=min_n)
        cohen_d = cohen_dz(differences)
        boot_ci = bootstrap_percentile_ci(
            differences,
            confidence=1.0 - ALPHA,
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
            min_valid_n=min_n,
        )
        sign_res = exact_sign_test(differences, min_valid_n=min_n)
        wilcox_res = exact_wilcoxon_signed_rank(differences, min_valid_n=min_n)

        finish_time = time.time()
        json_path = self.output_dir / "sac_adaptive_vs_fixed.json"
        csv_path = self.output_dir / "sac_adaptive_vs_fixed.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "replicate",
                    "training_seed",
                    "p_pre_mean",
                    "p0_mean",
                    "fixed_t_h",
                    "fixed_status",
                    "adaptive_t_h",
                    "adaptive_status",
                    "d_i",
                ]
            )
            for r in replicate_reports:
                writer.writerow(
                    [
                        r["replicate_index"],
                        r["training_seed"],
                        f"{r['fixed_recovery']['p_pre']:.4f}",
                        f"{r['fixed_recovery']['p0']:.4f}",
                        r["fixed_recovery"]["truncated_recovery_time"],
                        r["fixed_recovery"]["status"],
                        r["adaptive_recovery"]["truncated_recovery_time"],
                        r["adaptive_recovery"]["status"],
                        r["d_i"],
                    ]
                )

        manifest = create_manifest(
            experiment_name="sac_adaptive_vs_fixed_shift",
            config_dict={
                "environment": "drone_disturbed",
                "condition": "TEST-B",
                "algorithm": "sac",
                "training_seeds": self.training_seeds,
                "protocol_version": PROTOCOL_VERSION,
            },
            started_at=start_time,
            finished_at=finish_time,
            training_time_seconds=finish_time - start_time,
            artifacts={"csv_summary": csv_path},
        )

        report_dict: Dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "experiment": "sac_online_adaptation_vs_fixed_shift",
            "condition": "TEST-B",
            "num_replicates": len(replicate_reports),
            "differences": differences,
            "statistics": {
                "mean_d": float(t_res.mean),
                "std_d": float(t_res.std_dev),
                "se_d": float(t_res.standard_error),
                "t_statistic": float(t_res.t_statistic),
                "p_value_onesided": float(t_res.p_value),
                "confidence_interval_95": [float(ci_res[0]), float(ci_res[1])],
                "cohens_dz": float(cohen_d) if cohen_d is not None else None,
                "bootstrap_ci_95": [float(boot_ci[0]), float(boot_ci[1])],
                "sign_test_p": float(sign_res.p_value),
                "wilcoxon_p": float(wilcox_res.p_value),
            },
            "replicates": replicate_reports,
            "manifest": manifest.model_dump(),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        return report_dict


run_sac_adaptive_vs_fixed_experiment = AdaptiveSACShiftRunner

__all__ = [
    "AdaptiveSACShiftRunner",
    "execute_sac_adaptation_block",
    "run_sac_adaptive_vs_fixed_experiment",
    "run_sac_adaptive_vs_fixed_replicate",
]
