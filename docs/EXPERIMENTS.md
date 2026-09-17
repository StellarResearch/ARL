# AdaptiveRL — Experiment Lifecycle and Reproducibility

This document describes the experiment management system introduced in Phase 14,
including the output schema, seed management, manifest format, and comparison workflows.

---

## 1. Experiment Lifecycle

AdaptiveRL experiments are managed through the `ExperimentManager` class and the
`adaptive-rl experiment` CLI sub-commands.

### 1.1 Running an Experiment

```bash
adaptive-rl experiment run --config configs/gridworld_ppo.yaml
adaptive-rl experiment run --config configs/gridworld_astar.yaml
adaptive-rl experiment run --config configs/gridworld_ppo.yaml --seed 43
adaptive-rl experiment run --config configs/smoke_gridworld_ppo.yaml --timesteps 1000
```

Each run:
1. Validates the YAML configuration against the schema.
2. Generates a unique experiment ID.
3. Creates a structured output directory.
4. Records provenance metadata (git commit, Python version, packages).
5. Runs training (RL) or planning (A*).
6. Evaluates the trained agent/planner.
7. Saves all artifacts and a machine-readable manifest.

### 1.2 Experiment ID Format

```
YYYY-MM-DD_<environment>_<algorithm>_seed<seed>
```

Examples:
- `2026-09-17_gridworld_ppo_seed42`
- `2026-09-17_gridworld_astar_seed42`
- `2026-09-17_navigation_sac_seed43`

> [!NOTE]
> Experiment IDs are filesystem-safe and include the date, environment, algorithm,
> and seed. If you run the same experiment twice on the same day, the second run
> will create a duplicate directory. Use `--seed` to differentiate runs.

---

## 2. Output Directory Schema

Each experiment produces the following artifacts under `experiments/results/<experiment_id>/`:

```
experiments/results/<experiment_id>/
├── config.yaml          — copy of the configuration used (reproducibility)
├── manifest.json        — machine-readable provenance record
├── metrics.json         — evaluation metrics (JSON)
├── metrics.csv          — flat scalar metrics (CSV for spreadsheet analysis)
├── evaluation.json      — full evaluation report (RL only)
├── model/               — saved model weights (RL only)
│   └── <name>_final.zip
├── logs/                — training logs (RL only)
└── plots/               — reserved for future visualization artifacts
```

---

## 3. Manifest Schema

Every completed experiment writes `manifest.json` with the following fields:

```json
{
  "experiment_id": "2026-09-17_gridworld_ppo_seed42",
  "created_at": "2026-09-17T03:00:00+00:00",
  "algorithm": "ppo",
  "environment": "gridworld",
  "seed": 42,
  "config_path": "configs/gridworld_ppo.yaml",
  "training_timesteps": 5000,
  "git_commit": "abc1234",
  "python_version": "3.12.3 (main, ...)",
  "platform_info": "Linux 5.15.0 x86_64",
  "package_versions": {
    "adaptive-rl": "0.1.0",
    "gymnasium": "1.0.0",
    "stable-baselines3": "2.3.0",
    "torch": "2.2.0",
    "pydantic": "2.10.0"
  },
  "evaluation_seeds": [42, 43, 44],
  "artifact_paths": {
    "config": "experiments/results/.../config.yaml",
    "model": "experiments/results/.../model/..._final.zip",
    "metrics": "experiments/results/.../metrics.json",
    "metrics_csv": "experiments/results/.../metrics.csv",
    "evaluation": "experiments/results/.../evaluation.json"
  },
  "evaluation_status": "completed",
  "notes": ""
}
```

For planner experiments (A*), `training_timesteps` is `null`.

---

## 4. Metrics Schema

### 4.1 RL Algorithm Metrics (EvaluationMetrics)

| Field | Type | Description |
|:------|:-----|:------------|
| `episodes` | int | Total evaluation episodes |
| `mean_reward` | float | Mean cumulative episodic reward |
| `std_reward` | float | Reward standard deviation |
| `min_reward` | float | Minimum episodic reward |
| `max_reward` | float | Maximum episodic reward |
| `success_rate` | float | Fraction of successful episodes [0, 1] |
| `collision_rate` | float | Fraction of collision episodes [0, 1] |
| `mean_episode_length` | float | Mean steps per episode |
| `std_episode_length` | float | Episode length standard deviation |

### 4.2 Planner Metrics (PlannerEvaluationMetrics)

| Field | Type | Description |
|:------|:-----|:------------|
| `episodes` | int | Total evaluation episodes |
| `success_rate` | float | Fraction where a valid path was found |
| `mean_path_length` | Optional[float] | Mean path length (steps/distance) over successes (null if no successes) |
| `std_path_length` | Optional[float] | Path length standard deviation |
| `min_path_length` | Optional[float] | Minimum path length |
| `max_path_length` | Optional[float] | Maximum path length |
| `mean_planning_time` | float | Mean wall-clock planning time (seconds) |
| `std_planning_time` | float | Planning time standard deviation |
| `collision_rate` | Optional[float] | Always null for offline planners (no dynamic environment step execution) |

> [!IMPORTANT]
> Planner metrics are distinct from RL metrics. Path length and planning time
> are not applicable to RL agents. Reward is not applicable to planners.
> Never compare reward directly between RL and planners.

---

## 5. Seed Management

### 5.1 Unified Evaluation Seed Protocol

Every experiment uses a declared `seed` value and a centralized evaluation seeding protocol (`adaptive_rl.evaluation.seeding`):
- **Environment Procedural Generation**: Evaluation episodes for both RL agents and classical planners derive identical environment reset seeds:
  $$\text{seed}_{\text{env}}(i) = \text{derive\_evaluation\_seed}(\text{experiment\_seed}, i) = \text{experiment\_seed} + i$$
  This ensures benchmark fairness: an RL policy and a classical planner (A*, RRT*) are evaluated on the exact same procedural world layouts, start positions, and goal positions for episode $i$.
- **Planner-Internal Randomness**: Sampling-based planners (RRT*) decouple internal tree sampling randomness from environment generation via `derive_planner_seed`. Modifying planner parameters or sampling seeds does not alter the underlying test environment.
- **Manifest Provenance**: The complete list of evaluation seeds executed during the run is permanently recorded in `manifest.json` under `evaluation_seeds`.

### 5.2 Generalization Train/Test Split

The existing generalization framework (Phase 11) maintains strictly disjoint
seed distributions:
- Training seeds: `[1000, 1015)` (15 seeds)
- Test seeds: `[2000, 2015)` (15 seeds)

These ranges must never overlap. The framework includes an overlap check.

### 5.3 Determinism Limitations

The following components are fully deterministic given the same seed:
- A* planner path computation.
- GridWorld procedural generation.
- ContinuousNavigation2DEnv obstacle placement.

The following may not be bit-for-bit reproducible across platforms or torch versions:
- PPO/SAC neural network training (GPU/CPU floating-point differences).
- SB3 parallel environment sampling.

These limitations are documented honestly. We do not claim exact reproducibility
for neural network training, only seed-controlled statistical reproducibility.

---

## 6. Listing and Inspecting Experiments

```bash
# List all experiments
adaptive-rl experiment list

# Inspect a specific experiment
adaptive-rl experiment inspect 2026-09-17_gridworld_ppo_seed42

# Full terminal dashboard
adaptive-rl dashboard

# Compare two experiments
adaptive-rl dashboard --compare 2026-09-17_gridworld_ppo_seed42,2026-09-17_gridworld_astar_seed42
```

---

## 7. Benchmarking and Ablations

Multi-seed benchmarking runs the same configuration across multiple seeds
and reports aggregate statistics:

```bash
# Single config, default seeds [42, 43, 44]
adaptive-rl benchmark --config configs/gridworld_ppo.yaml

# Custom seeds
adaptive-rl benchmark --config configs/gridworld_ppo.yaml --seeds 42,43,44,45,46

# Ablation comparison
adaptive-rl benchmark \
  --config configs/gridworld_ppo.yaml \
  --compare configs/gridworld_astar.yaml \
  --seeds 42,43,44 \
  --output-report experiments/comparison_ppo_vs_astar.json
```

Reported statistics:
- **mean**: Average metric value across seeds.
- **std**: Standard deviation (uncertainty estimate).
- **min**: Best/worst case value.
- **max**: Best/worst case value.

> [!WARNING]
> Results from a small number of seeds (< 5) have high variance. Use more seeds
> for statistically meaningful comparisons. Never present a single-seed result
> as a reliable conclusion.

---

## 8. Logged Experiments

The following table records experiments that have been run during development.
Only actual runs are listed here — no fabricated results.

| Experiment | Phase | Environment | Algorithm | Timesteps | Status |
|:-----------|:------|:------------|:----------|:----------|:-------|
| Smoke tests | Phase 14 | GridWorld | A* | N/A (planner) | Infrastructure validated |

Full benchmark results require running `adaptive-rl benchmark` and will depend
on available compute. Sample configurations are provided in `configs/`.
