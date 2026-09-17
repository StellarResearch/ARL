# AdaptiveRL — Research Methodology and Baseline Design

This document describes the research approach, evaluation methodology, classical
baseline design, ablation framework, and limitations of the AdaptiveRL platform.

---

## 1. Overview

AdaptiveRL is a multi-environment reinforcement learning research platform, not a
claim to novel algorithmic contributions. Its value lies in:

- Providing reproducible training and evaluation infrastructure across diverse environments.
- Enabling principled comparison of RL algorithms against classical planning baselines.
- Supporting ablation studies on curriculum learning, environmental disturbances, and generalization.
- Encouraging honest reporting of what was measured versus what was claimed.

---

## 2. Supported Algorithms and Planners

### 2.1 RL Algorithms

| Algorithm | Type | Action Space | Source |
|:----------|:-----|:-------------|:-------|
| PPO | On-policy actor-critic | Discrete and continuous | Stable-Baselines3 |
| SAC | Off-policy max-entropy | Continuous only | Stable-Baselines3 |

Both algorithms are wrapped through the `BaseAlgorithm` interface. Hyperparameters
are configurable via YAML. No claim is made about optimality of default hyperparameters
— they are reasonable starting points for experimentation.

### 2.2 Classical Navigation Baselines

| Planner | Type | Environment | Deterministic |
|:--------|:-----|:------------|:--------------|
| A* | Shortest-path grid search | GridWorld (discrete) | Yes |
| RRT* | Sampling planner with rewiring heuristic | Navigation2D (continuous) | Probabilistic (seed-controlled) |

A* is implemented directly against the `GridWorldEnv` grid representation. It uses
the Manhattan distance heuristic, which is admissible and consistent for 4-connected
discrete grids. Euclidean and Chebyshev heuristics are also supported.

RRT* (Rapidly-exploring Random Tree Star) is implemented for continuous 2D motion planning
against `ContinuousNavigation2DEnv`.

#### Algorithmic Specification:
1. **Neighborhood Rule**: Euclidean ball of fixed radius `search_radius` centered at new sample $x_{\text{new}}$.
2. **Connection Radius Behavior**: Fixed connection radius ($r = \text{const}$). Bounded-neighborhood approximation; does not implement the shrinking radius $\gamma (\log(n)/n)^{1/d}$ required for theoretical asymptotic optimality (Karaman & Frazzoli, 2011).
3. **Nearest & Near-Node Selection**:
   - Nearest node: $x_{\text{nearest}} = \arg\min_{v \in V} \|v - x_{\text{rand}}\|_2$.
   - Near nodes: $V_{\text{near}} = \{v \in V \mid \|v - x_{\text{new}}\|_2 \le r\}$.
   - Best parent: $x_{\text{parent}} = \arg\min_{v \in V_{\text{near}}} \{c(v) + \|v - x_{\text{new}}\|_2 \mid \text{collision\_free}(v, x_{\text{new}})\}$.
4. **Rewiring Behavior**: For all $v \in V_{\text{near}}$, if $c(x_{\text{new}}) + \|x_{\text{new}} - v\|_2 < c(v)$ and segment is collision-free (without ancestor cycle), re-parents $v$ to $x_{\text{new}}$ and recursively propagates cost deltas down the subtree.
5. **Collision Checking Assumptions**: Linear interpolation at step resolution `collision_resolution`. Collision occurs if any sample falls within $r_{\text{obstacle}} + r_{\text{agent}}$ of a circular obstacle or outside arena perimeter walls. Kinodynamic/differential constraints are omitted (holonomic 2D).
6. **Stopping Criteria**: Terminates at `max_iterations`. Retains the lowest-cost path reaching within `goal_radius` of target coordinate, or reports explicit failure if unreachable.
7. **Theoretical Limitations**: Fixed connection radius does not provide formal asymptotic optimality as $n \to \infty$. Linear scan over tree nodes scales as $O(n^2)$ over iterations, suitable for benchmark iteration budgets ($\le 2000$ steps) rather than large-scale planning.

> [!IMPORTANT]
> Classical planners (A*, RRT*) are not RL algorithms. They have direct access to the geometry
> or obstacles at planning time. RL agents must learn a policy from interaction
> without direct access to the world map. This is a fundamental methodological
> difference and must be considered when interpreting comparisons.

---

## 3. Baseline Comparison Methodology

### 3.1 What Can Be Compared

| Metric | PPO | SAC | A* | RRT* |
|:-------|:----|:----|:---|:-----|
| Success Rate | ✓ | ✓ | ✓ | ✓ |
| Collision Rate | ✓ | ✓ | ✗ (null: offline search) | ✗ (null: offline search) |
| Episode Reward | ✓ | ✓ | ✗ (not applicable) | ✗ (not applicable) |
| Episode Length | ✓ | ✓ | ✗ (planner doesn't step) | ✗ (planner doesn't step) |
| Path Length | ✗ | ✗ | ✓ | ✓ |
| Planning Time | ✗ | ✗ | ✓ | ✓ |

### 3.2 Comparison Caveats

- **A* has full observability**: The A* planner receives the complete grid layout at
  planning time. RL agents observe only a local feature vector (normalized positions).
  This gives A* a structural advantage in success rate on solvable instances.
  
- **A* is not a policy**: A* computes a path per episode. It does not generalize
  a learned policy across novel grids. RL agents learn generalizable policies.
  
- **Success rate comparison is valid but limited**: A* should achieve near-100%
  success rate on any solvable grid. RL agents may achieve lower success rates
  but require no grid-specific knowledge.

- **Reward comparison is invalid**: A* does not accumulate reward in the RL sense.
  Comparing rewards between A* and PPO/SAC is methodologically incorrect.

### 3.3 Valid Research Questions

- "Does the RL policy approach A*'s success rate after sufficient training?"
- "How does the RL policy's path efficiency compare to the A*-optimal path?"
- "Does curriculum learning improve RL success rate faster than flat training?"
- "Does the agent generalize to unseen grid configurations?"

---

## 4. Ablation Study Design

### 4.1 Curriculum Ablation

Compare agents trained with and without curriculum learning:
- **Curriculum enabled**: Progressive obstacle density increase via `CurriculumTrainer`.
- **Curriculum disabled**: Fixed obstacle density throughout training.

Measure: success rate at fixed training budget (e.g. 50k timesteps).

### 4.2 Disturbance Ablation (Drone)

Compare drone agent performance under different conditions:
- **Baseline**: No wind or turbulence.
- **Wind**: Constant directional wind field.
- **Turbulence**: Ornstein-Uhlenbeck turbulence process.
- **Dynamic obstacles**: Moving obstacle agents.

Configs: `configs/drone.yaml`, `configs/drone_disturbed_ppo.yaml`,
`configs/drone_disturbed_sac.yaml`.

### 4.3 Algorithm Ablation

Compare PPO vs SAC on identical environments:
- **GridWorld**: PPO (discrete action space, SAC not applicable).
- **Navigation 2D**: Both PPO and SAC applicable.
- **Drone 3D**: Both PPO and SAC applicable.

### 4.4 Generalization Ablation

Measure performance degradation from training to held-out environments:
- **Train distribution**: Seeds [1000, 1015).
- **Test distribution**: Seeds [2000, 2015).
- **Overlap**: 0 (validated by design).

Metric: generalization gap = train_success_rate − test_success_rate.

---

## 5. Train/Test Protocol

All generalization experiments strictly separate training and evaluation seeds.

```
Training seeds: [1000, 1015)   (15 seeds)
Test seeds:     [2000, 2015)   (15 seeds)
Overlap:        0 seeds (verified)
```

Evaluation on training seeds measures optimization performance.
Evaluation on test seeds measures generalization.

The `GeneralizationDistribution` class enforces overlap validation:
```python
assert len(set(train_seeds) & set(test_seeds)) == 0
```

> [!CAUTION]
> Never evaluate on seeds that appeared during training. Doing so inflates
> test performance and produces unreliable generalization estimates.

---

## 6. Statistical Reporting Standards

### 6.1 Multi-Seed Evaluation

All benchmark results should be reported over at least 3 seeds (ideally 5+):
- Mean ± standard deviation.
- Min and max for range indication.

Single-seed results are acceptable for smoke tests and debugging but should not
be presented as scientific conclusions.

### 6.2 Metrics Reported

| Metric | Unit | Source |
|:-------|:-----|:-------|
| `success_rate` | fraction [0, 1] | Episode info `success` flag |
| `collision_rate` | fraction [0, 1] | Episode info `collision` flag |
| `mean_reward` | reward units | Cumulative per-episode reward |
| `std_reward` | reward units | Standard deviation over episodes |
| `mean_episode_length` | steps | Steps per episode |
| `mean_path_length` | steps | A* path length (steps, not reward) |
| `mean_planning_time` | seconds | Wall-clock per-episode planning time |
| `generalization_gap` | fraction | train_success − test_success |

### 6.3 Honest Language

This document and all associated README/documentation use the following conventions:

- **Implemented**: Code exists and passes tests.
- **Evaluated**: Results were actually measured and recorded.
- **Smoke tested**: A lightweight run was performed to verify the pipeline works.
- **Untested at scale**: Infrastructure exists but full training runs have not been recorded.

We do **not** use: "proves", "optimal", "state of the art", "production ready" without
supporting experimental evidence.

---

## 7. Limitations

### 7.1 Algorithmic Limitations

- **A* limited to discrete grids**: A* applies to discrete `GridWorldEnv`. For continuous 2D
  spaces (`Navigation2DEnv`), the platform uses RRT*. Continuous 3D planning (`Drone3DEnv`)
  currently relies on RL policies.

- **No true closed-loop planning**: A* computes an open-loop path at episode start.
  It does not replan if the environment changes mid-episode (as dynamic obstacles do).

- **No statistical significance testing**: The benchmarking framework reports
  descriptive statistics (mean ± std) but does not perform significance tests
  (e.g. Mann-Whitney U, t-test). Interpret comparisons accordingly.

### 7.2 Computational Limitations

- Training runs of 50k+ timesteps have not been systematically recorded in this
  codebase. The infrastructure exists; results require user-initiated training.

- Smoke tests use 500–2000 timesteps, which is insufficient for meaningful policy
  learning. They validate pipeline integrity, not algorithm performance.

### 7.3 Reproducibility Limitations

- PyTorch training may not be bit-for-bit reproducible across machines or versions.
- A* planning is fully deterministic given the same grid layout and seed.
- Environment generation is deterministic given the same seed.

---

## 8. Future Research Directions

The following are possible research extensions, not current claims:

- Extending sampling-based planners (e.g. RRT* or BIT*) to continuous 3D drone navigation.
- Multi-objective reward balancing for drone battery + success trade-offs.
- Meta-learning for faster adaptation to novel environments.
- Population-based training for hyperparameter optimization.
- Safety constraints and safe RL exploration.
