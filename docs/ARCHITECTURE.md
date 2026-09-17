# AdaptiveRL Architecture Specification

## 1. System Overview

**AdaptiveRL** is a modular reinforcement learning framework engineered to train, evaluate, and benchmark agents across multiple diverse environments. Rather than tightly coupling agent algorithms to a specific task, AdaptiveRL separates the problem into clear, orthogonal layers interacting through standardized interfaces.

```
                         +-----------------------------------+
                         |       Configuration Layer         |
                         |   (YAML Schemas & Seed Manager)   |
                         +-----------------+-----------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |  Environment Layer    |                     |    Algorithm Layer    |
        |  - Gymnasium API      |                     |  - BaseAlgorithm      |
        |  - Registry & Factory |                     |  - SB3 Wrappers (PPO) |
        +-----------+-----------+                     +-----------+-----------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
                         +-----------------------------------+
                         |          Training Engine          |
                         |  (Trainer, Callbacks, Checkpoints)|
                         +-----------------+-----------------+
                                           |
                                           v
                         +-----------------------------------+
                         |         Evaluation Engine         |
                         |  (Metrics, Scenarios, Benchmarks) |
                         +-----------------+-----------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        +-----------------------+                     +-----------------------+
        |   Model Management    |                     | Visualization Engine  |
        | (Artifacts & Metadata)|                     |  (Plots & Renderers)  |
        +-----------------------+                     +-----------------------+
```

---

## 2. Core Architectural Decisions

### 2.1 Why the `src/` Layout?
AdaptiveRL utilizes the `src/` layout (`src/adaptive_rl/`) instead of a flat root structure:
* **Import Parity:** Prevents tests and scripts from accidentally importing the raw working directory instead of the installed package.
* **Packaging Reliability:** Ensures editable installs (`pip install -e .`) and distributed wheel builds reflect the true installed artifact.
* **Tooling Compatibility:** Cleanly isolates tooling caches (`.pytest_cache`, `.mypy_cache`, `.ruff_cache`) and prevents namespace pollution.

### 2.2 Separation Between Environments and Algorithms
Algorithms never import or depend directly on concrete environments.
* **Gymnasium Contract:** All environments conform strictly to Farama Gymnasium semantics (`observation_space`, `action_space`, `reset(seed=...)`, `step(action)`).
* **Algorithm Invariance:** The RL algorithm operates purely on abstract tensor or array spaces without knowing whether the domain is a 2D GridWorld, continuous navigation, traffic intersection, or 3D drone simulation.

### 2.3 Environment Registry Concept
The environment registry acts as a dynamic service locator and factory:
* Environments register with unique identifier keys (e.g. `gridworld`, `navigation`, `drone`).
* The training engine requests `make_env("environment_name", config)` dynamically.
* Protects against duplicate registrations and gives helpful diagnostics when an unrecognized environment is requested.

### 2.4 Training and Evaluation Separation
* **Training (`adaptive_rl.training`):** Focuses exclusively on optimizing policy weights, updating value functions, logging progression, and managing periodic checkpoints.
* **Evaluation (`adaptive_rl.evaluation`):** Evaluates policies deterministically over fixed episode sets, computing uncorrupted metrics (success rate, collision rate, average episodic return, episode length). Training and evaluation scenarios remain strictly isolated.

### 2.5 Configuration System
* Declarative YAML configurations define all experiment parameters (algorithm hyperparameters, environment settings, training timesteps, seed, logging paths).
* Configurations are parsed into strongly-typed Pydantic models (`ExperimentConfig`), catching typos, invalid ranges, and missing fields early with actionable error messages.
* Determinism is enforced by setting seeds across Python `random`, NumPy, PyTorch, and Gymnasium spaces simultaneously.

### 2.6 Model Management
* Models are saved with comprehensive metadata (`ModelMetadata`) recording:
  * Algorithm and version
  * Training environment name
  * Total timesteps trained
  * Hyperparameter dictionary
  * Checkpoint timestamp and path
* Prevents attempting to evaluate or deploy models into mismatched action/observation spaces.

### 2.7 Visualization and Decoupled Rendering
* Rendering logic (`BaseRenderer`) is completely decoupled from environment physics.
* Environments can run in headless mode (e.g., in continuous integration) at maximum execution speed without requiring graphics displays or X11/OpenGL servers.
* Plotting utilities (`PlotManager`) generate training curve visualizations from structured logs without touching the active training process.

### 2.8 Experiment Lifecycle Management
* Every experiment run produces a self-contained artifact bundle in `experiments/results/` and `experiments/logs/`.
* The saved configuration file guarantees that any experiment can be reproduced by another researcher with a single CLI command.

### 2.9 Environment Abstraction & Factory Flow (Environment → Registry → Factory → Training Engine)

The relationship between environments and the training engine follows a strict unidirectional dependency flow mediated by the registry and factory:

```
+------------------+         registers         +------------------------+
| Concrete Env     | ----------------------->  |  Environment Registry  |
| (e.g. GridWorld) |                           |  (Name -> Factory Map) |
+------------------+                           +-----------+------------+
                                                           | resolves
                                                           v
+------------------+       instantiates        +------------------------+
| Training Engine  | <------------------------ |  Environment Factory   |
| (Trainer)        |                           |  (make_env / create)   |
+------------------+                           +------------------------+
```

#### Why the Training Engine Must Depend on the Interface Rather Than Concrete Environments:
1. **Zero Domain Coupling:** The training engine (`Trainer`) requires only that the target environment implements `gymnasium.Env` (`reset() -> (obs, info)`, `step(action) -> (obs, reward, terminated, truncated, info)`). It has zero knowledge of grid cells, lidar rays, traffic signals, or quadrotor equations of motion.
2. **Pluggable Architecture:** Any new environment (e.g. `TrafficEnv` or `DroneNavigationEnv`) can be plugged in simply by defining the Gymnasium subclass and registering it with `register("env_name", factory)`. No code in the training engine, algorithm wrappers, or checkpointing routines needs to change.
3. **Seamless Benchmark Portability:** Standard third-party environments (such as Gymnasium's `CartPole-v1`, `Pendulum-v1`, or `BipedalWalker-v3`) can be trained using the exact same CLI command and training engine without wrapping or rewriting them.
4. **Isolated Testability:** Test suites can use lightweight dummy environments (like `DummyTestEnv`) to test the registry, vectorization, and training loops rapidly without incurring heavy simulation overhead.

---

## 3. Phase 12–17: Classical Baselines, Registries, and Experiment Management

### 3.1 Planner Layer (`adaptive_rl.planners`)

Classical navigation planners are implemented as a distinct layer separate from
RL algorithms. This separation respects the fundamental difference between:

- **RL policies**: Learned, stochastic, model-free, generalize through interaction.
- **Classical planners**: Deterministic, model-based, compute solutions per-instance.

```
BasePlanner (ABC)
  └── AStarPlanner
        └── PlannerAdapter (GridWorldEnv integration)
```

Planners expose the same success/collision metrics as RL agents where applicable,
but have planner-specific metrics (path length, planning time) that RL agents do not.
The `PlannerAdapter` runs the planner on the same GridWorldEnv layouts as RL evaluation,
enabling direct success-rate comparison.

### 3.2 Algorithm Registry (`adaptive_rl.algorithms.registry`)

The `AlgorithmRegistry` mirrors the `EnvironmentRegistry` design:
- Both RL algorithms (PPO, SAC) and planners (A*) are registered.
- Each registration includes typed `AlgorithmMetadata` with capability flags.
- `AlgorithmKind.RL_POLICY` vs `AlgorithmKind.PLANNER` prevents type confusion.
- CLI commands (`adaptive-rl algorithm list`, `algorithm inspect <name>`) expose the registry.

```python
from adaptive_rl.algorithms.registry import algorithm_registry, AlgorithmKind
rl_algos = algorithm_registry.list_by_kind(AlgorithmKind.RL_POLICY)   # ['ppo', 'sac']
planners = algorithm_registry.list_by_kind(AlgorithmKind.PLANNER)     # ['astar']
```

### 3.3 Experiment Manager (`adaptive_rl.experiments.manager`)

`ExperimentManager` provides reproducible experiment orchestration:

```
run(config) → ExperimentResult
  ├── Generate experiment_id (date + env + algo + seed)
  ├── Create output directory structure
  ├── Record provenance (git commit, Python, packages, platform)
  ├── Save config copy
  ├── Run training (RL) or planning (planner)
  ├── Evaluate performance
  ├── Save metrics.json + metrics.csv
  └── Save manifest.json
```

Every experiment is self-contained and independently reproducible from its `config.yaml` copy.

### 3.4 Benchmarking Framework (`adaptive_rl.benchmarking`)

`BenchmarkRunner` extends `ExperimentManager` with multi-seed execution:
- Runs the same configuration across N seeds.
- Aggregates statistics (mean ± std, min, max) per metric.
- Supports two-arm comparisons (ablations).
- Saves `ComparisonReport` as JSON.

### 3.5 Dashboard (`adaptive_rl.visualization.dashboard`)

The terminal dashboard uses `rich` (already a project dependency) to render
experiment results from saved artifacts:

- **Overview**: All experiments with status, algorithm, environment, seed, timestamp.
- **Detail**: Metrics for a single experiment, including a text sparkline for reward.
- **Comparison**: Side-by-side metric table for multiple experiments.

No external plotting libraries are required. The dashboard is purely text-based,
works over SSH, and does not require a graphical environment.

---

## 4. Complete Module Reference

```
src/adaptive_rl/
├── __init__.py               — Package root and version
├── cli.py                    — Typer CLI (all commands)
├── config.py                 — Pydantic configuration schemas
├── algorithms/
│   ├── base.py               — BaseAlgorithm abstract interface
│   ├── ppo.py                — SB3 PPO wrapper
│   ├── sac.py                — SB3 SAC wrapper
│   └── registry.py           — AlgorithmRegistry (Phase 13)
├── planners/
│   ├── base.py               — BasePlanner + PlannerResult (Phase 12)
│   ├── astar.py              — A* planner implementation (Phase 12)
│   └── adapter.py            — PlannerAdapter for GridWorldEnv (Phase 12)
├── environments/
│   ├── base.py               — AdaptiveRLEnv abstract base
│   ├── registry.py           — EnvironmentRegistry
│   ├── metadata.py           — EnvironmentMetadata
│   ├── seeded_wrapper.py     — TrainingDistributionWrapper
│   ├── testing.py            — DummyTestEnv
│   ├── gridworld/            — GridWorld environment (Phase 3)
│   ├── navigation/           — ContinuousNavigation2D (Phase 6)
│   ├── traffic/              — TrafficSignalEnv (Phase 8)
│   └── drone/                — DroneNavigation3D + DroneDisturbance3D (Phase 9-10)
├── training/
│   ├── trainer.py            — PPOTrainer, SACTrainer, get_trainer
│   ├── callbacks.py          — MetricLoggerCallback, CheckpointCallback
│   └── checkpointing.py      — CheckpointManager
├── evaluation/
│   ├── evaluator.py          — Evaluator, BaseEvaluator
│   ├── metrics.py            — EvaluationMetrics (Phase 16)
│   ├── generalization.py     — GeneralizationReport, GeneralizationDistribution
│   └── scenarios.py          — EvaluationScenario
├── experiments/
│   ├── runner.py             — BaseExperimentRunner
│   ├── generalization_runner.py — GeneralizationExperimentRunner (Phase 11)
│   └── manager.py            — ExperimentManager (Phase 14)
├── benchmarking/
│   └── __init__.py           — BenchmarkRunner, AggregateStats (Phase 15)
├── curriculum/               — Curriculum learning (Phase 7)
├── rewards/                  — Reward function interfaces
├── models/                   — Model artifact management
└── visualization/
    ├── plots.py              — PlotManager (text summaries)
    ├── renderer.py           — BaseRenderer
    └── dashboard.py          — Rich terminal dashboard (Phase 17)
```
