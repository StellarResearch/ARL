# AdaptiveRL — Multi-Environment Reinforcement Learning Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**AdaptiveRL** is a modular, multi-environment reinforcement learning framework designed to train, evaluate, and benchmark adaptive agents across diverse problem domains—eventually scaling to autonomous 3D drone navigation in complex, dynamic obstacle fields.

---

## 1. Project Goals

* **Domain-Agnostic Core:** Decouple reinforcement learning algorithms from environment specifics using standardized Farama Gymnasium interfaces.
* **Algorithm Adapters:** Wrap battle-tested algorithms (such as Stable-Baselines3 PPO and SAC) behind unified agent interfaces rather than reinventing algorithms from scratch.
* **Reproducibility First:** Enforce deterministic seeding and declarative YAML configuration schemas for every experiment.
* **Progressive Benchmarking:** Progress through discrete GridWorld, continuous 2D navigation, traffic flow control, and autonomous 3D drone navigation.
* **Contributor Friendly:** Modern Python packaging (`src/` layout, `pyproject.toml`), automated test suites, type checking, and clean development workflows.

---

## 2. Current Platform

AdaptiveRL is a reusable Gymnasium-based RL platform with a drone-navigation
flagship and reproducible scientific evaluation. It currently provides:

* GridWorld, continuous 2D navigation, traffic control, Drone3D, and disturbed Drone3D environments.
* PPO and SAC adapters backed by Stable-Baselines3.
* A* and RRT* classical baselines for planner comparisons.
* Curriculum learning, multi-seed benchmarking, and disjoint-distribution generalization tests.
* Experiment manifests containing configuration, environment, seed, package, and Git provenance.
* CLI workflows for inspection, training, evaluation, experiments, benchmarking, and reporting.

---

## 3. Architecture at a Glance

```
adaptive-rl/
├── configs/                   # Declarative YAML experiment configurations
│   ├── ppo.yaml
│   ├── sac.yaml
│   ├── navigation.yaml
│   ├── curriculum_navigation.yaml
│   ├── curriculum_gridworld.yaml
│   └── drone.yaml
├── docs/                      # Architectural specifications and research design
│   ├── ARCHITECTURE.md
│   ├── RESEARCH.md
│   └── EXPERIMENTS.md
├── experiments/               # Experiment output directories (.gitignore tracked)
│   ├── results/
│   └── logs/
├── src/
│   └── adaptive_rl/           # Core platform package
│       ├── algorithms/        # Base algorithm interfaces and SB3 adapters (PPO, SAC)
│       ├── environments/      # Gymnasium contracts, registry, and environments
│       ├── planners/          # Classical baselines (A*, RRT*, make_planner, PlannerAdapter)
│       ├── benchmarking/      # Multi-seed benchmarking and ablation runner
│       ├── curriculum/        # Staged curriculum managers, wrappers, and callbacks
│       ├── rewards/           # Modular reward function base interfaces
│       ├── training/          # Trainers, callbacks, and checkpoint managers
│       ├── evaluation/        # Benchmark evaluators, metrics, and scenarios
│       ├── models/            # Model artifact storage and metadata management
│       ├── visualization/     # Renderers, plot generation, and terminal dashboard
│       ├── experiments/       # Experiment orchestration manager and manifests
│       ├── config.py          # Pydantic schema validation & YAML parser
│       └── cli.py             # Typer command-line interface
└── tests/                     # Automated pytest suite
```

For in-depth architectural principles, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Installation & Setup

### Prerequisites
* Python 3.10, 3.11, or 3.12 (tested and validated in CI)
* `git`

### Quick Start
```bash
# 1. Clone the repository
git clone https://github.com/ashishsinghbora/ARL.git
cd ARL

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install AdaptiveRL (minimal: planners, config, CLI)

# Or install the optional desktop research interface
pip install -e ".[studio]"
pip install -e .

# Or install for full development (RL engines, SB3, PyTorch, dev tools)
pip install -e ".[all]"
```

---

## 5. Command-Line Interface (CLI)

AdaptiveRL includes a CLI tool (`adaptive-rl`):

```bash
# View help and available commands
adaptive-rl --help

# Show installed version and current milestone
adaptive-rl version


# Launch AdaptiveRL Studio desktop interface
adaptive-rl studio
```

### AdaptiveRL Studio

AdaptiveRL Studio is an optional PySide6 desktop control center for the existing
framework. It provides an experiment overview, environment episode visualization,
background training through the existing trainer API, and experiment artifact
inspection. It does not implement a second RL engine.

Use `--output-dir` to point Studio at another experiment artifact directory:

```bash
adaptive-rl studio --output-dir experiments/results
```
# Inspect development roadmap and completed phases
adaptive-rl info

# List and inspect registered environments
adaptive-rl env list
adaptive-rl env inspect gridworld

# Run an interactive or simulated rollout
adaptive-rl env run gridworld --steps 15 --seed 42
```

---

## 6. Procedural GridWorld Environment

AdaptiveRL provides a procedurally generated 2D grid navigation environment compliant with the Farama Gymnasium contract.

* **Observation Space:** `Box(4,)` containing normalized coordinates `[agent_x, agent_y, goal_x, goal_y]`.
* **Action Space:** `Discrete(4)` corresponding to `UP (0)`, `DOWN (1)`, `LEFT (2)`, `RIGHT (3)`.
* **Rewards:** `+100.0` for reaching goal, `-100.0` for obstacle collision, `-1.0` per step.
* **Solvability Guarantee:** Breadth-First Search (BFS) path verification guarantees a valid collision-free path exists for every generated obstacle layout.

### Python Example

```python
from adaptive_rl.environments import make_env

# Instantiate via factory with custom dimensions and obstacle count
env = make_env("gridworld", width=6, height=5, num_obstacles=3, max_steps=50)

obs, info = env.reset(seed=42)
print("Initial observation:", obs)
env.render()

# Step through the environment
obs, reward, terminated, truncated, info = env.step(1)  # DOWN
env.close()
```

---

## 7. PPO Training Engine

AdaptiveRL features an end-to-end PPO training engine integrating Stable-Baselines3 with automated metric logging callbacks and model checkpoint management.

### Training via CLI

```bash
# Train on GridWorld with PPO for 5,000 steps
adaptive-rl train --config configs/gridworld_ppo.yaml --timesteps 5000

# Train on standard CartPole baseline
adaptive-rl train --config configs/ppo.yaml --timesteps 10000
```

### Training via Python API

```python
from adaptive_rl.config import load_config
from adaptive_rl.training import PPOTrainer

# 1. Load experiment configuration
config = load_config("configs/gridworld_ppo.yaml")

# 2. Instantiate trainer and run optimization
trainer = PPOTrainer(config=config)
result = trainer.fit()

print(f"Trained {result.total_timesteps} steps across {result.episodes_completed} episodes.")
print(f"Final model saved to: {result.final_model_path}")
print(f"Saved {len(result.checkpoints)} periodic checkpoints.")
```

---

## 8. Evaluation Engine & Standard Metrics

AdaptiveRL provides a standardized evaluation benchmark engine to measure policy performance across fixed episode sets and configurable scenarios, with automatic export to JSON reports.

* **Standard Metrics Tracked:** Mean episodic return ± standard deviation, min/max returns, success rate, collision rate, and mean episode length ± standard deviation.
* **Deterministic Seeding:** Enforces deterministic seeding across evaluation episodes and environments.
* **Scenario Testing:** Benchmarks agents across curated challenge scenarios (e.g. varying obstacle densities).

### Evaluation via CLI

```bash
# Evaluate a trained model over 20 deterministic episodes
adaptive-rl evaluate --config configs/gridworld_ppo.yaml --model experiments/results/models/gridworld_ppo_baseline_final.zip --episodes 20

# Export structured JSON metrics report
adaptive-rl evaluate --config configs/gridworld_ppo.yaml --model experiments/results/models/gridworld_ppo_baseline_final.zip --episodes 20 --output-report experiments/results/eval_report.json
```

### Evaluation via Python API

```python
from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation import Evaluator, EvaluationScenario

# 1. Instantiate environment and loaded agent
env = make_env("gridworld", width=6, height=5, num_obstacles=3)
algo = PPOAlgorithm.from_pretrained(
    "experiments/results/models/gridworld_ppo_baseline_final.zip", env=env
)

# 2. Run multi-episode evaluation
evaluator = Evaluator(algorithm=algo, env=env)
metrics = evaluator.evaluate(num_episodes=20, deterministic=True, base_seed=42)

print(f"Mean Return: {metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
print(f"Success Rate: {metrics.success_rate * 100:.1f}%")
print(f"Collision Rate: {metrics.collision_rate * 100:.1f}%")

# 3. Export JSON report
evaluator.save_report(metrics, "experiments/results/eval_report.json")
```

---

## 9. Continuous 2D Navigation & SAC Algorithm

Continuous navigation introduces continuous action space control and distance-based rangefinder (LiDAR) sensing.

* **Continuous Action Space:** `Box(-1.0, 1.0, shape=(2,))` governing continuous 2D planar velocity $[v_x, v_y]$.
* **14-Dimensional Observation Space:**
  * Normalized agent coordinates $[x/W, y/H] \in [0, 1]^2$
  * Normalized target goal coordinates $[g_x/W, g_y/H] \in [0, 1]^2$
  * Relative target offset vector $[(g_x - x)/W, (g_y - y)/H] \in [-1, 1]^2$
  * 8-Ray LiDAR distance readings normalized to $[0, 1]$ computed via analytical ray-casting against circular obstacles and arena perimeter walls.
* **Collision Physics & Rewards:** Non-overlapping procedural circular obstacles with safety margins around start/goal. Terminal rewards: $+100.0$ for goal arrival, $-100.0$ for collision with obstacle/wall, plus dense progress shaping.
* **Soft Actor-Critic (SAC) Engine:** SB3-backed `SACAlgorithm` wrapper and `SACTrainer` pipeline for sample-efficient continuous actor-critic optimization.

### Continuous Navigation via CLI

```bash
# Simulate 10 continuous navigation steps with ASCII visualization
adaptive-rl env run navigation --steps 10 --seed 42

# Train SAC continuous control agent on navigation
adaptive-rl train --config configs/navigation.yaml --timesteps 10000
```

### Continuous Navigation via Python API

```python
import numpy as np
from adaptive_rl.environments import make_env
from adaptive_rl.algorithms import SACAlgorithm

# 1. Instantiate continuous navigation environment
env = make_env("navigation", arena_width=20.0, arena_height=20.0, num_obstacles=5)
obs, info = env.reset(seed=42)

# 2. Train SAC agent
agent = SACAlgorithm(env=env, learning_rate=3e-4, buffer_size=50000)
agent.train(total_timesteps=10000)

# 3. Predict continuous velocity action
action, _ = agent.predict(obs, deterministic=True)
obs, reward, terminated, truncated, info = env.step(action)
print(f"Action: {action}, Reward: {reward:.2f}, Dist to Goal: {info['distance_to_goal']:.2f}")
env.close()
```

---

## 10. Curriculum Learning Engine

AdaptiveRL features a flexible, automated curriculum learning subsystem that gradually escalates task complexity based on empirical agent proficiency.

* **Curriculum Stages:** Encapsulate environmental complexity parameters (e.g. obstacle density, arena dimensions), advancement criteria (rolling success rate $\ge$ threshold, mean reward $\ge$ threshold), and maximum timestep timeouts.
* **Curriculum State Machine:** `Curriculum` tracks the active milestone, evaluates graduation rules over a rolling window, records stage transition events with timestamps, and exports JSON audit reports.
* **Dynamic Gymnasium Wrapper:** `CurriculumEnvWrapper` intercepts resets and steps, injecting the active stage's configuration parameters directly into the environment without re-instantiation.
* **Callback Coordination:** `CurriculumCallback` bridges training optimization loops and curriculum state, triggering seamless transitions when graduation thresholds are achieved.
* **Built-in Presets:**
  * **Navigation (4 tiers):** `Clear Corridor` (0 obstacles) $\rightarrow$ `Sparse Clutter` (2 obstacles) $\rightarrow$ `Standard Density` (5 obstacles) $\rightarrow$ `Dense Hazard Field` (8 obstacles).
  * **GridWorld (4 tiers):** `Open Grid` (4x4, 0 obstacles) $\rightarrow$ `Light Clutter` (5x5, 2 obstacles) $\rightarrow$ `Standard Grid` (6x5, 4 obstacles) $\rightarrow$ `Dense Labyrinth` (7x7, 6 obstacles).

### Curriculum via CLI

```bash
# List available curriculum presets
adaptive-rl curriculum list

# Inspect stages, progression thresholds, and parameters of a preset
adaptive-rl curriculum inspect navigation

# Train agent with automated curriculum progression
adaptive-rl train --config configs/curriculum_navigation.yaml
```

### Curriculum via Python API

```python
from adaptive_rl.config import load_config
from adaptive_rl.curriculum import CurriculumTrainer

# 1. Load experiment configuration with curriculum block enabled
config = load_config("configs/curriculum_navigation.yaml")

# 2. Train agent with automated curriculum transitions
trainer = CurriculumTrainer(config=config)
result = trainer.fit()

print(f"Trained {result.total_timesteps} steps across {result.episodes_completed} episodes.")
print(f"Final model saved to: {result.final_model_path}")
```

---

## 11. Traffic Signal Optimization Environment

Traffic demonstrates the domain-agnostic capability of AdaptiveRL through a discrete, non-spatial queuing optimization benchmark: a **4-way signalized intersection** (`TrafficSignalEnv`, registered as `traffic` and `traffic_signal`).

* **Intersection Queuing Dynamics:**
  * 4 directional approach lanes: **North (N)**, **South (S)**, **East (E)**, and **West (W)**.
  * Stochastic Poisson arrival process per approach with configurable arrival rates $\lambda = (\lambda_N, \lambda_S, \lambda_E, \lambda_W)$.
  * Saturation discharge throughput: Green approaches discharge up to `departure_rate` vehicles per step; Red approaches discharge 0.
  * FIFO vehicle delay tracking: Accurately records individual waiting time, cumulative delay, and maximum waiting times.
* **Farama Gymnasium Spaces:**
  * **Observation Space:** `Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)`
    * Normalized queue lengths: $[q_N, q_S, q_E, q_W] / \text{max\_queue}$
    * Normalized waiting times: $[w_N, w_S, w_E, w_W] / \text{max\_wait\_limit}$
    * Current signal phase: $0.0$ for North-South Green, $1.0$ for East-West Green
    * Phase duration ratio: $\min(\text{duration} / \text{max\_phase\_duration}, 1.0)$
  * **Action Space:** `Discrete(2)`
    * `0`: North-South Green (East & West Red)
    * `1`: East-West Green (North & South Red)
* **Multi-Objective Reward Function:**
  $$R_t = c_{\text{dep}} \cdot \Delta_{\text{departures}} - c_q \sum q_i - c_w \max(w_i) - c_{\text{switch}} \cdot \mathbb{I}_{\text{switch}} - c_{\text{prem}} \cdot \mathbb{I}_{\text{premature}}$$
  Incentivizes clearing vehicle queues while penalizing excessive signal flickering and premature phase switching before minimum green time.
* **ASCII Visualizer:**
```
+--------------------------------------------------+
|   4-WAY SIGNALIZED INTERSECTION OPTIMIZATION     |
+--------------------------------------------------+
| Phase: NORTH_SOUTH (Green)  Step: 012/100        |
| Duration: 04 | Switches: 02 | Cleared: 018       |
+--------------------------------------------------+
                 |   N   |                          
                 | Q:02  | (Wait: 03)             
                 |  [G]  |                          
  ---------------+       +---------------           
   W  Q:04  [R]              [R]  Q:01  E    
  (Wait: 08)                       (Wait: 02)     
  ---------------+       +---------------           
                 |  [G]  |                          
                 | Q:01  | (Wait: 01)             
                 |   S   |                          
+--------------------------------------------------+
  Queues: [N=2, S=1, E=1, W=4] | Total: 08
+--------------------------------------------------+
```

### Traffic Signal via CLI

```bash
# Simulate 10 traffic steps with live ASCII rendering
adaptive-rl env run traffic --steps 10 --seed 42

# Inspect spaces and metadata
adaptive-rl env inspect traffic

# Train PPO agent on traffic signal optimization
adaptive-rl train --config configs/traffic_ppo.yaml
```

### Traffic Signal via Python API

```python
from adaptive_rl.environments import make_env
from adaptive_rl.algorithms import PPOAlgorithm

# 1. Create 4-way traffic signal environment
env = make_env("traffic", max_steps=100, arrival_rates=(0.35, 0.35, 0.2, 0.2))
obs, info = env.reset(seed=42)

# 2. Train PPO policy
agent = PPOAlgorithm(env=env, learning_rate=3e-4)
agent.train(total_timesteps=10000)

# 3. Step environment with optimized signal controls
action, _ = agent.predict(obs, deterministic=True)
obs, reward, terminated, truncated, step_info = env.step(action)
print(
    f"Phase: {step_info['phase_name']}, Total Queue: {step_info['total_queue']}, Reward: {reward:.2f}"
)
env.close()
```

---

## 12. Autonomous 3D Drone Navigation Environment

The flagship environment is a continuous 3D quadrotor flight environment (`DroneNavigation3DEnv`, registered as `drone`, `drone_3d`, and `drone_navigation`), combining second-order translation kinematics, aerodynamic drag damping, procedural 3D spherical obstacle fields, and multi-directional 3D spherical LiDAR rangefinders.

* **3D Kinematic Physics Model:**
  * Translational state: position $\mathbf{p} = [x, y, z]^T \in [0, X_{\max}] \times [0, Y_{\max}] \times [0, Z_{\max}]$, velocity $\mathbf{v} = [v_x, v_y, v_z]^T$, and acceleration $\mathbf{a} = [a_x, a_y, a_z]^T$.
  * Equations of motion:
    $$\frac{d\mathbf{v}}{dt} = \mathbf{a} - c_d \mathbf{v}$$
    $$\frac{d\mathbf{p}}{dt} = \mathbf{v}$$
    Integrated via semi-implicit Euler integration with maximum velocity spherical clamping ($\|\mathbf{v}\| \le v_{\max}$).
* **Farama Gymnasium Spaces:**
  * **Continuous Action Space:** `Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)` representing commanded 3D accelerations $[a_x, a_y, a_z]$, scaled by $a_{\max} = 4.0\ \text{m/s}^2$.
  * **Continuous Observation Space:** `Box(low=-1.0, high=1.0, shape=(29,), dtype=np.float32)`
    * $[0:3]$: Normalized 3D position $\mathbf{p} / \mathbf{B} \in [0, 1]^3$
    * $[3:6]$: Normalized 3D velocity $\mathbf{v} / v_{\max} \in [-1, 1]^3$
    * $[6:9]$: Normalized 3D waypoint target $\mathbf{g} / \mathbf{B} \in [0, 1]^3$
    * $[9:12]$: Relative target vector $(\mathbf{g} - \mathbf{p}) / \mathbf{B} \in [-1, 1]^3$
    * $[12]$: Normalized Euclidean distance to goal $\|\mathbf{g} - \mathbf{p}\| / D_{\max} \in [0, 1]$
    * $[13:29]$: 16-ray 3D LiDAR distance rangefinder readings $\in [0, 1]^{16}$ (8 horizontal equatorial rays, 4 upper hemispheric rays $+45^\circ$, 4 lower hemispheric rays $-45^\circ$) computed via analytical 3D ray-sphere and ray-box slab intersections.
* **Procedural Obstacle Field:**
  * Generates non-overlapping spherical obstacles with guaranteed safe radius clearance around both drone takeoff position and destination waypoint.
* **Reward Structure:**
  * $+100.0$: Target waypoint reached within `target_radius`.
  * $-100.0$: Collision with obstacle sphere or bounding perimeter wall.
  * $+w_{\text{progress}} \cdot (d_{t-1} - d_t)$: Dense potential-based progress reward.
  * $-0.05$: Per-step time penalty.
  * $-0.01 \cdot \|\mathbf{a}\|^2$: Action effort / control smoothness regularization.
* **ASCII 3D Flight Deck:**
```
+----------------------------------------------------------------+
|               AUTONOMOUS 3D DRONE FLIGHT DECK                  |
+----------------------------------------------------------------+
| Step: 014/300 | Altitude (Z):  12.4m | Speed:  4.2 m/s         |
| Position [X, Y, Z]: [ 18.2,  22.1,  12.4]                      |
| Velocity [Vx,Vy,Vz]: [  2.8,   3.1,   0.5]                     |
| Waypoint [Gx,Gy,Gz]: [ 45.0,  45.0,  20.0]                     |
| Range to Target:  35.6m | Obstacles in Area: 08                |
+----------------------------------------------------------------+
  Flight Arena Boundaries: [0..50, 0..50, 0..25] m
+----------------------------------------------------------------+
```

### Drone Navigation via CLI

```bash
# Simulate 10 continuous 3D drone steps with live flight deck visualization
adaptive-rl env run drone --steps 10 --seed 42

# Inspect 3D observation and action spaces
adaptive-rl env inspect drone

# Train continuous PPO on 3D drone navigation
adaptive-rl train --config configs/drone.yaml

# Train continuous SAC on 3D drone navigation
adaptive-rl train --config configs/drone_sac.yaml
```

### Drone Navigation via Python API

```python
import numpy as np
from adaptive_rl.environments import make_env
from adaptive_rl.algorithms import SACAlgorithm

# 1. Instantiate 3D continuous drone environment
env = make_env("drone", bounds=(50.0, 50.0, 25.0), num_obstacles=8)
obs, info = env.reset(seed=42)

# 2. Train SAC agent for continuous 3D control
agent = SACAlgorithm(env=env, learning_rate=3e-4, buffer_size=50000)
agent.train(total_timesteps=100000)

# 3. Predict continuous 3D acceleration command
action, _ = agent.predict(obs, deterministic=True)
obs, reward, terminated, truncated, step_info = env.step(action)
print(
    f"Altitude: {step_info['altitude']:.1f}m, Distance: {step_info['distance_to_goal']:.1f}m, Reward: {reward:.2f}"
)
env.close()
```

---

## 13. Drone Disturbances and Constraints

The disturbed drone environment extends 3D quadrotor flight navigation with realistic atmospheric disturbances, electro-mechanical energy constraints, and moving obstacle hazards.

### Environmental Features:
1. **3D Atmospheric Wind Field**:
   - Prevailing steady wind current $[w_x, w_y, w_z]$.
   - Linear altitude shear gradient: wind velocity scales higher with altitude $z$.
   - **Ornstein-Uhlenbeck Stochastic Gust Turbulence**: mean-reverting continuous Brownian velocity perturbations ($dg = -\theta g\,dt + \sigma\sqrt{dt}N(0, I)$).
2. **Quadrotor Battery Depletion Model**:
   - Power consumption dynamically modeled: $P_{\text{total}} = P_{\text{base}} + c_{\text{thrust}}\|a\|^2 + c_{\text{speed}}\|v\|^2$.
   - Battery state-of-charge tracking ($\text{SoC} \in [0.0, 1.0]$) with termination penalty upon complete exhaustion.
3. **Dynamic 3D Moving Obstacles**:
   - Autonomous spherical obstacles with velocity vectors $[\dot{x}, \dot{y}, \dot{z}]$ and elastic perimeter reflection upon hitting arena boundary walls.
   - Combined static and dynamic obstacle distance measuring via unified 16-ray spherical LiDAR.
4. **Observation & Action Spaces**:
   - **Action**: Continuous 3D acceleration command $\mathbf{a} \in [-1.0, 1.0]^3$.
   - **Observation**: 33-dimensional normalized continuous vector including drone coordinates, velocities, goal vector, 16-ray LiDAR distances, battery charge level, and instantaneous 3D wind velocity.

### CLI Usage:
```bash
# Run 10 steps of disturbed drone simulation
adaptive-rl env run drone_disturbed --steps 10 --seed 42

# Inspect disturbed drone environment spaces and registration
adaptive-rl env inspect drone_disturbed

# Inspect 4-stage progressive disturbance curriculum
adaptive-rl curriculum inspect drone_disturbed

# Train PPO under wind disturbances and battery constraints
adaptive-rl train --config configs/drone_disturbed_ppo.yaml
```

---

## 14. Generalization to Unseen Environments Benchmark

AdaptiveRL includes rigorous empirical evaluation protocols to test whether trained reinforcement learning policies generalize to novel, unseen environment topologies or merely overfit to training layouts.

### Key Capabilities:
1. **Strict Train/Test Partitioning**:
   - Training environments are strictly constrained using `TrainingDistributionWrapper`, guaranteeing zero exposure to test seeds during optimization.
   - `GeneralizationDistribution` performs automated programmatic assertions ensuring $|D_{\text{train}} \cap D_{\text{test}}| = 0$.
2. **Generalization Gap Metrics**:
   - Quantifies performance degradation: $\Delta_{\text{success}} = S_{\text{train}} - S_{\text{unseen}}$ and $\Delta_{\text{reward}} = R_{\text{train}} - R_{\text{unseen}}$.
   - Calculates relative success retention percentage ($S_{\text{test}} / S_{\text{train}}$).
3. **Reproducible Experiment Runner**:
   - `GeneralizationExperimentRunner` executes end-to-end training and evaluation, persisting structured JSON reports.

### CLI Usage:
```bash
# Run generalization benchmark on GridWorld
adaptive-rl generalization --config configs/generalization_gridworld.yaml

# Run generalization benchmark on Continuous 2D Navigation
adaptive-rl generalization --config configs/generalization_navigation.yaml --train-count 20 --test-count 20
```

---

## 15. Running Tests

Execute the automated test suite with `pytest`:
```bash
# Run all tests
pytest -v tests/

# Run with coverage
pytest --cov=adaptive_rl tests/
```

---

## 16. Contributing

We welcome contributions! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming conventions, quality gates, and code formatting standards before opening a pull request.

---

## 17. License

This project is licensed under the [MIT License](LICENSE).








