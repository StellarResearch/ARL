# 6-DOF Quadrotor Reinforcement Learning Training Guide

This guide documents training, evaluating, and demonstrating reinforcement learning agents on the rigid-body 6-DOF quadrotor simulation environment (`Drone6DOFEnv`, registered as `drone-6dof`).

> [!NOTE]
> **Educational Research Platform**:
> This quadrotor dynamics model is an educational theoretical simulation. It is designed for algorithmic benchmarking and reinforcement learning experiments and is not tuned to specific physical drone hardware telemetry.

---

## 1. Quick Start Commands

### 1.1 Training PPO on 6-DOF
Train an on-policy PPO agent using the preconfigured YAML configuration:
```bash
adaptive-rl train --config configs/ppo_drone_6dof.yaml --timesteps 50000
```
Or with custom budget and seed:
```bash
adaptive-rl train --config configs/ppo_drone_6dof.yaml --timesteps 25000 --seed 42
```

### 1.2 Training SAC on 6-DOF
Train an off-policy SAC agent:
```bash
adaptive-rl train --config configs/sac_drone_6dof.yaml --timesteps 50000
```

### 1.3 Evaluating Trained Checkpoints
Evaluate the trained policy over 20 deterministic benchmark episodes:
```bash
adaptive-rl evaluate --model artifacts/models/ppo_drone_6dof_final.zip --config configs/ppo_drone_6dof.yaml --episodes 20
```

### 1.4 Command-Line 3D Demonstration
Run a deterministic flight demo rendering ASCII status and real-time telemetry:
```bash
adaptive-rl demo-drone --model artifacts/models/ppo_drone_6dof_final.zip --config configs/ppo_drone_6dof.yaml --seed 42 --max-steps 150
```

### 1.5 Interactive Streamlit 3D Flight Deck
Launch the interactive web-based 3D simulator:
```bash
adaptive-rl gui
```
Select **6-DOF Quadrotor** from the dynamics selector dropdown in the browser.

---

## 2. Configuration Files

Three configuration files are provided in `configs/`:

| Config File | Algorithm | Environment | Action Dim | Obs Dim |
| :--- | :--- | :--- | :--- | :--- |
| `configs/drone_6dof.yaml` | PPO | `drone-6dof` | 4 | 36 |
| `configs/ppo_drone_6dof.yaml` | PPO | `drone-6dof` | 4 | 36 |
| `configs/sac_drone_6dof.yaml` | SAC | `drone-6dof` | 4 | 36 |

### 2.1 PPO Configuration (`configs/ppo_drone_6dof.yaml`)
```yaml
name: "ppo_drone_6dof"
seed: 42

algorithm:
  name: "ppo"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 64
  parameters:
    n_steps: 1024
    n_epochs: 10
    clip_range: 0.2
    ent_coef: 0.01

environment:
  name: "drone-6dof"
  max_steps: 400
  parameters:
    bounds: [30.0, 30.0, 15.0]
    start_pos: [5.0, 5.0, 5.0]
    goal_pos: [25.0, 25.0, 10.0]
    num_obstacles: 4
    obstacle_radius: 2.0
    target_radius: 1.5
    collision_radius: 0.8
    progress_weight: 2.0
    action_penalty_weight: 0.01
    angular_rate_penalty_weight: 0.05
    tilt_penalty_weight: 0.1

training:
  total_timesteps: 50000
  checkpoint_freq: 10000
  log_interval: 10

evaluation:
  eval_episodes: 20
  deterministic: true

output_dir: "artifacts"
log_dir: "artifacts/logs"
```

### 2.2 SAC Configuration (`configs/sac_drone_6dof.yaml`)
```yaml
name: "sac_drone_6dof"
seed: 42

algorithm:
  name: "sac"
  learning_rate: 0.0003
  gamma: 0.99
  batch_size: 256
  parameters:
    buffer_size: 100000
    tau: 0.005
    learning_starts: 100
    ent_coef: "auto"

environment:
  name: "drone-6dof"
  max_steps: 400
  parameters:
    bounds: [30.0, 30.0, 15.0]
    num_obstacles: 4
```

---

## 3. Python API Usage

```python
from adaptive_rl.environments.registry import make_env
from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.evaluation.evaluator import Evaluator

# 1. Instantiate the 6-DOF environment
env = make_env("drone-6dof", num_obstacles=4, max_steps=400)

# 2. Instantiate and train PPO
algo = PPOAlgorithm(env=env, n_steps=1024, batch_size=64)
algo.train(total_timesteps=50000)

# 3. Save checkpoint
algo.save("artifacts/models/my_6dof_ppo.zip")

# 4. Evaluate deterministic performance
evaluator = Evaluator(algorithm=algo, env=env)
metrics = evaluator.evaluate(num_episodes=20, deterministic=True, base_seed=42)
print(f"Success Rate: {metrics.success_rate * 100:.1f}%, Mean Reward: {metrics.mean_reward:.2f}")

env.close()
```

---

## 4. Decomposed Reward Inspection

During stepping, the environment returns detailed reward components inside the `info` dictionary:

```python
obs, reward, terminated, truncated, info = env.step(action)

components = info["reward_components"]
print("Progress Reward:", components["progress_reward"])
print("Step Penalty:", components["step_penalty"])
print("Control Effort Penalty:", components["effort_penalty"])
print("Angular Rate Penalty:", components["rate_penalty"])
print("Tilt Angle Penalty:", components["tilt_penalty"])
print("Terminal Reward:", components["terminal_reward"])
```
