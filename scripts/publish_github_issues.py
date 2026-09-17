#!/usr/bin/env python3
"""Publish audited maintainer issues to GitHub repository (AryanXCode646/ARL).

Supports:
1. `gh` CLI (`gh issue create`)
2. GitHub REST API via GITHUB_TOKEN or GH_TOKEN
3. --dry-run mode for previewing issue payloads without writing to GitHub
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
from typing import Any, Dict, List

ISSUES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "title": "fix(planners): avoid representing failed path lengths as 0.0 in PlannerEvaluationMetrics.all_path_lengths",
        "labels": ["bug", "planners"],
        "body": """### Problem
In `PlannerAdapter`, when an evaluation episode fails (i.e. the planner finds no collision-free path to the goal), `0.0` is appended to `all_path_lengths`. This pollutes the raw episode trajectory data, giving the false impression that a zero-length path was traversed rather than signaling an unavailable/failed outcome.

### Current Behavior
In `src/adaptive_rl/planners/adapter.py`:
- Line 208 (GridWorld A*): `else: path_lengths.append(0.0)`
- Line 284 (Continuous 2D RRT*): `else: path_lengths.append(0.0)`

While `mean_path_length` and `std_path_length` are correctly calculated over `successful_path_lengths` only, `all_path_lengths` records `0.0` for all failed episodes. Downstream callers reading `metrics["all_path_lengths"]` or serializing them to tabular artifacts interpret failed episodes as 0-length trajectories.

### Expected Behavior
Failed episodes in `all_path_lengths` should be represented as `None` (JSON `null`) or `float("nan")`, or tracked in a dedicated `failed_episode_indices` mapping, consistent with AdaptiveRL's explicit policy that unavailable metrics are never silently defaulted to 0.

### Why This Matters
Empirical planning research relies on clean path distributions. Encoding search failures as 0-length paths distorts histogram calculations, box plots, and median path statistics if users compute summaries from raw episode records.

### Proposed Direction
1. Update `_evaluate_gridworld` and `_evaluate_continuous_nav` in `src/adaptive_rl/planners/adapter.py` to append `None` (or `float("nan")`) to `path_lengths` when `valid` is False.
2. Update `PlannerEvaluationMetrics.all_path_lengths` type annotation to `List[Optional[float]]`.
3. Add regression assertions in `tests/test_planners.py` verifying that failed episodes yield `None` entries in `all_path_lengths`.

### Acceptance Criteria
- [ ] Failed planning episodes append `None` instead of `0.0` to `all_path_lengths`.
- [ ] `PlannerEvaluationMetrics.to_dict()` outputs JSON-valid `null` values for failed path lengths.
- [ ] Existing path aggregations (`mean_path_length`, `std_path_length`) remain unskewed.
- [ ] Unit tests verify both complete success (all floats) and partial failure scenarios.

### Relevant Files
- `src/adaptive_rl/planners/adapter.py` (lines 208, 284, 325)
- `tests/test_planners.py`
""",
    },
    {
        "id": 2,
        "title": "fix(traffic): define success/overflow episode outcome semantics and track queuing telemetry in Evaluator",
        "labels": ["bug", "environments"],
        "body": """### Problem
`TrafficSignalEnv` never populates a `success` or `collision` key in `step_info`. As a result, the centralized `Evaluator.evaluate()` engine always records `success_rate: 0.0` and discards domain-specific queuing metrics (queue lengths, delay, vehicle throughput).

### Current Behavior
In `src/adaptive_rl/environments/traffic/intersection.py` (lines 268–276), `step()` returns an `info` dictionary containing queuing telemetry (`total_queue`, `overflow`, `total_departures`), but neither `success` nor `collision`.
In `src/adaptive_rl/evaluation/evaluator.py` (lines 117–120):
```python
if step_info.get("success", False):
    ep_success = True
```
Because `success` is never present, `ep_success` is permanently False. When running benchmarks on `TrafficSignalEnv`, the reported `success_rate` is always 0.0% regardless of agent performance. Furthermore, `Evaluator.evaluate()` discards `step_departures`, `total_queue`, and `max_wait`.

### Expected Behavior
1. `TrafficSignalEnv` should define explicit episode success semantics in `step_info`: e.g. completing `max_steps` without queue overflow (`info["success"] = not info["overflow"]`).
2. `Evaluator.evaluate()` and `StandardizedExperimentMetrics` should aggregate domain telemetry when evaluating traffic environments (e.g. `mean_queue_length`, `max_wait_time`, `total_departures`).

### Why This Matters
Users training RL policies on traffic signal control see misleading benchmark tables reporting 0.0% success rate, falsely indicating training failure when the agent may be optimizing throughput effectively.

### Proposed Direction
1. In `TrafficSignalEnv.step()`, set `info["success"] = (not telemetry.overflow)` when reaching `max_steps` or surviving without overflow.
2. In `Evaluator.evaluate()`, collect optional domain metrics from step infos (`queue_lengths`, `departures`) into `additional_metrics`.
3. In `StandardizedExperimentMetrics`, map traffic metrics into structured fields or `additional_metrics`.

### Acceptance Criteria
- [ ] `TrafficSignalEnv` step info contains a boolean `success` indicator.
- [ ] Evaluating a non-overflowing traffic policy yields `success_rate > 0.0`.
- [ ] Average queue length and throughput are preserved in `EvaluationMetrics.additional_metrics`.
- [ ] Regression test added in `tests/test_traffic.py` evaluating `TrafficSignalEnv` with `Evaluator`.

### Relevant Files
- `src/adaptive_rl/environments/traffic/intersection.py` (lines 268–280)
- `src/adaptive_rl/evaluation/evaluator.py` (lines 112–129)
- `src/adaptive_rl/evaluation/metrics.py`
- `tests/test_traffic.py`
""",
    },
    {
        "id": 3,
        "title": "fix(experiments): synchronize evaluation seed protocol between RL algorithms and classical planners",
        "labels": ["bug", "reproducibility"],
        "body": """### Problem
`ExperimentManager` uses asymmetrical evaluation seeds for RL algorithms versus classical planners. In comparative benchmarks under the same seed, RL agents and planners are evaluated on different procedural environment layouts.

### Current Behavior
In `src/adaptive_rl/experiments/manager.py`:
- Line 413 (RL evaluation): `base_seed = config.seed + 10000`
- Line 517 (Planner evaluation): `base_seed = config.seed`

When executing `adaptive-rl benchmark --config configs/gridworld_ppo.yaml --compare configs/gridworld_astar.yaml --seeds 42`:
- PPO evaluates on seeds `[10042, 10043, 10044, ...]`
- A* evaluates on seeds `[42, 43, 44, ...]`
Because `GridWorldEnv.reset(seed=...)` seeds procedural obstacle and goal generation, PPO and A* navigate entirely different maps during comparative evaluation.

### Expected Behavior
Both RL policies and classical planners must evaluate on the exact same sequence of evaluation seeds (e.g. `config.seed + eval_offset`, or configurable via `EvaluationConfig.eval_seed`).

### Why This Matters
Head-to-head comparisons between RL algorithms and classical planners are scientifically valid only if both paradigms are tested on identical environment instances. Testing on different seeds introduces uncontrollable instance-variance confounds into ablation results.

### Proposed Direction
1. Add an optional `eval_seed_offset: int = 10000` or `eval_seed: Optional[int] = None` to `EvaluationConfig` in `src/adaptive_rl/config.py`.
2. Standardize `ExperimentManager._run_planner_experiment` to use the same evaluation seed computation as `_run_rl_experiment`.
3. Verify via integration tests that `Evaluator` and `PlannerAdapter` receive identical seeds when given the same config seed.

### Acceptance Criteria
- [ ] RL and classical planners evaluate on identical episode seeds when configured with identical seeds.
- [ ] Evaluation seed generation is deterministic, configurable, and transparent.
- [ ] `tests/test_experiment_manager.py` verifies seed parity between RL and planner runs.

### Relevant Files
- `src/adaptive_rl/experiments/manager.py` (lines 413, 517)
- `src/adaptive_rl/config.py` (lines 208–224)
- `tests/test_experiment_manager.py`
""",
    },
    {
        "id": 4,
        "title": "refactor(planners): generalize BasePlanner interface to support continuous and 3D motion planners",
        "labels": ["refactor", "planners"],
        "body": """### Problem
`BasePlanner` in `src/adaptive_rl/planners/base.py` is hardcoded to 2D discrete grid parameters (`start: GridCoordinate`, `goal: GridCoordinate`, `width: int`, `height: int`). `RRTStarPlanner` cannot inherit from `BasePlanner`, breaking OOP polymorphism and necessitating type-switched dispatch in `PlannerAdapter`.

### Current Behavior
In `src/adaptive_rl/planners/base.py` (lines 75–90), `BasePlanner.plan()` accepts only integer grid dimensions.
In `src/adaptive_rl/planners/rrt_star.py` (line 42), `RRTStarPlanner` does not inherit from `BasePlanner` because continuous planning requires continuous coordinates, float bounds, and geometric obstacles.
In `src/adaptive_rl/planners/adapter.py` (lines 114–126, 151–157), `PlannerAdapter` uses runtime `isinstance` branches rather than a generic planner interface.

### Expected Behavior
A clean class hierarchy should distinguish discrete grid planning from continuous motion planning, allowing `AStarPlanner`, `RRTStarPlanner`, and future 3D planners to share a unified planning abstraction.

### Proposed Direction
1. Refactor `BasePlanner` into a generic abstract class or create `BaseGridPlanner` and `BaseContinuousPlanner`.
2. Define a unified `plan()` signature using type variables or parameter objects.
3. Have `RRTStarPlanner` inherit from the appropriate continuous planner base class.
4. Simplify `PlannerAdapter` dispatch to rely on interface contracts rather than concrete class type checks.

### Acceptance Criteria
- [ ] `RRTStarPlanner` inherits from a shared planner base class.
- [ ] `BasePlanner` contract is extensible to continuous 2D and 3D domains.
- [ ] `PlannerAdapter` type checks rely on base abstractions.
- [ ] All existing planner tests pass without regression.

### Relevant Files
- `src/adaptive_rl/planners/base.py`
- `src/adaptive_rl/planners/rrt_star.py`
- `src/adaptive_rl/planners/adapter.py`
- `tests/test_planners.py`
""",
    },
    {
        "id": 5,
        "title": "ci: add GitHub Actions workflow for automated testing, linting, formatting, and type checks",
        "labels": ["ci", "testing"],
        "body": """### Problem
The repository has no `.github/workflows/` directory. No continuous integration is configured to validate pull requests, test commits across Python versions, or enforce static code quality gates.

### Current Behavior
Contributors and maintainers must run checks manually. Regressions in tests, formatting, or typing can be merged into `main` without automated warnings.

### Expected Behavior
A comprehensive GitHub Actions workflow (`.github/workflows/ci.yml`) should trigger on every `push` to `main` and all `pull_request`s, verifying:
1. Linting with `ruff check .`
2. Format validation with `ruff format --check .`
3. Static type checks with `mypy src/`
4. Complete test suite with coverage: `pytest --cov=adaptive_rl tests/`
5. Compatibility matrix across supported Python versions (3.10, 3.11, 3.12, 3.13, 3.14).

### Why This Matters
CI is essential for professional open-source maintenance, preventing regressions and giving external contributors instant feedback on pull requests.

### Proposed Direction
1. Create `.github/workflows/ci.yml` using `astral-sh/setup-uv` for fast, reproducible dependency resolution.
2. Configure steps: setup Python, cache uv dependencies, run `ruff check`, run `ruff format --check`, run `mypy`, run `pytest`.
3. Add status badges to `README.md`.

### Acceptance Criteria
- [ ] `.github/workflows/ci.yml` is created and passes syntax validation.
- [ ] Matrix tests against supported Python versions.
- [ ] Quality gates (ruff, mypy, pytest) run automatically on PRs.

### Relevant Files
- `.github/workflows/ci.yml` (new)
- `pyproject.toml`
- `README.md`
""",
    },
    {
        "id": 6,
        "title": "test(typing): fix missing type annotations in test fixtures to enforce zero-error mypy checks",
        "labels": ["testing"],
        "body": """### Problem
Executing `uv run mypy tests/` fails with `[no-untyped-def]` due to an unannotated helper function in `tests/test_algorithm_registry.py`.

### Current Behavior
Running `uv run mypy tests/` fails with:
```text
tests/test_algorithm_registry.py:31: error: Function is missing a type annotation  [no-untyped-def]
Found 1 error in 1 file (checked 20 source files)
```
Because `pyproject.toml` configures `disallow_untyped_defs = true`, any CI or pre-commit hook checking tests fails with exit code 1.

### Expected Behavior
`uv run mypy tests/` should pass with 0 errors across all 20 test files, enabling test directory validation in CI.

### Why This Matters
Full type safety across both `src/` and `tests/` prevents subtle type bugs in test mocks and ensures strict CI type checking remains green.

### Proposed Direction
1. In `tests/test_algorithm_registry.py:31`, annotate `def _dummy_factory(self, **kwargs: Any) -> Any:`.
2. Add `tests/` to the default mypy target in CI.

### Acceptance Criteria
- [ ] `_dummy_factory` in `tests/test_algorithm_registry.py` is fully typed.
- [ ] `uv run mypy tests/` completes with exit code 0 and 0 errors.

### Relevant Files
- `tests/test_algorithm_registry.py` (line 31)
- `pyproject.toml`
""",
    },
    {
        "id": 7,
        "title": "style: resolve ruff formatting discrepancies across CLI, config, and planner modules",
        "labels": ["quality"],
        "body": """### Problem
Running `uv run ruff format --check .` flags 6 files that require reformatting to adhere to the repository's 100-character line length standard.

### Current Behavior
`uv run ruff format --check .` reports:
```text
6 files would be reformatted, 95 files already formatted:
- src/adaptive_rl/cli.py
- src/adaptive_rl/config.py
- src/adaptive_rl/planners/astar.py
- src/adaptive_rl/planners/rrt_star.py
- tests/test_configuration.py
- tests/test_experiment_manager.py
```
This causes any CI format check (`ruff format --check .`) to fail.

### Expected Behavior
All Python files should satisfy `ruff format --check .` cleanly without formatting violations.

### Why This Matters
Consistent formatting prevents noisy diffs in pull requests and ensures automated CI format checks pass cleanly.

### Proposed Direction
1. Run `uv run ruff format .` across the workspace.
2. Confirm `uv run ruff format --check .` passes with 0 files to reformat.
3. Include format checking in the CI workflow.

### Acceptance Criteria
- [ ] `uv run ruff format --check .` exits with code 0 across all repository files.
- [ ] No functional or logical code changes occur during reformatting.

### Relevant Files
- `src/adaptive_rl/cli.py`
- `src/adaptive_rl/config.py`
- `src/adaptive_rl/planners/astar.py`
- `src/adaptive_rl/planners/rrt_star.py`
- `tests/test_configuration.py`
""",
    },
    {
        "id": 8,
        "title": "fix(dashboard): handle unreadable manifests and incomplete experiment directories gracefully",
        "labels": ["enhancement", "dashboard"],
        "body": """### Problem
`ExperimentManager.list_experiments()` silently ignores experiment folders if `manifest.json` does not exist (e.g. when an experiment is cancelled, killed by OOM, or currently running). Additionally, if `manifest.json` is corrupted, the Rich terminal dashboard displays `?` placeholders without informative diagnostics.

### Current Behavior
In `src/adaptive_rl/experiments/manager.py` (lines 671–680):
```python
manifest_path = exp_dir / "manifest.json"
if manifest_path.exists():
    try: ...
    except Exception:
        experiments.append({"experiment_id": exp_dir.name, "error": "manifest unreadable"})
```
- Folders without `manifest.json` are skipped completely.
- If manifest JSON is corrupted, `render_overview` in `src/adaptive_rl/visualization/dashboard.py` prints `?` for Environment, Algorithm, and Seed, and status "unknown".
- `adaptive-rl dashboard --experiment <id>` on a corrupt manifest prints a generic "not found" message.

### Expected Behavior
1. Incomplete experiment folders (containing `config.yaml` or `logs/` but no manifest) should be listed with status `interrupted` or `in-progress`.
2. Corrupt manifests should display a clear warning with the path to the damaged file and suggested remediation.
3. The dashboard UI should offer an option to clean or archive failed/incomplete runs.

### Why This Matters
Experiment runs frequently get interrupted during long RL training sessions. Users need to see what happened to incomplete runs rather than wondering why they disappeared from the dashboard.

### Proposed Direction
1. Update `ExperimentManager.list_experiments()` to inspect folders lacking `manifest.json` for `config.yaml` and report status as `"interrupted"`.
2. In `dashboard.py`, render a distinct yellow or red status indicator for incomplete and corrupted runs.
3. Display error details when inspecting a corrupted experiment.

### Acceptance Criteria
- [ ] Interrupted experiment directories are visible in `adaptive-rl dashboard` and `adaptive-rl experiment list`.
- [ ] Corrupt manifests display explicit diagnostic messages rather than silent `?` fallbacks.
- [ ] Unit tests cover listing empty, interrupted, and corrupt experiment directories.

### Relevant Files
- `src/adaptive_rl/experiments/manager.py` (lines 657–685)
- `src/adaptive_rl/visualization/dashboard.py` (lines 80–126)
- `tests/test_dashboard_cli.py`
""",
    },
    {
        "id": 9,
        "title": "feat(benchmarking): incorporate hypothesis testing and confidence intervals into benchmark comparisons",
        "labels": ["feat", "benchmarking"],
        "body": """### Problem
`BenchmarkRunner.compare()` calculates only naive arithmetic mean differences (`delta = b_stats.mean - a_stats.mean`) and percentage changes across seeds. It provides no hypothesis testing, p-values, or confidence intervals to establish whether observed differences are statistically significant.

### Current Behavior
In `src/adaptive_rl/benchmarking/__init__.py` (lines 423–436):
```python
entry["delta"] = b_stats.mean - a_stats.mean
entry["pct_change"] = ((b_stats.mean - a_stats.mean) / abs(a_stats.mean) * 100 ...)
```
When running 3 to 10 seeds, stochastic variations in RL can easily produce positive deltas that are statistically meaningless (Agarwal et al., 2021, "Deep Reinforcement Learning at the Edge of the Statistical Precipice").

### Expected Behavior
`ComparisonReport` should provide:
1. Two-sample Welch's t-test or Mann-Whitney U test p-values for key metrics (`success_rate`, `mean_reward`).
2. Stratified bootstrap 95% confidence intervals for metric differences.
3. Clear indicators when sample size (number of seeds) is too small to draw statistically significant conclusions.

### Why This Matters
AdaptiveRL positions itself as a principled research platform. Providing rigorous statistical tests elevates benchmark credibility and guards users against reporting false-positive ablation findings.

### Proposed Direction
1. Add statistical comparison helpers in `src/adaptive_rl/benchmarking/statistics.py` (using `scipy.stats` or lightweight numpy implementations).
2. Populate `ComparisonReport.comparison[metric]["p_value"]` and `["ci_95"]`.
3. Display statistical significance stars or confidence bands in CLI comparison tables.
4. Document the methodology in `docs/RESEARCH.md`.

### Acceptance Criteria
- [ ] `ComparisonReport` includes p-values and 95% confidence intervals when seeds >= 3.
- [ ] CLI benchmark output highlights statistically significant deltas.
- [ ] Unit tests verify statistical test calculations on known synthetic distributions.

### Relevant Files
- `src/adaptive_rl/benchmarking/__init__.py` (lines 400–440)
- `src/adaptive_rl/cli.py` (benchmark reporting)
- `docs/RESEARCH.md`
- `tests/test_benchmarking.py`
""",
    },
    {
        "id": 10,
        "title": "fix(environments): register Gymnasium EnvSpecs to eliminate env_checker spec warnings and enable gym.make",
        "labels": ["enhancement", "environments"],
        "body": """### Problem
Custom environments (`GridWorldEnv`, `ContinuousNavigation2DEnv`, `TrafficSignalEnv`, `Drone3DEnv`, `DisturbedDroneEnv`) are only registered in AdaptiveRL's internal registry, not with Farama Gymnasium's global registry. Consequently, Gymnasium's `check_env` emits 9 `UserWarning`s during testing, and users cannot instantiate environments using standard `gymnasium.make("AdaptiveRL/GridWorld-v0")`.

### Current Behavior
Running `pytest` triggers 9 warnings:
```text
UserWarning: WARN: Not able to test alternative render modes due to the environment not having a spec.
Try instantiating the environment through `gymnasium.make`
```
Furthermore, third-party libraries expecting standard Gymnasium registration cannot load AdaptiveRL environments by ID.

### Expected Behavior
All AdaptiveRL environments should be registered with Farama Gymnasium via `gymnasium.register()` with semantic IDs (e.g. `AdaptiveRL/GridWorld-v0`, `AdaptiveRL/Navigation2D-v0`, `AdaptiveRL/Drone3D-v0`, `AdaptiveRL/Traffic-v0`), declaring valid render modes and default kwargs.

### Why This Matters
Full compliance with Farama Gymnasium standards eliminates pytest warning noise, enables compatibility with external RL libraries (CleanRL, Tianshou, Ray RLlib), and follows open-source best practices.

### Proposed Direction
1. In `src/adaptive_rl/environments/registry.py` (or `__init__.py`), call `gymnasium.register()` for each environment with appropriate `entry_point`, `max_episode_steps`, and `reward_threshold`.
2. Update tests to verify both `make_env(...)` and `gymnasium.make("AdaptiveRL/...")`.
3. Verify that `check_env` runs with zero warnings.

### Acceptance Criteria
- [ ] Environments are registered in Gymnasium's global registry.
- [ ] `gymnasium.make("AdaptiveRL/GridWorld-v0")` instantiates the environment correctly.
- [ ] All 9 Gymnasium spec warnings disappear during pytest execution.

### Relevant Files
- `src/adaptive_rl/environments/registry.py`
- `src/adaptive_rl/environments/__init__.py`
- `tests/test_registry.py`
""",
    },
    {
        "id": 11,
        "title": "feat(experiments): record execution duration, evaluation parameters, and CLI invocations in ExperimentManifest",
        "labels": ["feat", "experiments"],
        "body": """### Problem
`ExperimentManifest` records git commit and package versions, but omits total execution wall-clock time (`duration_seconds`), completion timestamp (`completed_at`), evaluation settings (`eval_episodes`, `deterministic`, `eval_base_seed`), and CLI command invocations.

### Current Behavior
In `src/adaptive_rl/experiments/manager.py` (lines 569–605), `_make_manifest` populates `created_at` but no timing duration or evaluation parameters. If training takes 45 minutes, there is no record in `manifest.json` indicating how long the run took or under what evaluation hyperparameters it was assessed.

### Expected Behavior
`ExperimentManifest` should include:
- `duration_seconds: float`
- `completed_at: str` (ISO 8601 UTC)
- `evaluation_settings: Dict[str, Any]` (num episodes, deterministic flag, eval seeds)
- `command_args: Optional[str]` (CLI string if launched via CLI)

### Why This Matters
Provenance tracking is incomplete without runtime metrics. Knowing the wall-clock cost of experiments is vital for compute budgeting, replication, and paper writing.

### Proposed Direction
1. Add fields `duration_seconds`, `completed_at`, and `evaluation_settings` to `ExperimentManifest`.
2. Measure `time.perf_counter()` in `ExperimentManager.run()` and populate these fields prior to saving `manifest.json`.
3. Surface execution duration in `adaptive-rl experiment inspect` and the dashboard manifest panel.

### Acceptance Criteria
- [ ] `manifest.json` contains `duration_seconds`, `completed_at`, and `evaluation_settings`.
- [ ] Dashboard displays experiment duration (e.g. `Duration: 3m 42s`).
- [ ] `tests/test_experiment_manager.py` verifies these fields are populated on success and failure.

### Relevant Files
- `src/adaptive_rl/experiments/manager.py` (lines 31–75, 569–605)
- `src/adaptive_rl/visualization/dashboard.py` (lines 150–175)
- `tests/test_experiment_manager.py`
""",
    },
    {
        "id": 12,
        "title": "docs(cli): update adaptive-rl train help text and documentation to reflect multi-algorithm (PPO & SAC) support",
        "labels": ["documentation"],
        "body": """### Problem
The CLI command `adaptive-rl train` is documented as "Train a reinforcement learning agent using PPO", despite the underlying implementation fully supporting Soft Actor-Critic (SAC).

### Current Behavior
In `src/adaptive_rl/cli.py` (line 437):
```python
def train(...) -> None:
    \"\"\"Train a reinforcement learning agent using PPO.\"\"\"
```
`adaptive-rl --help` displays:
```text
train    Train a reinforcement learning agent using PPO.
```
In reality, `train()` calls `get_trainer(config=exp_config)` (line 491), which checks `exp_config.algorithm.name` and seamlessly dispatches to either `PPOTrainer` or `SACTrainer`.

### Expected Behavior
Help text and docstrings should reflect multi-algorithm capabilities: "Train a reinforcement learning agent (PPO, SAC) from configuration."

### Why This Matters
Inaccurate CLI help confuses new users, giving the false impression that continuous-action SAC training requires a different unlisted command.

### Proposed Direction
1. Update `train()` docstring in `src/adaptive_rl/cli.py`.
2. Clarify in CLI help that algorithm choice is determined by the config YAML (`--config`).
3. Update `README.md` CLI examples to show both PPO and SAC training commands.

### Acceptance Criteria
- [ ] `adaptive-rl --help` and `adaptive-rl train --help` accurately state support for PPO and SAC.
- [ ] `README.md` documentation matches the updated CLI help text.
- [ ] `tests/test_cli.py` checks for accurate help text output.

### Relevant Files
- `src/adaptive_rl/cli.py` (line 437)
- `README.md`
- `tests/test_cli.py`
""",
    },
]


def check_existing_issues_api(repo: str, token: str) -> List[str]:
    """Fetch existing issue titles from GitHub REST API."""
    url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page=100"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "AdaptiveRL-Issue-Creator",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return [item.get("title", "") for item in data if "title" in item]
    except Exception as exc:
        print(f"[WARN] Failed to query GitHub API for existing issues: {exc}")
        return []


def check_existing_issues_gh(repo: str) -> List[str]:
    """Fetch existing issue titles using `gh` CLI."""
    try:
        out = subprocess.check_output(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                repo,
                "--state",
                "all",
                "--json",
                "title",
                "--limit",
                "100",
            ],
            text=True,
        )
        data = json.loads(out)
        return [item.get("title", "") for item in data if "title" in item]
    except Exception as exc:
        print(f"[WARN] Failed to query gh CLI for existing issues: {exc}")
        return []


def _extract_token_from_git_credentials() -> str:
    """Extract GitHub token from ~/.git-credentials if available."""
    cred_file = os.path.expanduser("~/.git-credentials")
    if os.path.exists(cred_file):
        try:
            with open(cred_file, encoding="utf-8") as f:
                for line in f:
                    if "github.com" in line and ":" in line and "@" in line:
                        part = line.strip().split("@")[0]
                        if ":" in part:
                            token = part.split(":")[-1]
                            if token.startswith("gho_") or token.startswith("ghp_"):
                                return token
        except Exception:
            pass
    return ""


def create_issue_api(repo: str, token: str, title: str, body: str, labels: List[str]) -> bool:
    """Create an issue via GitHub REST API with fallback if labels are rejected."""
    url = f"https://api.github.com/repos/{repo}/issues"
    payload = json.dumps({"title": title, "body": body, "labels": labels}).encode()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "AdaptiveRL-Issue-Creator",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"[SUCCESS] Created issue #{data.get('number')}: {data.get('html_url')}")
            return True
    except urllib.error.HTTPError as err:
        # If labels or permissions caused an error, retry without labels
        if err.code in (400, 403, 422):
            fallback_payload = json.dumps({"title": title, "body": body}).encode()
            fallback_req = urllib.request.Request(url, data=fallback_payload, headers=headers)
            try:
                with urllib.request.urlopen(fallback_req) as resp:
                    data = json.loads(resp.read().decode())
                    print(
                        f"[SUCCESS] Created issue #{data.get('number')} (without labels): {data.get('html_url')}"
                    )
                    return True
            except Exception as retry_err:
                print(f"[ERROR] Failed to create issue '{title}' on retry: {retry_err}")
                return False
        else:
            print(f"[ERROR] Failed to create issue '{title}': HTTP {err.code}: {err.reason}")
            return False
    except Exception as exc:
        print(f"[ERROR] Failed to create issue '{title}': {exc}")
        return False


def create_issue_gh(repo: str, title: str, body: str, labels: List[str]) -> bool:
    """Create an issue via gh CLI."""
    cmd = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body]
    for label in labels:
        cmd.extend(["--label", label])
    try:
        out = subprocess.check_output(cmd, text=True)
        print(f"[SUCCESS] Created issue via gh: {out.strip()}")
        return True
    except subprocess.CalledProcessError:
        # Fallback without labels
        cmd_no_label = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body]
        try:
            out = subprocess.check_output(cmd_no_label, text=True)
            print(f"[SUCCESS] Created issue via gh (without labels): {out.strip()}")
            return True
        except Exception as retry_exc:
            print(f"[ERROR] gh issue create failed: {retry_exc}")
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish AdaptiveRL issues to GitHub")
    parser.add_argument(
        "--repo", default="ashishsinghbora/ARL", help="Target GitHub repository (owner/repo)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print issue payloads without creating them"
    )
    args = parser.parse_args()

    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or _extract_token_from_git_credentials()
    )
    has_gh = shutil.which("gh") is not None

    print("=== AdaptiveRL GitHub Issue Publisher ===")
    print(f"Target Repository: {args.repo}")
    print(f"Total Issues in Backlog: {len(ISSUES)}")
    print(f"Dry Run: {args.dry_run}")
    print(
        f"Authentication: {'Token provided' if token else 'No token'}, gh CLI: {'Available' if has_gh else 'Not available'}"
    )
    print()

    if args.dry_run or (not token and not has_gh):
        if not args.dry_run:
            print("[INFO] No GitHub token or gh CLI detected. Running in --dry-run mode.")
            print("[INFO] Export GITHUB_TOKEN=... or authenticate with 'gh auth login' to publish.")
            print()
        for item in ISSUES:
            print(
                "--------------------------------------------------------------------------------"
            )
            print(f"[{item['id']}/12] {item['title']}")
            print(f"Labels: {', '.join(item['labels'])}")
            print(f"Body length: {len(item['body'])} chars")
        print("--------------------------------------------------------------------------------")
        print("\nAll 12 issues validated. Run with credentials to publish.")
        return

    # Check existing issues to avoid duplicates
    existing_titles: List[str] = []
    if token:
        existing_titles = check_existing_issues_api(args.repo, token)
    elif has_gh:
        existing_titles = check_existing_issues_gh(args.repo)

    created_count = 0
    skipped_count = 0

    for item in ISSUES:
        title = item["title"]
        if any(
            title.lower() in ext.lower() or ext.lower() in title.lower() for ext in existing_titles
        ):
            print(f"[SKIP] Issue already exists: {title}")
            skipped_count += 1
            continue

        print(f"Publishing: {title} ...")
        success = False
        if token:
            success = create_issue_api(args.repo, token, title, item["body"], item["labels"])
        elif has_gh:
            success = create_issue_gh(args.repo, title, item["body"], item["labels"])

        if success:
            created_count += 1

    print()
    print("=== Publication Summary ===")
    print(f"Successfully Created: {created_count}")
    print(f"Skipped Duplicates: {skipped_count}")


if __name__ == "__main__":
    main()
