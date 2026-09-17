"""Rich terminal dashboard for AdaptiveRL experiment results.

Provides an interactive terminal-based dashboard that reads saved experiment
artifacts and renders them using the Rich library (already a project dependency).
No additional dependencies are required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from adaptive_rl.experiments.manager import ExperimentManager

console = Console()


# ---------------------------------------------------------------------------
# Metric display helpers
# ---------------------------------------------------------------------------


def _fmt_pct(val: Any) -> str:
    """Format a [0, 1] fraction as a percentage string."""
    if val is None or not isinstance(val, (int, float)) or isinstance(val, bool) or np.isnan(val):
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt_float(val: Any, decimals: int = 2) -> str:
    """Format a float value, or return 'N/A' for missing/NaN values."""
    if val is None or not isinstance(val, (int, float)) or isinstance(val, bool) or np.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_time(val: Any) -> str:
    """Format a time duration in seconds."""
    if val is None or not isinstance(val, (int, float)) or isinstance(val, bool) or np.isnan(val):
        return "N/A"
    if val < 0.001:
        return f"{val * 1e6:.1f}µs"
    if val < 1.0:
        return f"{val * 1000:.2f}ms"
    return f"{val:.3f}s"


def _status_color(status: Optional[str]) -> str:
    """Return Rich color markup for an experiment status string."""
    if not status or not isinstance(status, str):
        return "[dim]unknown[/dim]"
    colors = {
        "completed": "[bold green]completed[/bold green]",
        "failed": "[bold red]failed[/bold red]",
        "pending": "[yellow]pending[/yellow]",
        "skipped": "[dim]skipped[/dim]",
    }
    return colors.get(status.lower(), f"[white]{status}[/white]")


# ---------------------------------------------------------------------------
# Dashboard views
# ---------------------------------------------------------------------------


def render_overview(manager: ExperimentManager) -> None:
    """Render the experiment overview table.

    Shows all available experiments with environment, algorithm, seed,
    status, and creation timestamp.

    Args:
        manager: ExperimentManager instance pointing to the results directory.
    """
    experiments = manager.list_experiments()

    if not experiments:
        console.print(
            Panel(
                "[yellow]No experiments found.[/yellow]\n\n"
                "Run [bold cyan]adaptive-rl experiment run --config <path>[/bold cyan] "
                "to create your first experiment.",
                title="AdaptiveRL Dashboard — No Experiments",
                border_style="yellow",
            )
        )
        return

    table = Table(
        title=f"AdaptiveRL Experiment Overview ({len(experiments)} experiments)",
        show_lines=True,
    )
    table.add_column("Experiment ID", style="cyan", no_wrap=True, max_width=40)
    table.add_column("Environment", style="green")
    table.add_column("Algorithm", style="magenta")
    table.add_column("Seed", style="blue", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Created At", style="dim")

    for exp in experiments:
        exp_id = str(exp.get("experiment_id") or "?")
        env = str(exp.get("environment") or "?")
        algo = str(exp.get("algorithm") or "?")
        seed = str(exp.get("seed") if exp.get("seed") is not None else "?")
        status = exp.get("evaluation_status")
        created_at = exp.get("created_at") or "?"
        # Truncate ISO timestamp to date+time without microseconds
        if isinstance(created_at, str) and "T" in created_at:
            created_at = created_at[:19].replace("T", " ") + " UTC"

        table.add_row(
            exp_id,
            env,
            algo.upper(),
            seed,
            _status_color(status),
            str(created_at),
        )

    console.print(table)


def render_metrics(experiment_id: str, manager: ExperimentManager) -> None:
    """Render detailed metrics for a specific experiment.

    Args:
        experiment_id: Experiment identifier string.
        manager: ExperimentManager instance.
    """
    manifest = manager.get_experiment(experiment_id)
    metrics = manager.get_metrics(experiment_id)

    if manifest is None:
        console.print(
            f"[bold red]Experiment '[cyan]{experiment_id}[/cyan]' not found.[/bold red]\n"
            f"Check the experiment ID with [bold]adaptive-rl experiment list[/bold]."
        )
        return

    # Manifest panel
    algo = str(manifest.get("algorithm") or "?")
    env = str(manifest.get("environment") or "?")
    seed = str(manifest.get("seed") if manifest.get("seed") is not None else "?")
    git_commit = str(manifest.get("git_commit") or "?")
    python_raw = str(manifest.get("python_version") or "?")
    python_ver = python_raw.split()[0] if python_raw else "?"
    status = manifest.get("evaluation_status")
    created_at = manifest.get("created_at") or "?"
    if isinstance(created_at, str) and "T" in created_at:
        created_at = created_at[:19].replace("T", " ") + " UTC"

    timesteps = manifest.get("training_timesteps")
    timesteps_str = f"{timesteps:,}" if timesteps else "N/A (planner)"

    manifest_text = (
        f"[bold]Environment:[/bold] {env}\n"
        f"[bold]Algorithm:[/bold] {algo.upper()}\n"
        f"[bold]Seed:[/bold] {seed}\n"
        f"[bold]Training Timesteps:[/bold] {timesteps_str}\n"
        f"[bold]Status:[/bold] {_status_color(status)}\n"
        f"[bold]Created:[/bold] {created_at}\n"
        f"[bold]Git Commit:[/bold] {git_commit}\n"
        f"[bold]Python:[/bold] {python_ver}"
    )

    console.print(
        Panel(
            manifest_text,
            title=f"Experiment: {experiment_id}",
            border_style="cyan",
        )
    )

    if metrics is None:
        console.print("[yellow]No metrics found for this experiment.[/yellow]")
        return

    # Metrics table
    table = Table(title="Evaluation Metrics", show_lines=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    metric_display = [
        ("success_rate", "Success Rate", lambda v: _fmt_pct(v)),
        ("collision_rate", "Collision Rate", lambda v: _fmt_pct(v)),
        ("mean_reward", "Mean Reward", lambda v: _fmt_float(v, 3)),
        ("std_reward", "Reward Std Dev", lambda v: f"± {_fmt_float(v, 3)}"),
        ("min_reward", "Min Reward", lambda v: _fmt_float(v, 3)),
        ("max_reward", "Max Reward", lambda v: _fmt_float(v, 3)),
        ("mean_episode_length", "Mean Episode Length", lambda v: _fmt_float(v, 1)),
        ("episodes", "Episodes", lambda v: str(int(v)) if v is not None else "N/A"),
        # Planner metrics
        ("mean_path_length", "Mean Path Length", lambda v: _fmt_float(v, 1)),
        ("std_path_length", "Path Length Std Dev", lambda v: f"± {_fmt_float(v, 1)}"),
        ("mean_planning_time", "Mean Planning Time", lambda v: _fmt_time(v)),
        ("std_planning_time", "Planning Time Std Dev", lambda v: _fmt_time(v)),
    ]

    for key, label, fmt in metric_display:
        val = metrics.get(key)
        if val is not None:
            table.add_row(label, fmt(val))

    console.print(table)

    # Reward curve (text sparkline if rewards available)
    all_rewards = metrics.get("all_rewards") or metrics.get("additional_metrics", {}).get(
        "all_rewards"
    )
    if all_rewards and isinstance(all_rewards, list) and len(all_rewards) > 1:
        _render_sparkline(all_rewards, title="Reward Distribution (evaluation episodes)")


def _render_sparkline(values: List[float], title: str = "Values") -> None:
    """Render a simple text sparkline of scalar values.

    Uses block characters to give a rough sense of the distribution.
    This is purely textual — no matplotlib or external deps needed.

    Args:
        values: List of scalar values to visualize.
        title: Display title for the sparkline.
    """
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return

    clean_values = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and not np.isnan(v)
    ]
    if not clean_values:
        return

    min_v = min(clean_values)
    max_v = max(clean_values)
    rng = max_v - min_v if max_v != min_v else 1.0

    sparkline = ""
    for v in clean_values:
        idx = int((v - min_v) / rng * (len(blocks) - 1))
        sparkline += blocks[idx]

    console.print(
        Panel(
            f"{sparkline}\n\n"
            f"Min: {min_v:.2f}  |  Max: {max_v:.2f}  |  "
            f"Avg: {sum(clean_values) / len(clean_values):.2f}  |  "
            f"n={len(clean_values)}",
            title=title,
            border_style="green",
        )
    )


def render_comparison(
    experiment_ids: List[str],
    manager: ExperimentManager,
) -> None:
    """Render a side-by-side metric comparison across multiple experiments.

    Args:
        experiment_ids: List of experiment identifiers to compare.
        manager: ExperimentManager instance.
    """
    if len(experiment_ids) < 2:
        console.print("[yellow]Specify at least 2 experiment IDs to compare.[/yellow]")
        return

    # Load all experiments
    experiments_data = []
    for exp_id in experiment_ids:
        manifest = manager.get_experiment(exp_id)
        metrics = manager.get_metrics(exp_id)
        if manifest is None:
            console.print(f"[bold red]Experiment '{exp_id}' not found — skipping.[/bold red]")
            continue
        experiments_data.append((exp_id, manifest, metrics or {}))

    if not experiments_data:
        return

    # Build comparison table
    table = Table(
        title=f"Experiment Comparison ({len(experiments_data)} runs)",
        show_lines=True,
    )
    table.add_column("Metric", style="cyan", no_wrap=True)

    for exp_id, manifest, _ in experiments_data:
        algo = str(manifest.get("algorithm") or "?").upper()
        env = str(manifest.get("environment") or "?")
        seed = manifest.get("seed")
        seed_str = str(seed) if seed is not None else "?"
        header = f"{algo}\n{env}\nseed={seed_str}"
        table.add_column(header, style="green", justify="right")

    compare_metrics = [
        ("success_rate", "Success Rate", _fmt_pct),
        ("collision_rate", "Collision Rate", _fmt_pct),
        ("mean_reward", "Mean Reward", lambda v: _fmt_float(v, 3)),
        ("mean_episode_length", "Mean Ep. Length", lambda v: _fmt_float(v, 1)),
        ("mean_path_length", "Mean Path Length", lambda v: _fmt_float(v, 1)),
        ("mean_planning_time", "Planning Time", _fmt_time),
    ]

    for key, label, fmt in compare_metrics:
        row_values = []
        has_any = False
        for _, _, metrics in experiments_data:
            val = metrics.get(key)
            if val is not None:
                row_values.append(fmt(val))
                has_any = True
            else:
                row_values.append("—")
        if has_any:
            table.add_row(label, *row_values)

    console.print(table)


# ---------------------------------------------------------------------------
# Top-level dashboard entry point
# ---------------------------------------------------------------------------


def run_dashboard(
    base_output_dir: Optional[Path] = None,
    experiment_id: Optional[str] = None,
    compare: Optional[List[str]] = None,
) -> None:
    """Entry point for the AdaptiveRL experiment dashboard.

    Renders the appropriate view depending on the arguments provided:
    - No arguments: Overview of all experiments.
    - experiment_id: Detailed metrics for one experiment.
    - compare: Side-by-side comparison of listed experiments.

    Args:
        base_output_dir: Root experiments directory. Defaults to
            ``experiments/results`` relative to the current directory.
        experiment_id: Show detail view for this experiment.
        compare: List of experiment IDs to compare side by side.
    """
    output_dir = base_output_dir or Path("experiments/results")
    manager = ExperimentManager(base_output_dir=output_dir)

    console.print()
    console.rule("[bold blue]AdaptiveRL Experiment Dashboard[/bold blue]")
    console.print(f"[dim]Results directory: {output_dir.resolve()}[/dim]\n")

    if compare and len(compare) >= 2:
        render_comparison(compare, manager)
    elif experiment_id:
        render_metrics(experiment_id, manager)
    else:
        render_overview(manager)

    console.print()
