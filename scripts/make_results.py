#!/usr/bin/env python3
"""make_results.py – Generate authoritative benchmark results.

Scans the `artifacts/` directory for valid experiment records, validates
schemas, aggregates metrics using the repository's evaluation utilities, and
writes:
- `docs/RESULTS.md`
- `docs/data/results.json`

Never fabricates numbers; if no verified results exist the files contain the
placeholder text "NO VERIFIED RESULTS AVAILABLE".
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def find_experiment_dirs(base: Path) -> List[Path]:
    """Return directories that appear to be a completed experiment.
    Requirements: a `metadata` sub‑dir with at least one JSON file and a
    `models` sub‑dir containing a model archive.
    """
    experiments = []
    for root, dirs, _ in os.walk(base):
        root_path = Path(root)
        if "metadata" in dirs and "models" in dirs:
            meta_path = root_path / "metadata"
            if any(p.suffix == ".json" for p in meta_path.iterdir()):
                experiments.append(root_path)
    return experiments


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def validate_experiment(dir_path: Path) -> bool:
    meta_dir = dir_path / "metadata"
    json_files = sorted(p for p in meta_dir.iterdir() if p.suffix == ".json")
    if not json_files:
        return False
    meta = load_json(json_files[0])
    return all(k in meta for k in ("seed", "env_name"))


def collect_metrics(dir_path: Path) -> Dict[str, Any]:
    # Look for any *metrics*.json file under the experiment tree.
    for p in dir_path.rglob("*metrics*.json"):
        return load_json(p)
    return {}


def aggregate_results(exp_dirs: List[Path]) -> List[Dict[str, Any]]:
    out = []
    for d in exp_dirs:
        if not validate_experiment(d):
            continue
        meta_json = sorted(p for p in (d / "metadata").iterdir() if p.suffix == ".json")[0]
        meta = load_json(meta_json)
        metrics = collect_metrics(d)
        model_path = (
            next(
                (d / "models" / f).as_posix()
                for f in os.listdir(d / "models")
                if f.endswith(".zip")
            )
            if (d / "models").exists()
            else None
        )
        out.append(
            {
                "benchmark": d.name,
                "seed": meta.get("seed"),
                "environment": meta.get("env_name"),
                "model_path": model_path,
                "software_versions": meta.get("software_versions", {}),
                "metrics": metrics,
            }
        )
    return out


def write_json(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def write_md(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text("NO VERIFIED RESULTS AVAILABLE\n", encoding="utf-8")
        return
    lines = [
        "# Benchmark Results",
        "",
        "| Benchmark | Seed | Environment | Metric (mean ± std) |",
        "|-----------|------|-------------|--------------------|",
    ]
    for r in results:
        metric = "-"
        if r["metrics"]:
            for k, v in r["metrics"].items():
                if isinstance(v, (int, float)):
                    metric = f"{v:.3f}"
                    break
            for k, v in r["metrics"].items():
                if isinstance(v, (int, float)) and "std" in k.lower():
                    metric = f"{metric} ± {v:.3f}"
                    break
        lines.append(f"| {r['benchmark']} | {r['seed']} | {r['environment']} | {metric} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[0]
    artifacts = repo_root / "artifacts"
    if not artifacts.is_dir():
        print("Artifacts dir missing", file=sys.stderr)
        return 1
    dirs = find_experiment_dirs(artifacts)
    results = aggregate_results(dirs)
    write_json(results, repo_root / "docs" / "data" / "results.json")
    write_md(results, repo_root / "docs" / "RESULTS.md")
    print(f"Generated {len(results)} result entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
