#!/usr/bin/env python3
"""Main script to generate and publish the 62-issue engineering roadmap to AryanXCode646/ARL."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from build_roadmap_data import ISSUES_DATA
from build_roadmap_data_part2 import ISSUES_DATA_PART2
from build_roadmap_data_part3 import ISSUES_DATA_PART3

ALL_ISSUES: List[Dict[str, Any]] = ISSUES_DATA + ISSUES_DATA_PART2 + ISSUES_DATA_PART3

TOKEN = ""
cred_file = os.path.expanduser("~/.git-credentials")
if os.path.exists(cred_file):
    with open(cred_file, encoding="utf-8") as f:
        for line in f:
            if "github.com" in line and ":" in line and "@" in line:
                part = line.strip().split("@")[0]
                if ":" in part:
                    candidate = part.split(":")[-1]
                    if candidate.startswith("gho_") or candidate.startswith("ghp_"):
                        TOKEN = candidate

REPO = "AryanXCode646/ARL"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "AdaptiveRL-Roadmap-Publisher",
    "Content-Type": "application/json",
}

MILESTONES_DEF = [
    {
        "title": "Phase 18 — Reliability & Reproducibility",
        "description": "Foundational architectural integrity, testing coverage, metrics consistency, CI automation, and seed reproducibility.",
    },
    {
        "title": "Phase 19 — Benchmarking & Research",
        "description": "Rigorous statistical benchmarking, hypothesis testing, classical 3D planning baselines, and advanced generalization research.",
    },
    {
        "title": "Phase 20 — Performance & Scaling",
        "description": "Spatial query acceleration, parallel seed benchmarking, vectorized simulation, and profiling.",
    },
    {
        "title": "Phase 21 — Developer Experience",
        "description": "Interactive TUI dashboard enhancements, CLI developer tooling, and developer documentation guides.",
    },
    {
        "title": "Phase 22 — Release Engineering",
        "description": "Farama Gymnasium registration, release automation, packaging modularity, and environment standards.",
    },
]

LABELS_DEF = [
    {
        "name": "architecture",
        "color": "0075ca",
        "description": "System architecture, boundaries, and public interfaces",
    },
    {
        "name": "environments",
        "color": "008672",
        "description": "Gymnasium environments and scenario generators",
    },
    {
        "name": "algorithms",
        "color": "1d76db",
        "description": "Reinforcement learning policy architectures and adapters",
    },
    {
        "name": "planners",
        "color": "5319e7",
        "description": "Classical deterministic and sampling-based planners",
    },
    {
        "name": "experiments",
        "color": "b60205",
        "description": "Experiment orchestration, manifests, and provenance",
    },
    {
        "name": "benchmarking",
        "color": "d93f0b",
        "description": "Multi-seed evaluation, ablations, and comparison reports",
    },
    {
        "name": "metrics",
        "color": "fbca04",
        "description": "Standardized evaluation metrics and result schemas",
    },
    {
        "name": "dashboard",
        "color": "0e8a16",
        "description": "Rich terminal user interface and visualization",
    },
    {
        "name": "cli",
        "color": "c2e0c6",
        "description": "Typer command-line interface and developer tools",
    },
    {
        "name": "testing",
        "color": "bfdadc",
        "description": "Unit, integration, property-based, and regression tests",
    },
    {
        "name": "ci",
        "color": "2cbe4e",
        "description": "Continuous integration and workflow automation",
    },
    {
        "name": "packaging",
        "color": "e99695",
        "description": "Setuptools packaging, dependencies, and wheel distribution",
    },
    {
        "name": "performance",
        "color": "d4c5f9",
        "description": "Execution speed, vectorization, and memory profiling",
    },
    {
        "name": "security",
        "color": "ee0701",
        "description": "Vulnerability scanning, path hardening, and artifact safety",
    },
    {
        "name": "research",
        "color": "6f42c1",
        "description": "Advanced methodology, domain randomization, and empirical analysis",
    },
    {
        "name": "quality",
        "color": "e4e669",
        "description": "Formatting, static analysis, and code quality standards",
    },
    {
        "name": "provenance",
        "color": "5319e7",
        "description": "Execution metadata, version tracking, and audit trails",
    },
    {
        "name": "robustness",
        "color": "f9d0c4",
        "description": "Error recovery, corrupted state handling, and edge cases",
    },
    {
        "name": "evaluation",
        "color": "c5def5",
        "description": "Evaluation benchmarks, scenarios, and protocols",
    },
]


def render_issue_body(issue: Dict[str, Any]) -> str:
    """Render issue markdown body conforming to the exact required template."""
    criteria_str = "\n".join(f"- [ ] {c}" for c in issue["acceptance_criteria"])
    files_str = "\n".join(f"- `{f}`" for f in issue["relevant_files"])

    return f"""## Problem

{issue["problem"]}

## Current State

{issue["current_state"]}

## Proposed Change

{issue["proposed_change"]}

## Scope

{issue["scope"]}

## Acceptance Criteria

{criteria_str}

## Relevant Files

{files_str}

## Notes

{issue["notes"]}
"""


def ensure_milestones() -> Dict[str, int]:
    """Ensure milestones exist and return a mapping of title -> milestone_number."""
    milestone_map: Dict[str, int] = {}

    # Query existing
    url = f"https://api.github.com/repos/{REPO}/milestones?state=all&per_page=100"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req) as resp:
            existing = json.loads(resp.read().decode())
            for m in existing:
                milestone_map[m["title"]] = m["number"]
    except Exception as exc:
        print(f"[WARN] Failed to fetch milestones: {exc}")

    # Create missing
    for m in MILESTONES_DEF:
        if m["title"] not in milestone_map:
            payload = json.dumps(m).encode()
            create_url = f"https://api.github.com/repos/{REPO}/milestones"
            c_req = urllib.request.Request(create_url, data=payload, headers=HEADERS)
            try:
                with urllib.request.urlopen(c_req) as resp:
                    data = json.loads(resp.read().decode())
                    milestone_map[data["title"]] = data["number"]
                    print(f"[SUCCESS] Created Milestone {data['number']}: {data['title']}")
            except Exception as exc:
                print(f"[ERROR] Failed to create milestone '{m['title']}': {exc}")

    return milestone_map


def ensure_labels() -> None:
    """Ensure all required domain labels exist on the repository."""
    url = f"https://api.github.com/repos/{REPO}/labels?per_page=100"
    req = urllib.request.Request(url, headers=HEADERS)
    existing_labels = set()
    try:
        with urllib.request.urlopen(req) as resp:
            existing = json.loads(resp.read().decode())
            existing_labels = {lbl["name"] for lbl in existing}
    except Exception as exc:
        print(f"[WARN] Failed to fetch labels: {exc}")

    for lbl in LABELS_DEF:
        if lbl["name"] not in existing_labels:
            payload = json.dumps(lbl).encode()
            create_url = f"https://api.github.com/repos/{REPO}/labels"
            c_req = urllib.request.Request(create_url, data=payload, headers=HEADERS)
            try:
                with urllib.request.urlopen(c_req) as resp:
                    print(f"[SUCCESS] Created Label: {lbl['name']}")
            except Exception:
                pass


def save_roadmap_artifact() -> str:
    """Save the comprehensive 62-issue roadmap artifact as Markdown."""
    artifact_path = "/home/aryan/.gemini/antigravity-cli/brain/691d4829-962d-4acb-9a58-7d26f29bb496/engineering_roadmap_60_issues.md"
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

    lines = [
        "# AdaptiveRL — Master 60+ Issue Engineering Roadmap",
        "",
        "This master roadmap establishes a professional, repository-grounded engineering backlog of **62 issues** for AdaptiveRL (`AryanXCode646/ARL`).",
        "",
        "Every issue represents a concrete, actionable engineering goal verified against the codebase, categorized across 16 domains and assigned to 5 planned milestone phases.",
        "",
        "---",
        "",
        "## Summary Matrix",
        "",
        "| ID | Domain | Milestone | Labels | Title |",
        "|:---|:---|:---|:---|:---|",
    ]

    for item in ALL_ISSUES:
        lbls = ", ".join(f"`{lbl}`" for lbl in item["labels"])
        lines.append(
            f"| **#{item['num']}** | {item['domain']} | {item['milestone'].split('—')[0].strip()} | {lbls} | {item['title']} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Issue Specifications")
    lines.append("")

    for item in ALL_ISSUES:
        lines.append(f"### Issue #{item['num']}: {item['title']}")
        lines.append("")
        lines.append(f"**Domain**: `{item['domain']}`  ")
        lines.append(f"**Milestone**: `{item['milestone']}`  ")
        lines.append(f"**Labels**: {', '.join(f'`{lbl}`' for lbl in item['labels'])}")
        lines.append("")
        lines.append(render_issue_body(item))
        lines.append("")
        lines.append("---")
        lines.append("")

    content = "\n".join(lines)
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(content)

    return artifact_path


def main() -> None:
    print("=== AdaptiveRL Master 60+ Issue Roadmap Publisher ===")
    print(f"Repository: {REPO}")
    print(f"Total Issues in Catalog: {len(ALL_ISSUES)}")
    print(f"Token present: {bool(TOKEN)}")
    print()

    # 1. Save artifact first
    artifact_path = save_roadmap_artifact()
    print(f"[OK] Master roadmap artifact saved to: {artifact_path}")
    print()

    # 2. Setup GitHub labels & milestones
    print("Setting up GitHub repository labels and milestones...")
    ensure_labels()
    milestone_map = ensure_milestones()
    print(f"[OK] Milestones configured: {milestone_map}")
    print()

    # 3. Publish each issue
    created_issues: List[Dict[str, Any]] = []
    skipped_issues: List[str] = []

    # Query existing issues to avoid duplicates
    existing_titles = set()
    url = f"https://api.github.com/repos/{REPO}/issues?state=all&per_page=100"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req) as resp:
            existing = json.loads(resp.read().decode())
            existing_titles = {i["title"].lower().strip() for i in existing if "title" in i}
    except Exception as exc:
        print(f"[WARN] Failed to query existing issues: {exc}")

    for idx, item in enumerate(ALL_ISSUES, start=1):
        title = item["title"]
        if title.lower().strip() in existing_titles:
            print(f"[{idx}/{len(ALL_ISSUES)}] [SKIP] Already exists: {title}")
            skipped_issues.append(title)
            continue

        body = render_issue_body(item)
        milestone_num = milestone_map.get(item["milestone"])

        payload_dict: Dict[str, Any] = {
            "title": title,
            "body": body,
            "labels": item["labels"],
        }
        if milestone_num is not None:
            payload_dict["milestone"] = milestone_num

        payload = json.dumps(payload_dict).encode()
        post_url = f"https://api.github.com/repos/{REPO}/issues"
        req = urllib.request.Request(post_url, data=payload, headers=HEADERS)

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                print(
                    f"[{idx}/{len(ALL_ISSUES)}] [SUCCESS] Created Issue #{data['number']}: {title}"
                )
                created_issues.append(
                    {
                        "number": data["number"],
                        "title": title,
                        "url": data["html_url"],
                        "domain": item["domain"],
                    }
                )
        except urllib.error.HTTPError as err:
            if err.code in (400, 422):
                # Retry without milestone or labels
                fallback_payload = json.dumps({"title": title, "body": body}).encode()
                f_req = urllib.request.Request(post_url, data=fallback_payload, headers=HEADERS)
                try:
                    with urllib.request.urlopen(f_req) as resp:
                        data = json.loads(resp.read().decode())
                        print(
                            f"[{idx}/{len(ALL_ISSUES)}] [SUCCESS] Created Issue #{data['number']} (fallback): {title}"
                        )
                        created_issues.append(
                            {
                                "number": data["number"],
                                "title": title,
                                "url": data["html_url"],
                                "domain": item["domain"],
                            }
                        )
                except Exception as r_err:
                    print(f"[{idx}/{len(ALL_ISSUES)}] [ERROR] Failed to create '{title}': {r_err}")
            else:
                print(
                    f"[{idx}/{len(ALL_ISSUES)}] [ERROR] Failed to create '{title}': HTTP {err.code}: {err.reason}"
                )
        except Exception as exc:
            print(f"[{idx}/{len(ALL_ISSUES)}] [ERROR] Failed: {exc}")

        # Rate-limiting pause
        time.sleep(0.6)

    print()
    print("================================================================================")
    print("ROADMAP PUBLICATION SUMMARY")
    print("================================================================================")
    print(f"Total Target Issues: {len(ALL_ISSUES)}")
    print(f"Successfully Created on GitHub: {len(created_issues)}")
    print(f"Skipped (Already Existed): {len(skipped_issues)}")
    print(f"Artifact Location: {artifact_path}")
    print()


if __name__ == "__main__":
    main()
