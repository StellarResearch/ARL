#!/usr/bin/env python3
"""Format and post the 62-issue engineering roadmap summary comment to PR #12 on ashishsinghbora/ARL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

# Ensure scripts dir is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from build_roadmap_data import ISSUES_DATA
from build_roadmap_data_part2 import ISSUES_DATA_PART2
from build_roadmap_data_part3 import ISSUES_DATA_PART3

ALL_ISSUES = ISSUES_DATA + ISSUES_DATA_PART2 + ISSUES_DATA_PART3

# Upstream tracked issue mapping on ashishsinghbora/ARL
UPSTREAM_MAP = {
    "standardize public abstract interface for classical": 16,
    "avoid representing failed path lengths as 0.0": 13,
    "define success/overflow episode outcome semantics": 14,
    "synchronize evaluation seed protocol": 15,
    "add github actions workflow for automated testing": 17,
    "fix missing type annotations in test fixtures": 18,
    "resolve ruff formatting discrepancies": 19,
    "handle unreadable manifests and incomplete experiment": 20,
    "incorporate hypothesis testing and confidence intervals": 21,
    "register gymnasium envspecs to eliminate": 22,
    "record execution duration, evaluation parameters": 23,
    "update adaptive-rl train help text and documentation": 24,
}


def find_upstream(title: str) -> int | None:
    t_lower = title.lower()
    for key, num in UPSTREAM_MAP.items():
        if key in t_lower:
            return num
    return None


def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    cred_file = os.path.expanduser("~/.git-credentials")
    if os.path.exists(cred_file):
        try:
            with open(cred_file, encoding="utf-8") as f:
                for line in f:
                    if "github.com" in line and ":" in line and "@" in line:
                        part = line.strip().split("@")[0]
                        if ":" in part:
                            candidate = part.split(":")[-1]
                            if candidate.startswith("gho_") or candidate.startswith("ghp_"):
                                return candidate
        except Exception:
            pass
    return ""


def build_comment_markdown() -> str:
    lines: list[str] = []
    lines.append("## 🗺️ AdaptiveRL — 62-Issue Comprehensive Engineering Roadmap")
    lines.append("")
    lines.append(
        "As part of the classical planning, benchmarking, and experiment framework introduced in this PR, here is the full **62-issue engineering roadmap** for AdaptiveRL. It spans **16 core engineering domains** aligned across **5 milestone phases** (Phases 18–22)."
    )
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append(
        "> Foundational reliability and bug-fix issues (#13–#24) are already tracked directly as individual issues on upstream `ashishsinghbora/ARL` and are cross-referenced below."
    )
    lines.append("")

    # Table of domains
    lines.append("### 📊 Domain Overview")
    lines.append("")
    lines.append("| # | Domain | Issue Range | Target Milestone | Key Focus |")
    lines.append("|---|---|---|---|---|")

    domains: dict[str, list[dict]] = {}
    for iss in ALL_ISSUES:
        d = iss.get("domain", "General")
        domains.setdefault(d, []).append(iss)

    domain_meta = {
        "Core Architecture": (
            "Phase 18 — Reliability",
            "Unified planner interfaces, capability metadata, API freeze tests",
        ),
        "Environment Engineering": (
            "Phase 18 & 22",
            "Farama Gym registration, traffic semantics, dynamic obstacles",
        ),
        "RL Algorithms": (
            "Phase 18 & 21",
            "SAC numerical stability, shared action distributions, recurrent models",
        ),
        "Classical Planners": (
            "Phase 18 & 19",
            "RRT* neighborhood radius, KD-Tree 3D RRT, hybrid neuro-symbolic",
        ),
        "Experiment Management": (
            "Phase 18 & 20",
            "Concurrency safety, manifest artifact ordering, traceback capture",
        ),
        "Benchmarking & Research": (
            "Phase 19 — Benchmarking",
            "Hypothesis testing, multi-seed aggregations, statistical reports",
        ),
        "Metrics & Evaluation": (
            "Phase 18 & 19",
            "Zero-metric preservation, flat CSV serializers, latency telemetry",
        ),
        "Dashboard & Visualization": (
            "Phase 21 — DevEx",
            "Malformed manifest recovery, Pareto frontier, comparative TUI",
        ),
        "CLI & Developer Experience": (
            "Phase 21 — DevEx",
            "Multi-algorithm help text, dry-run validation, completion scripts",
        ),
        "Testing & Quality": (
            "Phase 18 — Reliability",
            "Zero-error mypy annotations, property-based tests, stress fixtures",
        ),
        "CI/CD": (
            "Phase 18 & 22",
            "Matrix testing (3.10-3.12), artifact publishing, security linters",
        ),
        "Packaging & Release Engineering": (
            "Phase 22 — Release",
            "Setuptools/PyPI wheels, optional dependency extras, headless mode",
        ),
        "Documentation & Open Source": (
            "Phase 21 — DevEx",
            "Canonical upstream URLs, architecture diagrams, tutorial notebooks",
        ),
        "Performance & Profiling": (
            "Phase 20 — Performance",
            "Spatial query KD-Trees, vectorized rollouts, memory profiling",
        ),
        "Security & Repository Hygiene": (
            "Phase 18 — Reliability",
            "Path traversal guards, unpickling safety, dependency audits",
        ),
        "Future Research Capabilities": (
            "Phase 19 & 22",
            "Domain randomization, meta-RL, multi-agent navigation baselines",
        ),
    }

    for i, (d, items) in enumerate(domains.items(), 1):
        m_info = domain_meta.get(d, ("Phase 18", "Roadmap implementation"))
        first_num = items[0]["num"]
        last_num = items[-1]["num"]
        lines.append(
            f"| {i} | **{d}** | #{first_num} – #{last_num} ({len(items)} issues) | {m_info[0]} | {m_info[1]} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("### 📋 Detailed Engineering Backlog (62 Issues)")
    lines.append("")

    for d, items in domains.items():
        lines.append(f"<details><summary><b>{d} ({len(items)} Issues)</b></summary>")
        lines.append("")
        for iss in items:
            num = iss["num"]
            title = iss["title"]
            up = find_upstream(title)
            up_badge = f" *(Tracked upstream in #{up})*" if up else ""
            labels_str = ", ".join(f"`{lbl}`" for lbl in iss.get("labels", []))
            lines.append(f"#### #{num}: {title}{up_badge}")
            lines.append(f"- **Milestone**: `{iss.get('milestone', 'Phase 18')}`")
            lines.append(f"- **Labels**: {labels_str}")
            lines.append(f"- **Problem**: {iss.get('problem', '').strip()}")
            lines.append(f"- **Proposed Change**: {iss.get('proposed_change', '').strip()}")
            if iss.get("acceptance_criteria"):
                lines.append("- **Key Acceptance Criteria**:")
                for ac in iss["acceptance_criteria"][:3]:
                    lines.append(f"  - [ ] {ac}")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append("### 🎯 Milestone Alignment")
    lines.append("")
    lines.append(
        "- **Phase 18 — Reliability & Reproducibility**: Architectural integrity, seed synchronization, typing, zero-error linting, CI automation."
    )
    lines.append(
        "- **Phase 19 — Benchmarking & Research**: Rigorous statistical benchmarking, hypothesis testing, classical 3D baselines, generalization."
    )
    lines.append(
        "- **Phase 20 — Performance & Scaling**: Spatial indexing (KD-trees), parallel seed execution, memory and CPU profiling."
    )
    lines.append(
        "- **Phase 21 — Developer Experience**: Rich TUI dashboard enhancements, complete CLI developer tooling, interactive visualization."
    )
    lines.append(
        "- **Phase 22 — Release Engineering**: Farama Gymnasium global registry, automated PyPI release packaging, environment standardization."
    )

    return "\n".join(lines)


def post_comment(comment_text: str, token: str) -> None:
    url = "https://api.github.com/repos/ashishsinghbora/ARL/issues/12/comments"
    payload = json.dumps({"body": comment_text}).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "AdaptiveRL-Roadmap-Publisher",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"[SUCCESS] Posted comment to PR #12: {data.get('html_url')}")
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8")
        print(f"[ERROR] HTTP {err.code}: {err.reason}\n{err_body}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post 62-issue roadmap comment to PR #12")
    parser.add_argument("--post", action="store_true", help="Post comment to GitHub")
    args = parser.parse_args()

    comment_text = build_comment_markdown()
    print(f"Comment generated successfully. Total length: {len(comment_text)} characters.")

    if not args.post:
        print("\n--- PREVIEW (first 40 lines) ---")
        for line in comment_text.splitlines()[:40]:
            print(line)
        print("...\nRun with --post to submit to GitHub.")
        return

    token = get_github_token()
    if not token:
        print("[ERROR] No GitHub token found in GITHUB_TOKEN or ~/.git-credentials.")
        sys.exit(1)

    print("Posting comment to https://github.com/ashishsinghbora/ARL/pull/12...")
    post_comment(comment_text, token)


if __name__ == "__main__":
    main()
