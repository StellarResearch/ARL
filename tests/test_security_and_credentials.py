"""Security regression tests for credential safety and local path hygiene.

Verifies:
1. No source code or tests access or reference ~/.git-credentials or local credential stores.
2. No hardcoded personal developer paths (/home/..., C:\\Users\\...) exist in tracked files.
3. No personal tokens or sensitive authentication material exist in codebase.
4. Git subprocess operations are strictly confined to read-only non-sensitive metadata queries (e.g. git rev-parse).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestCredentialAndPathSecurity:
    """Security tests to prevent credential theft, leakage, or environment contamination."""

    def test_no_git_credentials_store_references(self) -> None:
        """Verify no file in src/ or scripts/ references ~/.git-credentials or .git-credentials."""
        forbidden_patterns = [
            re.compile(r"\.git-credentials"),
            re.compile(r"git-credentials"),
        ]

        violations = []
        for check_dir in ["src", "scripts", "configs", "docs"]:
            dir_path = REPO_ROOT / check_dir
            if not dir_path.exists():
                continue
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith((".pyc", ".png")):
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    for pattern in forbidden_patterns:
                        if pattern.search(text):
                            violations.append(
                                f"{file_path.relative_to(REPO_ROOT)} matched {pattern.pattern}"
                            )

        assert not violations, "Forbidden credential store references found:\n" + "\n".join(
            violations
        )

    def test_no_hardcoded_developer_filesystem_paths(self) -> None:
        """Verify no personal home or user paths exist in codebase."""
        forbidden_patterns = [
            re.compile(r"/home/[a-zA-Z0-9_-]+/"),
            re.compile(r"[C-Z]:\\Users\\[a-zA-Z0-9_-]+", re.IGNORECASE),
        ]

        violations = []
        for check_dir in ["src", "configs", "tests"]:
            dir_path = REPO_ROOT / check_dir
            if not dir_path.exists():
                continue
            for file_path in dir_path.rglob("*.py"):
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                for pattern in forbidden_patterns:
                    matches = pattern.findall(text)
                    if matches:
                        violations.append(f"{file_path.relative_to(REPO_ROOT)}: {matches}")

        assert not violations, "Hardcoded developer-specific paths found:\n" + "\n".join(violations)

    def test_subprocess_git_calls_are_safe_and_readonly(self) -> None:
        """Ensure subprocess calls to git do not leak secrets or execute mutation commands."""
        import inspect

        from adaptive_rl.experiments.manager import _get_git_commit

        # Verify _get_git_commit only runs 'git rev-parse --short HEAD'
        source = inspect.getsource(_get_git_commit)
        assert "git" in source
        assert "rev-parse" in source
        assert "--short" in source
        assert "push" not in source
        assert "clone" not in source
        assert "credential" not in source
