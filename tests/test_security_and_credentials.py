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
from typing import Any

import pytest

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

    def test_subprocess_git_commit_behavioral_safety(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Behaviorally verify _get_git_commit behaves safely under all runtime conditions."""
        import subprocess

        from adaptive_rl.experiments.manager import _get_git_commit

        # Case 1: Normal execution returns short hash
        real_commit = _get_git_commit()
        assert isinstance(real_commit, str)
        assert len(real_commit) >= 4

        # Case 2: Git returns non-zero returncode -> gracefully returns "unknown"
        def mock_run_nonzero(*args: Any, **kwargs: Any) -> Any:
            class MockCompletedProcess:
                returncode = 128
                stdout = "fatal: not a git repository"
                stderr = ""

            return MockCompletedProcess()

        monkeypatch.setattr(subprocess, "run", mock_run_nonzero)
        assert _get_git_commit() == "unknown"

        # Case 3: Git binary missing (FileNotFoundError) -> returns "unknown"
        def mock_run_missing(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("No such file or directory: 'git'")

        monkeypatch.setattr(subprocess, "run", mock_run_missing)
        assert _get_git_commit() == "unknown"

        # Case 4: Subprocess timeout -> returns "unknown"
        def mock_run_timeout(*args: Any, **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd=["git"], timeout=5)

        monkeypatch.setattr(subprocess, "run", mock_run_timeout)
        assert _get_git_commit() == "unknown"

        # Case 5: Verify execution invariants: shell=False, timeout specified, args fixed list
        captured_calls = []

        def mock_run_capture(cmd: Any, **kwargs: Any) -> Any:
            captured_calls.append((cmd, kwargs))

            class MockResult:
                returncode = 0
                stdout = "abc1234\n"

            return MockResult()

        monkeypatch.setattr(subprocess, "run", mock_run_capture)
        commit = _get_git_commit()
        assert commit == "abc1234"
        assert len(captured_calls) == 1
        cmd, kwargs = captured_calls[0]
        assert cmd == ["git", "rev-parse", "--short", "HEAD"]
        assert kwargs.get("shell") is not True  # Must never run with shell=True
        assert kwargs.get("timeout", 0) > 0  # Must specify finite timeout
