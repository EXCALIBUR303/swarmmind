"""Collects software/version information for run manifests (§21.2, §34)."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit() -> tuple[str | None, bool]:
    """Return (commit_hash, is_dirty). (None, False) if not a git repo."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return commit, bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def collect_versions() -> dict[str, object]:
    """Return a JSON-serializable dict of environment/version metadata."""
    commit, dirty = _git_commit()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _pkg_version(name)
            for name in (
                "numpy",
                "torch",
                "pettingzoo",
                "gymnasium",
                "pydantic",
                "fastapi",
                "scipy",
                "pandas",
            )
        },
        "git_commit": commit,
        "git_dirty": dirty,
    }
