"""Tolerant JSON / JSONL readers: a partial or corrupt file is a value
(`None` / skipped line), never an exception that takes the API down (§37)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from marl.utils.paths import REPO_ROOT


def read_json(path: Path) -> Any | None:
    try:
        with path.open("r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path, from_line: int = 0) -> list[dict[str, Any]]:
    """Complete, parseable lines only (a half-written last line is skipped)."""
    out: list[dict[str, Any]] = []
    try:
        with path.open("r") as f:
            for i, line in enumerate(f):
                if i < from_line or not line.endswith("\n"):
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


_REPO_PREFIX = str(REPO_ROOT) + os.sep
_HOME_PREFIX = str(Path.home()) + os.sep


def scrub(obj: Any) -> Any:
    """Recursively strip local absolute paths from API output: the repo root
    becomes a repo-relative path and the home directory becomes `~/`, so
    responses never reveal the host's directory layout or username."""
    if isinstance(obj, str):
        return obj.replace(_REPO_PREFIX, "").replace(_HOME_PREFIX, "~/")
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    return obj
