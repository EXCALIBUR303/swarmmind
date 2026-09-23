"""Central path resolution for the repository's data directories.

Backend routers use `safe_join` (see spec §37, Security) to resolve
user-supplied IDs against these roots without allowing path traversal.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = REPO_ROOT / "configs"
MAPS_DIR = CONFIGS_DIR / "maps"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
STUDIES_DIR = RESULTS_DIR / "studies"
REPLAYS_DIR = RESULTS_DIR / "replays"
MODELS_DIR = REPO_ROOT / "models"
DOCS_DIR = REPO_ROOT / "docs"

# Strict identifier pattern for anything used to build a filesystem path
# from an external request (run_id, replay_id, model_id, map_id).
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def is_safe_id(value: str) -> bool:
    """True iff `value` is safe to use as a single path segment."""
    return bool(_SAFE_ID_RE.match(value))


class UnsafePathError(ValueError):
    """Raised when a resolved path would escape its intended root."""


def safe_join(root: Path, *parts: str) -> Path:
    """Join `parts` onto `root`, refusing traversal outside of `root`.

    Every part must independently match `_SAFE_ID_RE`. This is stricter
    than only checking the final resolved path, and it rejects encoded
    traversal attempts (e.g. "..%2f") before they ever reach the filesystem.
    """
    for part in parts:
        if not is_safe_id(part):
            raise UnsafePathError(f"unsafe path segment: {part!r}")
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if not candidate.is_relative_to(root):
        raise UnsafePathError(f"resolved path {candidate} escapes root {root}")
    return candidate


def ensure_dirs() -> None:
    """Create the standard results/models directory tree if missing."""
    for d in (RUNS_DIR, STUDIES_DIR, REPLAYS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
