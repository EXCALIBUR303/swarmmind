"""Build the committed demo-data bundle (`demo_data/`) from `results/`.

`results/` is ~6 GB (training checkpoints) and gitignored, so a clean clone
has no trained models. This script copies the *real* artifacts the app and
the documentation need for study `main_L2` into `demo_data/`, byte for byte,
and writes `demo_data/MANIFEST.json` with a SHA-256 for every file.

Included (everything the backend, UI and RESULTS.md read):
  runs/main_L2/<algo>/<level>/<run>/{manifest.json, actor.pt, metrics.jsonl,
                                     eval_val.jsonl, eval/*.json}
  studies/main_L2/**            summary, baselines, analysis, leak probe, ablation
  replays/main_L2/**            stored evaluation replays (gz)
Excluded: `checkpoints/` (≈250 MB per run; only the *final* actor is needed
to run the app), pilot/gate runs (superseded by the study, some with stale
manifests) and the smoke study (an infrastructure dry run, never evidence).

Run manifests keep their original absolute `checkpoints[].path` strings
(evidence is copied unmodified); the files they point to are not shipped.
`scripts/demo_setup.sh` installs the bundle into `results/` and verifies
every hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDY = "main_L2"
RUN_FILES = ("manifest.json", "actor.pt", "metrics.jsonl", "eval_val.jsonl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select(results: Path) -> list[Path]:
    """Relative paths (under `results/`) that belong in the bundle."""
    rel: list[Path] = []
    runs = results / "runs" / STUDY
    for run in sorted(p for p in runs.glob("*/*/*") if p.is_dir()):
        manifest = json.loads((run / "manifest.json").read_text())
        if not (run / "actor.pt").is_file() or manifest.get("end_time") is None:
            raise SystemExit(f"refusing to bundle an unfinished run: {run}")
        rel += [(run / name).relative_to(results) for name in RUN_FILES]
        rel += sorted(f.relative_to(results) for f in (run / "eval").glob("*.json"))
    for sub in ("studies", "replays"):
        rel += sorted(f.relative_to(results) for f in (results / sub / STUDY).rglob("*") if f.is_file())
    return rel


def build(results: Path, out: Path) -> dict:
    files = select(results)
    if out.exists():
        shutil.rmtree(out)
    entries = {}
    for r in files:
        dst = out / "results" / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results / r, dst)
        entries[r.as_posix()] = {"sha256": sha256(dst), "bytes": dst.stat().st_size}
    manifest = {
        "description": (
            f"Real artifacts of study `{STUDY}` copied unmodified from results/ "
            "(see scripts/make_demo_bundle.py). Install with scripts/demo_setup.sh."
        ),
        "study": STUDY,
        "n_files": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries.values()),
        "files": entries,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "demo_data")
    args = ap.parse_args()
    m = build(args.results, args.out)
    print(f"bundled {m['n_files']} files, {m['total_bytes'] / 1e6:.1f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
