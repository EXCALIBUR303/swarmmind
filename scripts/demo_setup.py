"""Install the committed demo-data bundle (`demo_data/`) into `results/`.

Idempotent and non-destructive: a file already present with the right
SHA-256 is skipped; a missing file is copied; a present file with a
DIFFERENT hash aborts (nothing is overwritten unless `--force`). That makes
it safe on a machine that already holds the full 6 GB `results/` tree.
`--check` only verifies and never writes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from make_demo_bundle import REPO, sha256


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=REPO / "demo_data")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite files whose hash differs")
    args = ap.parse_args()

    manifest_path = args.bundle / "MANIFEST.json"
    if not manifest_path.is_file():
        print(f"error: {manifest_path} not found (is this a full checkout?)", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    copied = same = 0
    problems: list[str] = []
    root, dest_root = (args.bundle / "results").resolve(), args.results.resolve()
    for rel, meta in sorted(manifest["files"].items()):
        src, dst = (root / rel).resolve(), (dest_root / rel).resolve()
        if not (src.is_relative_to(root) and dst.is_relative_to(dest_root)):
            problems.append(f"unsafe path in manifest (refused): {rel}")
            continue
        if not src.is_file() or sha256(src) != meta["sha256"]:
            problems.append(f"bundle file missing or corrupt: {rel}")
            continue
        if dst.is_file():
            if sha256(dst) == meta["sha256"]:
                same += 1
            elif args.force and not args.check:
                shutil.copy2(src, dst)
                copied += 1
            else:
                problems.append(f"differs from bundle (not overwritten): {rel}")
        elif args.check:
            problems.append(f"missing in results/: {rel}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    if problems:
        print(f"FAILED: {len(problems)} problem(s)", file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
        return 1
    verb = "verified" if args.check else "installed"
    print(f"demo data {verb}: {manifest['n_files']} files ({copied} copied, {same} already present and identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
