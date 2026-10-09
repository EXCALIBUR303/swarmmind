#!/usr/bin/env bash
# Fresh-clone proof (Phase 14 gate): follow the README exactly in a brand-new directory and record
# what happened. If the working tree is clean this is a real `git clone` of HEAD; if it has
# uncommitted work, the tracked + untracked-but-not-ignored files are exported instead (what the
# next commit would contain) and the report says so.
#
#   scripts/fresh_clone_check.sh [--out docs/FRESH_CLONE_REPORT.md] [--keep]
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/docs/FRESH_CLONE_REPORT.md"
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in --out) OUT="$2"; shift 2;; --keep) KEEP=1; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac
done
export NODE_OPTIONS="${NODE_OPTIONS:---no-network-family-autoselection}"
WORK="$(mktemp -d -t sm_fresh.XXXXXX)"
DEST="$WORK/swarmmind"
LOG="$WORK/steps.tsv"
: > "$LOG"

if [ -z "$(git -C "$REPO" status --porcelain)" ]; then
  git clone --quiet "$REPO" "$DEST" || exit 2
  MODE="git clone of HEAD $(git -C "$REPO" rev-parse --short HEAD) (working tree clean)"
else
  mkdir -p "$DEST"
  python3 - "$REPO" "$DEST" <<'PY' || exit 2
import os, shutil, subprocess, sys
repo, dest = sys.argv[1:]
files = subprocess.run(["git", "-C", repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True, check=True).stdout.split(b"\0")
n = 0
for f in filter(None, files):
    src = os.path.join(repo, f.decode())
    if os.path.isfile(src):
        d = os.path.join(dest, f.decode()); os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(src, d); n += 1
print(f"exported {n} files")
PY
  MODE="export of the working tree at $(git -C "$REPO" rev-parse --short HEAD) + uncommitted changes (tracked and untracked, ignored files excluded) — equals the next commit"
fi

cd "$DEST" || exit 2
step() { # name, command...
  local name="$1"; shift
  local t0=$SECONDS
  echo "=== $name: $*" >&2
  ( "$@" ) > "$WORK/$(echo "$name" | tr ' /' '__').log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$(printf '%s' "$*" | tr '\n\t' '  ')" "$rc" "$((SECONDS - t0))" >> "$LOG"
  echo "    exit $rc in $((SECONDS - t0))s" >&2
  return 0
}

# Exactly the README quickstart, then every test layer.
step "1 make setup" make setup
step "2 demo data" scripts/demo_setup.sh
step "3 make lint" make lint
step "4 make test" make test
step "5 make test-slow" make test-slow
step "6 frontend tests" bash -c "cd frontend && npm test"
step "7 frontend lint" bash -c "cd frontend && npm run lint"
step "8 frontend build" bash -c "cd frontend && npm run build"
step "9 browser E2E" .venv/bin/python frontend/e2e/run_e2e.py --out "$WORK/E2E_REPORT.md"
step "10 real-data audit" .venv/bin/python scripts/audit_real_data.py --out "$WORK/REAL_DATA_AUDIT.md"
step "11 demo launcher" bash -c '
  scripts/demo.sh >/dev/null 2>&1 & pid=$!
  ok=0
  for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/api/health >/dev/null && curl -sf http://localhost:5173/ >/dev/null; then ok=1; break; fi
    sleep 1
  done
  kill $pid 2>/dev/null; pkill -P $pid 2>/dev/null; sleep 1
  [ $ok = 1 ]'

python3 "$REPO/scripts/fresh_clone_report.py" "$LOG" "$OUT" "$MODE"
RC=$?
[ "$KEEP" = 1 ] && echo "kept: $WORK" || rm -rf "$WORK"
exit $RC
