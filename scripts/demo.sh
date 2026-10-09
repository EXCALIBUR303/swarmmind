#!/usr/bin/env bash
# Start the backend (:8000) and the production frontend (:5173) for the demo; Ctrl-C stops both.
# Prerequisites: `make setup`, then `make demo-setup build-frontend` (`make demo` runs those last two for you).
set -euo pipefail
cd "$(dirname "$0")/.."
export NODE_OPTIONS="${NODE_OPTIONS:---no-network-family-autoselection}"
[ -x .venv/bin/python ] || { echo "no .venv — run 'make setup' first" >&2; exit 2; }
[ -f frontend/dist/index.html ] || { echo "no frontend build — run 'make build-frontend'" >&2; exit 2; }
.venv/bin/python scripts/demo_setup.py --check >/dev/null || { echo "demo data missing or altered — run 'make demo-setup'" >&2; exit 2; }

# Stop exactly the two servers started here (never the caller's process group).
pids=()
cleanup() { trap - EXIT INT TERM; kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM
.venv/bin/python -m uvicorn backend.app.main:app --port 8000 --ws-max-size 65536 & pids+=($!)
(cd frontend && exec ./node_modules/.bin/vite preview --port 5173 --strictPort) & pids+=($!)
echo "SWARMMIND demo: open http://localhost:5173 (Ctrl-C to stop)"
wait
