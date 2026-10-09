#!/usr/bin/env bash
# Install the real, committed demo data (study main_L2: 10 trained actors,
# summaries, analysis, replays) into results/ and verify every SHA-256.
# Usage: scripts/demo_setup.sh [--check] [--force]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
[ -x "$PY" ] || { echo "no .venv — run 'make setup' first" >&2; exit 2; }
exec "$PY" scripts/demo_setup.py "$@"
