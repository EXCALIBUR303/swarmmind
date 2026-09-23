# SWARMMIND

**Multi-Agent Robot Swarm Learning and Coordination Simulator**

A 2D grid simulator and MARL research pipeline in which a team of robots
learns — via Random / IPPO / MAPPO — to cooperatively explore an unknown
environment and rescue targets under partial observability.

The authoritative specification is
[`SWARMMIND_TECHNICAL_SPEC.md`](SWARMMIND_TECHNICAL_SPEC.md). Read it before
touching anything else — it defines the research question, environment
semantics, algorithms, evaluation protocol, UI system, repository layout and
the phase-by-phase implementation plan this project follows.

## Status

Implementation is in progress, one spec phase at a time (see
`docs/phase_reports/` for what's been done). Currently: **Phase 1 complete**
(repository scaffolding only — no environment or ML logic yet).

## Quickstart

Requires Python 3.11 and Node.js.

```bash
make setup          # creates .venv, installs the marl/backend package + dev deps, npm installs the frontend
make test           # runs the fast Python test suite
make lint           # ruff + mypy on marl/utils

make dev-backend    # FastAPI on http://127.0.0.1:8000 (GET /api/health)
make dev-frontend   # Vite dev server on http://localhost:5173
```

Frontend-only checks:

```bash
cd frontend
npm test            # vitest
npm run build       # mock-guard + typecheck + production build
```

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a short pointer, and
spec §5 for the full layered architecture and dependency rules.

## Results

Once experiments have run (Phase 7+), results are generated — never
hand-written — into `docs/RESULTS.md` by `scripts/aggregate_results.py`.
