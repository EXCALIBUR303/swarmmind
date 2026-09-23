# SWARMMIND — Architecture (short)

This is a short pointer document. The authoritative architecture, methodology
and design decisions live in
[`SWARMMIND_TECHNICAL_SPEC.md`](../SWARMMIND_TECHNICAL_SPEC.md), especially
§5 (System architecture) and §31 (Repository structure).

## Layers (spec §5.1)

```
frontend (React/Vite/TS)  →  backend (FastAPI, thin adapters)  →  marl/ (single source of truth)
                                                                        →  filesystem (configs/, results/, models/)
```

`marl/` never imports from `backend/`. `backend/` never re-implements
environment or ML logic — see the dependency table in spec §5.2, enforced
by `tests/test_import_boundaries.py`.

## Current status

Phase 1 complete: repository scaffolding, seeding/config/path/hash/version
utilities, the `EnvConfig` schema + level registry (L1–L4), a FastAPI health
endpoint, and a Vite/React/TS app shell with routing to six placeholder
pages. No environment dynamics, no ML, no real data yet.

See `docs/phase_reports/` for the full history of what was implemented,
tested and validated in each phase.

## Quickstart

```bash
make setup          # python venv + pip install -e ".[dev]" + npm install
make test           # python test suite (fast)
make dev-backend    # uvicorn on :8000
make dev-frontend   # vite dev server on :5173
```
