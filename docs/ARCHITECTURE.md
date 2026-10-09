# SWARMMIND — Architecture

The spec ([`SWARMMIND_TECHNICAL_SPEC.md`](../SWARMMIND_TECHNICAL_SPEC.md), §5, §12, §28) is the design of record.
This document describes the system **as built and verified**, and where each guarantee is enforced.

## 1. Layers

```
 frontend/  React + TS + Vite         renders what the backend sends; owns no simulation or ML logic
     │  REST (JSON) + WebSocket
 backend/app/  FastAPI                 routers (thin) → services → marl/ ; no ML or env code of its own
     │  imports
 marl/         library                 the single source of truth: env, algorithms, training, evaluation, analysis, replay
     │  reads/writes
 configs/ (frozen inputs)   results/ (run artifacts, git-ignored)   demo_data/ (committed real subset)
```

Dependency rules are enforced by `tests/test_import_boundaries.py` (an AST scan): `marl/` never imports
`backend/`; the actor module and `algorithms/common/policy.py` never import critic or global-state modules; no
global `numpy.random` in the environment; no source file beyond the soft line limit.

### `marl/`

| Package | Responsibility |
|---|---|
| `envs/core` | `SwarmWorld`: dynamics, no-follow movement rules, sensing, per-robot beliefs and communication modes, rewards, actor observations, **global state (critic only)**, procedural + frozen maps |
| `envs` | PettingZoo parallel env and the vectorised wrapper used for training |
| `algorithms` | IPPO, MAPPO (differ only in critic input), Random and the frontier heuristic; shared PPO update, GAE with truncation folding, value normalisation |
| `networks` | actor (local CNN encoder + masked categorical head), critics, init |
| `training` | trainer loop, checkpoints at fixed fractions, resume, validation evaluation, run directory + manifest |
| `evaluation` | `EpisodeRunner`/`run_episode`, final held-out evaluation with the access gatekeeper, coordination metrics, ablation |
| `analysis` | aggregation across seeds (Welch/t intervals, Hedges' g), hypothesis decisions, coordination metrics (DER, redundant pursuit, …), leak probe, figures, `RESULTS.md` rendering |
| `replay` | replay schema, `build_frame` (one frame builder), recorder, loader |
| `experiments` | study config (frozen), job planning/resume, workers, outputs |
| `utils` | seeding, config, paths (`safe_join`, `is_safe_id`), hashing, environment versions |

### `backend/app`

`create_app(Settings)` builds the app and a service container. Routers: health, env (maps), models, experiments,
studies, replays, sessions, training, plus two WebSocket endpoints (`/ws/sessions/{id}`, `/ws/training/{run_id}`).
Services: run index, model registry, replay store, study service, map service, session manager, job runner.
The full contract is [`API.md`](API.md) (generated from the OpenAPI schema by `scripts/export_schemas.py`; the frontend types
are generated from the same files and the build fails if they are stale).

### `frontend/src`

`api/` (typed client + `SimSocket` with backoff), `stores/` (zustand), `features/simulation` (frame reducer, timeline, canvas
renderers as pure functions, live controller), `features/{comparison,analysis,training,experiments}`, `components/charts`
(custom SVG kit), `pages/`, `design/` (tokens, motion, contrast). One rAF loop reads stores via `getState()`; the canvas
never causes React renders.

## 2. Data flow

```
scripts/train.py (or the UI's "Train Swarm")        scripts/run_study.py
   └─ marl.training ── writes ──> results/runs/<study>/<algo>/<level>/<run>/
                                    manifest.json  metrics.jsonl  eval_val.jsonl  checkpoints/  actor.pt
study pipeline ── final eval (held-out maps) ──> runs/*/eval/{test,structured}_L2.json
               ── aggregate ──> studies/<study>/summary.json ── render ──> docs/RESULTS.md (+ figures)
               ── record_replay ──> replays/<study>/<run>/*.json.gz
backend reads those files (never recomputes a result) ──> REST/WS ──> frontend
```

Every number the UI shows is read from `summary.json`, a run's files or a replay. `docs/REAL_DATA_AUDIT.md` re-derives each link
of this chain (actors → eval files → summary → RESULTS.md) and compares.

## 3. The information boundary (CTDE)

The actor sees only its **local observation**: an egocentric 8×11×11 window built from that robot's own belief and current
sensing (plus a 10-vector of own state), and a local action mask. The MAPPO critic additionally sees the **global state**
(full terrain, all robots, hidden targets) — available only during training. Exported models (`actor.pt`) contain actor weights
only and are loaded with `weights_only=True`; the evaluator, the backend and the UI never load a critic.

Guarded by tests: counterfactual invariance of actor observations to hidden targets, unknown terrain and unsensed teammates
(`tests/env/test_observations.py`), mode-A isolation, mask locality, the AST import boundary, actor-only inference, and a
statistical leak probe on trained models (`marl/analysis/leak_probe.py`, results in `results/studies/main_L2/leak_probe_L2.json`).

## 4. One code path: parity guarantees

- `EpisodeRunner` is the single stepper used by the batch evaluator (`run_episode`), by live backend sessions, and by the replay recorder.
- `build_frame` is the single replay-frame builder; live WebSocket frames are exactly the stored replay frame type.
- Session seeding reuses the evaluator's `episode_seed(map_id, seed)`.

Verified, not assumed: `tests/integration` runs a real server and shows that a live WebSocket session with the trained actor
reproduces **each of the 10 stored final-checkpoint study replays frame for frame**, and `frontend/e2e` shows the browser's live
session and replay end in the same state as the stored outcome. The frontend frame reducer reproduces the recorded outcome of
all 56 replays that existed during Phase 11 (see `docs/phase_reports/PHASE_11.md`).

## 5. Reproducibility

Seeds are explicit everywhere (`marl/utils/seeding.py`); per-episode evaluation seeds derive from `sha256(map_id:eval_seed)`.
Each run's `manifest.json` records its config hash, library versions, platform and git commit/dirty flag. Evaluation maps are frozen
and hashed (`configs/maps/manifest.json`, tested). Held-out maps are readable only for completed runs, and each access is logged.
Re-evaluating the trained actors reproduces the stored held-out episodes exactly on the development machine (audit row C1);
other hardware or library versions may differ at the floating-point level, which the audit would report rather than hide.

## 6. Security posture (local-first)

Binds to `127.0.0.1`; CORS and WebSocket `Origin` restricted to the dev origin; `Host` allow-list (DNS rebinding); JSON-only POST;
IDs validated (`is_safe_id`, `safe_join`) so no request builds an arbitrary path; stored manifests are scrubbed of absolute paths
before being served; bounded custom environment configs; at most 4 live sessions (idle TTL) and one training job; training runs as
a subprocess with an allow-listed, range-checked override set (no shell); one WebSocket driver per session (takeover closes the old
socket with 4409). Not designed for public hosting.

## 7. Verification map

| Layer | Where | Command |
|---|---|---|
| Environment rules, observations, rewards | `tests/env` | `make test` |
| PPO/GAE/IPPO/MAPPO maths and learning | `tests/marl` (smoke tests marked slow) | `make test`, `make test-slow` |
| Evaluation, statistics, coordination | `tests/evaluation`, `tests/analysis` | `make test` |
| Backend + API contract | `tests/backend` (in-process) | `make test` |
| Whole stack on real sockets and real data | `tests/integration` | `make test` |
| Frontend units + real-artifact reduction | `frontend/src/**/*.test.ts(x)` | `make test-frontend` |
| Browser flows against the results files | `frontend/e2e/run_e2e.py` | `make e2e` |
| Number provenance | `scripts/audit_real_data.py` | `make audit` |
| Fresh machine | `scripts/fresh_clone_check.sh` | `make fresh-clone` |

Known gaps are listed in the README ("Limitations") and in each phase report under "Remaining concerns".
