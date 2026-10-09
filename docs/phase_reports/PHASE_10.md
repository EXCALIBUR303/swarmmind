# PHASE 10 COMPLETE — FastAPI + WebSocket backend

## IMPLEMENTED
- **All §28.2 REST endpoints** (`/api/…`): health, levels, maps, sessions (create/get/step/reset/delete), models, experiments (list/get/metrics/eval), studies, replays (index + gzip passthrough), analysis heatmap, training jobs (launch/status/cancel). Extras: `GET /replays/{id}/initial`, `GET /training/jobs`.
- **Both WebSockets (§28.3):** `/ws/sessions/{id}` (server-owned clock, play/pause/step/reset/speed, resync `state` on connect, ≤30 steps/s, one driver per session) and `/ws/training/{run_id}` (tails `metrics.jsonl`/`eval_val.jsonl`, complete lines only, status changes, ends cleanly).
- **Services, no logic in routers:** `session_manager` (TTL, max 4 sessions), `model_registry`, `run_index` (tolerates corrupt runs → `invalid`), `replay_store` (index-resolved, `safe_join`), `study_service`, `map_service` (held-out maps via the gatekeeper + access log), `job_runner` (real `scripts/train.py` subprocess, 1 job, SIGTERM→interrupted, SIGKILL after 15 s).
- **Single code path for parity (§28.4):** extracted `EpisodeRunner` (evaluator) and `build_frame` (recorder). `run_episode` now runs on `EpisodeRunner`; live sessions use the same runner and frame builder. Sessions seed the policy with `episode_seed(map_id, seed)` — the batch evaluator's stream.
- **Contract export:** `scripts/export_schemas.py` → `docs/openapi.json`, `docs/schemas/{replay,ws_messages}.schema.json`, `docs/API.md` (a test fails if they go stale).

## FILES CREATED/CHANGED
- New: `backend/app/{container.py, api/deps.py, api/routers/{env,sessions,models,experiments,studies,replays,training}.py, api/ws/{common,sim_ws,train_ws}.py, schemas/{sessions,training}.py, services/{errors,jsonio,run_index,model_registry,replay_store,study_service,map_service,session_manager,job_runner}.py}`, `scripts/export_schemas.py`, `docs/{API.md,openapi.json,schemas/*}`, `tests/backend/{conftest,test_registry_endpoints,test_sessions,test_parity,test_security,test_ws_sim,test_ws_training,test_training_jobs,test_export_schemas}.py`.
- Changed: `backend/app/{main,settings}.py`, `api/routers/health.py`, `marl/evaluation/evaluator.py` (EpisodeRunner), `marl/replay/{recorder,schema}.py` (build_frame, optional `initial`), `marl/utils/paths.py` (reject dot-leading ids), `scripts/train.py` (`--runs-dir`), `pyproject.toml` (ruff B008 for FastAPI), `.gitignore`, `tests/backend/test_health.py`.

## TESTS RUN
- `tests/backend`: **143 passed** (2 marked slow/integration run real training subprocesses), **backend coverage 96 %**; ruff + `mypy backend` clean.
- Rest of the suite (before the final small edits): **345 passed** in 306 s; after the `marl` refactors I re-ran replay/evaluation/boundary tests with the backend: 213 passed.
- **Live check against real `uvicorn`:** real MAPPO actor over a real WebSocket at 8×: 25 frames, 0.033 s min gap (= the 30/s cap), **frames identical to the stored Phase 7 study replay**; `Host: evil.example` → 400; traversal → 404.

## RESULTS (the parity gate)
| Check | Result |
|---|---|
| Tiny exported actor, stochastic and greedy, session vs `record_episode` | every field of every frame equal; outcome metrics equal |
| Random policy session vs `run_episode` + `evaluate_policy` | actions, positions, success, steps, collisions equal |
| Procedural map by seed vs `run_episode(seed=…)` | actions equal |
| **Real MAPPO (s0, 999 424 steps) session vs the stored `main_L2` replay** | **25/25 frames equal, outcome metrics equal** |
| Known map rebuilt from derived `initial` + stored deltas vs live final known map | equal |

## PROBLEMS FOUND / FIXED
1. **Replay format gap (affects Phase 9 & 11).** §27.2's `initial` was never implemented, and the reset-time sensing pass is in *no* frame's `newly_known`, so a viewer that only accumulates deltas leaves the start area fogged for the whole replay. The Phase 9 prototype has this bug. Fix: optional `Replay.initial` (new recordings) + `GET /replays/{id}/initial`, which derives it by re-running reset for the 56 existing replays (no re-recording; study artifacts untouched). Verified against the live session. **Phase 11 must use it.**
2. **Run-id collisions:** the early pilot runs are all named `seed0`; keying by name hid 9 of 30 runs. Colliding runs are now `<study>.<algo>.<level>.<name>`.
3. **`is_safe_id("..")` was true** (the character class admits dots; only `safe_join`'s resolve caught it). Dot-leading ids are now rejected everywhere.
4. **Security review** (the `security-review` skill would not load: `git diff origin/HEAD` fails, there is no remote — so I reviewed the diff by hand). Fixed: (a) real manifests/summaries contain `/Users/sid/...` paths — now scrubbed from all responses; (b) client `env_config` could request a 100 000-cell observation window or a 10⁹-step episode — now bounded; (c) a bool/string was accepted as a WS number — strict typing; (d) several sockets could drive one session — one driver, takeover closes the old with 4409; (e) a recycled PID could make a dead training job look alive and block the single job slot / receive SIGTERM — process start-time token; (f) policy fault during play leaked nothing but would have killed the socket — now a generic `internal_error`; (g) `Host` allowlist (DNS rebinding), WS `Origin` check, JSON-only POST (verified `text/plain` → 422), no credentials in CORS.
5. Spec deviation: heatmaps are precomputed per (study, level, algo, map) over seeds, so the endpoint is `/analysis/{study}/heatmap`, not `{run_id}`.

## REMAINING CONCERNS
- `code-review` (medium) was a self-review of the diff, not an independent reviewer.
- Held-out `GET /maps?set=test|structured` appends to the access log on every call (audit trail, but a chatty client grows it).
- No auth and no REST rate limiting (local-first, bound to 127.0.0.1; per spec). WebSocket message size is bounded in-app (4 KiB) and, if you launch with `--ws-max-size 65536`, at the protocol layer — that flag is documented in `docs/API.md` but not enforced.
- Real-artifact tests skip on a fresh clone (`results/` is gitignored, 6 GB). The tiny-actor parity tests always run.
- The two job tests train for real (~10 s and ~15 s).
- Frame schema is a delta format; the browser reconstruction rules (keyframe cache) are Phase 11.

## VALIDATION GATE
**PASS.** Backend tests pass; coverage 96 % (≥ 80); parity passes (including against a real study replay); OpenAPI + JSON Schema exported and checked by test; `uvicorn` serves real models (26) and real results (30 runs, 56 replays, 2 studies).

## CURRENT PROJECT STATUS
Phases 0–9 committed (Phase 9 approved). **Phase 10 is implemented and uncommitted.**

## NEXT PHASE
Phase 11 — React simulator (Swarm Lab + replay), against the approved Phase 9 design.

## RECOMMENDED MODEL
Sonnet 5.5

## RECOMMENDED EFFORT
High

## WHY
Visual/interaction work with a measured performance gate (≥ 50 fps), canvas coordinate and DPI pitfalls, and browser verification at four widths.

## USE OPUS ONLY IF
Never expected for Phase 11.

## RECOMMENDED SKILLS / CONNECTORS
Built-in browser tools for verification; `emil-animate` / `animation-principles` for motion. Run `python scripts/export_schemas.py` first so `types.gen.ts` can be generated from `docs/openapi.json` and `docs/schemas/*`.

## ACTION REQUIRED
Reply **CONTINUE** to start Phase 11, and **commit** if you want Phase 10 committed first (nothing is committed yet).

## STOP HERE.
