# PHASE 14 COMPLETE — End-to-end integration

## IMPLEMENTED
- **A committed, real demo-data bundle** (`demo_data/`, 133 files, 38 MB, SHA-256 `MANIFEST.json`): the 10 trained `main_L2` actors, per-run manifests/metrics/eval files, the study summary + analysis, and the 40 recorded replays, copied byte for byte from `results/` by `scripts/make_demo_bundle.py`. `scripts/demo_setup.sh` installs it into the (gitignored, 6 GB) `results/`, verifies every hash, never overwrites a differing file, and refuses manifest paths that escape `results/`. This is what makes "a clean checkout with only real data" possible: `results/` and `*.pt` are not in git.
- **`make demo`**: one command (demo data → production build → backend `:8000` + frontend `:5173`); `requirements.lock` pins the exact versions that trained and audited the models; `make setup` uses it.
- **Integration tests** (`tests/integration`, 10 tests): a real uvicorn process on a results tree containing *only* the bundle. Live **WebSocket** sessions with the trained actors reproduce **each of the 10 stored final-checkpoint study replays frame for frame**; a fresh procedural-map session equals a direct `record_episode`; every number the API serves for the study equals the file; no endpoint leaks a local path; WS Origin/Host/JSON-only rules hold on real sockets; the training job streams live metrics over the training socket and cancels (`interrupted`); bundle tamper detection; the launcher. (Mutation-checked: a wrong seed fails the parity test.)
- **Browser end-to-end run** (`frontend/e2e/run_e2e.py`, `make e2e`): production build + bundle-only backend + headless Chromium over the DevTools protocol (no Playwright: it would add a second browser download). 7 steps: replay and live MAPPO on a held-out map end in the same state as the stored outcome (t=25, 2/2 rescued, 93% coverage, 0 collisions); the Compare table's 18 cells equal `summary.json` (mean/SD/CI); the analysis caption and the 4 Swarm Evolution panels equal the stored replays; a real UI-launched training job streams metrics and is cancelled; history lists every run with its manifest values; settings; animated rendering 60 fps. **0 console or network errors.** → `docs/E2E_REPORT.md`
- **Real-data audit** (`scripts/audit_real_data.py`, `make audit`) → `docs/REAL_DATA_AUDIT.md`: re-derives each link of actor → eval file → summary → RESULTS.md → API/UI and compares (table below).
- **Performance** (`scripts/perf_check.py`) → `docs/PERFORMANCE.md`; **fresh-clone check** (`scripts/fresh_clone_check.sh`) → `docs/FRESH_CLONE_REPORT.md`.
- **Docs**: README rewritten (quickstart verified, results stated with their limits, limitations), `docs/ARCHITECTURE.md` final, `docs/DEMO.md` (8-minute script), Makefile targets (`demo`, `e2e`, `audit`, `perf`, `fresh-clone`, `test-all`).

## TESTS RUN (final, on the fresh export)
| Layer | Result |
|---|---|
| `make lint` (ruff + mypy) | clean |
| `make test` (fast Python incl. integration) | **490 passed**, 1 skipped (launcher test needs the UI build, which the next steps create); coverage of `marl`+`backend` **94%** |
| `make test-slow` (learning smoke) | **12 passed** (7.6 min) |
| Vitest / oxlint / `npm run build` (mock guard, token audit, fresh generated types) | **203 passed** (1 opt-in skipped) / clean / clean |
| Browser E2E | **7/7**, 0 console/network errors, ~37 s |
| Real-data audit | **10 PASS, 1 PARTIAL, 0 FAIL** |
| Demo launcher | both servers up in 3 s; stopping it leaves the caller alive |

## RESULTS (gate)
| Gate | Result |
|---|---|
| Fresh clone following the README: setup + full test suite | **PASS** — 11 of 11 steps exit 0 (`docs/FRESH_CLONE_REPORT.md`). The tree was exported with the uncommitted work (it equals the next commit); no `git clone` yet because the tree is dirty. Final run used the exact pinned versions. |
| E2E flows | **PASS** (list above) |
| Real-data audit table complete | **PASS** (one PARTIAL row, explained below) |
| Demo script < 10 min | **PASS as written**: `docs/DEMO.md` is an 8-minute script; its machine steps run end to end in about 37 s under `make e2e`. The 8 minutes is a per-step estimate for a presenter, **not a stopwatch run by a person**. |

**Real-data audit rows** — every row re-derives a number from a more primary artifact:
| # | Claim | Result |
|---|---|---|
| A1/A2 | bundle = manifest (133 SHA-256); all 10 runs completed from a clean git tree, ids carry their config hash | PASS |
| B1 | `summary.json` **re-aggregated from the per-run eval files is identical** (every leaf; only the timestamp differs) | PASS |
| B2 | `docs/RESULTS.md` **re-rendered from summary.json is identical** (111 lines) | PASS |
| C1/C2 | the **actors re-run on held-out maps reproduce the stored episodes exactly**: 240 structured-map episodes (all 10 runs) + 400 test-set episodes (2 runs), 0 differences | PASS |
| D1 | each stored final-checkpoint replay's outcome equals the same episode in the eval file | PASS |
| E1 | 15 frontend fixtures verified as sub-structures of real files/payloads | **PARTIAL** — the fixtures also contain pilot/gate runs, the `smoke` study and 16 replays that are not in the bundle (they exist only in the developer's full `results/`); those parts cannot be checked from a clean clone |
| E2 | production frontend imports no mock and has no hard-coded result number | PASS |
| F1 | every result number quoted in README/ARCHITECTURE/DEMO is registered and matched to a *named* `summary.json` value (an unregistered number fails; mutation-checked) | PASS |
| G1 | API payloads = files; live sessions = stored replays, on real sockets | PASS |

**Cross-version reproducibility (unplanned, useful):** the first fresh install resolved unpinned dependencies to *newer* versions (torch 2.14.1 vs the 2.14.0 that trained the models, newer fastapi/ruff/mypy). Every step passed except the lint item (defect 2) and the launcher step (defect 3 killed its caller), **including the held-out re-evaluation: all 640 episodes identical**. The second and third runs used the new lock file.

**Performance (one laptop, measured, `docs/PERFORMANCE.md`):** environment ≈2,500 env-steps/s; one trained-actor episode step ≈1.1 ms; backend REST step median 2.2 ms, session creation 4 ms; WebSocket delivers 3.97 frames/s at 1× and 29 frames/s at 8× (cap 30); canvas 60 fps animated, draw 0.2–0.3 ms; production JS 146 KiB gzipped; backend 247 MB resident. Training throughput from the manifests: IPPO ≈57, MAPPO 43–92 env-steps/s.

## PROBLEMS FOUND / FIXED
Product defects (found only by running the whole stack):
1. **Live Play was silently dropped while the session WebSocket was still connecting** — the button looked enabled, the click was ignored (`SimSocket.send` returns `false`). Controls are now disabled until the socket is open (`features/simulation/controls.ts`, regression test).
2. **A clean install failed `make lint`**: mypy needs `types-PyYAML`, which had only ever been installed by hand. Added to the dev dependencies.
3. **`scripts/demo.sh` used `kill 0`**, which killed its whole process group, including whatever script launched it. Now stops only its own two servers (regression test).
4. **Dependencies were version ranges**, so a fresh install could differ from the environment that trained the models → `requirements.lock`.
5. Hardening: the demo installer refuses manifest paths that escape `results/` (test).

Tooling / environment findings:
- The **installed Google Chrome is terminated ~35 s after launch** on this machine (its self-updater), which was the cause of the "flaky headless Chrome" noted in Phases 11–13. The E2E uses puppeteer's `chrome-headless-shell`, which is stable (`CHROME_BIN` overrides).
- Several of my own E2E/test-script bugs (a select race, an element returned by value, a tamper test that changed nothing) were found and fixed; none were product code.
- The fresh-clone script's own report crashed once on a multi-line command; fixed (and one run's report was re-rendered from its preserved step log before the final complete run).
- `make setup` took **26 minutes on a cold cache** over a ≈140 kB/s link (PyTorch ≈130 MB) and ~38 s with warm caches. The README says so rather than promising minutes.

## REMAINING CONCERNS
- **Not committed, so no literal `git clone` was tested.** The fresh check exported the working tree (equal to the next commit). After you commit, `make fresh-clone` will do a real clone; I expect the same result.
- **The bundle adds 38 MB of binary weights to git history.** Fine for a university repo, but it is a permanent size cost; Git LFS or a GitHub release asset are alternatives if that matters.
- The human **demo timing is estimated, not stopwatch-measured.**
- Verified on **macOS arm64 only** (Linux and Windows untested), one machine, CPU only; Safari/Firefox untested; the E2E needs a Chromium and the Chrome-updater issue above may affect others.
- **Audit row E1 is PARTIAL** (fixtures also reference data outside the bundle). The pilot/gate runs with stale "running" manifests and the `smoke` study are deliberately not bundled.
- **I did not run the `code-review` or `security-review` skills** (`security-review` needs a git remote; none exists). I reviewed the new surfaces by hand (path handling, subprocess use, ports, manifests); Phase 15 should do the independent review.
- The audit's held-out re-evaluations use the frozen test/structured maps through the gatekeeper with a private access log; they are reproducibility checks, not tuning, but they are additional reads of held-out maps.
- `GET /api/health` costs ≈100 ms (it shells out to git for the version block); harmless, not changed.
- A third-party deprecation warning (Starlette `TestClient` with `httpx`) will need attention if dependencies are ever upgraded; the lock avoids it for now.
- GitHub push not done (`plugin:engineering:github` still needs authorization; `gh` CLI not used).

## VALIDATION GATE
**PASS** — with the caveats above (uncommitted export instead of a literal clone; demo timing estimated).

## CURRENT PROJECT STATUS
Phases 0–14 built; Phases 0–13 committed (`68324be`), Phase 14 uncommitted. 490 + 12 Python tests, 203 frontend tests, 7-step browser run and 11-row audit all green on a fresh install. Remaining: Phase 15 (independent final audit).

## NEXT PHASE
**Phase 15 — Final MARL + system audit** (no new features): adversarial review of environment rules, information boundaries, IPPO/MAPPO against spec §15/§16, evaluation fairness, claims vs. evidence, documentation accuracy; writes `docs/AUDIT.md`.

## RECOMMENDED MODEL
**Opus 5.5.**

## RECOMMENDED EFFORT
**Extra High (xhigh)** if available, otherwise High.

## WHY
It is an adversarial methodological review where a missed leak or overclaim is the failure; the spec assigns Opus. Phase 14's audit tooling makes numbers cheap to re-derive, so the audit can concentrate on judgement.

## USE OPUS ONLY IF
n/a — Phase 15 is already an Opus phase. (For Phase 14 no discrepancy implicating model/env parity appeared: every parity check matched exactly.)

## RECOMMENDED SKILLS/CONNECTORS
`code-review` (max), `security-review` (needs a git remote or manual use), `data:statistical-analysis`. GitHub only if you want to push: it needs authorization in your claude.ai connector settings.

## ACTION REQUIRED
Commit Phase 14 (say "commit"), then reply **CONTINUE** for Phase 15. Decide whether the 38 MB bundle should stay in git or move to LFS/a release.
