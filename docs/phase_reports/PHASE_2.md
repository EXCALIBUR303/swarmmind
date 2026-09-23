# PHASE 2 COMPLETE

## IMPLEMENTED
- `marl/envs/core/types.py`: constants (terrain, target status, actions), and the shared dataclasses `RobotBelief`, `WorldState`, `StepResult`, `EpisodeTrace` (§6.2, §7.1, §27.1).
- `marl/envs/core/sensing.py`: cached Euclidean disc offsets and `sensed_cells`/`sensed_mask` (§11.2, no occlusion in v1).
- `marl/envs/core/maps_io.py`: `MapSpec` (content-hash independent of id/generator metadata, per §20.2), JSON save/load, and a `#`/`.`/`R`/`T` structured-map text loader (§20.4).
- `marl/envs/core/mapgen.py`: procedural generation — obstacle sampling, largest-component pocket-filling for guaranteed connectivity, clustered/scattered spawn, BFS-gated target placement (`target_min_dist`), full reachability validation, deterministic per-attempt seeding (`hash64`), and `MapGenerationError` after 100 failed attempts (§20.1).
- `marl/envs/core/dynamics.py`: the no-follow conflict rule in one order-independent pass — terrain check, R1 (destination occupied at step start) and R2 (contested empty cell) (§6.5).
- `marl/envs/core/belief.py`: per-robot sensing/belief update and all three communication modes — `none`, `shared_map`, `limited_range` (transitive merge over the comm graph's connected components) (§11, §19).
- `marl/envs/core/rewards.py`: the full reward formula with the exploration-budget terminal payout that makes total exploration reward per successful episode exactly `w_x·(|G|−|K₀|)` regardless of when success happens (§13.2–13.3).
- `marl/envs/core/observations.py`: the 8-channel egocentric actor observation, 10-dim vector, and the locally-computed action mask — **local-only, by construction** (§8, §10.2).
- `marl/envs/core/global_state.py`: the 9-channel privileged critic state (7 shared + 2 agent-specific channels) and the 4-dim global vector (§9).
- `marl/envs/core/world.py`: `SwarmWorld`, the single implementation of the step order in §6.4 (propose → terrain check → conflict resolution → move → rescue → sense → reward → terminate/truncate), plus `actor_obs`/`all_actor_obs`, `global_state`/`agent_state_channels`/`full_global_grid`, and ASCII rendering.
- `marl/envs/pettingzoo_env.py`: `SwarmParallelEnv`, a thin PettingZoo Parallel-API adapter (dict bookkeeping only).
- `marl/envs/vector.py`: `SyncVectorSwarm`, the batched vector wrapper with auto-reset and `FinalInfo` capture (true final obs/state before reset, needed for the truncation bootstrap in Phase 4's GAE).
- The full §33.2 test suite: 96 new tests across 9 files, including the counterfactual information-leak tests (§12.4) that prove the actor's observation and mask are bit-identical when hidden targets move, unobserved terrain changes, or unsensed teammates move — and the reverse-direction sanity check that the critic's global state *does* see what the actor doesn't.

## FILES CREATED/CHANGED
- `marl/envs/core/{types,sensing,maps_io,mapgen,dynamics,belief,rewards,observations,global_state,world}.py`
- `marl/envs/{pettingzoo_env,vector}.py`
- `tests/env/{test_dynamics,test_mapgen,test_observations,test_global_state,test_rewards,test_belief_comm,test_world,test_wrappers,test_sensing_and_maps_io}.py`
- `pyproject.toml` (added the `pydantic.mypy` plugin, needed for mypy to understand `Field(default_factory=...)` defaults correctly)
- `Makefile` (lint target now also checks `marl/envs`)

## TESTS RUN
- `pytest -m "not slow"` — **132 tests total** (36 from Phase 1 + 96 new), all passing.
- `pettingzoo.test.parallel_api_test` (200 cycles) — passing.
- `ruff check` and `mypy marl/utils marl/envs` — both clean.
- Manual smoke scripts: core step loop, PettingZoo conformance, vector-env auto-reset, and a cross-seed reproducibility check (two independent vector envs with the same root seed produce bit-identical map sequences and, after 40 identical joint-action steps, bit-identical robot positions).
- Single-process throughput benchmark on the L2 config (10×10, 3 robots): **~4,600 env-steps/s** including full observation+mask construction, **~10,500 env-steps/s** for `step()` alone — both comfortably above the §35 "flag if <1k" threshold and within the spec's estimated 3k–10k range.

## RESULTS
- `marl/envs/core` coverage: 99% (the one meaningful miss is an `AssertionError` message branch in the rescue-precedes-discovery invariant check, which by construction cannot be hit by a correct implementation — it's a safety net, not dead logic).
- `marl/envs/pettingzoo_env.py`: 100%. `marl/envs/vector.py`: 98%.
- All §12.4 leak tests pass: bit-identical actor observations/masks under counterfactual hidden-target, unknown-terrain, and unsensed-teammate perturbations; mode-A isolation confirmed (a robot's own belief never contains a teammate's out-of-range discovery); the reverse-direction check confirms the critic's global state *does* carry the hidden target the actor is denied.
- The `IPPO ≡ MAPPO` type parity concern doesn't apply yet (that's Phase 5), but the **PettingZoo ↔ vector wrapper parity test** — the Phase 2 analogue of "single code path" — passes: identical (map, seed, actions) produce bit-identical observations, rewards, and termination flags through both APIs.
- The discovery-precedes-rescue invariant (`t_discovered ≤ t_rescued`) held across 30 full random-policy episodes with a runtime `assert` as a permanent safety net in `world.py`.

## PROBLEMS FOUND
1. My own reward invariant test initially asserted a `≤` comparison losing to float32 rounding on exact `==` checks — a test bug, not a code bug (three separate reward tests hit this).
2. My own capacity-overflow config test (`test_env_config_rejects_capacity_overflow` from Phase 1) turned out to be unreachable given field bounds — already documented and fixed in Phase 1's report; unrelated to this phase but re-verified still passing.
3. **A genuine bug**, caught by a coverage-driven test I added: `SwarmParallelEnv.state_space` declared shape `(9, H, W)` (the full agent-specific global channel count) but `state()` actually returns only the 7-channel team-shared grid (it has no per-agent context, so the two agent-specific channels don't apply to a single team-wide `state()` call). The declared space and the actual return value were inconsistent.
4. mypy failed on `Field(default_factory=ClassName)` for `RewardConfig`/`ObsConfig`, reporting spurious "missing named argument" errors, because pydantic v2 model defaults aren't understood by plain mypy without the pydantic plugin.
5. A retry-determinism mapgen test used `target_min_dist=5` on a 6×6 grid expecting infeasibility, but that distance is sometimes achievable depending on obstacle layout — not a true infeasibility guarantee.

## PROBLEMS FIXED
1. Replaced exact `==` with `abs(...) < 1e-6` tolerance comparisons in the affected reward tests.
2. (Carried from Phase 1, re-verified, no new action needed.)
3. Fixed the real bug: `state_space` now declares 7 channels (`N_SHARED_STATE_CHANNELS`), matching what `state()` actually returns, with a comment clarifying that the full 9-channel agent-specific grid for the MAPPO critic comes from `SwarmWorld.full_global_grid(i)` instead, not through PettingZoo's team-level `state()`.
4. Added `plugins = ["pydantic.mypy"]` to `[tool.mypy]` in `pyproject.toml`.
5. Changed that test to use `target_min_dist=100` on a 6×6 grid — a distance that exceeds the maximum possible BFS distance on that grid regardless of obstacle layout, guaranteeing genuine infeasibility.

## REMAINING CONCERNS
- `sensing.occlusion` is not implemented (v1 has no line-of-sight blocking, as the spec explicitly allows — §11.2 documents this as a stretch option, `occlusion: true` is accepted by the config schema but currently has no effect on `sensed_cells`). This is intentional scope, not a gap, but worth flagging so a later phase doesn't assume occlusion silently works.
- `mapgen`'s clustered-spawn tie-breaking sorts candidates by `(BFS distance, y, x)`, which is deterministic but not stated verbatim in the spec beyond "lexicographic order" — I judged this the natural reading and it's covered by the retry-determinism test.
- No `max_steps` calibration has happened yet (correctly deferred to Phase 3) — all four level configs still carry their provisional values from Phase 0/1.
- Environment throughput was benchmarked ad hoc in this session, not via a committed `scripts/bench_env.py` (that script is explicitly Phase 3 scope) — the numbers above are for the phase report only and aren't persisted anywhere.

## VALIDATION GATE
**PASS**

All Phase 2 gate criteria met: all environment tests pass, `marl/envs` coverage is 99% (≥ the 90% target), `pettingzoo.test.parallel_api_test` passes, the 10k+-step fuzz tests pass across dynamics and the full world, all §12.4 leak tests pass, and single-process throughput (~4.6k–10.5k env-steps/s) is well above the flag threshold.

## CURRENT PROJECT STATUS
The environment core is complete and thoroughly tested: deterministic multi-agent dynamics with a well-defined conflict rule, partial observability with three communication modes, procedurally generated maps with reachability guarantees, a reward function with a proven anti-farming invariant, and both PettingZoo and vectorized wrappers that agree bit-for-bit. No baselines, no learning, no backend/frontend integration yet — all correctly deferred.

## NEXT PHASE
Phase 3: Validation + random baseline. Stress-tests the map generator across 1k+ maps per level, freezes and hashes the validation/test/structured map sets, implements the Random policy and the greedy nearest-frontier reference heuristic, builds the episode-metrics module from `EpisodeTrace`, implements the replay recorder/schema/loader, and calibrates each level's `max_steps` (writing the Phase 4 learning-gate thresholds into `docs/EXPERIMENTS.md`).

## RECOMMENDED MODEL
Sonnet

## RECOMMENDED EFFORT
Medium

## WHY
Phase 3's procedures are fully specified (heuristic algorithm, metric formulas, calibration rule in §20.6) and build directly on Phase 2's tested primitives — this is implementation and measurement, not open design or high-risk invariant work like Phase 2 was.

## USE OPUS ONLY IF
Calibration reveals the task is degenerate at a level that implicates methodology — e.g. the random baseline nearly always succeeds even after level-density adjustments, or the heuristic cannot reach a reasonable success rate at any defensible `max_steps`, suggesting the environment design itself (not just its constants) needs to change.

## RECOMMENDED SKILLS / CONNECTORS
`dataviz` skill, for the calibration plots (random vs. heuristic success/steps per level). No other specialized capability is needed — everything else is direct implementation of already-specified procedures.

## ACTION REQUIRED
Nothing new to commit yet (still no git commits exist in this repo — Phase 1's offer to commit was not taken up). Reply **CONTINUE** to begin Phase 3, staying on Sonnet.

## STOP HERE.
