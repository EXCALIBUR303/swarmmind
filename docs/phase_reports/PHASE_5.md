# PHASE 5 COMPLETE

## IMPLEMENTED
- **`MAPPOAgent`** ([mappo/agent.py](../../marl/algorithms/mappo/agent.py)): the same `ActorPolicy` as IPPO, plus the `CentralCritic` fed the agent-specific global state from §9.3. That state has 7 shared privileged channels, the agent's own position and belief-mask channels, a 4-dim global vector, and the agent's own local observation. The actor is built first, so for a given seed IPPO and MAPPO start with bitwise-identical actors.
- **Global-state data path through the trainer.** A single `PPOAgent.uses_global_state` flag switches it on, so IPPO pays nothing for it.
  - `SyncVectorSwarm` can optionally emit the agent-specific channels, in both the live batch and the pre-reset `FinalInfo`.
  - The rollout buffer stores shared channels once per `(t,e)` and agent channels per `(t,e,n)` as uint8 (§16.5).
  - One function, `build_agent_specific_state`, builds the 9-channel critic grid. It is used unchanged at rollout time, at the truncation bootstrap, and at update time.
- **Statistical leak probe** ([leak_probe.py](../../marl/analysis/leak_probe.py), §12.4). It measures paired, pre-discovery bias toward the real target against a phantom target drawn under the same placement rules, using a one-sided z-test plus a materiality threshold.
  - Calibrated before use. It stays quiet on Random (−0.004) and the heuristic (−0.008).
  - It fires on an oracle cheater (+0.265, z=10.8) and on a cheater that uses hidden-target information only 30% of the time (+0.046, p=6e-4).
- **Evaluation-isolation guard (§20.3) wired into training.** The vector env rejects and redraws any training map whose content hash matches a frozen validation or test map. The run manifest records the guarded hash count and the number of rejections.
- **Active-sample masking (§16.6)** in the policy loss, value loss, entropy mean and ValueNorm update. `masked_mean` has an exact fast path, so it is bitwise-identical to before whenever every sample is active, which is always true in v1.
- **Map-generator bug fix** (`MAPGEN_VERSION = 2`). See PROBLEMS FOUND.
- Scripts:
  - `scripts/run_l1_gate.py`: resumable, works for either algorithm.
  - `scripts/audit_eval_isolation.py`: replays a run's map stream against the frozen eval hashes.
  - `scripts/plot_l1_gate.py`: draws the preliminary figure.
  - `scripts/make_eval_maps.py` now refuses to overwrite frozen sets unless given `--force`.

## FILES CREATED/CHANGED
- New:
  - `marl/algorithms/mappo/agent.py`
  - `marl/analysis/leak_probe.py`
  - `scripts/{run_l1_gate,audit_eval_isolation,plot_l1_gate}.py`
  - `docs/figures/phase5_l1_preliminary.png`
- Changed:
  - `marl/envs/vector.py`: agent-state channels and the eval-hash guard.
  - `marl/buffers/rollout_buffer.py`: global-state storage and assembly.
  - `marl/algorithms/common/agent.py`
  - `marl/algorithms/common/ppo_update.py`: masking.
  - `marl/training/trainer.py`: critic inputs and the guard.
  - `marl/envs/core/mapgen.py`: `_hash64` fix, `bfs_distances`, version stamp.
  - `scripts/make_eval_maps.py`
  - `docs/EXPERIMENTS.md`: new section on the mapgen bug and eval contamination.
- Tests:
  - New: `tests/marl/test_mappo.py`, `tests/marl/test_mappo_smoke.py`, `tests/analysis/test_leak_probe.py`.
  - Extended: `tests/env/{test_wrappers,test_mapgen}.py`, `tests/marl/{test_rollout_buffer,test_ppo_update,test_ippo_smoke}.py`.

## TESTS RUN
- `ruff check`: clean. `mypy`: clean (72 source files).
- `pytest`: **274 passed, 0 failed.** That is 264 fast tests in ~26 s plus 10 slow tests in ~4.5 min. Coverage is **97%**.
- **Bitwise parity test.** Running the MAPPO data path (global state stored, assembled and passed through every stage) with a local critic reproduces IPPO's parameters byte for byte after several updates, including truncation bootstraps. The negative control, MAPPO with its real central critic, diverges, which shows the test can fail.
- **MAPPO bitwise resume**: 2 updates, save, resume, 2 more updates gives the same result as 4 straight updates.
- **Full-scale L1 gate, run for real**: IPPO and MAPPO, 3 seeds each, 300k env steps, identical configs and seeds. Study `l1_gate_v2`, re-run after the mapgen fix.
- **Eval-isolation audit** of every run's map stream against the frozen hashes (see PROBLEMS FOUND).
- **Leak probe** on all 6 trained actors at high power: 5 episodes per validation map, about 1,600 to 1,770 pairs each.

## RESULTS
Final checkpoint, 50 frozen L1 validation maps, stochastic policy:

| | Seed 0 | Seed 1 | Seed 2 | Untrained (0% ckpt) |
|---|---|---|---|---|
| IPPO success / censored steps | 1.00 / 8.22 | 1.00 / 7.54 | 1.00 / 8.78 | 0.60 / 0.64 / 0.60 |
| MAPPO success / censored steps | 1.00 / 8.40 | 1.00 / 7.44 | 1.00 / 8.58 | 0.60 / 0.64 / 0.60 |

- **Gate** (success ≥ 0.85, censored steps ≤ 50.0, at least 2 of 3 seeds): IPPO 3/3, MAPPO 3/3. Random reference: 0.58 / 71.48.
- The untrained checkpoints are identical per seed across the two algorithms, as expected from identical actor initialisation.
- **Critic fit.** Mean explained variance over the last 10 updates was about 0.74 for IPPO and 0.79–0.83 for MAPPO. No NaNs in any log. Approx-KL and entropy behaved normally.
- **Leak probe.** All six diffs were ≤ +0.001, 99% upper bounds were ≤ +0.035, and none were flagged. IPPO leans slightly negative (−0.02 to −0.03). That is the opposite direction to a leak, and plausibly because the policy explores away from cells it has already seen.
- **Parameter counts.** The actors are identical at 732,005 parameters. The critics are 730,977 (IPPO) vs **1,183,777 (MAPPO), 1.62×**.
- **Preliminary learning curves** are in `docs/figures/phase5_l1_preliminary.png`. At about 20k env steps the MAPPO mean was 0.90 vs IPPO 0.78. Both saturate at 1.00 well before 300k. **This is not a result.** With n=3 on a level both algorithms solve, no comparative claim is made; the study is Phase 7 on L2.
- Throughput: about 140 env-steps/s for IPPO and 110 for MAPPO, single-threaded on CPU.

## PROBLEMS FOUND
1. **Map-generator precision bug, which contaminated evaluation (serious).**
   - In `_hash64`, `np.uint64 % int` was promoted to float64 under NumPy 1.x, which zeroed the low 11 bits of every per-episode map seed. Only about 2^20 distinct maps per level could be generated.
   - The new eval-isolation audit found that **8 frozen eval maps (4 validation, 4 test) were drawn as training maps** in every v1 IPPO (Phase 4) and MAPPO run. An exact replay of the generator state confirmed it.
   - Full impact table: `docs/EXPERIMENTS.md`.
2. **The §20.3 training-side eval guard was never wired in.** Phase 3 had deferred it, which is why problem 1 went unnoticed.
3. **Inactive samples were not masked** in the loss means or the ValueNorm update (§16.6, checklist item 7). This had no effect in v1 because every agent is active every step, but it would have been silently wrong later.
4. A Phase 4 buffer comment said `value_old` was normalized. It is denormalized, and the code was already correct.
5. **Leak-probe threshold.** A first `min_diff=0.05` missed the 30% cheater (+0.046).
6. **Smoke-test sizing.**
   - The first MAPPO smoke test (25 updates × 8 envs, judged on 12 maps) tied its untrained baseline, 0.58 vs 0.58.
   - The IPPO smoke test with that same sizing began failing (0.50 vs 0.58) once mapgen v2 changed which maps it drew. It had passed in Phase 4 only because of that particular map stream.

## PROBLEMS FIXED
1. **Mapgen bug.**
   - `_hash64` now does the modulo on a Python int. There is a full-width regression test, and `MAPGEN_VERSION = 2` is stamped into the map metadata.
   - The frozen eval sets were **not** regenerated. They stay exactly as committed, v1 data identified by content hash.
   - All six gate runs were re-run cleanly as `l1_gate_v2`, with 0 eval maps rejected or drawn. The v1 runs stay on disk, marked superseded. For reference, v1 MAPPO was 1.00 / 1.00 / 0.96 success, so the contamination did not change the verdict. That is luck, not validation.
2. **Eval guard.** Training now rejects any frozen eval hash, backed by tests with a forced collision and a no-op check.
3. **Masking.** `masked_mean` is threaded through all three losses and ValueNorm. Tests check exactness, zero gradient from inactive samples, and exclusion from the value loss.
4. The comment is corrected.
5. **Leak-probe threshold** set to 0.03. That is still about 3× the leak-free noise of ±0.01, and it now flags the 30% cheater.
6. **Both smoke tests resized from the measured clean curves** to 16 envs × 64 steps × 20 updates (MAPPO) or 30 updates (IPPO), judged on 25 maps. The reasoning is in each docstring.

## REMAINING CONCERNS
- **Capacity confound.** MAPPO's critic has 1.62× the parameters of IPPO's. Any MAPPO advantage in Phase 7 could come partly from critic capacity rather than centralized information. Phase 7 should report it, and ideally add a width-matched IPPO-critic ablation.
- **The frozen eval sets are v1 data.** Their recorded seeds no longer regenerate them under mapgen v2. They are loaded from JSON and identified by hash, which is correct, but they must never be regenerated. `make_eval_maps.py` now refuses to overwrite them without `--force`.
- **L1 saturates**, so it cannot tell the algorithms apart. Separating them needs L2 or higher (Phase 7).
- The spec's full 1M-step budget was not run. 300k was enough for the gate, and the L2 budgets belong to Phase 6/7 planning.
- **The spec's §30.3 teal `#0E7490` fails the dataviz chroma check.** The Phase 5 figure uses validated colours instead. Flag this for the Phase 9 UI palette.
- IPPO's slight negative probe bias is not a leak, but it is worth re-checking on L2.
- CPU-only throughput of about 110–140 env-steps/s will make Phase 7's seed × level grid expensive. Phase 6 should plan the parallelism.

## §16.8 COMMON-MISTAKES CHECKLIST (audited item by item)

| # | Mistake | Status | Evidence |
|---|---|---|---|
| 1 | Global state (or anything derived from it) reaching the actor, incl. via the action mask | ✅ Clear | `ActorPolicy.forward(grid, vec, mask)` has no other input; `LearnedActorPolicy` only calls `world.all_actor_obs()`; AST import-boundary test (actor.py / policy.py never import `global_state`/`critics`); Phase 2 counterfactual tests; new `test_mappo_critic_sees_hidden_target_but_actor_does_not`; behavioural leak probe negative on all 6 trained actors |
| 2 | GAE after flattening agents/envs together | ✅ Clear | `compute_gae` on `[T,E,N]`; `flatten_batch` only after; `test_gae_no_leakage_across_agents/envs` |
| 3 | Bootstrapping from the auto-reset observation | ✅ Clear | Truncation bootstrap reads `FinalInfo` (captured pre-reset) for the actor obs AND the MAPPO global state + agent channels; `test_vector_env_final_info_captures_pre_reset_agent_state`; parity test runs with `max_steps=8` so bootstraps happen in every update |
| 4 | Truncation vs termination confusion | ✅ Clear | Fold only when `truncated and not terminated`; env makes them mutually exclusive (success precedence, §6.4); hand-computed GAE truncation test |
| 5 | Action mask not stored/re-applied at update | ✅ Clear | Buffer stores `mask`; update calls `agent.evaluate(..., flat.mask[idx])` |
| 6 | Recomputing `logp_old` after the policy changed | ✅ Clear | Stored at collection time only |
| 7 | Inactive/padded samples in advantage normalization or loss means | ⚠️→✅ **Fixed this phase** | Advantage normalization was masked, but loss means and the ValueNorm update were not (spec §16.6 requires both). Added `masked_mean` with an exact fast path when all samples are active (bitwise identical in v1) + tests with an artificial mask |
| 8 | Mixing normalized and denormalized values | ✅ Clear | Rollout values, truncation bootstrap and `last_value` all denormalized; GAE on denormalized; ValueNorm `update` then `normalize` targets; critic loss compares normalized to normalized. (Fixed a misleading Phase 4 buffer comment that said `value_old` was normalized — it's denormalized.) |
| 9 | `-inf` logits → NaN entropy | ✅ Clear | `-1e9` fill; `test_entropy_is_finite_not_nan`; zero NaNs in all training logs |
| 10 | Concatenating all agents' observations as critic input | ✅ Clear | Agent-specific state (shared channels + own 2 channels + own obs), size-agnostic in N (shape test for N = 2, 3, 4) |
| 11 | Agent-specific input omitting "which agent am I" | ✅ Clear | Channel 7 = own position; `test_agent_specific_channels_identify_each_robot`; buffer per-sample assembly test |
| 12 | Different budgets / seeds / maps / hyperparameters between IPPO and MAPPO | ✅ Clear | `ippo.yaml` vs `mappo.yaml` differ only in `name` and `critic.input`; same run seeds → same training-map streams; identical gate configs; actors bitwise-identical at init; parity test |
| 13 | Counting agent-steps as env-steps | ✅ Clear | Budgets in env steps; manifest records both separately |
| 14 | Greedy evaluation of symmetric shared policies | ✅ Clear | Validation is stochastic by default |
| 15 | Evaluating on training maps / selecting on test | ⚠️→✅ **Fixed this phase** | Validation uses the frozen validation set and nothing reads the test set — but the §20.3 guard that stops TRAINING from ever drawing a frozen eval map was never wired in (Phase 3 deferred it). Wired into the vector env; retroactive replay audit of every earlier run's map stream: see results |
| 16 | Shared reward summed in one place, averaged in another | ✅ Clear | Every agent receives the full team reward; GAE per agent; logged `mean_reward` is a plain mean |

## VALIDATION GATE
**PASS.**
- MAPPO trains to the L1 gate on 3/3 seeds, with identical configs, seeds and budget to IPPO.
- Bitwise IPPO-parity and resume tests pass.
- The actor provably never sees global state: structural tests, a counterfactual test, and a behavioural leak probe that is negative on all six actors.
- The §16.8 checklist is fully audited: 14 items were already clear and 2 were fixed.
- ruff, mypy and all 274 tests are green.

## CURRENT PROJECT STATUS
Phases 0–5 are complete. Phases 0–4 are committed; Phase 5 is not committed yet. Random, the heuristic, IPPO and MAPPO all run on the same env, evaluator and training loop, with eval isolation enforced.

## NEXT PHASE
Phase 6: experiment infrastructure. That means sweep configs, a multi-seed/multi-level runner with parallelism, aggregation, and the pre-registered statistics pipeline.

## RECOMMENDED MODEL
Sonnet 5

## RECOMMENDED EFFORT
Medium

## WHY
Phase 6 is orchestration and bookkeeping: run matrices, process pools, result aggregation and statistics plumbing. It needs no new RL mathematics, and the correctness-critical pieces already exist and are tested.

## USE OPUS ONLY IF
The statistics design, meaning the pre-registered tests, multiple-comparison handling or the capacity-matched ablation design, turns out to be ambiguous. Or if any Phase 6 run hits an unexplained divergence between IPPO and MAPPO data paths.

## RECOMMENDED SKILLS / CONNECTORS
None required. The GitHub connector still needs authorization if you want pushes done from here.

## ACTION REQUIRED
Reply **CONTINUE** to start Phase 6. Say whether you want Phase 5 committed first.

## STOP HERE.
