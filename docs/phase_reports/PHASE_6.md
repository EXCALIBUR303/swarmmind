# PHASE 6 COMPLETE (infrastructure) — pre-registration awaiting approval

## IMPLEMENTED
- **Study runner** (`marl/experiments/{config,jobs,workers,runner}.py`, `scripts/run_study.py`).
  - It expands levels × algos × seeds into jobs and runs them in a `spawn` process pool, with one torch thread per process.
  - It is idempotent. A completed run is skipped (and only if its budget matches), an interrupted or failed run is resumed from `latest.pt`, and a run another live process holds (checked via a `.lock` pid) is left alone.
  - A crashed worker process is recorded as a failed job and doesn't abort the study.
  - SIGTERM or Ctrl-C leaves a resumable run marked `interrupted`.
  - A `study_manifest.json` records every job, its config hash, the sha256 of every input file, the git state and a code fingerprint.
- **Pre-registration freezing.** `--freeze` writes the sha256 of the study YAML, both algo YAMLs, the level YAML and the maps manifest to `configs/experiments/FROZEN.json`. A frozen study whose inputs changed refuses to run, unless `--allow-deviation` is given, in which case the change is recorded as a deviation.
- **Final held-out evaluation** (`marl/evaluation/final_eval.py`, §22.1).
  - Every map is played with evaluation seeds {0, 1}. Stochastic is primary; greedy is secondary and stored separately.
  - Common random numbers: the sampling seed is derived from sha256 of the map and evaluation seed.
  - Random is evaluated with 5 sampling seeds, plus the heuristic as a reference.
  - A gatekeeper only releases held-out maps for runs whose manifest says `completed`, and appends every access to `results/heldout_access_log.jsonl`.
- **Statistics** (`marl/evaluation/stats.py`): t-intervals, Welch difference CIs with Welch's p, Hedges' g, normalized AUC, and interpolation onto a common grid. All are checked against scipy.
- **Aggregation, figures and report** (`marl/analysis/{aggregate,figures,report}.py`, `scripts/aggregate_results.py`).
  - `summary.json` holds per-seed values, intervals, pairwise comparisons, learning curves with 95% CI bands, validation AUC, and H1/H2/H4 decisions whose wording is generated from the rules. H3 is marked pending until Phase 8.
  - The figures are a validation learning-curve plot and per-seed dot plots.
  - `docs/RESULTS.md` is generated, with a SMOKE banner on smoke studies.
- **Swarm Evolution replays** (`scripts/record_replays.py`): the same maps and evaluation seed at the 0/10/50/100% checkpoints.
- **Structured G2 maps.** These were a Phase 3 deliverable that never happened.
  - There are 12 handmade 10×10 L2 maps (rooms ×3, corridors ×2, open ×2, maze ×3, bottleneck ×2), frozen with hashes into `configs/maps/structured/L2.json`.
  - They pass a new `validate_map_spec`, which applies the procedural generator's guarantees.
  - They are included in the training isolation guard.
  - The heuristic solves them all (1.00); Random succeeds in 0.10 of episodes.
- **File-level immutability** of every frozen map set, via `configs/maps/manifest.json` and a test.
- **Study configs**: `smoke`, `main_L2`, `scale_L3` and `comm_L4`.
- **Pre-registration draft** in `docs/EXPERIMENTS.md`, backed by a validation-only pilot on seed 1000 and a set of explicit decisions (D1–D4).
- **Trainer additions:**
  - A validation point at env step 0.
  - The budget and the steps actually trained, both recorded in the manifest.
  - A resume history.
  - An `interrupted` status.

## FILES CREATED/CHANGED
- **New:**
  - `marl/experiments/{__init__,config,jobs,workers,runner,outputs}.py`
  - `marl/evaluation/{stats,final_eval}.py`
  - `marl/analysis/{aggregate,figures,report}.py`
  - `scripts/{run_study,aggregate_results,record_replays,summarize_pilot}.py`
  - `configs/experiments/{main_L2,scale_L3,comm_L4}.yaml` (and `smoke.yaml` rewritten)
  - `configs/maps/structured/L2/*.txt` (12 maps), `configs/maps/structured/L2.json`, `configs/maps/manifest.json`
  - `tests/evaluation/{test_stats,test_aggregate}.py`, `tests/experiments/test_runner.py`
  - `docs/RESULTS.md` (smoke), `docs/figures/smoke/*.png`
- **Changed:**
  - `marl/training/trainer.py`: step-0 validation, fraction fix, interrupted status, manifest fields.
  - `marl/envs/core/mapgen.py`: `validate_map_spec`.
  - `marl/evaluation/eval_sets.py`: structured split, manifest.
  - `scripts/{train,evaluate,make_eval_maps}.py`
  - `tests/{test_import_boundaries.py, evaluation/test_eval_sets.py, marl/test_checkpoint.py}`
  - `docs/EXPERIMENTS.md`, `Makefile`, `.gitignore`

## TESTS RUN
- `ruff` and `mypy` (73 files): both clean.
- `pytest` (fast + slow): **305 passed, 0 failed**. Coverage is **95%**; the new modules are at 82–96%.
- New tests cover:
  - the statistics against scipy;
  - aggregation on a synthetic study, where every summary number can be recomputed from its inputs;
  - null results worded as "no detectable difference";
  - missing runs being reported;
  - the H2 metric and AUC window following the config;
  - an end-to-end tiny study that is idempotent on re-run and has correct manifest hashes;
  - an interrupted run being resumed rather than restarted;
  - a budget mismatch being refused;
  - a frozen input change being detected;
  - the held-out gate and its access log;
  - common-random-number seeding;
  - evolution replays re-simulating exactly;
  - SIGTERM leading to `interrupted`;
  - the structured maps' validity and immutability;
  - the final checkpoint being present when the budget isn't a multiple of the update size.
- **Smoke study, end to end and unattended** (`scripts/run_study.py --study smoke --parallel 4`):
  - 4 training runs, 4 held-out evaluations, 6 baseline evaluations, the aggregate, 3 figures and 16 evolution replays;
  - 14/14 jobs completed with 0 manual steps;
  - it was re-run from scratch on the final code (see RESULTS).
- A spawn-pool check ran 4 parallel runs plus evaluations and baselines.
- Pilots used validation maps only, on seed 1000: L2 with IPPO and MAPPO (471k / 334k steps); L3 and L4 with IPPO only (266k steps each).

## RESULTS
- **The smoke gate passes.** `results/studies/smoke/summary.json`, `docs/RESULTS.md` (marked SMOKE) and the figures were all generated with no manual steps. Two re-runs from scratch reproduced every aggregate number exactly.
- **The pilot changes the picture for the main study** (full tables in `docs/EXPERIMENTS.md` §4):
  - On L2, IPPO and MAPPO both reach **1.00 validation success at 82k env steps**, under 3% of the spec's 3M budget.
  - Censored steps settle at about 11–12 by roughly 125k. The heuristic takes 14.7 and Random succeeds 0.54 of the time.
  - L3 also saturates (1.00 by 123k). Only L4 (no communication) is still learning at 250k.
- **Measured compute.**
  - Per run: IPPO about 79 env-steps/s and MAPPO about 56, with ~8 processes running together.
  - The time goes into the CNN's forward and backward passes; environment stepping is about 6%.
  - More torch threads make it *slower* (103 → 95 → 81 env-steps/s for 1/2/4 threads).
  - Main study wall time is about 5.5 h per 1M env steps of budget; roughly 17 h at the spec's 3M.

## PROBLEMS FOUND
1. **The final checkpoint was never saved** whenever the budget wasn't a multiple of 16 × 128 (which includes the spec's 3M). Checkpoint fractions were measured against the nominal budget rather than the steps actually trained, so fraction 1.0 was never reached. The smoke study's evolution replays exposed it by having no 100% frame.
2. **The structured G2 maps never existed.** They were a Phase 3 task, and H4 and the `structured` evaluation depend on them.
3. **A SIGTERM'd or crashed run left a stale `running` manifest.** Background processes ignore SIGINT, so the trainer's cleanup never ran. Also, the step-0 checkpoint and evaluation sat outside the trainer's `try` block, which is a race the new test caught as a flaky failure.
4. **A dead worker process would have aborted the whole study.** `fut.result()` raising wasn't handled.
5. **The L2 main-study design is at a ceiling.** Success saturates almost immediately, so H2 on success cannot distinguish the algorithms. A whole-budget AUC is ~0.99 for both, and 2/3 of a 3M budget would be spent on a plateau.
6. **The spec's §35 compute estimate was ~20× too optimistic.**
7. `runner.py` grew past the ~400-line guideline (§5.2).
8. Smaller fixes:
   - figure banners were clipped and a label overlapped the data;
   - `p=0.000` was printed instead of `p<0.001`;
   - the Phase 4 resume test slowed down once the step-0 evaluation was added.

## PROBLEMS FIXED
1. Checkpoint fractions are now measured against `trained_env_steps`, with a regression test. The smoke study was re-run, and its replays now include 100%.
2. The 12 structured maps were built, validated (by the generator's own rules plus the heuristic's 1.00), frozen, hash-guarded and tested.
3. A SIGTERM handler now turns SIGTERM into `SystemExit`, the trainer records `interrupted`, and the step-0 work moved inside the `try`. There is a subprocess test for it.
4. `drain()` records a dead worker as a failed job, so a re-run resumes it.
5. **Not fixed unilaterally.** Instead, `StudyConfig.analysis` (the H2 metric and the AUC window) makes the choice explicit and frozen, with defaults exactly as in the spec. Decisions D1–D4 are laid out with pilot evidence and a recommendation each.
6. Measured numbers were recorded in `docs/EXPERIMENTS.md` §5.
7. `runner.py` was split into `jobs.py`, `workers.py` and `runner.py`. The largest is now 235 lines.
8. The smaller issues were all fixed; the resume test now uses 3 validation maps and also shows that validation never perturbs training.

## REMAINING CONCERNS
- **The pre-registration is not frozen.** It needs your answers to D1–D4. Until then `main_L2.yaml` holds the spec defaults: 3M steps, success as H2's metric, whole-budget AUC.
- **Ceiling risk remains even with D1.** If the final policies of both algorithms converge to the same censored steps, H2 will honestly be "no detectable difference". L4 is the level most likely to separate them, and it is a stretch study.
- The pilot is one seed per cell. Its validation points carry about ±1 step of noise from 50-episode evaluations.
- **The learned L2 policies finish about 22% faster than the greedy heuristic** (11–12 vs 14.7 steps). That is plausible, since the heuristic isn't optimal, but Phase 7's sanity check ("no method implausibly faster than the heuristic's envelope") and the leak probe on L2 must confirm it.
- The frozen eval sets are still the v1 generator's data. That is fine, but they must never be regenerated.
- No multiplicity correction is applied across H1 (×2) and H2. This is stated in the pre-registration.
- The smoke study's L1 has no structured set, so the structured-split path of the real runner was exercised only by the tiny-study tests and by the L2 gatekeeper test, not by the smoke run.
- CPU only.

## VALIDATION GATE
**Partial pass: infrastructure PASS; freeze PENDING user approval.**
- **Passed:** the smoke study ran end to end, unattended, and produced `summary.json`, a SMOKE-marked `RESULTS.md` and figures. The statistics match scipy; aggregation, idempotent skip/resume and manifest integrity are all tested; all 305 tests are green.
- **Not yet done:** "EXPERIMENTS.md is frozen and committed (with user approval)". That requires your answers to D1–D4, then `--freeze`, then a commit.

## CURRENT PROJECT STATUS
- Phases 0–4 are committed.
- Phase 5 (MAPPO) and Phase 6 are complete but **uncommitted**.
- The study tooling is ready. Phase 7 is blocked only on the frozen pre-registration.

## NEXT PHASE
Phase 7: controlled experiments (Random vs IPPO vs MAPPO on the frozen `main_L2`). You launch it as a background terminal process, and progress is checked from the study manifest.

## RECOMMENDED MODEL
Sonnet 5

## RECOMMENDED EFFORT
Medium

## WHY
Phase 7 is orchestration and honest reporting on tooling that is already tested. The statistics and wording are mechanical and pre-registered.

## USE OPUS ONLY IF
A Phase 7 sanity check fails. That includes:
- the learned policies turning out implausibly faster than the heuristic on test;
- a positive L2 leak probe;
- MAPPO being worse than IPPO *and* having a lower critic explained variance;
- bimodal seeds.

## RECOMMENDED SKILLS / CONNECTORS
- `data:statistical-analysis` (a second look at the stats procedure) and `dataviz` (figures).
- GitHub still needs authorization if you want pushes done from here.

## ACTION REQUIRED
1. Answer D1–D4 in `docs/EXPERIMENTS.md` §6. The recommendations are: censored steps as H2's metric, a 1M budget, an AUC window of the first 245,760 steps, and L2 kept as primary.
2. Say whether to commit Phases 5 and 6.
3. Reply **CONTINUE**. I will then apply your answers, freeze and commit the pre-registration, and start Phase 7.

## STOP HERE.
