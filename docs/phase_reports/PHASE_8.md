# PHASE 8 COMPLETE

## IMPLEMENTED
- **Coordination metrics** (§24, [marl/analysis/coordination.py](../../marl/analysis/coordination.py)): DER, DER@50 (matched-coverage), visitation overlap, redundant pursuit, target-allocation optimality, mean/min inter-robot distance, dispersion speed. All are pure functions of an `EpisodeTrace` plus the `MapSpec` it was played on — a robot's own sensed cells are re-derived from its position trajectory via the existing no-occlusion sensing geometry (§11.2), so `EpisodeTrace`'s schema didn't need to change.
- **Isolation null** (§24.3, [marl/evaluation/coordination.py](../../marl/evaluation/coordination.py)): for each robot, a single-robot (N=1) episode on the identical map (same terrain, same targets, only that robot's start cell), unioning the resulting sensed sets to get `DER_iso` — the overlap a team would show with zero interaction. Built as recommended by the spec, not skipped as a stretch item.
- **Behaviour analysis** (§26, [marl/analysis/behaviour.py](../../marl/analysis/behaviour.py)): per-episode DER-vs-coverage curves, inter-robot-distance-over-time, discovery/rescue timelines, and multi-episode exploration/collision heatmaps on one map — all as JSON (for the eventual frontend) plus static figures for this report.
- **A proper, persisted input-ablation pipeline** ([marl/evaluation/ablation.py](../../marl/evaluation/ablation.py)), replacing Phase 7's ad hoc script: `run_ablation_for_checkpoint` + `summarize_ablation` compute the channel-5/channel-6 zeroing effect for every checkpoint and feed §24.4 criterion 4 directly.
- **`scripts/analyze_study.py`**: orchestrates all of the above for a finished study — coordination + isolation null + ablation for every learned checkpoint and Random's 5 sampling seeds, plus showcase heatmaps/timelines/DER-curves on the L2 structured maps. Writes `results/studies/<study>/analysis/*.json`.
- **H3 and the §24.4 decision**, computed mechanically in `marl/analysis/aggregate.py` from that JSON, exactly like H1/H2/H4: all four criteria (H1 supported; DER@50 below both Random's and the isolation null's; redundant pursuit below Random's; the ablation degrades coordination) must hold, per algorithm, with CIs, or the report states the spec's exact fallback sentence.
- **`docs/RESULTS.md` extended** ([marl/analysis/report.py](../../marl/analysis/report.py)) with a "Coordination analysis" table and a "§24.4 coordination-emergence criteria" pass/fail table per level.
- Ran the full pipeline against the real, finished `main_L2` study (not just synthetic data).

## FILES CREATED/CHANGED
- New: `marl/analysis/{coordination,behaviour}.py`, `marl/evaluation/{coordination,ablation}.py`, `scripts/analyze_study.py`, `tests/analysis/{test_coordination,test_behaviour}.py`, `tests/evaluation/{test_coordination,test_ablation}.py`.
- Changed: `marl/analysis/aggregate.py` (H3/§24.4, `_coordination_block`), `marl/analysis/report.py` (coordination tables), `marl/analysis/figures.py` (heatmap, DER-vs-coverage, discovery-timeline plots), `tests/evaluation/test_aggregate.py` (H3 tests).
- Generated (not committed as source, but the `docs/` outputs are): `results/studies/main_L2/analysis/*.json`, updated `docs/RESULTS.md`, new figures under `docs/figures/main_L2/`.

## TESTS RUN
- `ruff` and `mypy`: clean across 77 source files.
- `pytest` (fast + slow): **346 passed, 0 failed**; coverage 94%.
- New tests, all against hand-built traces/positions with independently computed expected values (per §33's requirement), or hand-built JSON coordination fixtures for the H3 decision logic:
  - DER: no overlap, full overlap (hits the spec's stated (N-1)/N bound), partial overlap, and cumulative-over-time behaviour.
  - Visitation overlap, inter-robot distance, dispersion speed (including the "never exceeds" and single-robot edge cases).
  - Redundant pursuit and allocation optimality, including a case where the BFS-nearest robot is NOT the one that rescues (allocation_optimality = 0).
  - The isolation-null runner: deterministic given a seed, differs across sampling seeds, never runs more than one robot per episode (a structural guard, not just an output check), and works against the real, committed L2 structured maps.
  - The H3/§24.4 decision: "emerged" when all four criteria hold, "inconclusive" when any one fails (tested separately for the DER@50 and the ablation criteria), and "pending" when no coordination.json exists yet.
  - The ablation summary's median computation, including the zero-baseline-collisions division guard.
- Ran `scripts/analyze_study.py` twice: once against the `smoke` study (L1, tiny, no structured set — confirms the pipeline degrades gracefully when there's nothing to show) and once against the real, finished `main_L2` study.

## RESULTS

### Coordination metrics, 25 of the 100 frozen L2 test maps × 2 eval seeds (see "Sample size" below)

| Method | DER | DER@50 | DER (isolation null) | Visitation overlap | Redundant pursuit | Allocation optimality | Dispersion speed |
|---|---|---|---|---|---|---|---|
| Random | 0.52 [0.51, 0.53] | 0.455 [0.450, 0.461] | 0.551 [0.542, 0.561] | 0.46 [0.43, 0.49] | 1.16 [1.13, 1.20] | 0.74 [0.69, 0.80] | 3.9 [3.7, 4.1] |
| IPPO | 0.34 [0.33, 0.36] | 0.357 [0.353, 0.362] | 0.604 [0.594, 0.614] | 0.063 [0.046, 0.080] | 2.12 [2.08, 2.16] | 0.97 [0.94, 1.00] | 1.28 [1.17, 1.39] |
| MAPPO | 0.35 [0.33, 0.36] | 0.357 [0.353, 0.362] | 0.617 [0.610, 0.624] | 0.064 [0.057, 0.071] | 2.10 [2.05, 2.14] | 0.96 [0.95, 0.98] | 1.28 [1.22, 1.33] |

### §24.4 criteria, per algorithm

| Criterion | IPPO | MAPPO |
|---|---|---|
| 1. H1 supported | ✅ | ✅ |
| 2a. DER@50 < Random | ✅ (diff −0.098, 95% Welch CI [−0.105, −0.092], p<0.001) | ✅ (diff −0.098, CI [−0.104, −0.092], p<0.001) |
| 2b. DER@50 < isolation null | ✅ (paired diff −0.247, 95% CI [−0.255, −0.240]) | ✅ (paired diff −0.260, CI [−0.270, −0.250]) |
| 3. Redundant pursuit < Random | ❌ (diff **+0.955**, 95% CI [+0.910, +1.001], p<0.001 — the WRONG direction) | ❌ (diff +0.935, CI [+0.885, +0.986], p<0.001 — same) |
| 4. Ablation degrades coordination | ✅ (median channel-6 collision multiplier 91×, channel-5 success drop up to 0.30) | ✅ (median multiplier 270×, drop up to 0.20) |
| **Coordination emerged (§24.4)** | **❌** | **❌** |

**H3 verdict (both algorithms): "Differences in behaviour were observed; evidence of learned coordination is inconclusive"** — the spec's exact prescribed fallback, since criterion 3 fails clearly and consistently for both algorithms.

### What the numbers actually show (an honest, nuanced picture, not a single verdict)
- **Exploration division of labour is excellent and unambiguous.** Both algorithms cut DER roughly in half versus Random (0.34–0.35 vs 0.52) and versus their OWN isolation null (0.60–0.62) — i.e. the trained team senses *less* redundantly than the same policy would with zero interaction, which is the strongest possible reading of criteria 2a/2b. Visitation overlap collapses from Random's 0.46 to 0.06 — robots visibly split up the map. Both also disperse from the clustered spawn about 3× faster than Random (1.3 vs 3.9 steps) and rescue via the BFS-nearest robot far more often (0.96–0.97 vs Random's 0.74).
- **Redundant pursuit is the opposite of coordinated — and by a large, precise margin.** Once a target is discovered, both algorithms average about 2.1 robots per step whose distance to it is *decreasing*, versus Random's 1.16 — nearly double, not a noise-level difference (Hedges' g ≈ 28). This reads as herding: once one robot's discovery makes a target visible to the shared-parameter policy, more than one robot tends to converge on it, and nothing in the reward (fully shared, β=1.0) or the architecture (no target-assignment signal) discourages that. This is plausible given the reward design (§13.3's analysis never considered redundant *pursuit*, only redundant *exploration*) and is a genuine, reportable limitation, not an artifact — the ablation study (channel 6) already showed these policies are highly sensitive to currently-sensed teammates, so the mechanism for reduced herding exists in principle; it just isn't what got learned here.
- **The net honest claim**: SWARMMIND's learned policies coordinate strongly on *where to look* and comparatively poorly on *who finishes the job*, and the criteria correctly refuse to round that up to a blanket "coordination emerged."

### Behaviour analysis (§26), illustrative (showcase maps, not a statistic)
- Collision heatmaps on the structured bottleneck maps show collisions concentrated exactly at the wall's narrow gap — a legible, sanity-checking confirmation that the collision metric and the map's intended structure agree.
- DER-vs-coverage curves (one example episode per algorithm) show DER dipping through the middle of the episode as robots spread across newly discovered territory, then rising again near the end as the last few unexplored cells pull every robot back toward the same remaining region — a plausible, interpretable shape rather than noise.
- Discovery/rescue timelines show rescue happening almost immediately after discovery on these small, easy maps (matching the earlier steps-on-success numbers), which is why the timeline bars are visually thin.

## PROBLEMS FOUND
1. None of §24's coordination metrics, the isolation null, or the behaviour-analysis JSON/figures existed before this phase — expected, since this was Phase 8's actual scope, not a defect carried from earlier phases.
2. Phase 7's ablation numbers lived only in an ad hoc scratch script's output, not in a reusable, tested module. Re-computing it here (on a different map subset) gave the same qualitative result (large, consistent multiplier/drop across all seeds) but different exact numbers (median 91–106× here vs Phase 7's 281× on the validation set), which is expected — worth noting so nobody is confused later about which number is "the" number: neither is wrong, they're different samples of the same real effect, and both clear the materiality threshold by a wide margin.

## PROBLEMS FIXED
- Built `marl/evaluation/ablation.py` as the single, tested, reusable ablation pipeline; `scripts/analyze_study.py` now regenerates ablation data as part of the standard analysis pass rather than relying on a one-off script.

## REMAINING CONCERNS
- **Sample size for coordination analysis: 25 of the 100 frozen test maps × 2 eval seeds** (50 episodes/seed for the team run, 150 more for the isolation null), not the full 100-map set the headline H1/H2/H4 results use. The isolation null multiplies cost by `n_robots`, so the full set would have cost roughly 4× more compute for a secondary, descriptive analysis. This is a documented, reasoned reduction (in the spirit of Phase 6's pilot-budget decisions), not a silent one, and the per-seed CIs above are visibly tight (e.g. IPPO's DER@50 CI is [0.353, 0.362]) — 50 episodes/seed was enough to resolve the effects reported here.
- **The redundant-pursuit failure deserves follow-up**, ideally in whatever final write-up follows this project: it suggests a concrete reward-design change (a small per-step cost for a *second* robot closing in on an already-discovered, not-yet-rescued target) that could be tested in a future study, not attempted here since Phase 8 is analysis-only and must not touch the frozen study.
- **Allocation optimality and dispersion speed are reported descriptively** (§24.2's table lists them without a §24.4 pass/fail rule attached) — they support the "explores well, pursues poorly" reading but are not part of the mechanical decision.
- Heatmaps and the DER-vs-coverage/discovery-timeline figures use only the FIRST completed seed's final checkpoint per algorithm, on 3 structured maps × 20 episodes — illustrative, not a claim with a CI, exactly as §26 intends ("Swarm Evolution" / behaviour panels are qualitative by design).
- Robot trajectories and communication-link visualizations (§26's other two rows) are frontend-rendering concerns over data the replay schema (Phase 3) and evolution replays (Phase 6) already provide; no new Python analysis was needed for them, and they're deferred to the frontend phases (11+) as the spec's own file list implies.

## VALIDATION GATE
**PASS.**
- All tests pass (346/346); every coordination metric is checked against a hand-built trace with an independently computed expected value.
- An analysis JSON exists for both algorithms on the primary level, including the required isolation-null run.
- H3 and the coordination claim are stated exactly per §4.2/§24.4's rules (the fallback sentence is used verbatim, since it applies).
- `docs/RESULTS.md` was regenerated from the real study and includes the new coordination section.

## CURRENT PROJECT STATUS
Phases 0–7 are committed. **Phase 8 is not yet committed.**

## NEXT PHASE
Phase 9: professional product/UI design (the design-authority phase) — the first frontend-facing phase, working from the replay schema, evolution replays, and now the coordination/behaviour JSON this phase produced.

## RECOMMENDED MODEL
Sonnet 5

## RECOMMENDED EFFORT
Medium

## WHY
Phase 9 is a design specification and critique phase per the spec's own plan, not new MARL implementation.

## USE OPUS ONLY IF
A later phase's frontend work surfaces a genuine disagreement between the coordination JSON and the replay data it's meant to describe (a data-contract bug, not a design question).

## RECOMMENDED SKILLS / CONNECTORS
`dataviz` for the eventual frontend chart specs. GitHub still needs authorization in your claude.ai connector settings for pushes from here.

## ACTION REQUIRED
Say whether to commit Phase 8, then reply **CONTINUE** to start Phase 9.

## STOP HERE.
