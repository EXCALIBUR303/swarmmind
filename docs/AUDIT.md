# SWARMMIND — Final MARL and system audit (Phase 15)

**Scope.** Spec §38 Phase 15: environment correctness, information boundaries and CTDE, IPPO/MAPPO against
§15/§16 (every §16.8 item), GAE/PPO maths, reward design and hacking evidence, evaluation fairness, seed
reproducibility, generalization claims, metric definitions against code, checkpoint integrity,
backend/model parity, frontend real-data use, test coverage, architecture boundaries and documentation
accuracy. No new features; fixes limited to defects.

**Method.** I read the code of each layer against the spec, and re-derived each number I could from a
more primary artifact, rather than trusting the test suite. Machine-checkable evidence is in two
generated reports:
- [`REAL_DATA_AUDIT.md`](REAL_DATA_AUDIT.md) (`make audit`): everything that can be checked from the
  committed demo bundle.
- [`AUDIT_CHECKS.md`](AUDIT_CHECKS.md) (`scripts/audit_checks.py`): checks that need the full `results/`
  tree with checkpoints. These are re-training, checkpoint integrity, replay re-recording and the
  independent statistics.

**Limitation of this audit.** It was done by the same assistant (a different model, Opus 5.5) that built
the system under the user's direction. That makes it an adversarial self-review, not an independent
external one. I did not run the `security-review` skill: it diffs against a git remote, and none exists.
The security posture was reviewed by hand and is covered by `tests/backend/test_security.py` and
`tests/integration`.

**Severity.**
- **Critical:** invalidates a headline result.
- **High:** a reported claim or criterion is unsupported, or a reported comparison is methodologically
  invalid.
- **Medium:** a defect in a reported secondary number, or a displayed number or label is wrong.
- **Low:** documentation, provenance or process gaps.
- **Info:** observations and spec tensions, no defect.

## Outcome

**No Critical findings. One High finding, fixed. No open Critical or High findings.** All hypothesis
verdicts are unchanged: H1 supported for both, H2 not supported, H3 inconclusive, H4 descriptive. After
the fixes the held-out evaluation was re-run under evaluation protocol v2 (no retraining). Across all
4,480 learned-run episodes, success, steps, coverage, return and collisions are identical to the
originals. Only the three fields the fixes target changed.

## Findings

| ID | Sev. | Area | Finding | Fix | Status |
|---|---|---|---|---|---|
| A-01 | **High** | Coordination (§24.3–24.4) | Criterion 2b ("DER@50 lower than the isolation null") compared the team's DER at 50% coverage with the isolated robots' DER over their **whole solo episodes**. Lone robots run far longer than a team, and §24.1 itself says DER is confounded by episode length. So the null was inflated and the criterion was biased towards passing. RESULTS.md reported 2b as ✅ on that basis. | `run_isolation_null` now also returns **DER_iso@50**: the solo trajectories are played on one shared clock and DER is taken at the first step where their combined sensing covers 50% of free cells, exactly how the team's DER@50 is measured. Criterion 2b pairs the two by seed. The whole-episode value is kept as a labelled descriptive column. Regression test added. | Fixed. **Result:** 2b still holds under the fair comparison (paired DER@50 − DER_iso@50: IPPO −0.162 [−0.168, −0.157], MAPPO −0.180 [−0.204, −0.156]). The H3 verdict is unchanged (inconclusive, because criterion 3 fails). |
| A-02 | Medium | Metrics, replay, UI | A target already sensed by the **reset-time** sensing pass appears in no step's `discovered` events. So it was missing from `discover_rate` and rescue latency, from the redundant-pursuit and allocation windows, and from the discovery timeline, and the UI drew it as hidden until rescued. Visible symptom: one stored run had discover rate 0.9875 < rescue rate 0.9975, which is impossible. A frontend comment claimed this "cannot occur because target_min_dist (3) > sensor_radius (2)". That is false: target_min_dist is a BFS distance and sensing is a Euclidean disc, so an obstacle in between allows it. | `EpisodeTrace.initial_discovered` is set by the single stepper and used by metrics, coordination and behaviour code. Both `initial` objects (`/replays/{id}/initial`, session snapshot) gain `discovered`, and the frontend reducer seeds those targets as DISCOVERED. Evaluation protocol bumped to v2. Regression tests in Python and Vitest. | Fixed. Scope: 1 of 100 validation targets, 2 of 200 test targets, 0 of 24 structured targets. Identical across methods (same maps), so no differential bias. Not used in any hypothesis. |
| A-03 | Medium | Metrics (§23), UI | The **workload share** did not follow §23. It split each step's exploration reward among robots that *moved*, including the terminal exploration payout, which is a reward and not exploration by anyone. The UI captioned it "share of total distance travelled", which is a third definition. The old metric made work look almost perfectly even (learned Gini ≈ 0.03). | The stepper now records per-step **observer credit**: each newly team-known free cell is credited to the robot(s) whose sensing revealed it, split among simultaneous observers. Workload share and Gini use it, and the caption is corrected. Regression tests check that credit sums to the cells actually revealed. | Fixed. Learned Gini on test is now 0.24 (IPPO) and 0.23 (MAPPO) vs Random 0.30. No conclusion relied on it; it is a secondary column on the Compare page. |
| A-04 | Low | Held-out isolation (§20.3) | `scripts/analyze_study.py` read the test and structured maps through `load_eval_set`, bypassing the gatekeeper, so those held-out accesses were not logged. | Goes through `load_heldout_maps`, which logs. | Fixed |
| A-05 | Low | Provenance | `leak_probe_L2.json` (Phase 7) had no generating script in the repository. | `scripts/run_leak_probe.py` reproduces it **identically** (`--check`) and adds a pooled estimate. | Fixed. Pooled over 10 runs, 11,013 pairs: bias −0.0005, SE 0.0054. No leak. |
| A-06 | Low | Reporting (§16.3, §16.7) | RESULTS.md did not report the parameter counts or the theoretical caveat on state-conditioned critics. Both are required by the spec. | Computed in the aggregator (`parameter_counts`) and rendered by the generator: actor 732,005; critics 730,977 vs 1,183,777 (1.62×). | Fixed |
| A-07 | Low | Claims (§14.2, §4.2) | The README did not say that the hand-written heuristic beats the learned policies on the structured maps (success 1.00 vs 0.82), as §14.2 requires. | README now states both directions (learned slightly faster on test, worse on structured). The audit registers those numbers. | Fixed |
| A-08 | Low | Single code path (§5.3) | `record_episode` duplicated the stepping and trace loop instead of using `EpisodeRunner`, so trace changes could diverge between replays and evaluation. | The recorder steps through `EpisodeRunner`. All 40 stored replays re-record identically (frames and outcomes). | Fixed |
| A-09 | Low | UI label | Oscillation rate was labelled "fraction of steps"; §23 defines it per moved step. | "fraction of moves that reverse the previous one" | Fixed |
| A-10 | Info | Reward (§13.2) | The exploration budget is defined over free cells (w_x = 1/\|F\|), not all in-bounds cells \|G\| as §13.2's text says. This is consistent with the coverage metric, and the budget invariant holds on 399 real successful episodes (max error 6.7e-8, float32 accumulation). | — | Documented |
| A-11 | Info | Observation (§8.1 vs §11.4) | Under shared-map communication, channel 5 (team-visited cells) reveals teammates' trails, including the cell each just entered, to every actor. That sits in tension with §11.4 ("positions of unsensed teammates" unknown). It is information communicated at execution time, identical for IPPO and MAPPO, and not a train/test (CTDE) asymmetry. | — | Documented |
| A-12 | Info | Coordination (§24.4 c4) | The ablation materiality thresholds (2× collisions, 2-point success drop) were set in Phase 8 after Phase 7's ablation was seen. Observed medians are 91× (IPPO) and 270× (MAPPO), so the outcome does not depend on the threshold, and H3 is decided by criterion 3 anyway. `ablation_L2.json` (Phase 7) is superseded by the analysis-file ablation and supports no claim. | — | Documented |
| A-13 | Info | Statistics | Random's per-seed variance contains sampling noise only (no training variance). That makes the learned-vs-Random CIs narrower than a like-for-like comparison would be. This is by design (§14.1) and disclosed in the RESULTS protocol line. H1's margin (+0.56, CI excludes 0 by ~0.5) is far from that concern. | — | Documented |
| A-14 | Info | Leak probe power | Per run (~1,100 pairs) the probe flags only biases ≳ 0.04. The pooled estimate (A-05) is 0.0005 ± 0.0054, so it would detect a ≈ 0.013 bias. | — | Documented |

## Verified, no defect found

| Audit item (§38 Phase 15) | Evidence |
|---|---|
| Environment rules | `dynamics.py` re-derived against §6.4–6.5: R1 (occupied at start of step), R2 (contested empty cell), single order-independent pass, collision attribution per the §6.5 table, WAIT never penalised. Rescue requires discovery (asserted). Truncation precedence is correct. `tests/env/*` |
| Reward invariants | Components sum to the reward; the team terms are identical across agents; terminal payout makes total exploration constant on success (R6: 399 real episodes). No high-return failures. |
| Partial observability / information boundary | Observations use only `belief[i]`, current sensing and own state, and the mask uses only sensed neighbour cells (`observations.py`). Counterfactual tests cover hidden target, unknown terrain and unsensed teammate. AST import boundary. Actor-only export loaded with `weights_only=True`. Leak probe on all 10 final actors reproduced and pooled (A-05). |
| CTDE | The critic receives global state only through `CriticInputs.state_grid/state_vec`, which the actor's `forward(grid, vec, mask)` cannot accept. `test_mappo_critic_sees_hidden_target_but_actor_does_not`. |
| IPPO/MAPPO vs §15/§16 | Network shapes, orthogonal init gains, separate optimizers, ValueNorm (β = 0.99999, var floor 1e-2), MAPPO critic fusion 256+256+4 → 256 → 1. Bitwise parity test (MAPPO data path with a local critic equals IPPO), with a negative control. |
| GAE / PPO maths | `gae.py` matches §15.5 on `[T,E,N]`, with truncation folded using the denormalised critic on the true final state. Clipped objective, approx-KL `mean((ρ−1)−log ρ)`, advantage normalisation once per update before minibatching, per-network gradient clipping. Hand-computed GAE tests. |
| Reward hacking | Every run's per-update reward and validation success rise together. Final validation success is 0.98–1.00 with ~10–15 steps. No late return-up / success-down divergence (scan of all 10 `metrics.jsonl` / `eval_val.jsonl`). |
| Evaluation fairness | R1: configs identical except `critic.input`; 999,424 env steps / 488 updates each; seeds 0–4 for both; 462 frozen eval-map hashes guarded, 0 rejected. Every method is evaluated on the identical 200 (map, eval seed) pairs with common random numbers. Final checkpoint is primary. No tuning: defaults for both, with deviations D1–D3 applied to both. |
| Seed reproducibility | R4: re-training seed 0 reproduces the first 6 logged updates **bit for bit** for IPPO and MAPPO, and the initial actor weights. 640 held-out episodes re-run from the actors equal the stored records (REAL_DATA_AUDIT C1/C2), also under PyTorch 2.14.1 on a fresh install (Phase 14). |
| Checkpoint integrity | R2: each `actor.pt` equals its final checkpoint's actor. R3: all 40 replays re-record identically from the 0/10/50/100% checkpoints. |
| Statistics | R5: H1 and H2 recomputed from the raw per-episode files with scipy match `summary.json` to 1e-9. Welch df, t-interval and Hedges' J are correct by inspection. |
| Generalization claims | H4 is reported as descriptive retention with CIs, and the absolute structured-map success and the heuristic comparison are now stated (A-07). |
| Metric definitions | All of §23/§24 checked against `metrics.py` and `coordination.py`. Three deviations fixed (A-01, A-02, A-03); one documented (A-10). |
| Backend/model parity | `tests/integration`: live WebSocket sessions with the real actors reproduce each of the 10 stored final-checkpoint study replays frame for frame. |
| Frontend real-data use | Browser E2E checks the pages against the results files. `check_no_mocks`; no hard-coded result numbers. Fixtures are verified as sub-structures of real payloads (REAL_DATA_AUDIT E1) and re-captured by `frontend/scripts/refresh_fixtures.py`. |
| Test coverage | 495 fast + 12 slow Python tests (all pass); 94% line coverage of `marl` + `backend`; 204 Vitest tests pass (+1 opt-in skipped); browser E2E 7/7. |
| Architecture boundaries | `tests/test_import_boundaries.py` (dependency rules, actor purity, no global NumPy RNG in the env, file-size soft limit). |
| Documentation accuracy | Every result number in README/ARCHITECTURE/DEMO is registered and matched to a *named* `summary.json` value (REAL_DATA_AUDIT F1; an unregistered number fails, mutation-checked). RESULTS.md re-renders identically from `summary.json` (B2), which re-aggregates identically from the per-run files (B1). |

### §16.8 checklist (MAPPO implementation mistakes)

| # | Mistake | Status and evidence |
|---|---|---|
| 1 | Global state reaches the actor (incl. via mask) | ✅ Clear: `ActorPolicy.forward(grid, vec, mask)`; mask from current sensing only; counterfactual tests; leak probe |
| 2 | GAE after flattening | ✅ Clear: `compute_gae` on `[T,E,N]`; flattening after GAE; cross-agent/env leakage tests |
| 3 | Bootstrapping from the auto-reset observation | ✅ Clear: `FinalInfo` captured before reset; trainer folds `γ·V(final)` using it (incl. MAPPO global state) |
| 4 | Truncation vs termination confused | ✅ Clear: fold only when truncated ∧ ¬terminated; both end the GAE recursion; folding test |
| 5 | Mask not stored | ✅ Clear: stored in the buffer and re-applied at update |
| 6 | `logp_old` recomputed | ✅ Clear: stored at collection, never recomputed |
| 7 | Inactive samples in normalisation/loss | ✅ Clear: `masked_mean`, active-only advantage stats and ValueNorm update; artificial-mask test |
| 8 | Normalised/denormalised values mixed | ✅ Clear: `value_old` and GAE in reward scale; targets normalised after `update()` |
| 9 | `-inf` logits | ✅ Clear: −1e9 |
| 10 | All agents' observations concatenated | ✅ Clear: agent-specific state (shared channels + own position + own belief mask + own obs) |
| 11 | No "which agent am I" | ✅ Clear: own-position channel; `test_agent_specific_channels_identify_each_robot` |
| 12 | Budget/seed/map/hparam asymmetry | ✅ Clear: AUDIT_CHECKS R1 |
| 13 | Agent-steps vs env-steps | ✅ Clear: budgets in env steps (999,424); agent steps 2,998,272 for both |
| 14 | Deterministic evaluation deadlock | ✅ Clear: stochastic primary, greedy secondary and labelled |
| 15 | Training-map evaluation / test-based selection | ✅ Clear: hash guard; final checkpoint; held-out gatekeeper with access log (A-04 closed a logging gap) |
| 16 | Shared reward summed in one place, averaged in another | ✅ Clear: per-agent rewards everywhere; episode return = mean over agents |

## What changed in the reported results

| Artifact | Change |
|---|---|
| `docs/RESULTS.md` main tables, hypothesis statements | **None** |
| RESULTS.md coordination table | New column *DER@50 (isolation null)*: Random 0.499, IPPO 0.520, MAPPO 0.537. The old whole-episode null is relabelled. Redundant pursuit and allocation CIs moved in the second decimal (reset-discovered targets now included). |
| RESULTS.md | New paragraph: parameter counts and the §16.7 caveat |
| Per-run eval files (protocol v2) | `workload_gini` (4,468 of 4,480 episodes), `discover_rate` and `rescue_latency` (80 episodes = 2 test maps × 2 seeds × 2 modes × 10 runs). Nothing else. |
| `summary.json` | `workload_gini` blocks, coordination blocks, H3 criterion details, `parameter_counts` |
| Demo bundle, frontend fixtures | Rebuilt / re-captured from the regenerated files |

## Open items (none Critical or High)

- Low-severity process point: this is a self-audit (see Limitation). An external reviewer, or the
  `security-review` skill once the repository has a remote, would strengthen it.
- The comm-limited and larger levels (L3/L4) and the stretch hypothesis H5 were never studied, so nothing
  about them is claimed.
