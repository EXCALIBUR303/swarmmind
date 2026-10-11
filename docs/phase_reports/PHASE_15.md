# PHASE 15 COMPLETE — Final MARL + system audit

(Model: Opus 5.5, as the spec recommends. This is an adversarial self-review by the same assistant
lineage that built the system, not an independent external audit; see `docs/AUDIT.md`.)

## IMPLEMENTED
- **`docs/AUDIT.md`**: 14 findings with severity, evidence, fix and status; the full Phase 15 checklist;
  the §16.8 table; what changed in the results.
- **`scripts/audit_checks.py` → `docs/AUDIT_CHECKS.md`**: committed, re-runnable versions of every check
  that needs the full `results/` tree. Seven checks, R1–R7.
- **Defect fixes** (no features):
  - A-01: coverage-matched isolation null for coordination criterion 2b.
  - A-02: targets sensed at reset counted as discovered, in Python, the API and the UI.
  - A-03: workload share per the §23 definition, from per-step observer credit.
  - A-04: held-out access by the analysis script is now logged.
  - A-05: `scripts/run_leak_probe.py` reproduces the stored leak-probe file.
  - A-06: parameter counts and the §16.7 caveat in RESULTS.md.
  - A-07: honest heuristic comparison in the README.
  - A-08: the replay recorder now steps through the single `EpisodeRunner`.
  - A-09: oscillation-rate label.
- **Regeneration, no retraining:**
  - Held-out evaluation and baselines re-run under evaluation protocol v2.
  - Coordination analysis, `summary.json`, RESULTS.md and figures regenerated.
  - Demo bundle rebuilt; the five affected frontend fixtures re-captured by the new
    `frontend/scripts/refresh_fixtures.py`.

## TESTS RUN
- `make lint` clean.
- `make test`: **495 passed**, coverage 94%.
- `make test-slow`: **12 passed**.
- Vitest **204 passed** (+1 opt-in skipped); `tsc`, `oxlint` and `npm run build` clean.
- Browser E2E **7/7**, 0 console or network errors.
- `make audit`: **PASS** (10 pass, 1 partial, 0 fail).
- `scripts/audit_checks.py`: **7/7 PASS**.
- Fresh-clone check (README followed in a new directory): **11/11 PASS**, `docs/FRESH_CLONE_REPORT.md`.

## RESULTS
- **No Critical findings. One High (A-01), fixed. No open Critical or High findings.**
- **Hypothesis verdicts unchanged:** H1 supported for IPPO and MAPPO, H2 not supported, H3 inconclusive,
  H4 descriptive.
- **Fair comparison still passes:** criterion 2b holds under the matched-coverage null. Paired DER@50 −
  DER_iso@50 is IPPO −0.162 [−0.168, −0.157] and MAPPO −0.180 [−0.204, −0.156].
- **Re-evaluation changed only the targeted fields:** across all 4,480 learned-run episodes, success,
  steps, coverage, return and collisions are identical to the originals. Only workload Gini, discover rate
  and rescue latency changed.
- **Reproducibility:**
  - Re-training seed 0 reproduces the first 6 logged updates bit for bit, for both algorithms.
  - Each actor equals its final checkpoint.
  - All 40 replays re-record identically.
  - 640 held-out episodes re-run identically.
  - H1 and H2 recomputed with scipy from the raw files match to 1e-9.
- **No leak:** the leak probe is reproducible and, pooled over 11,013 pairs, shows a bias of −0.0005
  (SE 0.0054).

## PROBLEMS FOUND
A-01 to A-09 (fixed) and A-10 to A-14 (documented observations); details in `docs/AUDIT.md`. The two
that changed reported numbers:
- **Isolation null (A-01):** it used to be measured over whole solo episodes.
- **Workload Gini (A-03):** the old definition made work look almost perfectly even. Under the spec's
  definition the learned Gini on test is about 0.24, against Random's 0.30. No conclusion relied on it.

## PROBLEMS FIXED
All of A-01 to A-09, each with a regression test where code changed. Two audit-tooling fixes on the way:
- The claim audit now also checks one-decimal numbers.
- The claim audit no longer mistakes "§16.8"-style section references for results.

## REMAINING CONCERNS
- **Self-audit:** this is not an independent review. An external reviewer, or the `security-review`
  skill once the repository has a git remote, would strengthen it.
- **Unstudied scope:** L3, L4 and H5 were never studied and nothing is claimed about them.
- **Bundle size:** the demo bundle stays in git (38 MB), per the "commit and continue" after Phase 14.
- **`audit_checks.py` needs the full results:** it requires the developer's full `results/` tree
  (checkpoints), so it cannot run on a clean clone. The bundle-only audit (`make audit`) can.

## VALIDATION GATE
**PASS**: no open Critical or High findings; every claim in RESULTS.md and the README traces to evidence
(B1, B2, F1); AUDIT.md is complete.

## CURRENT PROJECT STATUS
All 16 phases (0–15) built. Phase 15 uncommitted.

## NEXT PHASE
None. The project is complete per spec §38.

## RECOMMENDED MODEL / EFFORT
No further phase. For follow-up maintenance, Sonnet at Medium effort is enough.

## WHY
The remaining work is housekeeping: commit, and optionally push.

## USE OPUS ONLY IF
A future change touches the training, evaluation or statistics methodology.

## RECOMMENDED SKILLS / CONNECTORS
`security-review` once a git remote exists. GitHub only if you want to push: it needs authorization in
your claude.ai connector settings.

## ACTION REQUIRED
Say **commit** to commit Phase 15.

STOP HERE.
