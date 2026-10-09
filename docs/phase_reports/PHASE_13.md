# PHASE 13 COMPLETE — UI polish

(Model note: this was done on Sonnet throughout; the spec allows Haiku for the mechanical token replacement, which I did not delegate.)

## IMPLEMENTED
- **Token audit tool** `frontend/scripts/check_tokens.mjs` (part of `npm run build`): found **66 raw literals in 8 files**; all replaced by tokens (21 new layout/size tokens in `design/tokens.css`). Remaining raw values live only in `src/design/` (`ramp.ts`, `motion.ts`), with a test that fails if they drift from `tokens.css`.
- **Contrast**: a WCAG test over the tokens found **two real AA failures** (success and warning text on the sunken surface, 4.43:1) — fixed. axe-core on every route at 1440 and 375 px: **0 violations** (it initially found two missing `<h1>`).
- **Keyboard**: automated Tab walk — all interactive elements reachable on all 7 routes at both widths, **0 without a visible focus indicator**.
- **Micro-interactions per §30.7**: fixed-duration collision (400 ms), discovery (500 ms), rescue (800 ms) effects; the missing blocked-move nudge and comm-link pulse; movement tween = min(80% of interval, 160 ms). All durations in `design/motion.ts`, boundary-tested.
- **Responsive**: SVG charts now measure their container and draw 1:1, so axis text at 375 px is full size (was ≈8 px).
- **Settings page** (decision: built now, since the UI spec defines it and "every page keyboard navigable" is the gate): API base URL validated against `/api/health` before saving, default playback speed, cell-coordinate overlay, motion override (system / reduce / allow) that also gates CSS, theme shown disabled with a reason.
- `docs/design/POLISH_CHECKLIST.md` (every item with how it was checked), reusable audit scripts in `frontend/scripts/a11y/`.

## TESTS RUN
- vitest **200 passed** (1 opt-in e2e skipped), backend **148 passed**, `tsc`, `oxlint`, `npm run build` (mock guard + token audit + fresh types) clean.
- Real browser: axe (14 route×width runs, 0 violations), keyboard walk (14 runs), no console errors, screenshots after.

## RESULTS (gate)
| Gate | Result |
|---|---|
| Token audit passes | ✓ 0 findings (66 before) |
| No AA contrast failures | ✓ token test + axe (2 real failures fixed on the way) |
| Every page keyboard navigable | ✓ 7/7 routes × 2 widths |
| Before/after screenshots | ✓ `docs/design/screens/phase11-*`, `phase12-*` (before) and `after/phase13-*` (after) |

## PROBLEMS FOUND / FIXED
Success/warning text contrast; missing `<h1>` on Swarm Lab and Settings; hard-coded colours in the canvas token reader; micro-interaction durations that followed the step interval instead of the spec; missing nudge and comm pulse; tiny chart text at 375 px; a flaky self-made headless-Chrome harness (not product code).

## REMAINING CONCERNS
- axe cannot judge reading order or announcement quality; **no VoiceOver walkthrough** was done.
- The comm-link pulse and layer are verified only with synthetic edges (no real limited-range replay exists).
- Reduced-motion override is tested in unit tests and by reading the attribute; I did not record a browser session with the OS setting on.
- The Swarm Lab playback group still wraps to a second toolbar row at 1440 px.
- The axe/keyboard scripts drive a local headless Chrome by hand-rolled CDP and are not wired into CI (Phase 14 may add Playwright).
- Forced-colors mode and print styles are untouched.
- The `frontend/dist` build is not committed (gitignored).

## VALIDATION GATE
**PASS.**

## CURRENT PROJECT STATUS
Phases 0–12 committed (Phase 12 = `b7651bf`). **Phase 13 uncommitted.**

## NEXT PHASE
Phase 14 — end-to-end integration: clean-checkout proof with only real data, complete documentation.

## RECOMMENDED MODEL / EFFORT
Sonnet 5.5 · High (it touches every layer and must catch cross-layer drift).

## WHY
Integration bugs live between components; the work is mostly running everything from scratch and fixing what breaks.

## USE OPUS ONLY IF
A clean-checkout run reveals a reproducibility discrepancy in results (not plumbing).

## RECOMMENDED SKILLS / CONNECTORS
`engineering:testing-strategy`, `engineering:documentation`; GitHub connector would be needed to push (not authorised in this environment).

## ACTION REQUIRED
Reply **CONTINUE** for Phase 14 (add **commit** to commit Phase 13 first).

## STOP HERE.
