# PHASE 12 COMPLETE — Analytics + training UI

## IMPLEMENTED
- **Algorithm Comparison** (`/compare`): study / evaluation-set / action-selection filters; results table (mean ± SD [95% CI], n seeds, higher/lower-is-better) printed straight from `summary.json`; six per-seed dot-and-whisker plots; learning curves (mean ± CI band, thin per-seed lines, Random/heuristic as dashed neutral references, AUC window stated); hypothesis panel quoting the pipeline's statements **and** the frozen rule text from `docs/EXPERIMENTS.md` verbatim; pairwise-difference table; coordination table; missing-runs list.
- **Swarm Analysis** (`/analysis`): heatmaps (visits/collisions) for IPPO and MAPPO on one shared colour scale; one illustrative episode (coverage, DER, inter-robot distance mean + closest pair, discovery/rescue timeline, workload with Gini, idle) labelled "illustrative episode, not a statistic"; **Swarm Evolution**: the same map at 0/10/50/100% checkpoints on one shared clock, each panel with its own mission strip; single canvas + checkpoint selector below 1024 px.
- **Train Swarm** (`/train`): form whose bounds come from the backend (`GET /api/training/schema`); Enter never launches; job card with progress, **measured** steps/s and ETA, Cancel, failure log tail; live charts over `/ws/training/{run}` with polling fallback; adopts a job already running after a reload.
- **Experiment History** (`/experiments`, `/experiments/:runId`): sortable/filterable table (cards on small screens), invalid manifests shown as invalid, run detail with manifest, training curves and links into Swarm Lab replay mode (`/?replay=<id>`, new).
- **Chart kit** (`components/charts`): `ChartFrame` (title, caption, legend for ≥2 series, explicit loading/empty/error, View-data table), `LineChart`, `DotWhisker`, `Heatmap`, `Bars`, `DiscoveryTimeline` — custom SVG, no dual axes.
- **Backend additions:** `GET /api/studies/{id}/analysis/{name}`, `GET /api/preregistration`, `GET /api/training/schema`, `log_tail` on failed/cancelled jobs.

## TESTS RUN
- vitest **163 passed** (1 opt-in e2e skipped): chart components on real fixture JSON; every page's loading/empty/error states; formatting utilities; **training-form validation checked against the backend's own schema at every advertised bound**; stream de-duplication and WebSocket→polling fallback; shared-clock Evolution; sort/filter logic.
- Backend **148 passed**, ruff + mypy clean; `tsc`, `oxlint`, `npm run build` (mock guard + fresh generated types) clean.
- Browser (real backend, real data): all four pages render with **no console errors**, no horizontal overflow at 1440 and 375 px; launched a **real training job from the UI** (IPPO L1, 30k steps): live charts over WebSocket, measured 289–341 steps/s, completed, appeared at the top of Experiment History. I deleted that toy run afterwards so it is not mixed into the study results.

## §30.8 CHECKLIST (verified, with exceptions stated)
| Rule | Status |
|---|---|
| Title, axis labels with units, tooltips, loading/empty/error | ✓ (`ChartFrame` requires a state; tests cover all three). Bars and the discovery timeline have no hover tooltip: values are printed on the marks / in `<title>` and in View data |
| Legend when >1 series | ✓ (tested: none for one series) |
| Learning-curve x axis 0, 500k, 1M | ✓ (`fmtSteps`, tested) |
| Rates on fixed [0,100%] | ✓ for success/coverage/DER etc. The learning-curve y domain is [0, 1.05] so the CI band is not clipped |
| Bars start at 0; zoomed dot plots show the break | ✓ bars; dot plots with an auto-zoomed axis state "axis does not start at 0 (starts at X)" in the axis title (a text note, not a drawn break glyph) |
| Raw faint 1px + rolling mean 2px, window stated | ✓ training reward ("20-update trailing mean") |
| No dual axes / 3D / gradients in marks | ✓ |
| Tokens; sequential single-hue heatmap with numeric colour bar | ✓ IPPO/MAPPO pair re-validated with `validate_palette.js` on the white surface: all checks pass (CVD ΔE 14.7, normal 16.3). Tritan ΔE 6.7 is in the legal band because every series is also legended and direct-labelled |
| "View data" on every chart | ✓ (tested) |

## PROBLEMS FOUND / FIXED
1. **Real data shape ≠ my assumption:** `inter_robot_distance_over_time` is `{mean, min}` per step; the first draft crashed. Typed and plotted as two series.
2. **a11y bug:** a segmented control inside a `<label>` lent its whole text to the first button's accessible name. Now a `role="group"` (`FieldGroup`).
3. **Duplicate React keys** when a CI is zero-width (success 1.000 in every seed); regression test added.
4. Direct labels overprinted on the learning curve (MAPPO / heuristic both at 1.0): labels are now pushed apart. Charts were scaled to unreadable sizes in narrow cards: widened grid cells and capped line-chart width.
5. Heatmaps originally had independent colour scales per algorithm (misleading side by side): now shared.

## REMAINING CONCERNS
- **Settings page is still a placeholder** (not in Phase 12's list; it is in the UI spec). Left for Phase 13 or your call.
- Experiment History lists the four **pilot runs from 2026-09-24 as "running"**: their manifests were left stale by an early killed process (before the SIGTERM handler). That is what the data says, so the UI shows it; I did not edit those files.
- At 375 px the SVG charts scale down with the card, so axis text is small (≈8 px). Fixing needs per-breakpoint chart geometry: Phase 13.
- The hypothesis panel quotes rule text from `docs/EXPERIMENTS.md`; the H3 rule there still reads "evaluated in Phase 8", while the verdict statement cites the §24.4 criteria (the doc was frozen before Phase 8). Shown as is.
- Table-row hover/focus tooltips on the dot plots work with the pointer only; the same numbers are in View data.
- No contrast/axe audit yet (Phase 13). I did not use the `data-visualization` skill; the `dataviz` skill was read and its validator run.
- Swarm Evolution replays at the fixed 4 steps/s base rate; speeds 0.5–4×.

## VALIDATION GATE
**PASS:** every chart renders from real endpoints (verified live), no mock imports outside `mocks/` (build guard), §30.8 checklist above, screenshots at `docs/design/screens/phase12-*.png`.

## CURRENT PROJECT STATUS
Phases 0–11 committed (Phase 11 = `63e5df4`). **Phase 12 uncommitted.**

## NEXT PHASE
Phase 13 — UI polish (token audit, accessibility, micro-interactions, responsive fixes; no new features).

## RECOMMENDED MODEL / EFFORT
Sonnet 5.5 · Medium.

## WHY
Mechanical consistency and accessibility work against an existing design system.

## USE OPUS ONLY IF
Never expected.

## RECOMMENDED SKILLS / CONNECTORS
`accessibility-audit`, `design-token-audit`-style grep checks, built-in browser; optionally `emil-animate` for the motion items skipped in Phase 11.

## ACTION REQUIRED
Reply **CONTINUE** for Phase 13 (say **commit** to commit Phase 12 first). Tell me whether the Settings page should be built in Phase 13.

## STOP HERE.
