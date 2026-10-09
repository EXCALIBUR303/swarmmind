# SWARMMIND UI specification (Phase 9)

Companion to `DESIGN_SYSTEM.md`. Elaborates spec §30.2–§30.9 into the level
of detail a frontend implementer needs: every page's purpose, real data
source, layout, components with their exact props/states, interactions,
keyboard support, and responsive behaviour. Every data source named below
is a real backend endpoint (Phase 10) or a real study/run artifact (Phases
6–8) — never a fabricated one.

---

## Global: App Shell

**Purpose.** Navigation + a persistent backend-health indicator.

**Layout.** Left nav (`--sm-nav-width-expanded` ≥1024px, `--sm-nav-width-
collapsed` <1024px, icons + tooltips) + main content (`<Outlet/>`).

**Component: `Nav`**
| Prop | Type | Notes |
|---|---|---|
| `items` | `{to, label, icon, end}[]` | Fixed, from §30.2's IA — never user-configurable |
| `collapsed` | `boolean` | Driven by viewport width, not user toggle, per §30.5 |
| `backendStatus` | `"checking" \| "ok" \| "unreachable"` | From `GET /api/health` |

**States.** `backendStatus` renders as a status dot + label (already built,
Phase 1); `unreachable` uses `--sm-danger`, never blocks navigation — every
page must still render its own empty/error state independently (§30.8).

**Keyboard.** Standard nav: each link is a real `<a>`/`NavLink`, reachable
by Tab, `Enter` activates. No custom keyboard handling needed here.

**Responsive.** §30.5's breakpoints govern nav width; content area always
fills the remainder.

---

## Page 1 — Swarm Lab (`/`, default route)

**Purpose.** The flagship live + replay simulator (§30.4).

**Data sources (Phase 10 endpoints, per §28.2):**
- `POST /api/sessions` (live sessions), `GET/DELETE /api/sessions/{id}`, `POST /api/sessions/{id}/reset`
- `WS /ws/sessions/{id}` (live frame stream; the server owns the clock)
- `GET /api/replays`, `GET /api/replays/{id}`, `GET /api/replays/{id}/initial` (recorded replays and their reset-time known cells)
- `GET /api/models?level=` (learned policies for the Policy selector), `GET /api/levels`
- `GET /api/maps?set=validation|test|structured&level=` (map selector)

*(Phase 11 correction: this list originally named endpoint paths from before the backend existed; the above are the real ones, see `docs/API.md`.)*

**Layout.** Exactly §30.4's ASCII diagram: Toolbar → Mission Strip → Canvas
→ Timeline (replay mode only), with a 320px Right Rail. See
`docs/design/screens/desktop-1440.png` for the built prototype.

### Components

**`Toolbar`**
| Prop | Type |
|---|---|
| `mode` | `"live" \| "replay"` |
| `policySelector` | `{algo, level, seed, step}` (populated from `/api/models`) |
| `mapSelector` | `{level, split, mapId}` |
| `playback` | `{playing, speed, onPlay, onPause, onStep, onStepBack, onReset}` |
| `onOpenScenarioDrawer` | `() => void` |

States: disabled play/step controls while a session is loading; `mapSelector`
disabled in live mode until a scenario is configured via the drawer.

**`MissionStrip`**
| Prop | Type |
|---|---|
| `t`, `maxT` | `number` |
| `rescued`, `totalTargets` | `number` |
| `coveragePct` | `number` (0–100, fixed y-domain per §30.8) |
| `collisions` | `number` |
| `status` | `"running" \| "paused" \| "success" \| "failure"` |

`aria-live="polite"` on the whole strip (throttled to state-change events,
not every frame, §30.9). Status dot pulses at 1.5s only while `"running"`.

**`SimulationCanvas`**
| Prop | Type |
|---|---|
| `grid` | `{width, height, terrain}` |
| `robots` | `{id, pos, colorToken, focused}[]` |
| `targets` | `{id, pos, status: "hidden"\|"discovered"\|"rescued"}[]` |
| `layers` | `{sensor, paths, visited, unknown, commLinks, groundTruth}` (booleans) |
| `focusedRobot` | `number \| null` |
| `onRobotClick` | `(id: number) => void` |

Cell rendering, robot/target/sensor/path tokens: exactly §30.3's simulation-
layer table. Square, fit-to-container, ≥55% of viewport width at desktop
(§30.1's critique of the reference image — verified in the prototype, not
just asserted).

States: `loading` (skeleton grid, same dimensions, no data), `empty` ("No
session — start one from the toolbar"), `error` (session/WS failure,
retry action).

**`Timeline`** (replay mode only)
| Prop | Type |
|---|---|
| `t`, `maxT` | `number` |
| `events` | `{t, kind: "discover"\|"rescue"\|"collision", robotId?}[]` |
| `onScrub` | `(t: number) => void` |

Event markers per §30.4: ◆ discover, ★ rescue, ✕ collision.

**`RightRail`** — tabs `Robots | Live | Layers` (§30.4).
- **Robots tab**: table of `{id, pos, state, distanceTravelled, idlePct}`.
  `state` is an **observable fact** (`Moving`, `Blocked`, `Waiting`,
  `Rescued T2 @ t=41`), never a fabricated intent label (§30.1 — this is a
  direct, binding correction of the reference image, not a suggestion).
- **Live tab**: coverage `C(t)` sparkline, reward components (from
  `metrics.jsonl`/replay frames only — never invented).
- **Layers tab**: the six toggles as real checkboxes/switches (`Sensor
  radius`, `Paths`, `Visited cells`, `Unknown (fog)`, `Communication links`,
  `Ground truth` — the last is off by default and labelled "Not visible to
  robots", §30.4).

**`ScenarioDrawer`** — level, grid size, robots, targets, density, comm
mode, map seed. Opens over the right rail; **never** shrinks the canvas
(§30.4 — a layout constraint the prototype's CSS enforces with a fixed
overlay, not a reflow).

### Keyboard (§30.4, exact bindings)
`Space` play/pause · `→` step · `←` step back (replay only) · `R` reset ·
`1–6` focus that robot number if present · `L` toggle the Layers tab open.

### Responsive
§30.5's table applies verbatim. <768px: canvas full-width square, controls
in a sticky bottom bar, right rail becomes a bottom sheet.

### Robot focus (§30.4)
Clicking a robot: highlights it, dims others to 40% opacity, and outlines
its own 11×11 observation window on the canvas — this is the one feature
that makes partial observability visually concrete, and it must use the
real `obs_radius` from the env config, never a hardcoded 11.

---

## Page 2 — Train Swarm (`/train`)

**Purpose.** Launch and monitor a training run (§30.6).

**Data sources:** `POST /api/train` (launch), `GET /api/train/{job_id}`
(status/SPS/ETA), `GET /api/runs/{run_id}/metrics` (streams `metrics.jsonl`).

**Layout.** Left column: config form. Right: live small-multiple charts +
job status card.

**`TrainConfigForm`** fields (each with min/max bounds + helper text, from
`AlgoConfig`/`EnvConfig`'s own pydantic constraints — the form's bounds are
generated from those schemas, never duplicated by hand): algorithm, level,
seed, total env steps, `lr_actor`/`lr_critic`, entropy coef, clip, rollout
length, comm mode.

**`MetricsSmallMultiples`** — one chart per metric **that exists in
`metrics.jsonl`** (§30.6: "only metrics that exist... are rendered" — a
hard rule, not a suggestion: a metric absent from the log renders nothing,
not a zero-filled placeholder). Success, coverage, targets rescued,
collisions, episode return (raw faint 1px + rolling mean 2px solid, window
size stated in the caption, §30.8), policy loss, value loss, entropy,
approx-KL, explained variance. Single y-axis each — never combined into one
dual-axis chart (§30.1's binding correction of the reference image).

**`JobStatusCard`** — SPS and ETA **computed from measured SPS**, never a
static estimate; status `queued | running | completed | failed`.

**States.** No active job → empty state with a "configure a run" prompt,
form still usable. Job failed → error state showing the traceback from the
manifest, not a generic message.

**Keyboard.** Standard form tabbing; `Enter` in the form does not submit
early (prevents accidental launches — an explicit "Start training" button
click or `Space` on it is required).

**Responsive.** <1024px: form and charts stack vertically (form first).

---

## Page 3 — Algorithm Comparison (`/compare`)

**Purpose.** The Phase 7 study results, exactly (§30.6).

**Data sources:** `GET /api/studies`, `GET /api/studies/{id}` — this reads
`results/studies/<study>/summary.json` **verbatim**; the frontend never
recomputes a statistic client-side.

**Layout.** Study selector → results table → dot-and-whisker plots → learning
curves → Hypothesis status panel.

**`ResultsTable`** — mean ± std, 95% CI, n seeds, one row per method
(Random/Heuristic reference/IPPO/MAPPO), one column set per primary metric
— structurally identical to the tables already generated in `docs/RESULTS.md`.

**`DotWhiskerPlot`** — per-seed dots + mean + CI, one per metric, matching
`marl/analysis/figures.py`'s `plot_heldout_dots` exactly (same data, same
convention: black diamond/whisker for the mean, coloured dots per seed).

**`LearningCurveChart`** — mean ± CI band, thin per-seed lines, x-axis in
environment steps formatted `0, 500k, 1M, …` (§30.8, not raw integers).

**`HypothesisPanel`** — quotes `summary.json`'s `hypotheses.H1-H4` verdict
and `statement` fields **verbatim**. No editorializing text is added by the
frontend (§30.6: "quoting the pre-registered decision rules... with no
editorializing" — this is a hard content rule, not a style preference).
Random and the heuristic render as the dashed neutral reference line
described in §3.1 of `DESIGN_SYSTEM.md`, never a hue slot.

**States.** No completed study yet → empty state naming the exact command
(`scripts/run_study.py`) to run one — an actionable empty state, not a
dead end. A study with `missing` entries (per Phase 6/7's manifest) shows
them exactly as `docs/RESULTS.md`'s "Missing/incomplete" section does —
never silently hidden.

**Responsive.** <1024px: dot-whisker plots and learning curves stack to
one per row; the results table scrolls horizontally rather than reflowing
(a stats table's row/column meaning must not scramble on resize).

---

## Page 4 — Swarm Analysis (`/analysis`)

**Purpose.** Phase 8's coordination and behaviour outputs (§30.6).

**Data sources:** `results/studies/<study>/analysis/*.json` (coordination,
heatmaps, showcase episode bundles) via `GET /api/studies/{id}/analysis`.

**Layout.** Run + map selector → heatmap grid → coverage-over-time → DER
vs. coverage → discovery timeline → workload/idle bars → inter-robot
distance → Swarm Evolution (2×2 synchronized replays).

**`HeatmapGrid`** — sequential single-hue scale (light → `--sm-accent`),
obstacle cells hatched out (matching `marl/analysis/figures.py`'s
`plot_heatmap`), colour bar with numeric scale (§30.8).

**`DerVsCoverageChart`** — one line per illustrative episode, explicitly
labelled "illustrative episode, not a statistic" (matching Phase 8's own
report language) — this label is not optional, since §24.1's curve is a
diagnostic, not an aggregate with a CI.

**`DiscoveryTimeline`** — Gantt-style strip, ◆ discover / ★ rescue markers,
one row per target (matches `plot_discovery_timeline`).

**`WorkloadBars`** / **`IdleBars`** — per-robot bars, sorted, Gini
coefficient annotated on the workload chart (§26).

**`SwarmEvolutionPanel`** — 4 synchronized canvases (0/10/50/100%
checkpoints) sharing one timeline scrubber and each with its own compact
mission strip; built from the Phase 6 evolution replays, never re-simulated.

**States.** Level/study combinations with no structured set (e.g. L1, L3,
L4 in the current studies) show an explicit "no structured (G2) maps for
this level" empty state on the heatmap/showcase sections, not a blank
chart — matching Phase 8's own script behaviour (`analyze_study.py` logs
and skips, never fabricates).

**Responsive.** <1024px: the 2×2 Swarm Evolution grid becomes a single
canvas with a checkpoint selector instead of 4 simultaneous panels (four
270°+ canvases cannot fit meaningfully below tablet width).

---

## Page 5 — Experiment History (`/experiments`)

**Purpose.** Every run, sortable/filterable, with a detail view (§30.6).

**Data sources:** `GET /api/runs` (from every `manifest.json` under
`results/runs/`), `GET /api/runs/{id}` (full manifest + metrics + checkpoints
+ replays list).

**`RunsTable`** — columns: status, algo, level, seed, steps, final val
success, date. Sortable on every column, filterable by status/algo/level.

**`RunDetail`** (row click) — resolved config, manifest fields (§21.2, all
of them — git commit/dirty, versions, config_hash, wall time), learning
curves (reuses `LearningCurveChart`), checkpoints list with download/inspect
links, associated replays list linking into Swarm Lab's replay mode.

**States.** A run whose `manifest.json` fails to parse renders as `invalid`
in the table (matching the backend's own "tolerates partial/corrupt runs,
marks them invalid, never crashes" rule, §37) rather than breaking the
whole table.

**Responsive.** <1024px: table becomes a card list (one card per run, same
fields); detail view is a full-screen navigation, not a side panel.

---

## Page 6 — Settings (`/settings`)

**Purpose.** Local client preferences only — no server-side state (§30.6).

**Fields:** API base URL, default playback speed, reduced-motion override
(in addition to honouring the OS-level `prefers-reduced-motion`, §30.9),
cell coordinate display (on/off overlay), theme (`light` only in v1 — the
control exists and is disabled with a tooltip explaining why, rather than
being omitted, so the setting's future existence is discoverable).

**Persistence.** `localStorage` only; there is no backend endpoint for
settings — this page never calls the API except to validate a new base URL
(a lightweight `GET /api/health` against the candidate URL before saving).

**States.** Invalid API base URL → inline validation error, setting not
saved until it resolves or the user explicitly overrides.

---

## Cross-cutting: charts (§30.8, instantiated)

Every chart component above (`MetricsSmallMultiples`, `DotWhiskerPlot`,
`LearningCurveChart`, `HeatmapGrid`, `DerVsCoverageChart`) shares:
- Title, axis labels **with units**, a legend when >1 series, tooltips with
  exact values.
- Loading (skeleton), empty ("no data for this selection"), and error
  (with the actual fetch error, not a generic message) states — all three
  are required props on the shared `Chart` wrapper, not optional.
- A **"View data"** table toggle (accessibility — required, not a stretch
  extra, §30.8's last line).
- No dual y-axes, no 3D, no gradients inside marks, no truncated bar axes.

## Cross-cutting: loading/empty/error state component

**`AsyncBoundary`**
| Prop | Type |
|---|---|
| `state` | `"loading" \| "empty" \| "error" \| "ready"` |
| `loadingSlot`, `emptySlot`, `errorSlot` | `ReactNode` |
| `error` | `Error \| null` (rendered inside `errorSlot`, not swallowed) |

Every page and every chart wraps its data-dependent content in this, so
"what does this look like with no data / while loading / when the backend
is down" is answered once, consistently, rather than per-page improvisation.
