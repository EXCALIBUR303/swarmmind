# PHASE 9 COMPLETE — awaiting explicit user approval (§30.10)

**This phase's stop condition is different from every earlier phase: it
ends only with your explicit sign-off on the design system and the Swarm
Lab spec, not just a reply of CONTINUE.** Phase 11 (the real React
simulator) must not start until you approve. See "Action required" below
for exactly what to confirm.

## IMPLEMENTED
- **Validated every categorical colour in spec §30.3** against the dataviz
  skill's `validate_palette.js` (OKLCH lightness/chroma/CVD checks under
  simulated colour-vision deficiency, a normal-vision floor, and contrast
  vs. surface) — nothing in §30.3 had actually been run through this
  before. Found and fixed two real failures, disclosed one real limitation
  rather than hiding it. Full detail and every command's output:
  [`docs/design/DESIGN_SYSTEM.md`](../design/DESIGN_SYSTEM.md) §3.
- **Checked WCAG 2.2 text and graphical contrast** for every text/semantic
  pair and the simulation-layer tokens (target fill/outline, grid lines,
  robot colours vs. free cells). All text pairs pass comfortably (4.5:1+,
  most 5–18:1); two "naive failures" (grid lines, target fill alone) are
  correctly out of the non-text-contrast rule's scope, explained in the doc
  so nobody "fixes" a correctly-designed pair later.
- **Wrote `docs/design/UI_SPEC.md`**: purpose, real data source, layout,
  every component's exact props/states, interactions, keyboard bindings,
  and responsive behaviour for all 6 pages (§30.2's IA) plus the shared
  chart and async-state components — the level of detail §30 itself
  doesn't go into.
- **Built a real, working static Swarm Lab prototype**
  (`frontend/src/pages/_prototype/SwarmLabPrototype.tsx`), from a **real
  recorded replay** — Phase 7's `main_L2` study, MAPPO seed 0's near-final
  checkpoint (999,424 of 1M steps), a successful 25-step, zero-collision
  episode on the frozen structured map `L2-struct-bottleneck_vertical` —
  never mock data. It has working local playback (play/pause/step/scrub/
  speed), all 6 keyboard shortcuts, the robot-focus interaction (dim +
  11×11 observation-window outline), all 6 layer toggles, the fog-of-war
  reconstruction from `newly_known` deltas (matching §27.4's rule exactly),
  and the Robots/Live/Layers right-rail tabs.
- **Screenshotted it at all 4 required breakpoints** (1440/1024/768/375px)
  — `docs/design/screens/*.png` — and, in doing so, **found and fixed three
  real responsive-layout bugs** that a document alone would never have
  caught (detailed in `DESIGN_SYSTEM.md` §5.1): a sticky-toolbar/overflow
  interaction that hid content off-screen at <768px, a fixed-pixel canvas
  that overflowed a 375px viewport, and a flex-grow rule that left the
  canvas floating with large empty gaps on mobile. Each was fixed and
  re-verified by direct DOM measurement, not just re-screenshotted.
- **Ran a design critique pass** against §30.1's own reference-image
  critique and the dataviz skill's anti-patterns catalogue (no dual axes,
  no cycled/rainbow categorical colours, no 3D/gradients in marks, canvas
  dominance actually measured in the built prototype, legends/labels
  present everywhere a WARN-band colour needs them, status colours never
  repurposed). No unaddressed anti-pattern found.

## FILES CREATED/CHANGED
- New: `docs/design/DESIGN_SYSTEM.md`, `docs/design/UI_SPEC.md`,
  `docs/design/screens/{desktop-1440,desktop-1024,tablet-768,mobile-375}.png`,
  `frontend/src/pages/_prototype/{SwarmLabPrototype.tsx,SwarmLabPrototype.css,replay-fixture.json}`.
- Changed: `frontend/src/design/tokens.css` (full token set, two colour
  fixes, one disclosed limitation, all documented inline with the exact
  validation commands); `frontend/src/app/routes.tsx` (one extra route,
  `/_prototype/swarm-lab`, outside the real app shell — not part of the
  production IA, removed or promoted in Phase 11).
- No other production code touched, per this phase's own scope ("the only
  code is tokens plus one static prototype").

## TESTS RUN
- `tsc -b --noEmit`: clean.
- `oxlint`: clean (one real warning found and fixed — a setState-in-effect
  pattern in the play/pause timer, restructured to avoid it).
- `vitest run`: 2 passed, 1 file (pre-existing `EmptyState` test; the
  prototype is intentionally not wired into any test suite, per its own
  "design-review only" scope).
- 4 breakpoint screenshots taken via headless Chrome at the exact spec
  widths (1440/1024/768/375), each cross-checked against a live DOM
  measurement (`getBoundingClientRect`, computed styles) in an interactive
  browser session — not screenshots alone.
- Colour validation: every command and its full output is in
  `DESIGN_SYSTEM.md` §3, runnable again by anyone with the dataviz skill's
  `validate_palette.js`.

## RESULTS
Screenshots are attached to this message (`desktop-1440.png`,
`desktop-1024.png`, `tablet-768.png`, `mobile-375.png`). Summary of what
they show:
- **1440px**: full expanded nav (220px, labelled), canvas dominant, right
  rail (320px) with working tabs — matches §30.4's layout exactly.
- **1024px**: nav correctly collapses to 64px icons-only; right rail at
  300px — matches §30.5's 1024–1439 rule.
- **768px**: nav stays at 64px icons; right rail becomes a stacked panel
  below the canvas rather than beside it — matches §30.5's rule in spirit
  (I read "bottom tab sheet" as "stacked below, tabs visible", not
  necessarily a modal overlay; flag if you intended something more literal).
- **375px**: single column, nav hidden, canvas full-width and square,
  controls in a genuinely fixed bottom bar — matches §30.5's <768 rule,
  after the three fixes in `DESIGN_SYSTEM.md` §5.1.

## PROBLEMS FOUND
1. **§30.3's `--sm-algo-ippo` (`#0E7490`) fails the dataviz chroma floor**
   (0.094 < 0.10 — too desaturated to do reliable series-identity work).
   This was flagged as a risk back in Phase 5 and confirmed here for the
   first time with an actual validator run.
2. **Robot 5's colour (`#56B4E9`) fails WCAG non-text contrast** against a
   free cell (2.31:1, below the 3:1 floor) — genuinely too light to see,
   independent of any CVD concern.
3. **A real, hard information-theoretic limit**: no ordering of a 6-colour
   robot palette clears an all-pairs CVD floor of 15 — confirmed by
   testing, not assumed, and consistent with the dataviz skill's own
   documented finding about its curated 8-hue reference palette.
4. **Three responsive-layout bugs** in the <768px rules, found only by
   actually building and screenshotting the prototype (§5.1 of
   `DESIGN_SYSTEM.md`) — a pure documentation pass would not have caught
   any of them.

## PROBLEMS FIXED
1. `--sm-algo-ippo` snapped to `#0891B2` (same teal/cyan hue family, one
   step up in chroma) — all checks pass against MAPPO's blue.
2. Robot 5 darkened to `#2E93CB` — clears 3:1 (3.41:1).
3. **Not "fixed" by picking arbitrary colours** — disclosed as a real limit
   that no shipped level actually hits (max 4 robots), with a concrete
   mitigation on record (a secondary shape/pattern encoding) if a 5-6
   robot level is ever added, rather than pretending a colour search could
   solve it.
4. All three layout bugs fixed and re-verified (DOM measurement + a fresh
   screenshot at the same breakpoint) — see `DESIGN_SYSTEM.md` §5.1 for
   the exact CSS change and reasoning behind each.

## REMAINING CONCERNS
- **MAPPO's series colour equals the UI's primary accent colour**
  (`#2563EB`), by the spec's own original design, not a Phase 9 change.
  Flagged as a minor, accepted ambiguity (mitigated by mandatory legends
  and direct labels) rather than force a full accent re-theme this late —
  see `DESIGN_SYSTEM.md` §3.1 for the reasoning; tell me if you'd rather
  revisit it.
- **One small cosmetic gap remains at 375px** (a stretched grid row leaves
  a bit of empty space between the timeline and the Robots tab) — every
  element is still reachable by scrolling; it's a one-line polish item for
  Phase 11, not a blocker.
- **"Bottom tab sheet" at 768px** was implemented as a stacked panel below
  the canvas, not a modal/drawer sheet — a reasonable reading of §30.5's
  one-line spec, but worth confirming it's what you meant.
- The prototype is local-only (no backend calls) since Phase 10 doesn't
  exist yet, exactly as planned — Phase 11 replaces the local replay
  fixture with real session/replay endpoints.
- The Train Swarm/Algorithm Comparison/Swarm Analysis/Experiment
  History/Settings pages are specified in full in `UI_SPEC.md` but were
  not built as prototypes — the spec's file list asks for one static
  prototype (Swarm Lab, the flagship), not one per page.

## VALIDATION GATE
**Infrastructure PASS; final gate PENDING your approval**, per §30.10's
explicit requirement. Everything on the checklist that doesn't require
your sign-off is done: `DESIGN_SYSTEM.md` and `UI_SPEC.md` are complete,
tokens are implemented (and validated, not just asserted), the prototype
renders correctly at all 4 required widths, and accessibility contrast
checks pass.

## CURRENT PROJECT STATUS
Phases 0–8 are committed. **Phase 9 is not committed and not approved
yet.**

## NEXT PHASE
Phase 11 (React simulator, Swarm Lab + replay) — but **only after your
approval**. Phase 10 (FastAPI + WebSocket backend) has no design
dependency and could in principle proceed in parallel if you'd rather
sequence it that way; the spec puts it directly after Phase 9 in numbering
but doesn't require this design's approval as a precondition for it.

## RECOMMENDED MODEL
Sonnet 5

## RECOMMENDED EFFORT
Medium (Phase 10's endpoints are fully specified in §28; High if Phase 11
starts instead, matching the design/frontend work's own scope)

## WHY
Phase 10 is implementing a fully-specified REST/WebSocket contract against
code that already exists (the trained checkpoints, the evaluator, the
replay schema) — not new design or algorithmic work.

## USE OPUS ONLY IF
A live WebSocket session's behavior needs to exactly reproduce the batch
evaluator's results in a way that's hard to verify (the "backend ↔
evaluator parity" requirement in §33.5) and a subtle discrepancy shows up
that Sonnet can't localize.

## RECOMMENDED SKILLS / CONNECTORS
None further for the design package itself. GitHub still needs
authorization in your claude.ai connector settings for pushes from here.

## ACTION REQUIRED — please confirm before I continue

1. **Look at the 4 screenshots** (sent alongside this report).
2. **Approve or amend**, specifically:
   - The two colour fixes (IPPO teal, robot 5) and the disclosed 5-6-robot
     limitation (§3.2) — OK as-is, or do you want a different mitigation?
   - MAPPO-color-equals-accent-color — leave as the spec originally
     specified, or revisit?
   - The 768px "right rail becomes a stacked panel" reading of "bottom tab
     sheet" — is that what you meant, or do you want an actual overlay
     sheet?
3. Say whether to **commit Phase 9** (the two file changes plus the new
   design docs/prototype).
4. Once approved, reply **CONTINUE** and tell me whether to start Phase 10
   (backend) or wait for Phase 11 planning.

## STOP HERE.
