# Phase 13 polish checklist

Every item states how it was checked. "✓" means checked in a real browser or by an automated test that fails when it regresses.

## Token compliance
- ✓ `npm run audit:tokens` (part of `npm run build`): **no raw colour (`#hex`, `rgb()`, `hsl()`) or px literal outside `src/design/`**. Initial run: 66 findings in 8 files; all now read tokens. Deliberate exceptions (documented in `scripts/check_tokens.mjs`): hairline/ring widths 1–3 px on border/outline/stroke, `@media` breakpoints (CSS variables are not allowed there), `0`, `#fff` on accent/dark fills.
- ✓ Values the DOM/canvas cannot read from CSS variables live in `src/design/ramp.ts` (heatmap ramp, canvas fallbacks); `design.test.ts` fails if they drift from `tokens.css`.
- ✓ New layout/size tokens (`--sm-control-h`, `--sm-hit-min`, `--sm-card-min`, ...) replace magic numbers.

## Contrast (WCAG 2.2 AA)
- ✓ `design/contrast.test.ts`: every text token on every surface ≥ 4.5:1; white on accent and accent-hover; nav text on both ends of the nav gradient; algorithm colours ≥ 3:1 graphics; robots 1,2,3,5 ≥ 3:1 (robot 4 = documented 2.98:1 WARN case, always labelled).
- **Real failure found and fixed:** `--sm-success` (#15803d) and `--sm-warning` (#b45309) were 4.43:1 on `--sm-bg-sunken`. Darkened to #166534 and #92400e.
- ✓ axe-core (WCAG 2a/2aa/21a/21aa/22aa + best-practice) in headless Chrome on all 7 routes at 1440 and 375 px: **0 violations** (initially 2: no `<h1>` on Swarm Lab and Settings).

## Keyboard
- ✓ Programmatic Tab walk on all 7 routes at both widths: every interactive element is reachable (e.g. 56/56 on run detail) and **every focused element shows an outline or ring** (0 without).
- ✓ Segmented controls are `role="group"` (not a `<label>` that renamed the first button).
- ✓ Shortcuts do not fire while a control has focus (tested).
- ✓ Settings: new page, fully keyboard operable; theme control shown disabled with a reason.

## Reduced motion
- ✓ OS `prefers-reduced-motion` honoured by the canvas (no tween, effects static, ✕ glyph remains), nav entrance and CSS animations.
- ✓ New Settings override (system / always reduce / always allow): wins over the OS in JS and, via `html[data-motion="reduce"]`, in CSS (tested).

## Micro-interactions (§30.7)
| Interaction | Spec | Now |
|---|---|---|
| Robot movement | min(interval×0.8, 160 ms) ease-out | ✓ `moveDurationMs` (tested) |
| Blocked move | 2 px nudge and back, 120 ms | ✓ (was missing) |
| Collision | ring 0→0.6 cell, 400 ms | ✓ fixed 400 ms (was step-derived) |
| Target discovery | scale 0.6→1 + gold ring, 500 ms | ✓ fixed 500 ms |
| Target rescue | star → hollow grey, 800 ms | ✓ fixed 800 ms |
| Comm link | one opacity pulse, 300 ms | ✓ (was missing; synthetic edges only, no real L4 replay exists) |
| Status change / sensor toggle / panel transitions | CSS 150–200 ms | ✓ (Phase 11) |
Durations live in `design/motion.ts`; effect progress is unit-tested at the boundary of each duration.

## Responsive and typography
- ✓ Charts measure their container and draw at 1:1 (`useWidth`): at 375 px axis text is full size (it was ≈8 px when a 640-px viewBox was scaled down).
- ✓ No horizontal overflow at 1440 / 1024 / 768 / 375 (Swarm Lab, measured) and at 1440 / 375 (other pages).
- Not changed: at 1440 the Swarm Lab playback group wraps to a second toolbar row (functional; not worth a layout rewrite).

## Copy
- ✓ Empty/error states name the next action or command (no study → `scripts/run_study.py`; no replays → `record_replays.py`; backend down → the uvicorn command + Retry; failed job → its real output).

## Before / after
Before: `docs/design/screens/phase11-*.png`, `phase12-*.png`. After: `docs/design/screens/after/phase13-*.png`.

## Not done (honest list)
- Screen-reader walkthrough with VoiceOver; axe cannot judge reading order or announcement quality.
- Contrast of the *rendered* canvas cells (grid lines vs free cells are decorative by design, §3.3 of DESIGN_SYSTEM).
- Forced-colors / high-contrast mode and print styles.
