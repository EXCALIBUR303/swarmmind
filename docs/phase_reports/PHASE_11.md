# PHASE 11 COMPLETE — React simulator (Swarm Lab + replay)

## IMPLEMENTED
- **Canvas 2D simulator**, outside React state: one rAF loop reads the stores via `getState()`, draws only when something changed or is animating, handles `devicePixelRatio` (backing store = 2× CSS at dpr 2) and resizes with a `ResizeObserver`. Nine pure renderers `(ctx, view, viewport, tokens)`: terrain + hatched fog, visited, ground truth (debug), comm links, paths, targets, sensors, focus window (the focused robot's real `(2·obs_radius+1)²` window), robots (✕ glyph on a blocked move).
- **`frameReducer` + `Timeline`** (spec §27.4): delta accumulation with a keyframe every 25 steps; coverage series built incrementally; event markers (◆ discovery, ★ rescue, ✕ collision) derived from the frames. Never re-simulates.
- **Replay playback** (client clock, 0.25×–8×), **live sessions** (REST create → WebSocket; the server owns the clock; the client only sends play/pause/step/reset/speed), **WS client with reconnect** (exponential backoff, no retry on 4403/4404/4409), resync after missed frames.
- **UI:** toolbar (mode, policy/step/map selectors fed by `/api/replays`, `/api/models`, `/api/maps`), mission strip, timeline with scrubber, right rail (Robots with observable facts only / Live coverage sparkline + the frame's real reward components / Layers), scenario drawer over the rail, keyboard (`Space → ← R 1–6 L`), loading/empty/error states, approved nav (graded background, anime.js entrance, reduced-motion honoured), responsive per §30.5.
- **Contract types** generated from the backend: `npm run gen:types` (`types.gen.ts`, `wire.gen.ts` from `docs/openapi.json` + JSON Schemas); `npm run build` fails if they are stale or any mock import exists.
- The Phase 9 prototype is removed (promoted).

## FILES CREATED/CHANGED
`frontend/src/{api/{client,ws,types.gen,wire.gen}.ts, api/endpoints/*, stores/{simStore,uiStore,scenarioStore}.ts, features/simulation/*, features/replay/Timeline.tsx, components/{Select,StateViews}.tsx, lib/*, pages/SwarmLab.{tsx,css}, app/{AppShell.tsx,AppShell.css,routes.tsx}}`, `frontend/scripts/gen_types.mjs`, `frontend/package.json`, `docs/design/{UI_SPEC,DESIGN_SYSTEM}.md`, `docs/design/screens/phase11-*.png`. Backend: WS rejections now accept-then-close with an application code (a pre-handshake refusal reaches browsers only as 1006); tests updated.

## TESTS RUN
- **vitest: 88 passed** (87 + 1 opt-in e2e). Coverage areas: reducer on two **real** replays (final state == recorded outcome; every step identical in any access order across keyframes; initial-cell rule), timeline/events/resync, viewport & hit-testing on a **non-square** grid (no x/y transposition), interpolation, 9 renderers with a mock canvas context, stores (replay clock, live frames, resync, reset, command sink), WS client (backoff, fatal codes, give-up, close), components (toolbar, timeline, rail, drawer, mission strip), and page-level flows with a stubbed API (auto-load, keyboard, map switch, Live ↔ Replay, reconnect/expired, backend down, no replays, failed load).
- **Opt-in e2e against the live backend: all 56 recorded study replays reduce exactly to their own outcomes** (steps, rescued, discovered, collisions, coverage).
- `tsc -b`, `oxlint`: clean. `npm run build`: clean (mock guard OK, types fresh, no mock modules in the bundle). Backend: 144 tests still pass.

## RESULTS (gate)
| Gate item | Result |
|---|---|
| Live session, real MAPPO (s0, L2), 1× | **59.5–60 fps** (Performance API, dpr 2); draw 0.25 ms mean, 0.5 ms max, p95 frame 17.6 ms |
| Replay of a real episode plays and scrubs | play/pause/step ±1/scrub verified in the browser; mission strip t/coverage/collisions consistent |
| Robot focus, layers | focus dims others + outlines the real 11×11 window; 6 layer toggles |
| Reconnect | killed the backend mid-session: banner "Connection lost. Reconnecting…", status DISCONNECTED (never RUNNING); after restart the client learned the session was gone (4404) and offered "Start a new session" |
| Screenshots | 1440 / 1024 / 768 / 375 in `docs/design/screens/phase11-swarm-lab-*.png`; zero horizontal overflow at all four (measured) |

## PROBLEMS FOUND / FIXED
1. **Fog bug from the Phase 9 prototype confirmed and fixed** (initial known cells now come from `/replays/{id}/initial`; a test proves skipping them leaves cells unknown forever).
2. A test found a real crash: key events whose target is not an `HTMLElement` threw in the keyboard handler. Fixed.
3. The mission strip said **RUNNING while disconnected**. Now DISCONNECTED.
4. WS refusals before the handshake reached the browser as 1006, so "session expired" looked like "backend down". Fixed server-side (accept, then close with 4403/4404).
5. Layout: a gap above the rail at 375 px (padding on the wrong element), clipped "Algorithm Comparison" nav label (now wraps), clipped "SWARMM" title at 64 px (now "SM").

## REMAINING CONCERNS
- The fps was measured in the Claude browser pane (60 Hz, dpr 2); I did not test a 120 Hz display or a throttled CPU.
- **Micro-interactions are partial.** Implemented: robot tween, collision ring, discovery pulse/scale, rescue fade, ✕ glyph. Their durations follow the step interval (≈250 ms at 1×), not the spec's fixed 400/500/800 ms; the 2 px blocked-move nudge and the comm-link pulse are not implemented. I did not invoke the `emil-animate` / `animation-principles` skills.
- **Reduced motion** is implemented (no tween/animation, instant nav) and the pure parts are unit-tested, but I did not run the browser with the OS setting on.
- **Comm links** render only for frames that carry `comm_edges`; no real limited-range (L4) replay exists yet, so that path is tested with synthetic edges only.
- A live client that reconnects mid-episode keeps the server's counts but loses earlier path history and per-robot distances (restart from the resync point).
- A target discovered by the reset-time sensing pass would not show (impossible for the shipped levels: `target_min_dist 3 > sensor_radius 2`; documented in `frameReducer.ts`).
- At 1440 px the playback group wraps to a second toolbar row (functional, a little unbalanced); no axe audit yet. Both are Phase 13.
- The 768 px "bottom tab sheet" is still the approved stacked panel.
- Settings/other pages are still Phase-1 placeholders (Phase 12/13).

## VALIDATION GATE
**PASS** (tests pass; build clean with mock guard; live MAPPO session ≥ 50 fps; real replay plays and scrubs; screenshots attached).

## CURRENT PROJECT STATUS
Phases 0–10 committed (Phase 10 = `065798a`). **Phase 11 uncommitted.**

## NEXT PHASE
Phase 12 — Analytics + training UI (Train Swarm, Algorithm Comparison, Swarm Analysis incl. Swarm Evolution, Experiment History) from real endpoints.

## RECOMMENDED MODEL
Sonnet 5.5

## RECOMMENDED EFFORT
Medium

## WHY
Data-to-chart work against endpoints and a design system that already exist; the hard problems (contract, reducer, canvas) are solved.

## USE OPUS ONLY IF
Never expected.

## RECOMMENDED SKILLS / CONNECTORS
`dataviz` skill (mandatory read before any chart code), `data-visualization`, browser tools for verification.

## ACTION REQUIRED
Reply **CONTINUE** for Phase 12; say **commit** if you want Phase 11 committed first.

## STOP HERE.
