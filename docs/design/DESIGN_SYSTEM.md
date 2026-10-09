# SWARMMIND design system (Phase 9)

**Status: APPROVED by the user (spec §30.10).** This document, `UI_SPEC.md`,
`frontend/src/design/tokens.css`, and the static prototype together form the
design package Phase 11 builds from. Nothing here may change after approval
except through a new, documented design decision.

Spec §30 is the **binding baseline**. This document does not repeat it in
full; it records what Phase 9 verified, fixed, and elaborated on top of it.
Where this document is silent, §30 governs directly.

## 1. What Phase 9 did

1. Ran every categorical colour in §30.3 through the dataviz skill's
   validator (OKLCH lightness band, chroma floor, CVD separation under
   Machado-Oliveira-Fernandes 2009 simulated protanopia/deuteranopia/
   tritanopia, a normal-vision floor, and contrast vs. surface) plus WCAG
   2.2 text and non-text (graphical) contrast checks — nothing in §30.3 had
   actually been run through either check before now.
2. Found and fixed two real failures (§3). Documented one open limitation
   rather than hiding it (§3.3).
3. Wrote the full component/state/props inventory and the per-page
   specification (`UI_SPEC.md`) that §30 describes only at the layout level.
4. Built a static Swarm Lab prototype from a **real MAPPO replay** (Phase 7's
   `main_L2` study, seed 0's near-final checkpoint, a successful 25-step
   episode on a structured map) — no mock data.
5. Ran a design critique pass against the reference-image critique already
   in §30.1 and against the dataviz skill's anti-patterns catalogue.

## 2. Information architecture, layout, tokens, interactions, responsive rules, a11y

Unchanged from spec §30.2 (IA), §30.4 (Swarm Lab layout), §30.5 (responsive
breakpoints), §30.6 (other pages), §30.7 (micro-interactions), §30.8
(data-viz rules), §30.9 (accessibility). Read those sections directly; this
document does not restate them.

## 3. Colour validation results

All commands below use the dataviz skill's `scripts/validate_palette.js`,
run from that skill's base directory. Full transcripts are in this phase's
session log; the headline results:

### 3.1 Algorithm colours — FIXED

The categorical set for algorithm identity is **IPPO vs. MAPPO only**.
Random and the Heuristic reference are neutral reference lines (grey solid /
black dashed), not categorical series — per the dataviz skill's own scope
note, a lone reference colour takes a WCAG contrast check, not the
categorical six-check validator. This matches how every dataviz figure in
this project already renders them (Phases 5–8).

```
node validate_palette.js "#0E7490,#2563EB" --mode light --pairs all
→ FAIL: chroma floor (#0E7490 = 0.094, below the 0.10 floor — reads as grey)
```

Snapped one step up the same teal/cyan hue family, per the skill's
snap-to-passing method (same hue, move lightness/chroma until the floor
clears):

```
node validate_palette.js "#0891B2,#2563EB" --mode light --pairs all
→ ALL CHECKS PASS (chroma 0.10+ both; CVD ΔE 14.7 worst deutan; normal-
  vision floor 16.3, comfortably above the 15 gate)
```

`--sm-algo-ippo` is now `#0891B2` (Tailwind cyan-600). `--sm-algo-mappo`
stays `#2563EB`.

**Accepted, not fixed:** MAPPO's colour is identical to `--sm-accent`, the
UI's primary interactive colour (buttons, links, focus rings). This was a
deliberate choice in the Phase 0 spec, not an oversight, and it is common
practice to reuse a brand's primary hue for its flagship series. The
mitigation the spec already requires — a legend and direct labels on every
chart (§30.8) — makes the practical ambiguity low. Re-theming the entire
app's primary accent this late would be a much larger, out-of-scope change
for a phase whose job is to refine §30, not rewrite it. Flagged here so it
is a recorded decision, not a silent one.

### 3.2 Robot identity colours — FIXED (for the shipped range), LIMITED beyond it

Robots can appear anywhere relative to each other on the canvas, so the
correct check is `--pairs all`, not just adjacent pairs.

```
node validate_palette.js "#0072B2,#D55E00,#009E73,#CC79A7" --mode light --pairs all   # robots 1-4
→ ALL CHECKS PASS or WARN-in-the-legal-band:
  - CVD: worst pair (#CC79A7↔#009E73) ΔE 7.6 — the 6-8 WARN band, legal
    only with secondary encoding, which robots always have (mandatory
    labels, §30.1/§30.9).
  - Contrast: #CC79A7 (robot 4, pink) at 2.98:1 vs. white — WARN, "relief
    required (visible labels or table view)" — again already satisfied by
    the mandatory label.
  - Normal-vision floor: 16.4, clears the hard 15 gate.
```

No shipped level uses more than 4 robots (L1=2, L2=3, L3/L4=4), so **robots
1-4 are validated for every level this product actually ships.**

Robot 5 (`#56B4E9`) independently failed a real, unconditional check —
graphical (non-text) contrast against a free cell — at 2.31:1, well under
the 3:1 WCAG floor. This is not a CVD-pairing issue, it is simply too light
to see. Darkened along the same hue:

```
contrast("#2E93CB", "#FFFFFF") = 3.41:1  (PASS)
```

**Open limitation, disclosed rather than hidden:** at the full 6-robot set,
`--pairs all` cannot clear the CVD normal-vision floor (robot 6 `#7A5195`
vs. robot 1 `#0072B2`, ΔE 13.7 < 15) no matter how robot 5/6 are re-stepped
— re-darkening robot 5 to fix its contrast pushed it into collision with
robot 1's blue instead. This matches the dataviz skill's own documented
finding that not even its curated 8-hue reference palette clears an
all-pairs floor of 15 past three slots; 6-way pairwise CVD safety is not
achievable by re-ordering or re-stepping alone. **No level currently ships
more than 4 robots**, so this never affects a real user today. If a level
with 5-6 robots is ever added, the fix is a **secondary encoding** (a
numeral badge or distinct outline pattern per robot), not another colour
search — robots are already required to be labelled unconditionally
(§30.1: "never distinguished by colour alone"), so this is a defence in
depth for an edge case, not a live accessibility bug.

### 3.3 Text, semantic, and simulation-layer contrast — all pass, two scoping notes

Every text/semantic pair (body text, muted text, nav text-on-dark, the four
status colours' text form, the accent's white-on-blue button form) clears
WCAG AA (≥ 4.5:1), most by a wide margin (5.0–17.7:1).

Two pairs read as "failing" a naive check but are **out of scope for the
non-text-contrast rule**, and are noted here so nobody "fixes" a
correctly-designed pair later:
- **Grid lines vs. free cells** (1.24:1): grid lines are decorative cell-
  boundary hints, not information-bearing — cells are already distinguished
  by fill (white/hatched-grey/dark), matching WCAG 1.4.11's scope ("graphical
  objects required to understand content"), which a faint divider is not.
  This is also why §30.3 already hides them below 14px cells.
- **Target fill vs. free cell** (1.83:1): the target's own **1.5px dark
  outline** (`#7A5A00`, 6.38:1) is the graphical-contrast mechanism — a
  filled shape with a well-contrasted outline satisfies the non-text
  contrast rule at its boundary, which is exactly why §30.3 specifies both
  a fill and an outline for targets rather than a fill alone.

## 4. Component inventory (see `UI_SPEC.md` for full props/states)

Toolbar, Mission Strip, Simulation Canvas, Right Rail (Robots / Live /
Layers tabs), Timeline (replay), Scenario Drawer, Nav, Stat Tile, Dot-and-
whisker plot, Small-multiple chart, Data table, Empty/Loading/Error state.
Every component's exact props and states are in `UI_SPEC.md`.

## 5. Design critique pass

Checked against §30.1's own reference critique (already resolved by the
Phase 0 spec) and the dataviz skill's anti-patterns catalogue:

- **No dual-axis charts** anywhere in the spec's chart list (§30.6 explicitly
  splits reward+success into small multiples) — confirmed, not reintroduced.
- **No rainbow/cycled categorical colours** — algorithm and robot palettes
  are both fixed-order, validated sets (§3), never generated per-series.
- **No 3D, no gradients inside data marks** — none specified; the prototype
  introduces none.
- **The canvas dominates** (≥55% width at desktop, confirmed in the built
  prototype's actual layout, not just the spec's ASCII diagram) — see
  `docs/design/screens/desktop-1440.png`.
- **Every chart gets a legend + direct labels rule carries through** to the
  robot-4/robot-5 WARN-band colours (§3.2) — this is not decorative, it is
  the thing that makes those colours legal.
- **Status colours stay reserved**: `--sm-danger`/`--sm-success` are used
  only for collision/rescue events, never repurposed as a "4th algorithm"
  or "3rd robot" colour.

No anti-pattern from the catalogue was found unaddressed.

## 5.1 What building the prototype found (not just asserted)

Building a real, working static prototype — rather than only writing this
document — surfaced three genuine responsive-layout bugs in §30.5's rules
that a design doc alone would not have caught, all now fixed and re-verified
by re-rendering at the same breakpoint:

1. **<768px: the sticky-bottom toolbar clipped content instead of showing
   it.** `position: sticky` combined with a `flex order` reassignment (to
   visually move the toolbar to the bottom without reordering the DOM)
   interacted badly with the flex column's `overflow: hidden`, hiding the
   mission strip above the fold with no way to scroll to it. Fixed by
   making the toolbar a genuine `position: fixed` bar with normal document
   flow and reserved padding above it, instead of the sticky+order
   combination.
2. **<768px: the simulation canvas overflowed the viewport horizontally.**
   The canvas SVG had a fixed pixel size (44px/cell × 10 cells = 440px),
   wider than a 375px viewport, and several ancestors defaulted to
   `min-width: auto` (the standard flex/grid "won't shrink below content"
   trap), which no single `max-width: 100%` on the SVG alone fixed. Fixed
   by making the SVG scale via its `viewBox` (`width: 100%`, `height:
   auto`) and adding `min-width: 0` at every ancestor in the chain.
3. **<768px: the canvas floated with large, confusing empty gaps above and
   below it instead of sitting under the mission strip.** The canvas
   wrapper's `flex: 1` (which correctly lets it fill leftover space and
   center nicely at desktop) instead consumed a CSS Grid row that had
   stretched to fill the full viewport height once the toolbar left normal
   flow. Fixed by dropping the grow at <768px, so the mobile page stacks
   and scrolls (matching "canvas full-width square", not "canvas
   vertically centered with dead space").

All three were verified by direct DOM measurement (`getBoundingClientRect`,
zero horizontal overflow at 375/768/1024/1440px) in addition to the
screenshots in `docs/design/screens/`, not by eyeballing alone.

**One remaining minor, disclosed item:** at 375px there is a small,
cosmetic vertical gap between the timeline and the Robots-tab section
below it (the right rail's own grid row still stretches slightly). It does
not hide or block any content — every element remains reachable by
scrolling — and is a one-line polish fix for whoever builds the real Phase
11 component; it does not block this design's approval.

## 5.2 Nav redesign (requested during review)

The left nav's items were enlarged (`--sm-font-md` → `--sm-font-lg`, icons
18px → 20px, more padding) and given a colour-graded background: a
`linear-gradient` from a faint `--sm-accent` tint at the top through
`--sm-bg-nav` to a deepened navy at the bottom, built by mixing the
existing nav-surface and accent tokens (`color-mix`) rather than
introducing a new arbitrary colour. The active item now reads as an
accent-tinted gradient pill with a left accent border instead of a flat
8%-white fill; hover lifts the row slightly and nudges its icon. An
entrance animation (anime.js — `animate`/`stagger`/`createScope`) fades
and slides the wordmark and nav items in once on mount; it is skipped
entirely (items render at their settled state, zero duration) under
`prefers-reduced-motion` (§30.9), and the effect's `Scope` is reverted on
unmount so it never runs a second time or leaks. `docs/design/screens/*`
were re-captured after this change; nothing else in the layout moved.

## 6. Approval (granted)

Per §30.10, Phase 11 (React simulator) does not start until the user
explicitly approves this design system and the Swarm Lab spec. See the
Phase 9 completion report for what specifically needs approval and the
open limitation (§3.2) that ships regardless of the answer.
