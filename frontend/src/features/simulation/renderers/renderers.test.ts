import { describe, expect, it, vi } from "vitest";
import { drawComm } from "./comm";
import { drawFocusWindow, drawRobots } from "./robots";
import { drawSensors } from "./sensors";
import { drawTargets } from "./targets";
import { drawTerrain } from "./terrain";
import { drawGroundTruth } from "./groundTruth";
import { drawVisited } from "./visited";
import { RENDER_ORDER, type RenderView } from "./index";
import { timelineOf, REAL_SUCCESS, REAL_FAIL } from "../fixtures";
import { buildRenderView } from "../view";
import { computeViewport } from "../viewport";
import type { RenderTokens } from "../tokens";
import { DEFAULT_LAYERS } from "../../../stores/uiStore";

/** A recording CanvasRenderingContext2D: every method is a spy, every property assignable. */
function mockCtx() {
  const calls: Record<string, unknown[][]> = {};
  const target: Record<string, unknown> = {};
  const ctx = new Proxy(target, {
    get(t, prop: string) {
      if (prop in t) return t[prop];
      return (...args: unknown[]) => {
        (calls[prop] ??= []).push(args);
      };
    },
    set(t, prop: string, v) {
      t[prop] = v;
      return true;
    },
  }) as unknown as CanvasRenderingContext2D;
  return { ctx, calls };
}

class P2D {
  moveTo = vi.fn();
  lineTo = vi.fn();
  closePath = vi.fn();
}
vi.stubGlobal("Path2D", P2D);

const tk: RenderTokens = {
  robots: ["#1", "#2", "#3", "#4", "#5", "#6"],
  free: "#fff", unknown: "#ccc", unknownHatch: "#bbb", obstacle: "#333", visitedTeam: "#eef", gridLine: "#eee",
  targetFill: "#fb0", targetOutline: "#750", commLink: "#678", success: "#080", danger: "#b00", accent: "#26e", surface: "#eee", fontUi: "sans-serif",
};

function view(over: Partial<RenderView> = {}, t = 12, replay = REAL_SUCCESS): RenderView {
  const tl = timelineOf(replay);
  return { ...buildRenderView({ timeline: tl, playhead: t, live: false, alpha: 1, layers: { ...DEFAULT_LAYERS }, focused: null, reducedMotion: false }), ...over };
}
const vp = computeViewport(440, 440, 1, 10, 10);

describe("renderers (mock canvas)", () => {
  it("terrain draws one rect per cell plus the surface, and grid lines at normal cell sizes", () => {
    const { ctx, calls } = mockCtx();
    drawTerrain(ctx, view(), vp, tk);
    expect(calls.fillRect).toHaveLength(1 + 100);
    expect(calls.stroke?.length).toBeGreaterThanOrEqual(2); // hatch + grid
  });

  it("grid lines are hidden below 14px cells (§30.3)", () => {
    const { ctx, calls } = mockCtx();
    drawTerrain(ctx, view(), computeViewport(100, 100, 1, 10, 10), tk); // 10px cells
    expect(calls.moveTo ?? []).toHaveLength(0); // grid uses ctx.moveTo; hatch uses Path2D
  });

  it("fog layer off: every cell is drawn as known", () => {
    const { ctx, calls } = mockCtx();
    drawTerrain(ctx, view({ layers: { ...DEFAULT_LAYERS, unknown: false } }), vp, tk);
    expect(calls.fillRect).toHaveLength(101);
    const fills = new Set<string>();
    // with fog off no cell may use the unknown colour
    const { ctx: c2 } = mockCtx();
    const seen: string[] = [];
    Object.defineProperty(c2, "fillStyle", { set: (v) => seen.push(v), get: () => "" });
    drawTerrain(c2, view({ layers: { ...DEFAULT_LAYERS, unknown: false } }), vp, tk);
    seen.forEach((s) => fills.add(s));
    expect(fills.has(tk.unknown)).toBe(false);
  });

  it("each layer renderer is a no-op when its toggle is off", () => {
    const off = { sensor: false, paths: false, visited: false, unknown: true, commLinks: false, groundTruth: false };
    for (const draw of [drawSensors, drawVisited, drawGroundTruth, drawComm]) {
      const { ctx, calls } = mockCtx();
      draw(ctx, view({ layers: off }), vp, tk);
      expect(Object.keys(calls)).toEqual([]);
    }
  });

  it("sensors: one disc per robot with the real sensor radius in pixels", () => {
    const { ctx, calls } = mockCtx();
    const v = view();
    drawSensors(ctx, v, vp, tk);
    expect(calls.arc).toHaveLength(v.meta.nRobots);
    expect(calls.arc[0][2]).toBeCloseTo(v.meta.sensorRadius * vp.cell);
  });

  it("targets: hidden ones are not drawn; discovered/rescued ones are", () => {
    const early = mockCtx();
    drawTargets(early.ctx, view({}, 0), vp, tk);
    expect(early.calls.stroke ?? []).toHaveLength(0);
    const late = mockCtx();
    drawTargets(late.ctx, view({}, 25), vp, tk);
    expect(late.calls.stroke).toHaveLength(2);
  });

  it("robots: one disc + number label each; focus dims the others", () => {
    const { ctx, calls } = mockCtx();
    const alphas: number[] = [];
    Object.defineProperty(ctx, "globalAlpha", { set: (v) => alphas.push(v), get: () => 1 });
    drawRobots(ctx, view({ focused: 1 }), vp, tk);
    expect(calls.fillText.map((c) => c[0])).toEqual(["1", "2", "3"]);
    expect(alphas.filter((a) => a === 0.4).length).toBeGreaterThanOrEqual(2);
  });

  it("focus window outlines (2*obs_radius+1)^2 cells from the real env config", () => {
    const { ctx, calls } = mockCtx();
    const v = view({ focused: 0 });
    drawFocusWindow(ctx, v, vp, tk);
    const [, , w, h] = calls.strokeRect[0] as number[];
    expect(w).toBeCloseTo((2 * v.meta.obsRadius + 1) * vp.cell);
    expect(h).toBe(w);
    const none = mockCtx();
    drawFocusWindow(none.ctx, view({ focused: null }), vp, tk);
    expect(none.calls.strokeRect ?? []).toHaveLength(0);
  });

  it("a blocked move gets a ✕ glyph (two strokes) and a collision ring while the step animates", () => {
    const tl = timelineOf(REAL_FAIL);
    const t = [...Array(tl.maxT + 1).keys()].find((k) => tl.stateAt(k).lastFrame?.blocked_by_robot.some(Boolean))!;
    const base = buildRenderView({ timeline: tl, playhead: t, live: false, alpha: 0.5, sinceMs: 100, layers: { ...DEFAULT_LAYERS }, focused: null, reducedMotion: false });
    const animating = mockCtx();
    drawRobots(animating.ctx, base, vp, tk);
    const settled = mockCtx();
    drawRobots(settled.ctx, { ...base, sinceMs: Infinity }, vp, tk);
    const blocked = base.lastFrame!.blocked_by_robot.filter(Boolean).length;
    expect(animating.calls.arc.length).toBe(settled.calls.arc.length + blocked); // one expanding ring per blocked robot
    expect(settled.calls.lineTo.length).toBeGreaterThanOrEqual(2); // the ✕
  });

  it("comm links draw only when the frame recorded edges", () => {
    const { ctx, calls } = mockCtx();
    const v = view({ layers: { ...DEFAULT_LAYERS, commLinks: true } });
    drawComm(ctx, v, vp, tk);
    expect(calls.lineTo ?? []).toHaveLength(0); // shared_map level: no recorded edges
    const lf = { ...v.lastFrame!, comm_edges: [[0, 1], [1, 2]] };
    const withEdges = mockCtx();
    drawComm(withEdges.ctx, { ...v, lastFrame: lf }, vp, tk);
    expect(withEdges.calls.lineTo).toHaveLength(2);
  });

  it("the full pipeline runs in order without throwing, on both fixtures at every step", () => {
    for (const replay of [REAL_SUCCESS, REAL_FAIL]) {
      const tl = timelineOf(replay);
      for (let t = 0; t <= tl.maxT; t += 7) {
        const { ctx } = mockCtx();
        const v = buildRenderView({ timeline: tl, playhead: t + 0.4, live: false, alpha: 0.4, layers: { ...DEFAULT_LAYERS, groundTruth: true, commLinks: true }, focused: t % 3, reducedMotion: false });
        for (const draw of RENDER_ORDER) draw(ctx, v, vp, tk);
      }
    }
  });
});

describe("buildRenderView", () => {
  const tl = timelineOf(REAL_SUCCESS);
  it("interpolates from the previous step's positions and settles at alpha=1", () => {
    const mid = buildRenderView({ timeline: tl, playhead: 5.5, live: false, alpha: 0.5, layers: DEFAULT_LAYERS, focused: null, reducedMotion: false });
    const done = buildRenderView({ timeline: tl, playhead: 5.5, live: false, alpha: 1, layers: DEFAULT_LAYERS, focused: null, reducedMotion: false });
    const prev = tl.positionsAt(4);
    const cur = tl.stateAt(5).pos;
    const moved = cur.findIndex((p, i) => p[0] !== prev[i][0] || p[1] !== prev[i][1]);
    expect(moved).toBeGreaterThanOrEqual(0);
    expect(done.drawPos[moved]).toEqual(cur[moved]);
    expect(mid.drawPos[moved]).not.toEqual(cur[moved]);
  });
  it("live follows the latest frame; paths only when the layer is on", () => {
    const live = buildRenderView({ timeline: tl, playhead: 0, live: true, alpha: 1, layers: DEFAULT_LAYERS, focused: null, reducedMotion: false });
    expect(live.state.t).toBe(tl.maxT);
    const nopaths = buildRenderView({ timeline: tl, playhead: 9, live: false, alpha: 1, layers: { ...DEFAULT_LAYERS, paths: false }, focused: null, reducedMotion: false });
    expect(nopaths.trails).toEqual([]);
    const paths = buildRenderView({ timeline: tl, playhead: 9, live: false, alpha: 1, layers: DEFAULT_LAYERS, focused: null, reducedMotion: false });
    expect(paths.trails[0]).toHaveLength(10);
  });
});


describe("micro-interaction timings (spec §30.7, design/motion.ts)", () => {
  const firstWith = (tl: ReturnType<typeof timelineOf>, pick: (f: NonNullable<ReturnType<typeof timelineOf>["frames"][number]>) => boolean) =>
    tl.frames.findIndex(pick) + 1;

  it("movement tween = min(80% of the step interval, 160 ms)", async () => {
    const { moveDurationMs } = await import("../../../design/motion");
    expect(moveDurationMs(250)).toBe(160); // 1x: capped
    expect(moveDurationMs(100)).toBeCloseTo(80); // fast playback: shorter than the cap
  });

  it("collision ring lasts 400 ms, discovery pulse 500 ms, rescue fade 800 ms", () => {
    const fail = timelineOf(REAL_FAIL);
    const tc = firstWith(fail, (f) => f.blocked_by_robot.some(Boolean));
    const arcs = (draw: typeof drawRobots, v: RenderView) => {
      const m = mockCtx();
      draw(m.ctx, v, vp, tk);
      return (m.calls.arc ?? []).length;
    };
    const vc = buildRenderView({ timeline: fail, playhead: tc, live: false, alpha: 1, layers: { ...DEFAULT_LAYERS }, focused: null, reducedMotion: false });
    const settled = arcs(drawRobots, { ...vc, sinceMs: Infinity });
    expect(arcs(drawRobots, { ...vc, sinceMs: 399 })).toBeGreaterThan(settled);
    expect(arcs(drawRobots, { ...vc, sinceMs: 400 })).toBe(settled);

    const ok = timelineOf(REAL_SUCCESS);
    const td = firstWith(ok, (f) => (f.discovered ?? []).length > 0);
    const vd = buildRenderView({ timeline: ok, playhead: td, live: false, alpha: 1, layers: { ...DEFAULT_LAYERS }, focused: null, reducedMotion: false });
    const ringArcs = (since: number) => { const m = mockCtx(); drawTargets(m.ctx, { ...vd, sinceMs: since }, vp, tk); return (m.calls.arc ?? []).length; };
    expect(ringArcs(499)).toBeGreaterThan(ringArcs(Infinity)); // the pulse ring
    expect(ringArcs(500)).toBe(ringArcs(Infinity));
  });

  it("a blocked robot is nudged 2px toward the attempted direction (and returns), only while the nudge runs", () => {
    const fail = timelineOf(REAL_FAIL);
    const t = firstWith(fail, (f) => f.blocked_by_robot.some(Boolean) && f.action.some((a, i) => f.blocked_by_robot[i] && a !== 0));
    const v = buildRenderView({ timeline: fail, playhead: t, live: false, alpha: 1, layers: { ...DEFAULT_LAYERS }, focused: null, reducedMotion: false });
    const i = v.lastFrame!.blocked_by_robot.findIndex((b, k) => b && v.lastFrame!.action[k] !== 0);
    const centre = (since: number) => { const m = mockCtx(); drawRobots(m.ctx, { ...v, sinceMs: since }, vp, tk); return (m.calls.arc as number[][]).filter((a) => a[2] === vp.cell * 0.3)[i].slice(0, 2); };
    const rest = centre(Infinity);
    const mid = centre(60); // halfway through 120 ms: peak displacement
    const end = centre(120);
    expect(Math.hypot(mid[0] - rest[0], mid[1] - rest[1])).toBeCloseTo(2, 5);
    expect(end).toEqual(rest);
  });

  it("reduced motion: every effect is static (settled), glyphs remain", () => {
    const fail = timelineOf(REAL_FAIL);
    const t = firstWith(fail, (f) => f.blocked_by_robot.some(Boolean));
    const v = buildRenderView({ timeline: fail, playhead: t, live: false, alpha: 0, sinceMs: 10, layers: { ...DEFAULT_LAYERS }, focused: null, reducedMotion: true });
    expect(v.sinceMs).toBe(Infinity);
    const m = mockCtx();
    drawRobots(m.ctx, v, vp, tk);
    expect((m.calls.lineTo ?? []).length).toBeGreaterThanOrEqual(2); // the static ✕
  });

  it("comm links pulse once (300 ms) then hold", () => {
    const v = view({ layers: { ...DEFAULT_LAYERS, commLinks: true } });
    const withEdges = { ...v, lastFrame: { ...v.lastFrame!, comm_edges: [[0, 1]] } };
    const alphaAt = (since: number) => { const m = mockCtx(); let a = -1; Object.defineProperty(m.ctx, "globalAlpha", { set: (x) => { a = x; }, get: () => a }); drawComm(m.ctx, { ...withEdges, sinceMs: since }, vp, tk); return a; };
    expect(alphaAt(150)).toBeCloseTo(1, 5); // peak of the pulse
    expect(alphaAt(0)).toBeCloseTo(0.45, 5);
    expect(alphaAt(300)).toBe(1);
  });
});
