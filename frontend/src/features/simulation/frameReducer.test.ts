import { describe, expect, it } from "vitest";
import { applyFrame, cloneState, coverage, createInitialState, robotFact } from "./frameReducer";
import { framesOf, metaOf, REAL_FAIL, REAL_SUCCESS, timelineOf } from "./fixtures";
import { DISCOVERED, RESCUED } from "./model";
import { KEYFRAME_EVERY } from "./timeline";

describe.each([
  ["real MAPPO success replay", REAL_SUCCESS],
  ["real untrained failure replay", REAL_FAIL],
])("frameReducer on the %s", (_name, replay) => {
  const meta = metaOf(replay);

  it("final state equals the replay's recorded outcome", () => {
    const s = createInitialState(meta, replay.initial!.known_team);
    for (const f of framesOf(replay)) applyFrame(meta, s, f);
    const m = replay.outcome.metrics as Record<string, number>;
    expect(s.t).toBe(replay.outcome.steps);
    expect(s.rescued).toBe(m.n_rescued);
    expect(s.discovered).toBe(m.n_discovered);
    expect(s.collisions).toBe(m.n_collisions);
    expect(coverage(meta, s)).toBeCloseTo(m.coverage_final, 10);
  });

  it("every state matches sequential application, in any access order (keyframes)", () => {
    const tl = timelineOf(replay);
    const seq = createInitialState(meta, replay.initial!.known_team);
    const expected = [cloneState(seq)];
    for (const f of framesOf(replay)) {
      applyFrame(meta, seq, f);
      expected.push(cloneState(seq));
    }
    const order = [...expected.keys()];
    // forward, backward, then a deterministic shuffle: jumps across keyframe boundaries
    const shuffled = order.map((_, i) => order[(i * 37 + 11) % order.length]);
    for (const t of [...order, ...order.slice().reverse(), ...shuffled]) {
      const got = tl.stateAt(t);
      const want = expected[t];
      expect(got.t).toBe(want.t);
      expect(got.pos).toEqual(want.pos);
      expect(Array.from(got.known)).toEqual(Array.from(want.known));
      expect(Array.from(got.visited)).toEqual(Array.from(want.visited));
      expect(got.collisions).toBe(want.collisions);
      expect(got.rescued).toBe(want.rescued);
      expect(got.knownFree).toBe(want.knownFree);
    }
  });

  it("the timeline's incremental coverage series equals the reducer's coverage at every step", () => {
    const tl = timelineOf(replay);
    expect(tl.coverageSeries).toHaveLength(tl.maxT + 1);
    for (let t = 0; t <= tl.maxT; t++) expect(tl.coverageSeries[t]).toBeCloseTo(coverage(meta, tl.stateAt(t)), 12);
  });

  it("coverage is monotone non-decreasing and bounded", () => {
    const tl = timelineOf(replay);
    let prev = -1;
    for (let t = 0; t <= tl.maxT; t++) {
      const c = coverage(meta, tl.stateAt(t));
      expect(c).toBeGreaterThanOrEqual(prev);
      expect(c).toBeLessThanOrEqual(1);
      prev = c;
    }
  });
});

describe("initial known cells", () => {
  it("skipping `initial` under-reports coverage at t=0 and leaves cells unknown forever", () => {
    const meta = metaOf(REAL_SUCCESS);
    const withInit = createInitialState(meta, REAL_SUCCESS.initial!.known_team);
    const without = createInitialState(meta, []);
    expect(withInit.knownFree).toBeGreaterThan(0);
    expect(without.knownFree).toBe(0);
    for (const f of framesOf(REAL_SUCCESS)) {
      applyFrame(meta, withInit, f);
      applyFrame(meta, without, f);
    }
    const missing = Array.from(withInit.known).filter((k, i) => k && !without.known[i]).length;
    expect(missing).toBeGreaterThan(0); // start-area cells that no delta ever reveals
  });

  it("start cells are counted as visited at t=0", () => {
    const meta = metaOf(REAL_SUCCESS);
    const s = createInitialState(meta, REAL_SUCCESS.initial!.known_team);
    for (const [y, x] of meta.robotStarts) expect(s.visited[y * meta.width + x]).toBeGreaterThan(0);
  });
});

describe("targets and robot facts", () => {
  it("tracks discovered then rescued, with the rescuing robot", () => {
    const meta = metaOf(REAL_SUCCESS);
    const tl = timelineOf(REAL_SUCCESS);
    const final = tl.stateAt(tl.maxT);
    expect(Array.from(final.targetStatus)).toEqual([RESCUED, RESCUED]);
    const rescuer = REAL_SUCCESS.frames.flatMap((f) => f.rescued ?? []);
    for (const [k, r] of rescuer) expect(final.rescuedBy[k]).toBe(r);
    const firstDisc = REAL_SUCCESS.frames.findIndex((f) => (f.discovered ?? []).length > 0) + 1;
    expect(tl.stateAt(firstDisc).targetStatus.some((s) => s === DISCOVERED || s === RESCUED)).toBe(true);
    expect(tl.stateAt(0).targetStatus.every((s) => s === 0)).toBe(true);
    expect(meta.targets).toHaveLength(2);
  });

  it("facts are observable states only", () => {
    const tl = timelineOf(REAL_SUCCESS);
    const facts = new Set<string>();
    for (let t = 0; t <= tl.maxT; t++) for (let r = 0; r < 3; r++) facts.add(robotFact(tl.stateAt(t), r).replace(/T\d+ @ t=\d+/, "T# @ t=#"));
    for (const f of facts) expect(f).toMatch(/^(Waiting|Moving|Blocked|Rescued T# @ t=#)$/);
    expect(facts.has("Rescued T# @ t=#")).toBe(true);
  });
});

describe("timeline", () => {
  it("builds keyframes every KEYFRAME_EVERY steps and rejects out-of-order frames", () => {
    expect(KEYFRAME_EVERY).toBe(25);
    const tl = timelineOf(REAL_FAIL);
    expect(tl.maxT).toBe(150);
    expect(() => tl.append(framesOf(REAL_FAIL)[0])).toThrow(/out of order/);
  });

  it("derives event markers from the frames (discover / rescue / collision)", () => {
    const tl = timelineOf(REAL_FAIL);
    expect(tl.events.filter((e) => e.kind === "collision")).toHaveLength(15);
    const ok = timelineOf(REAL_SUCCESS);
    expect(ok.events.filter((e) => e.kind === "rescue")).toHaveLength(2);
    expect(ok.events.filter((e) => e.kind === "discover")).toHaveLength(2);
  });

  it("a resynced live timeline (base.t > 0) indexes frames correctly", () => {
    const meta = metaOf(REAL_SUCCESS);
    const frames = framesOf(REAL_SUCCESS);
    const mid = createInitialState(meta, REAL_SUCCESS.initial!.known_team);
    for (const f of frames.slice(0, 10)) applyFrame(meta, mid, f);
    const tl = new (timelineOf(REAL_SUCCESS).constructor as typeof import("./timeline").Timeline)(meta, mid, frames.slice(10));
    expect(tl.startT).toBe(10);
    expect(tl.maxT).toBe(25);
    const full = timelineOf(REAL_SUCCESS).stateAt(25);
    expect(tl.stateAt(25).rescued).toBe(full.rescued);
    expect(Array.from(tl.stateAt(25).known)).toEqual(Array.from(full.known));
    expect(tl.trail(0, 25)).toHaveLength(16);
  });

  it("clamps out-of-range t", () => {
    const tl = timelineOf(REAL_SUCCESS);
    expect(tl.stateAt(-5).t).toBe(0);
    expect(tl.stateAt(999).t).toBe(25);
  });
});
