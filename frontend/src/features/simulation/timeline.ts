/**
 * Frame store with a keyframe cache every `KEYFRAME_EVERY` steps (spec §27.4),
 * so scrubbing to any t costs at most KEYFRAME_EVERY applications instead of t.
 *
 * `base` is the state at `base.t` (0 for a replay or a fresh live session; >0
 * when a live client resyncs mid-episode after a reconnect and has not seen the
 * earlier frames). Frame i advances base.t + i to base.t + i + 1.
 */
import { applyFrame, cloneState } from "./frameReducer";
import type { Cell, Frame, SimMeta, SimState } from "./model";

export const KEYFRAME_EVERY = 25;

export type TimelineEvent = { t: number; kind: "discover" | "rescue" | "collision"; robot?: number };

export class Timeline {
  readonly meta: SimMeta;
  readonly frames: Frame[] = [];
  private readonly base: SimState;
  private readonly keyframes: SimState[] = []; // keyframes[k] is the state at base.t + k*KEYFRAME_EVERY
  private cursor: SimState; // last state handed out, for O(1) forward stepping
  private _events: TimelineEvent[] = [];
  /** Team coverage after each step: coverageSeries[i] is the value at t = startT + i. */
  readonly coverageSeries: number[] = [];
  private knownFreeNow: number;

  constructor(meta: SimMeta, base: SimState, frames: Frame[] = []) {
    this.meta = meta;
    this.base = cloneState(base);
    this.keyframes.push(cloneState(base));
    this.cursor = cloneState(base);
    this.knownFreeNow = base.knownFree;
    this.coverageSeries.push(this.ratio());
    for (const f of frames) this.append(f);
  }

  private ratio(): number {
    return this.meta.nFree === 0 ? 0 : this.knownFreeNow / this.meta.nFree;
  }

  get startT(): number {
    return this.base.t;
  }
  get maxT(): number {
    return this.base.t + this.frames.length;
  }
  get events(): readonly TimelineEvent[] {
    return this._events;
  }

  append(frame: Frame): void {
    const expected = this.maxT + 1;
    if (frame.t !== expected) throw new Error(`frame out of order: expected t=${expected}, got ${frame.t}`);
    this.frames.push(frame);
    for (const [y, x] of frame.newly_known) if (this.meta.terrain[y][x] === ".") this.knownFreeNow++;
    this.coverageSeries.push(this.ratio());
    for (const _k of frame.discovered) this._events.push({ t: frame.t, kind: "discover" });
    for (const [, r] of frame.rescued) this._events.push({ t: frame.t, kind: "rescue", robot: r });
    frame.blocked_by_robot.forEach((b, r) => {
      if (b) this._events.push({ t: frame.t, kind: "collision", robot: r });
    });
    // Build keyframes eagerly as frames arrive so scrubbing never pays the full cost.
    const relT = this.frames.length;
    if (relT % KEYFRAME_EVERY === 0) this.keyframes.push(cloneState(this.stateAt(this.base.t + relT)));
  }

  /** State after `t` steps. The returned object is shared: do not mutate it. */
  stateAt(t: number): SimState {
    const target = Math.min(Math.max(Math.round(t), this.base.t), this.maxT);
    if (target === this.cursor.t) return this.cursor;
    let from: SimState;
    if (target > this.cursor.t && target - this.cursor.t <= KEYFRAME_EVERY) {
      from = this.cursor; // cheap forward step
    } else {
      const k = Math.min(Math.floor((target - this.base.t) / KEYFRAME_EVERY), this.keyframes.length - 1);
      from = cloneState(this.keyframes[k]);
    }
    for (let i = from.t - this.base.t; from.t < target; i++) applyFrame(this.meta, from, this.frames[i]);
    this.cursor = from;
    return from;
  }

  /** Positions at t (cheap; used for interpolation's "previous" side). */
  positionsAt(t: number): Cell[] {
    const rel = Math.min(Math.max(Math.round(t), this.base.t), this.maxT) - this.base.t;
    return rel === 0 ? this.base.pos : this.frames[rel - 1].pos.map(([y, x]) => [y, x] as Cell);
  }

  /** A robot's path from the earliest known position up to step t. */
  trail(robot: number, t: number): Cell[] {
    const end = Math.min(Math.max(Math.round(t), this.base.t), this.maxT) - this.base.t;
    const out: Cell[] = [this.base.pos[robot]];
    for (let i = 0; i < end; i++) out.push(this.frames[i].pos[robot] as Cell);
    return out;
  }
}
