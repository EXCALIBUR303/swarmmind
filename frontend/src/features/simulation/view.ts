/** Turns (store state, clock) into the `RenderView` one canvas frame draws. Pure and unit-tested. */
import { interpolatePositions } from "./interpolation";
import type { Layers } from "./model";
import type { RenderView } from "./renderers";
import type { Timeline } from "./timeline";

export interface ViewInput {
  timeline: Timeline;
  playhead: number; // replay: fractional step; live: ignored (follows the latest frame)
  live: boolean;
  alpha: number; // 0..1 progress of the current step (already time-derived)
  layers: Layers;
  focused: number | null;
  reducedMotion: boolean;
}

export function buildRenderView(i: ViewInput): RenderView {
  const { timeline } = i;
  const curT = i.live ? timeline.maxT : Math.min(Math.max(Math.floor(i.playhead), timeline.startT), timeline.maxT);
  const state = timeline.stateAt(curT);
  const prev = timeline.positionsAt(curT - 1 < timeline.startT ? curT : curT - 1);
  const alpha = curT >= timeline.maxT && !i.live ? 1 : i.alpha;
  const drawPos = interpolatePositions(prev, state.pos, alpha, i.reducedMotion);
  const trails = i.layers.paths ? state.pos.map((_, r) => timeline.trail(r, curT)) : [];
  return {
    meta: timeline.meta,
    state,
    drawPos,
    trails,
    layers: i.layers,
    focused: i.focused,
    lastFrame: state.lastFrame,
    stepProgress: i.reducedMotion ? 1 : Math.min(Math.max(alpha, 0), 1),
  };
}
