import type { Cell, Frame, Layers, SimMeta, SimState } from "../model";
import type { RenderTokens } from "../tokens";
import type { Viewport } from "../viewport";

/** Everything one canvas frame needs. Renderers are pure `(ctx, view, vp, tokens) => void`. */
export interface RenderView {
  meta: SimMeta;
  state: SimState;
  drawPos: Cell[]; // interpolated, possibly fractional
  trails: Cell[][]; // per robot, grid path up to the current step
  layers: Layers;
  focused: number | null;
  lastFrame: Frame | null;
  /** Milliseconds since the displayed step became current (Infinity when settled or under reduced motion).
   * Drives the one-shot event effects with their own spec durations (§30.7, design/motion.ts). */
  sinceMs: number;
}

export type Renderer = (ctx: CanvasRenderingContext2D, view: RenderView, vp: Viewport, tokens: RenderTokens) => void;
