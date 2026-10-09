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
  /** 0..1 progress through the current step while animating; 1 when settled or under reduced motion.
   * Drives the one-shot event effects (collision ring, discovery pulse, rescue fade, §30.7). */
  stepProgress: number;
}

export type Renderer = (ctx: CanvasRenderingContext2D, view: RenderView, vp: Viewport, tokens: RenderTokens) => void;
