import { useEffect, useRef } from "react";
import { DEFAULT_LAYERS } from "../../stores/uiStore";
import { RENDER_ORDER } from "../simulation/renderers";
import { readTokens } from "../simulation/tokens";
import type { Timeline } from "../simulation/timeline";
import { buildRenderView } from "../simulation/view";
import { backingSize, computeViewport } from "../simulation/viewport";

/** A settled (non-animated) replay view driven by a parent's clock: used where several replays share one timeline. */
export function ReplayCanvas({ timeline, t, label }: { timeline: Timeline; t: number; label: string }) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const tRef = useRef(t);

  useEffect(() => {
    const w = wrap.current;
    const c = canvas.current;
    if (!w || !c) return;
    const draw = () => {
      const r = w.getBoundingClientRect();
      const vp = computeViewport(Math.max(1, r.width), Math.max(1, r.height), window.devicePixelRatio || 1, timeline.meta.width, timeline.meta.height);
      const { width, height } = backingSize(vp);
      if (c.width !== width) c.width = width;
      if (c.height !== height) c.height = height;
      c.style.width = `${vp.cssWidth}px`;
      c.style.height = `${vp.cssHeight}px`;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(vp.dpr, 0, 0, vp.dpr, 0, 0);
      const view = buildRenderView({ timeline, playhead: Math.min(tRef.current, timeline.maxT), live: false, alpha: 1, layers: { ...DEFAULT_LAYERS, sensor: false }, focused: null, reducedMotion: true });
      const tokens = readTokens();
      for (const d of RENDER_ORDER) d(ctx, view, vp, tokens);
    };
    draw();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(draw) : null;
    ro?.observe(w);
    (c as HTMLCanvasElement & { __redraw?: () => void }).__redraw = draw;
    return () => ro?.disconnect();
  }, [timeline]);

  useEffect(() => {
    tRef.current = t;
    (canvas.current as (HTMLCanvasElement & { __redraw?: () => void }) | null)?.__redraw?.();
  }, [t]);

  return (
    <div ref={wrap} style={{ position: "relative", width: "100%", aspectRatio: "1 / 1" }}>
      <canvas ref={canvas} role="img" aria-label={label} style={{ position: "absolute", inset: 0 }} />
    </div>
  );
}
