import { cellCenter } from "../viewport";
import type { Renderer } from "./types";

export const drawPaths: Renderer = (ctx, { trails, drawPos, layers, focused }, vp, tk) => {
  if (!layers.paths) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  trails.forEach((trail, i) => {
    if (trail.length < 2) return;
    ctx.strokeStyle = tk.robots[i % tk.robots.length];
    ctx.globalAlpha = focused === null || focused === i ? 0.5 : 0.15;
    ctx.beginPath();
    trail.forEach(([y, x], k) => {
      const c = cellCenter(vp, y, x);
      if (k === 0) ctx.moveTo(c.px, c.py);
      else ctx.lineTo(c.px, c.py);
    });
    // finish at the (interpolated) robot so the line meets the disc mid-step
    const head = drawPos[i];
    if (head) {
      const c = cellCenter(vp, head[0], head[1]);
      ctx.lineTo(c.px, c.py);
    }
    ctx.stroke();
  });
  ctx.restore();
};
