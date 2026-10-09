import { cellCenter } from "../viewport";
import type { Renderer } from "./types";

export const drawSensors: Renderer = (ctx, { meta, drawPos, layers, focused }, vp, tk) => {
  if (!layers.sensor) return;
  ctx.save();
  ctx.setLineDash([3, 3]);
  drawPos.forEach(([y, x], i) => {
    const { px, py } = cellCenter(vp, y, x);
    const colour = tk.robots[i % tk.robots.length];
    ctx.globalAlpha = focused === null || focused === i ? 1 : 0.4;
    ctx.beginPath();
    ctx.arc(px, py, meta.sensorRadius * vp.cell, 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.globalAlpha *= 0.08;
    ctx.fill();
    ctx.globalAlpha = (focused === null || focused === i ? 1 : 0.4) * 0.35;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1;
    ctx.stroke();
  });
  ctx.restore();
};
