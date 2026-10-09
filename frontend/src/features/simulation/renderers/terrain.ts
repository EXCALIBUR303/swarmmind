import { cellOrigin } from "../viewport";
import type { Renderer } from "./types";

/** Cells: free / obstacle / unknown (fog, hatched so it isn't colour-only), plus grid lines. */
export const drawTerrain: Renderer = (ctx, { meta, state, layers }, vp, tk) => {
  ctx.fillStyle = tk.surface;
  ctx.fillRect(0, 0, vp.cssWidth, vp.cssHeight);

  const hatch = new Path2D();
  for (let y = 0; y < meta.height; y++) {
    for (let x = 0; x < meta.width; x++) {
      const { px, py } = cellOrigin(vp, y, x);
      const isObstacle = meta.terrain[y][x] === "#";
      const known = !layers.unknown || state.known[y * meta.width + x] === 1;
      if (known) {
        ctx.fillStyle = isObstacle ? tk.obstacle : tk.free;
      } else {
        ctx.fillStyle = tk.unknown;
      }
      ctx.fillRect(px, py, vp.cell, vp.cell);
      if (!known) {
        const s = vp.cell;
        for (let d = s / 3; d < 2 * s; d += s / 3) {
          // 45° hatch clipped to the cell by endpoint math
          const x0 = Math.min(d, s);
          const y0 = d - x0;
          const x1 = d - Math.min(d, s);
          const y1 = Math.min(d, s);
          hatch.moveTo(px + x0, py + y0);
          hatch.lineTo(px + x1, py + y1);
        }
      }
    }
  }
  ctx.strokeStyle = tk.unknownHatch;
  ctx.lineWidth = 1;
  ctx.stroke(hatch);

  if (vp.cell >= 14) {
    // Grid lines are decorative cell-boundary hints (hidden below 14px per §30.3).
    ctx.strokeStyle = tk.gridLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= meta.width; x++) {
      const { px, py } = cellOrigin(vp, 0, x);
      ctx.moveTo(px, py);
      ctx.lineTo(px, py + meta.height * vp.cell);
    }
    for (let y = 0; y <= meta.height; y++) {
      const { px, py } = cellOrigin(vp, y, 0);
      ctx.moveTo(px, py);
      ctx.lineTo(px + meta.width * vp.cell, py);
    }
    ctx.stroke();
  }
};
