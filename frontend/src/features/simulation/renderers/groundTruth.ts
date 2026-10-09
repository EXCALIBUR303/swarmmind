import { cellOrigin } from "../viewport";
import type { Renderer } from "./types";

/** Debug overlay: what the robots do NOT know (true obstacles under fog, hidden targets).
 * Off by default and labelled "Not visible to robots" in the Layers tab (§30.4). */
export const drawGroundTruth: Renderer = (ctx, { meta, state, layers }, vp, tk) => {
  if (!layers.groundTruth) return;
  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = tk.danger;
  ctx.lineWidth = 1;
  for (let y = 0; y < meta.height; y++) {
    for (let x = 0; x < meta.width; x++) {
      if (meta.terrain[y][x] === "#" && !state.known[y * meta.width + x]) {
        const { px, py } = cellOrigin(vp, y, x);
        ctx.strokeRect(px + 2, py + 2, vp.cell - 4, vp.cell - 4);
      }
    }
  }
  meta.targets.forEach(([y, x], k) => {
    if (state.targetStatus[k] === 0) {
      const { px, py } = cellOrigin(vp, y, x);
      ctx.strokeRect(px + 3, py + 3, vp.cell - 6, vp.cell - 6);
    }
  });
  ctx.restore();
};
