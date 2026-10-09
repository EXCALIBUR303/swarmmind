import { cellOrigin } from "../viewport";
import type { Renderer } from "./types";

export const drawVisited: Renderer = (ctx, { meta, state, layers }, vp, tk) => {
  if (!layers.visited) return;
  ctx.fillStyle = tk.visitedTeam;
  for (let y = 0; y < meta.height; y++) {
    for (let x = 0; x < meta.width; x++) {
      if (state.visited[y * meta.width + x] > 0 && meta.terrain[y][x] === ".") {
        const { px, py } = cellOrigin(vp, y, x);
        ctx.fillRect(px + 1, py + 1, vp.cell - 2, vp.cell - 2);
      }
    }
  }
};
