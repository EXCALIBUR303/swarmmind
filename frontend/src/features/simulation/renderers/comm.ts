import { MOTION, progress } from "../../../design/motion";
import { cellCenter } from "../viewport";
import type { Renderer } from "./types";

/** Communication links (comm modes A/C record `comm_edges` per frame; empty otherwise). */
export const drawComm: Renderer = (ctx, { drawPos, layers, lastFrame, sinceMs }, vp, tk) => {
  if (!layers.commLinks || !lastFrame || lastFrame.comm_edges.length === 0) return;
  ctx.save();
  ctx.strokeStyle = tk.commLink;
  // One opacity pulse when the link set appears (a merge), 300 ms (§30.7); static afterwards.
  const p = progress(sinceMs, MOTION.commPulseMs);
  ctx.globalAlpha = p < 1 ? 0.45 + 0.55 * Math.sin(Math.PI * p) : 1;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  for (const [i, j] of lastFrame.comm_edges) {
    if (!drawPos[i] || !drawPos[j]) continue;
    const a = cellCenter(vp, drawPos[i][0], drawPos[i][1]);
    const b = cellCenter(vp, drawPos[j][0], drawPos[j][1]);
    ctx.moveTo(a.px, a.py);
    ctx.lineTo(b.px, b.py);
  }
  ctx.stroke();
  ctx.restore();
};
