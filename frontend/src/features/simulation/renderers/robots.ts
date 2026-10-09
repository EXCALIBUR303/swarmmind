import { MOTION, progress } from "../../../design/motion";
import { ACTION_DELTA } from "../model";
import { cellCenter, cellOrigin } from "../viewport";
import type { Renderer } from "./types";

/** The focused robot's own observation window: (2*obs_radius+1)² cells, from the real env config. */
export const drawFocusWindow: Renderer = (ctx, { meta, drawPos, focused }, vp, tk) => {
  if (focused === null || !drawPos[focused]) return;
  const [y, x] = drawPos[focused];
  const { px, py } = cellOrigin(vp, y - meta.obsRadius, x - meta.obsRadius);
  const side = (2 * meta.obsRadius + 1) * vp.cell;
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = tk.robots[focused % tk.robots.length];
  // Clip to the grid: the window is partly off-map near walls.
  ctx.beginPath();
  ctx.rect(vp.offsetX, vp.offsetY, vp.cols * vp.cell, vp.rows * vp.cell);
  ctx.clip();
  ctx.strokeRect(px, py, side, side);
  ctx.restore();
};

export const drawRobots: Renderer = (ctx, { drawPos, focused, lastFrame, sinceMs }, vp, tk) => {
  drawPos.forEach(([y, x], i) => {
    const c0 = cellCenter(vp, y, x);
    let px = c0.px;
    let py = c0.py;
    const dim = focused !== null && focused !== i;
    ctx.save();
    ctx.globalAlpha = dim ? 0.4 : 1;
    const r = vp.cell * 0.3;
    const collided = !!lastFrame && lastFrame.blocked_by_robot[i];
    const pColl = progress(sinceMs, MOTION.collisionMs);
    if (collided && pColl < 1) {
      // Collision: a danger ring expands 0 -> 0.6 cell and fades (400 ms in the spec).
      ctx.save();
      ctx.globalAlpha = (dim ? 0.4 : 1) * (1 - pColl);
      ctx.strokeStyle = tk.danger;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, r + vp.cell * 0.6 * pColl, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    if (lastFrame && (lastFrame.blocked_by_robot[i] || lastFrame.bumped_wall[i])) {
      // Blocked move: nudge 2px toward the attempted direction and back (120 ms, §30.7).
      const [dy, dx] = ACTION_DELTA[lastFrame.action[i]] ?? [0, 0];
      const n = Math.sin(Math.PI * progress(sinceMs, MOTION.nudgeMs)) * MOTION.nudgePx;
      px += dx * n;
      py += dy * n;
    }
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = tk.robots[i % tk.robots.length];
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = `600 ${Math.max(9, Math.round(vp.cell * 0.28))}px ${tk.fontUi}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(i + 1), px, py + 0.5);
    // A blocked move gets a ✕ glyph (shape, not just colour) in the danger token.
    if (lastFrame && (lastFrame.blocked_by_robot[i] || lastFrame.bumped_wall[i])) {
      const k = r * 0.55;
      const cx = px + r * 0.95;
      const cy = py - r * 0.95;
      ctx.strokeStyle = tk.danger;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx - k, cy - k);
      ctx.lineTo(cx + k, cy + k);
      ctx.moveTo(cx + k, cy - k);
      ctx.lineTo(cx - k, cy + k);
      ctx.stroke();
    }
    ctx.restore();
  });
};
