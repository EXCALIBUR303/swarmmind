import { DISCOVERED, RESCUED } from "../model";
import { cellCenter } from "../viewport";
import type { Renderer } from "./types";

export function starPath(cx: number, cy: number, r: number): Path2D {
  const p = new Path2D();
  for (let i = 0; i < 10; i++) {
    const a = (Math.PI / 5) * i - Math.PI / 2;
    const rr = i % 2 === 0 ? r : r * 0.45;
    const x = cx + rr * Math.cos(a);
    const y = cy + rr * Math.sin(a);
    if (i === 0) p.moveTo(x, y);
    else p.lineTo(x, y);
  }
  p.closePath();
  return p;
}

/** Discovered targets are filled stars; rescued ones are faded outlines (shape + opacity, not hue alone). */
export const drawTargets: Renderer = (ctx, { meta, state, lastFrame, stepProgress }, vp, tk) => {
  meta.targets.forEach(([y, x], k) => {
    const status = state.targetStatus[k];
    if (status !== DISCOVERED && status !== RESCUED) return;
    const { px, py } = cellCenter(vp, y, x);
    const justFound = !!lastFrame && lastFrame.discovered.includes(k) && stepProgress < 1;
    const justRescued = !!lastFrame && lastFrame.rescued.some(([rk]) => rk === k) && stepProgress < 1;
    // Discovery: star scales 0.6 -> 1.0 with one gold pulse ring. Rescue: fades to a hollow grey star (§30.7).
    const scale = justFound ? 0.6 + 0.4 * stepProgress : 1;
    const star = starPath(px, py, vp.cell * 0.3 * scale);
    if (justFound) {
      ctx.save();
      ctx.globalAlpha = 1 - stepProgress;
      ctx.strokeStyle = tk.targetFill;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, vp.cell * (0.3 + 0.4 * stepProgress), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    ctx.save();
    ctx.globalAlpha = status === RESCUED ? (justRescued ? 1 - 0.65 * stepProgress : 0.35) : 1;
    if (status === DISCOVERED || justRescued) {
      ctx.fillStyle = tk.targetFill;
      ctx.fill(star);
    }
    ctx.strokeStyle = tk.targetOutline;
    ctx.lineWidth = 1.5;
    ctx.stroke(star);
    ctx.restore();
  });
};
