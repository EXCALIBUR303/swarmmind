/**
 * Rendering performance, measured with the Performance API (Phase 11 gate:
 * >= 50 fps at 1x on L2). Only frames drawn while the scene is ANIMATING are
 * counted, so an idle page cannot inflate the number. Exposed on
 * `window.__swarmmindPerf()` for browser verification; shown on screen with ?perf=1.
 */
const WINDOW = 120;
const deltas: number[] = [];
const draws: number[] = [];
let lastAnimFrame: number | null = null;
let total = 0;

export function recordAnimatedFrame(now: number, drawMs: number): void {
  if (lastAnimFrame !== null && now - lastAnimFrame < 250) {
    deltas.push(now - lastAnimFrame);
    if (deltas.length > WINDOW) deltas.shift();
  }
  lastAnimFrame = now;
  draws.push(drawMs);
  if (draws.length > WINDOW) draws.shift();
  total++;
}

export function resetPerf(): void {
  deltas.length = 0;
  draws.length = 0;
  lastAnimFrame = null;
  total = 0;
}

export interface PerfStats {
  frames: number;
  samples: number;
  fps: number | null;
  frameMsP95: number | null;
  drawMsMean: number | null;
  drawMsMax: number | null;
}

export function getPerfStats(): PerfStats {
  const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
  const sorted = [...deltas].sort((a, b) => a - b);
  const m = mean(deltas);
  return {
    frames: total,
    samples: deltas.length,
    fps: m ? 1000 / m : null,
    frameMsP95: sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] : null,
    drawMsMean: mean(draws),
    drawMsMax: draws.length ? Math.max(...draws) : null,
  };
}

declare global {
  interface Window {
    __swarmmindPerf?: () => PerfStats;
    __swarmmindPerfReset?: () => void;
  }
}

if (typeof window !== "undefined") {
  window.__swarmmindPerf = getPerfStats;
  window.__swarmmindPerfReset = resetPerf;
}
