/**
 * Presentation helpers only. Statistics (means, CIs, tests) are computed by the
 * Python pipeline and shown verbatim; nothing here recomputes a study statistic.
 */
export const fmtNum = (x: number | null | undefined, digits = 3): string => (x == null || Number.isNaN(x) ? "–" : x.toFixed(digits));
export const fmtSigned = (x: number | null | undefined, digits = 3): string => (x == null ? "–" : `${x >= 0 ? "+" : ""}${x.toFixed(digits)}`);

/** 0, 500k, 1M, 1.5M ... (§30.8: never raw integers on a steps axis). */
export function fmtSteps(n: number): string {
  if (n === 0) return "0";
  if (Math.abs(n) >= 1e6) return `${trim(n / 1e6)}M`;
  if (Math.abs(n) >= 1e3) return `${trim(n / 1e3)}k`;
  return String(n);
}
const trim = (x: number): string => String(Math.round(x * 100) / 100);

export interface Interval {
  n: number;
  mean: number;
  std: number | null;
  lo: number | null;
  hi: number | null;
  conf: number;
}

/** "0.999 ± 0.002 [0.996, 1.002]"; a single seed has no interval and says so. */
export function fmtInterval(iv: Interval, digits = 3): string {
  const sd = iv.std == null ? "" : ` ± ${iv.std.toFixed(digits)}`;
  const ci = iv.lo == null || iv.hi == null ? " (n=1, no CI)" : ` [${iv.lo.toFixed(digits)}, ${iv.hi.toFixed(digits)}]`;
  return `${iv.mean.toFixed(digits)}${sd}${ci}`;
}

/** Round-number ticks covering [min, max] with about `count` intervals. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!(max > min)) return [min];
  const raw = (max - min) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

export function linear(domain: [number, number], range: [number, number]): (v: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

/** Trailing mean over `window` points (training reward smoothing; the window is stated in the caption). */
export function rollingMean(values: number[], window: number): number[] {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    out.push(sum / Math.min(i + 1, window));
  }
  return out;
}

export { sequential } from "../design/ramp";
