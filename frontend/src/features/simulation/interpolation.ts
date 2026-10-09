/** Smooth robot motion between two discrete steps (§30.7). Pure and unit-tested. */
import type { Cell } from "./model";

export const easeOutCubic = (a: number): number => 1 - Math.pow(1 - Math.min(Math.max(a, 0), 1), 3);

/**
 * Positions at fraction `alpha` of the way from `prev` to `cur`. A robot that
 * did not move, or that "teleports" more than one cell (a seek / reset, not a
 * step), is drawn at `cur` — never smeared across the map.
 */
export function interpolatePositions(prev: readonly Cell[], cur: readonly Cell[], alpha: number, reducedMotion = false): Cell[] {
  const a = reducedMotion ? 1 : easeOutCubic(alpha);
  return cur.map(([cy, cx], i) => {
    const p = prev[i];
    if (!p || (p[0] === cy && p[1] === cx)) return [cy, cx] as Cell;
    if (Math.abs(p[0] - cy) + Math.abs(p[1] - cx) > 1) return [cy, cx] as Cell;
    return [p[0] + (cy - p[0]) * a, p[1] + (cx - p[1]) * a] as Cell;
  });
}
