export { fmtSteps } from "../../lib/stats";
export const pctText = (x: number): string => `${Math.round(x * 100)}%`;

import type { ReplayIndexEntry } from "../../api/endpoints/catalog";

/** Per checkpoint fraction, the replay of one run on one map; in training order. */
export function evolutionSet(index: ReplayIndexEntry[], runId: string, mapId: string): ReplayIndexEntry[] {
  const byFrac = new Map<number, ReplayIndexEntry>();
  for (const r of index) if (r.run_id === runId && r.map_id === mapId) byFrac.set(r.fraction, r);
  return [...byFrac.values()].sort((a, b) => a.fraction - b.fraction);
}
