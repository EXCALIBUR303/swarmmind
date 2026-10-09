import type { RunSummary } from "../../api/endpoints/analytics";

export type SortKey = "status" | "algo" | "level" | "seed" | "trained_env_steps" | "final_val" | "start_time";
export interface Filters {
  status: string;
  algo: string;
  level: string;
  study: string;
}
export const EMPTY_FILTERS: Filters = { status: "", algo: "", level: "", study: "" };

export const finalVal = (r: RunSummary): number | null => r.final_val_metrics?.success_rate ?? null;

export function filterRuns(runs: RunSummary[], f: Filters): RunSummary[] {
  return runs.filter((r) => (!f.status || r.status === f.status) && (!f.algo || r.algo === f.algo) && (!f.level || r.level === f.level) && (!f.study || r.study === f.study));
}

/** Stable sort; missing values always sink to the bottom whichever direction is chosen. */
export function sortRuns(runs: RunSummary[], key: SortKey, dir: "asc" | "desc"): RunSummary[] {
  const val = (r: RunSummary): string | number | null => (key === "final_val" ? finalVal(r) : ((r as unknown as Record<string, string | number | null | undefined>)[key] ?? null));
  const sign = dir === "asc" ? 1 : -1;
  return runs
    .map((r, i) => ({ r, i, v: val(r) }))
    .sort((a, b) => {
      if (a.v === null && b.v === null) return a.i - b.i;
      if (a.v === null) return 1;
      if (b.v === null) return -1;
      const c = typeof a.v === "number" && typeof b.v === "number" ? a.v - b.v : String(a.v).localeCompare(String(b.v));
      return c !== 0 ? sign * c : a.i - b.i;
    })
    .map((x) => x.r);
}

export const distinct = (runs: RunSummary[], k: "status" | "algo" | "level" | "study"): string[] => [...new Set(runs.map((r) => r[k]))].sort();
