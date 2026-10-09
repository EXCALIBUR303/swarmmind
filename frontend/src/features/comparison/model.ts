/** Pure selectors over the study summary. Values are passed through verbatim from summary.json. */
import type { Difference, LevelSummary, MetricStat, ModeStats, StudySummary } from "../../api/endpoints/analytics";

export const METHODS = ["random", "heuristic", "ippo", "mappo"] as const;
export type Method = (typeof METHODS)[number];
export const METHOD_LABEL: Record<Method, string> = { random: "Random", heuristic: "Frontier heuristic", ippo: "IPPO", mappo: "MAPPO" };
/** Learned algorithms are the only categorical series; Random/heuristic are neutral references (DESIGN_SYSTEM §3.1). */
export const METHOD_COLOR: Record<Method, string> = {
  random: "var(--sm-algo-random)",
  heuristic: "var(--sm-algo-heuristic)",
  ippo: "var(--sm-algo-ippo)",
  mappo: "var(--sm-algo-mappo)",
};

export interface MetricDef {
  key: string;
  label: string;
  unit: string;
  digits: number;
  better: "higher" | "lower";
  domain?: [number, number];
}
export const METRICS: MetricDef[] = [
  { key: "success_rate", label: "Success rate", unit: "fraction of episodes", digits: 3, better: "higher", domain: [0, 1] },
  { key: "censored_steps", label: "Steps to completion (censored)", unit: "steps (failures count max_steps)", digits: 2, better: "lower" },
  { key: "steps_on_success", label: "Steps on success", unit: "steps", digits: 2, better: "lower" },
  { key: "coverage", label: "Coverage", unit: "fraction of free cells known", digits: 3, better: "higher", domain: [0, 1] },
  { key: "return_mean", label: "Episode return", unit: "mean per robot", digits: 2, better: "higher" },
  { key: "rescue_rate", label: "Rescue rate", unit: "fraction of targets", digits: 3, better: "higher", domain: [0, 1] },
  { key: "collisions_per_100_agent_steps", label: "Collisions", unit: "per 100 agent-steps", digits: 2, better: "lower" },
  { key: "exploration_efficiency", label: "Exploration efficiency", unit: "new cells per agent-step", digits: 3, better: "higher" },
  { key: "idle_fraction", label: "Idle fraction", unit: "fraction of steps", digits: 3, better: "lower", domain: [0, 1] },
  { key: "oscillation_rate", label: "Oscillation rate", unit: "fraction of steps", digits: 3, better: "lower", domain: [0, 1] },
  { key: "workload_gini", label: "Workload Gini", unit: "0 = even, 1 = one robot", digits: 3, better: "lower", domain: [0, 1] },
];
export const PRIMARY_KEYS = ["success_rate", "censored_steps", "coverage", "return_mean", "collisions_per_100_agent_steps", "rescue_rate"];

/** `splits[split][mode]`, tolerating methods that lack it (a study with a missing arm). */
export function modeStats(level: LevelSummary, method: Method, split: string, mode: string): ModeStats | null {
  const sp = level.methods[method]?.splits?.[split] as Record<string, ModeStats> | undefined;
  return sp?.[mode] ?? null;
}
export const metricOf = (m: ModeStats | null, key: string): MetricStat | null => {
  const v = m?.[key];
  return v && typeof v === "object" && "interval" in v ? (v as MetricStat) : null;
};

export function availableSplits(level: LevelSummary): string[] {
  const s = new Set<string>();
  for (const m of Object.values(level.methods)) for (const k of Object.keys(m.splits ?? {})) s.add(k);
  return ["validation", "test", "structured"].filter((k) => s.has(k));
}
export function availableModes(level: LevelSummary, split: string): string[] {
  const s = new Set<string>();
  for (const m of Object.values(level.methods)) for (const k of Object.keys(m.splits?.[split] ?? {})) s.add(k);
  return ["stochastic", "greedy"].filter((k) => s.has(k));
}

export function comparisonRows(level: LevelSummary, split: string, mode: string): { pair: string; metric: string; d: Difference }[] {
  const block = level.comparisons[`${split}/${mode}`];
  if (!block) return [];
  return Object.entries(block).flatMap(([pair, metrics]) => Object.entries(metrics).map(([metric, d]) => ({ pair, metric, d })));
}

export function levelOf(summary: StudySummary, level: string): LevelSummary | null {
  return summary.levels[level] ?? null;
}

export function badgeClass(verdict: string): "ok" | "bad" | "warn" {
  return verdict === "supported" ? "ok" : verdict === "not_supported" ? "bad" : "warn";
}
