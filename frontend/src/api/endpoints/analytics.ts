import { postJson, request } from "../client";
import type { Interval } from "../../lib/stats";

// ---- studies (Algorithm Comparison / Swarm Analysis) -------------------------
export interface MetricStat {
  per_seed: number[];
  interval: Interval;
}
export type ModeStats = { n_seeds: number; n_episodes_per_seed: number[] } & Record<string, MetricStat | number | number[]>;
export interface ValidationCurve {
  grid_env_steps: number[];
  seeds: number[];
  per_seed: number[][];
  mean: number[];
  lo: number[];
  hi: number[];
  auc_x_max: number;
  auc: Interval;
}
export interface MethodSummary {
  seeds_completed?: number[];
  splits: Record<string, Record<string, ModeStats> | { stochastic?: ModeStats }>;
  validation_curve?: ValidationCurve;
}
export interface Difference {
  n_a: number;
  n_b: number;
  diff: number;
  lo: number | null;
  hi: number | null;
  df: number | null;
  p_value: number | null;
  hedges_g: number | null;
  conf: number;
  ci_excludes_zero: boolean;
}
export interface Hypothesis {
  verdict: string;
  statement: string;
  [k: string]: unknown;
}
export interface LevelSummary {
  methods: Record<string, MethodSummary>;
  comparisons: Record<string, Record<string, Record<string, Difference>>>;
  hypotheses: Record<string, Hypothesis | Record<string, Hypothesis>>;
  coordination?: { protocol: Record<string, unknown>; methods: Record<string, Record<string, Interval | number>> };
}
export interface StudySummary {
  study: string;
  smoke: boolean;
  generated: string;
  primary: Record<string, unknown>;
  levels: Record<string, LevelSummary>;
  missing: unknown[];
}
export interface StudyDetail {
  summary: StudySummary;
  manifest: Record<string, unknown> | null;
  analysis_files: string[];
}
export interface StudyListItem {
  study: string;
  smoke: boolean | null;
  generated: string | null;
  levels: string[];
}

export interface ExampleEpisode {
  map_id: string;
  success: boolean;
  final_t: number;
  coverage_over_time: number[];
  der_over_time: number[];
  inter_robot_distance_over_time: { mean: number; min: number }[];
  discovery_timeline: { target: number; t_discovered: number | null; t_rescued: number | null; rescued_by: number | null }[];
  workload_share_per_robot: number[];
  workload_gini: number;
  idle_fraction_per_robot: number[];
  distance_travelled_per_robot: number[];
}
export interface HeatmapData {
  study: string;
  level: string;
  algo: string;
  map_id: string;
  kind: "visits" | "collisions";
  n_episodes: number;
  obstacles: boolean[][];
  counts: number[][];
}

export const studiesApi = {
  list: () => request<StudyListItem[]>("/api/studies"),
  get: (id: string) => request<StudyDetail>(`/api/studies/${encodeURIComponent(id)}`),
  analysis: <T>(id: string, name: string) => request<T>(`/api/studies/${encodeURIComponent(id)}/analysis/${encodeURIComponent(name)}`),
  heatmap: (id: string, level: string, algo: string, mapId: string, kind: "visits" | "collisions") =>
    request<HeatmapData>(`/api/analysis/${encodeURIComponent(id)}/heatmap?level=${level}&algo=${algo}&map_id=${encodeURIComponent(mapId)}&kind=${kind}`),
  rules: () => request<{ source: string; rules: { id: string; text: string }[] }>("/api/preregistration"),
};

// ---- runs (Experiment History / Train Swarm) ------------------------------
export interface RunSummary {
  run_id: string;
  study: string;
  algo: string;
  level: string;
  seed?: number | null;
  status: string;
  total_env_steps?: number | null;
  trained_env_steps?: number | null;
  budget_env_steps?: number | null;
  updates?: number | null;
  start_time?: string | null;
  end_time?: string | null;
  wall_time_seconds?: number | null;
  final_val_metrics?: Record<string, number> | null;
  has_actor?: boolean;
}
export interface RunDetail {
  summary: RunSummary;
  manifest: Record<string, unknown> | null;
}
export type MetricRow = Record<string, number>;

export const runsApi = {
  list: () => request<RunSummary[]>("/api/experiments"),
  get: (id: string) => request<RunDetail>(`/api/experiments/${encodeURIComponent(id)}`),
  metrics: (id: string, from = 0) => request<MetricRow[]>(`/api/experiments/${encodeURIComponent(id)}/metrics?from_update=${from}`),
  valEval: (id: string) =>
    request<{ validation_history: MetricRow[] } & Record<string, unknown>>(`/api/experiments/${encodeURIComponent(id)}/eval`),
};

// ---- training jobs ------------------------------------------------------------
export interface TrainSchema {
  levels: string[];
  algos: string[];
  seed: { min: number; max: number };
  total_env_steps: { min: number; max: number };
  overrides: Record<string, { type: "int" | "float"; min: number; max: number }>;
}
export interface TrainRequestBody {
  level: string;
  algo: string;
  seed: number;
  total_env_steps: number;
  overrides: Record<string, number>;
}
export interface Job {
  job_id: string;
  status: "starting" | "running" | "completed" | "failed" | "cancelled";
  run_id: string | null;
  request: TrainRequestBody;
  created_at: string;
  finished_at: string | null;
  exit_code: number | null;
  run_status: string | null;
  progress: { update: number | null; env_steps: number | null; total_env_steps: number } | null;
  log_tail: string[] | null;
}
export const trainingApi = {
  schema: () => request<TrainSchema>("/api/training/schema"),
  launch: (body: TrainRequestBody) => postJson<Job>("/api/training/jobs", body),
  jobs: () => request<Job[]>("/api/training/jobs"),
  job: (id: string) => request<Job>(`/api/training/jobs/${encodeURIComponent(id)}`),
  cancel: (id: string) => postJson<Job>(`/api/training/jobs/${encodeURIComponent(id)}/cancel`),
};
