import type { Job } from "../../api/endpoints/analytics";

export function formatEta(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "–";
  if (seconds < 90) return `${Math.round(seconds)} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

export interface Measure {
  jobId: string;
  t0: number; // ms, when this page first saw progress
  steps0: number;
  sps: number | null;
}

/**
 * Steps/s MEASURED from the progress this page has observed (never an a-priori estimate). Needs >= 3 s of
 * observation and positive progress before it reports a number; the ETA follows from it.
 */
export function nextMeasure(prev: Measure | null, job: Job, nowMs: number): Measure | null {
  const steps = job.progress?.env_steps;
  if (steps == null) return prev && prev.jobId === job.job_id ? prev : null;
  if (!prev || prev.jobId !== job.job_id) return { jobId: job.job_id, t0: nowMs, steps0: steps, sps: null };
  const dt = (nowMs - prev.t0) / 1000;
  return dt >= 3 && steps > prev.steps0 ? { ...prev, sps: (steps - prev.steps0) / dt } : prev;
}
