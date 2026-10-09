import { Link } from "react-router-dom";
import type { Job } from "../../api/endpoints/analytics";
import { fmtSteps } from "../../lib/stats";
import { formatEta } from "./measure";

const BADGE: Record<Job["status"], string> = { starting: "warn", running: "ok", completed: "ok", failed: "bad", cancelled: "warn" };

export function JobCard({ job, sps, onCancel }: { job: Job; sps: number | null; onCancel: () => void }) {
  const done = job.progress?.env_steps ?? 0;
  const total = job.progress?.total_env_steps ?? job.request.total_env_steps;
  const active = job.status === "starting" || job.status === "running";
  return (
    <section className="sm-hyp" aria-label="Job status">
      <div className="sm-hyp-head">
        <strong>{job.request.algo.toUpperCase()} · {job.request.level} · seed {job.request.seed}</strong>
        <span className={`sm-badge ${BADGE[job.status]}`}>{job.status}</span>
      </div>
      <div>
        <progress value={done} max={total} aria-label="Training progress" style={{ width: "100%" }} />
        <div className="sm-section-note" style={{ margin: 0 }}>
          {fmtSteps(done)} / {fmtSteps(total)} env steps
          {active && (sps ? ` · ${sps.toFixed(0)} steps/s measured · ETA ${formatEta((total - done) / sps)}` : " · measuring speed…")}
        </div>
      </div>
      {job.run_id && <div className="sm-section-note" style={{ margin: 0 }}>Run <Link className="sm-link" to={`/experiments/${job.run_id}`}>{job.run_id}</Link>{job.run_status ? ` (${job.run_status})` : ""}</div>}
      {active && <button className="sm-btn danger" onClick={onCancel}>Cancel job</button>}
      {(job.status === "failed" || job.status === "cancelled") && job.log_tail && job.log_tail.length > 0 && (
        <details open={job.status === "failed"}>
          <summary className="sm-section-note" style={{ margin: 0, cursor: "pointer" }}>Last lines of the job's output{job.exit_code != null ? ` (exit code ${job.exit_code})` : ""}</summary>
          <pre className="sm-log">{job.log_tail.join("\n")}</pre>
        </details>
      )}
    </section>
  );
}
