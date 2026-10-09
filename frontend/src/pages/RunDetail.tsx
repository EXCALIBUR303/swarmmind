import { Link, useParams } from "react-router-dom";
import { runsApi } from "../api/endpoints/analytics";
import { catalogApi } from "../api/endpoints/catalog";
import { Page, Section } from "../components/PageLayout";
import { ErrorState, LoadingState } from "../components/StateViews";
import { TrainingCharts } from "../features/training/TrainingCharts";
import { algoLabel, kSteps } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { StatusBadge } from "../components/StatusBadge";
import "./pages.css";

const scalar = (v: unknown): v is string | number | boolean | null => v === null || ["string", "number", "boolean"].includes(typeof v);

export function RunDetail() {
  const { runId = "" } = useParams();
  const detail = useAsync(() => runsApi.get(runId), [runId]);
  const metrics = useAsync(() => runsApi.metrics(runId), [runId]);
  const val = useAsync(() => runsApi.valEval(runId), [runId]);
  const replays = useAsync(() => catalogApi.replays(), []);
  const back = <Link className="sm-link" to="/experiments">← All runs</Link>;
  if (detail.loading) return <Page title="Run"><LoadingState title="Loading run…" /></Page>;
  if (detail.error || !detail.data) return <Page title="Run"><p>{back}</p><ErrorState title="Could not load this run" detail={detail.error ?? undefined} action={{ label: "Retry", onClick: detail.reload }} /></Page>;
  const { summary, manifest } = detail.data;
  const mine = (replays.data ?? []).filter((r) => r.run_id === summary.run_id);
  const entries = Object.entries(manifest ?? {}).filter(([, v]) => scalar(v));
  const objects = Object.entries(manifest ?? {}).filter(([, v]) => !scalar(v));
  return (
    <Page title={`${algoLabel(summary.algo)} · ${summary.level} · seed ${summary.seed ?? "?"}`} subtitle={summary.run_id}>
      <p>{back} <StatusBadge status={summary.status} /></p>
      {summary.status === "invalid" && <p className="sm-quote">This run's manifest.json could not be read, so only its location is known.</p>}
      <Section title="Training curves" note="Read from the run's metrics.jsonl and eval_val.jsonl.">
        <TrainingCharts metrics={metrics.data ?? []} val={val.data?.validation_history ?? []} />
      </Section>
      <Section title="Manifest">
        <dl className="sm-facts">
          {entries.map(([k, v]) => (<><dt key={`${k}-k`}>{k}</dt><dd key={`${k}-v`}>{String(v)}</dd></>))}
        </dl>
        {objects.map(([k, v]) => (
          <details key={k}><summary className="sm-section-note" style={{ cursor: "pointer" }}>{k}</summary><pre className="sm-log">{JSON.stringify(v, null, 2)}</pre></details>
        ))}
        <p className="sm-section-note">{kSteps(summary.trained_env_steps)} of {kSteps(summary.budget_env_steps ?? summary.total_env_steps)} budgeted env steps trained. {summary.has_actor ? "An exported actor exists (usable in Swarm Lab)." : "No exported actor yet."}</p>
      </Section>
      <Section title="Recorded replays" note="Open one in Swarm Lab's replay mode.">
        {mine.length === 0 ? <p className="sm-section-note">No replays were recorded for this run.</p> : (
          <ul>
            {mine.map((r) => (
              <li key={r.replay_id}><Link className="sm-link" to={`/?replay=${encodeURIComponent(r.replay_id)}`}>{r.map_id.replace(/^L\d-(struct-)?/, "")} · {kSteps(r.checkpoint_step)} steps ({Math.round(r.fraction * 100)}%)</Link> {r.success ? "✓" : "✗"} {r.steps} steps</li>
            ))}
          </ul>
        )}
      </Section>
    </Page>
  );
}
