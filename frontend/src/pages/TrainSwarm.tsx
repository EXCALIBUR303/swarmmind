import { useCallback, useEffect, useState } from "react";
import { trainingApi, type Job, type TrainRequestBody } from "../api/endpoints/analytics";
import { ApiError } from "../api/client";
import { EmptyState } from "../components/EmptyState";
import { Page, Section } from "../components/PageLayout";
import { ErrorState, LoadingState } from "../components/StateViews";
import { JobCard } from "../features/training/JobCard";
import { nextMeasure, type Measure } from "../features/training/measure";
import { TrainForm } from "../features/training/TrainForm";
import { TrainingCharts } from "../features/training/TrainingCharts";
import { useTrainingStream } from "../features/training/useTrainingStream";
import { useAsync } from "../lib/useAsync";
import "./pages.css";

const ACTIVE = new Set(["starting", "running"]);

export function TrainSwarm() {
  const schema = useAsync(() => trainingApi.schema(), []);
  const [job, setJob] = useState<Job | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const jobs = useAsync(() => trainingApi.jobs(), []);
  const [measure, setMeasure] = useState<Measure | null>(null);

  // Adopt a job that is already running (e.g. after a page reload) so the page never loses it.
  const existing = jobs.data?.find((j) => ACTIVE.has(j.status)) ?? null;
  const current = job ?? existing ?? (jobs.data && jobs.data.length ? jobs.data[0] : null);
  const jobId = current?.job_id ?? null;
  const active = !!current && ACTIVE.has(current.status);

  useEffect(() => {
    if (!jobId || !active) return;
    const id = setInterval(() => {
      trainingApi
        .job(jobId)
        .then((j) => {
          setJob(j);
          setMeasure((m) => nextMeasure(m, j, performance.now()));
        })
        .catch(() => undefined);
    }, 1500);
    return () => clearInterval(id);
  }, [jobId, active]);

  const sps = measure && measure.jobId === jobId ? measure.sps : null;
  const stream = useTrainingStream(current?.run_id ?? null);

  const start = useCallback(async (body: TrainRequestBody) => {
    setStarting(true);
    setServerError(null);
    try {
      setJob(await trainingApi.launch(body));
    } catch (e) {
      setServerError(e instanceof ApiError ? e.message : "could not start the job");
    } finally {
      setStarting(false);
    }
  }, []);

  const cancel = async () => {
    if (!jobId) return;
    try {
      setJob(await trainingApi.cancel(jobId));
    } catch (e) {
      setServerError(e instanceof ApiError ? e.message : "could not cancel the job");
    }
  };

  if (schema.loading) return <Page title="Train Swarm"><LoadingState title="Loading training options…" /></Page>;
  if (schema.error || !schema.data) return <Page title="Train Swarm"><ErrorState title="Backend unreachable" detail={schema.error ?? undefined} action={{ label: "Retry", onClick: schema.reload }} /></Page>;
  return (
    <Page title="Train Swarm" subtitle="Launch a training run as a background process and watch its real metrics arrive. One job at a time; budgets are bounded by the backend.">
      <div className="sm-train-layout">
        <aside>
          <TrainForm schema={schema.data} busy={starting} disabledReason={active ? "A job is running. Cancel it or wait for it to finish." : null} serverError={serverError} onStart={start} />
        </aside>
        <div style={{ minWidth: 0 }}>
          {current ? (
            <>
              <JobCard job={current} sps={sps} onCancel={cancel} />
              <Section title="Training metrics" note={stream.mode === "websocket" ? "Live over WebSocket." : stream.mode === "polling" ? "Live by polling (WebSocket unavailable)." : undefined}>
                <TrainingCharts metrics={stream.metrics} val={stream.val} live={active} />
              </Section>
            </>
          ) : (
            <EmptyState title="No training job yet" detail="Configure a run on the left and press Start training. Metrics appear here as the trainer logs them." />
          )}
        </div>
      </div>
    </Page>
  );
}
