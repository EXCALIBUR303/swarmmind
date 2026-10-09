import { useMemo, useState } from "react";
import { studiesApi } from "../api/endpoints/analytics";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, LoadingState } from "../components/StateViews";
import { Field, FieldGroup, Page, Section } from "../components/PageLayout";
import { Select } from "../components/Select";
import { DotPlots } from "../features/comparison/DotPlots";
import { HypothesisPanel, PairwiseTable } from "../features/comparison/Hypotheses";
import { LearningCurves } from "../features/comparison/LearningCurves";
import { availableModes, availableSplits, comparisonRows, levelOf, METHOD_LABEL, type Method } from "../features/comparison/model";
import { ResultsTable } from "../features/comparison/ResultsTable";
import { fmtInterval } from "../lib/stats";
import { useAsync } from "../lib/useAsync";
import "./pages.css";

export function AlgorithmComparison() {
  const studies = useAsync(() => studiesApi.list(), []);
  const [picked, setStudyId] = useState<string | null>(null);
  const [levelId, setLevelId] = useState<string | null>(null);
  const [split, setSplit] = useState("test");
  const [mode, setMode] = useState("stochastic");
  const rules = useAsync(() => studiesApi.rules(), []);

  const real = (studies.data ?? []).filter((s) => !s.smoke);
  const options = real.length ? real : (studies.data ?? []);
  const studyId = picked ?? options[0]?.study ?? null;
  const detail = useAsync(() => studiesApi.get(studyId!), [studyId], !!studyId);
  const summary = detail.data?.summary;
  const levels = Object.keys(summary?.levels ?? {});
  const level = summary ? levelOf(summary, levelId && levels.includes(levelId) ? levelId : (levels[0] ?? "")) : null;
  const splits = useMemo(() => (level ? availableSplits(level) : []), [level]);
  const effSplit = splits.includes(split) ? split : (splits.includes("test") ? "test" : splits[0]);
  const modes = useMemo(() => (level && effSplit ? availableModes(level, effSplit) : []), [level, effSplit]);
  const effMode = modes.includes(mode) ? mode : modes[0];

  if (studies.loading) return <Page title="Algorithm Comparison"><LoadingState title="Loading studies…" /></Page>;
  if (studies.error) return <Page title="Algorithm Comparison"><ErrorState title="Backend unreachable" detail={studies.error} action={{ label: "Retry", onClick: studies.reload }} /></Page>;
  if (!options.length)
    return (
      <Page title="Algorithm Comparison">
        <EmptyState title="No completed study yet" detail="Run one with: python scripts/run_study.py --config configs/experiments/main_L2.yaml, then python scripts/aggregate_results.py." />
      </Page>
    );

  const missing = (summary?.missing ?? []) as unknown[];
  return (
    <Page
      title="Algorithm Comparison"
      subtitle="Random vs IPPO vs MAPPO from the pre-registered study. Every number is read from the study's summary.json; the unit of analysis is the training seed."
      filters={
        <>
          <Field label="Study">
            <Select label="Study" className="" value={studyId ?? ""} options={options.map((s) => ({ value: s.study, label: s.study + (s.smoke ? " (smoke test)" : "") }))} onChange={setStudyId} />
          </Field>
          {levels.length > 1 && (
            <Field label="Level"><Select label="Level" className="" value={levelId ?? levels[0]} options={levels.map((l) => ({ value: l, label: l }))} onChange={setLevelId} /></Field>
          )}
          <FieldGroup label="Evaluation set"><div className="sm-seg">{splits.map((s) => <button key={s} aria-pressed={s === effSplit} onClick={() => setSplit(s)}>{s}</button>)}</div>
          </FieldGroup>
          <FieldGroup label="Action selection"><div className="sm-seg">{modes.map((m) => <button key={m} aria-pressed={m === effMode} onClick={() => setMode(m)}>{m}{m === "greedy" ? " (secondary)" : ""}</button>)}</div>
          </FieldGroup>
        </>
      }
    >
      {detail.loading && <LoadingState title="Loading study…" />}
      {detail.error && <ErrorState title="Could not load the study" detail={detail.error} action={{ label: "Retry", onClick: detail.reload }} />}
      {summary && level && effSplit && effMode && (
        <>
          {summary.smoke && <p className="sm-quote">This is a smoke-test study (tiny budget): its numbers demonstrate the pipeline, not a result.</p>}
          <Section title="Results" note={`${effSplit} maps, ${effMode} actions. mean ± sample SD [95% t-interval]. Lower or higher is better as marked.`}>
            <ResultsTable level={level} split={effSplit} mode={effMode} />
          </Section>
          <Section title="Per-seed results" note="Dots are individual training seeds; the black diamond is the mean and the bar its 95% CI. The dashed line is the heuristic reference.">
            <DotPlots level={level} split={effSplit} mode={effMode} />
          </Section>
          <Section title="Learning curves"><LearningCurves level={level} /></Section>
          <Section title="Hypotheses" note="Verdicts and statements are generated by the pipeline's frozen decision rules and quoted as is.">
            {rules.data ? <HypothesisPanel level={level} rules={rules.data.rules} source={rules.data.source} /> : <HypothesisPanel level={level} rules={[]} source="docs/EXPERIMENTS.md" />}
          </Section>
          <Section title="Pairwise differences" note="Welch 95% CI of the difference of per-seed means; no multiplicity correction (pre-registered).">
            <PairwiseTable rows={comparisonRows(level, effSplit, effMode)} />
          </Section>
          {level.coordination && (
            <Section title="Coordination metrics" note={`DER and related metrics over ${String(level.coordination.protocol.n_maps)} maps × ${(level.coordination.protocol.eval_seeds as unknown[]).length} evaluation seeds (a documented reduction of the held-out set).`}>
              <div className="sm-table-wrap" tabIndex={0} role="region" aria-label="Coordination metrics">
                <table className="sm-table">
                  <thead><tr><th scope="col">Metric</th>{Object.keys(level.coordination.methods).map((m) => <th key={m} scope="col">{METHOD_LABEL[m as Method] ?? m}</th>)}</tr></thead>
                  <tbody>
                    {Object.keys(Object.values(level.coordination.methods)[0]).filter((k) => k !== "n_seeds").map((k) => (
                      <tr key={k}>
                        <th scope="row" className="left">{k}</th>
                        {Object.values(level.coordination!.methods).map((row, i) => {
                          const v = row[k];
                          return <td key={i}>{v && typeof v === "object" ? fmtInterval(v, 3) : "–"}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}
          <Section title="Missing or incomplete runs">
            {missing.length === 0 ? <p className="sm-section-note">None: every planned run completed.</p> : <ul>{missing.map((m, i) => <li key={i}>{JSON.stringify(m)}</li>)}</ul>}
          </Section>
        </>
      )}
    </Page>
  );
}
