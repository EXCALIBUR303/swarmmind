import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { runsApi, type RunSummary } from "../api/endpoints/analytics";
import { EmptyState } from "../components/EmptyState";
import { Field, Page } from "../components/PageLayout";
import { Select } from "../components/Select";
import { ErrorState, LoadingState } from "../components/StateViews";
import { StatusBadge } from "../components/StatusBadge";
import { distinct, EMPTY_FILTERS, filterRuns, finalVal, sortRuns, type Filters, type SortKey } from "../features/experiments/runs";
import { algoLabel, kSteps } from "../lib/format";
import { fmtNum } from "../lib/stats";
import { useAsync } from "../lib/useAsync";
import { useMedia } from "../lib/useMedia";
import "./pages.css";

const COLS: { key: SortKey; label: string; num?: boolean }[] = [
  { key: "status", label: "Status" },
  { key: "algo", label: "Algorithm" },
  { key: "level", label: "Level" },
  { key: "seed", label: "Seed", num: true },
  { key: "trained_env_steps", label: "Env steps", num: true },
  { key: "final_val", label: "Final val success", num: true },
  { key: "start_time", label: "Started" },
];

export function ExperimentHistory() {
  const runs = useAsync(() => runsApi.list(), []);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "start_time", dir: "desc" });
  const cards = useMedia("(max-width: 1023px)");
  const all = useMemo(() => runs.data ?? [], [runs.data]);
  const rows = useMemo(() => sortRuns(filterRuns(all, filters), sort.key, sort.dir), [all, filters, sort]);
  const setF = (k: keyof Filters) => (v: string) => setFilters((f) => ({ ...f, [k]: v }));
  const toggle = (key: SortKey) => setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "asc" }));

  if (runs.loading) return <Page title="Experiment History"><LoadingState title="Loading runs…" /></Page>;
  if (runs.error) return <Page title="Experiment History"><ErrorState title="Backend unreachable" detail={runs.error} action={{ label: "Retry", onClick: runs.reload }} /></Page>;
  return (
    <Page
      title="Experiment History"
      subtitle="Every training run found under results/runs. A run whose manifest cannot be read is listed as invalid rather than hidden."
      filters={
        <>
          {(["status", "algo", "level", "study"] as const).map((k) => (
            <Field key={k} label={k[0].toUpperCase() + k.slice(1)}>
              <Select label={k} className="" value={filters[k]} options={[{ value: "", label: "All" }, ...distinct(all, k).map((v) => ({ value: v, label: v }))]} onChange={setF(k)} />
            </Field>
          ))}
          <span className="sm-section-note" style={{ margin: 0 }} role="status">{rows.length} of {all.length} runs</span>
        </>
      }
    >
      {all.length === 0 ? (
        <EmptyState title="No runs yet" detail="Train one from the Train Swarm page or with scripts/train.py." />
      ) : rows.length === 0 ? (
        <EmptyState title="No run matches these filters" detail="Clear a filter to see more." />
      ) : cards ? (
        <ul className="sm-cardlist">
          {rows.map((r) => (
            <li key={r.run_id} className="sm-hyp">
              <div className="sm-hyp-head"><Link className="sm-link" to={`/experiments/${r.run_id}`}>{algoLabel(r.algo)} · {r.level} · s{r.seed ?? "?"}</Link><StatusBadge status={r.status} /></div>
              <div className="sm-section-note" style={{ margin: 0 }}>{r.study} · {kSteps(r.trained_env_steps)} steps · val success {fmtNum(finalVal(r), 2)}</div>
            </li>
          ))}
        </ul>
      ) : (
        <div className="sm-table-wrap" tabIndex={0} role="region" aria-label="Runs">
          <table className="sm-table">
            <thead>
              <tr>
                {COLS.map((c) => (
                  <th key={c.key} scope="col" aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"} className={c.num ? "" : "left"}>
                    <button className="sm-th-btn" onClick={() => toggle(c.key)}>{c.label}<span aria-hidden="true">{sort.key === c.key ? (sort.dir === "asc" ? "▲" : "▼") : ""}</span></button>
                  </th>
                ))}
                <th scope="col" className="left">Run</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r: RunSummary) => (
                <tr key={r.run_id}>
                  <td className="left"><StatusBadge status={r.status} /></td>
                  <td className="left">{algoLabel(r.algo)}</td>
                  <td className="left">{r.level}</td>
                  <td>{r.seed ?? "–"}</td>
                  <td>{r.trained_env_steps == null ? "–" : kSteps(r.trained_env_steps)}</td>
                  <td>{fmtNum(finalVal(r), 3)}</td>
                  <td className="left">{r.start_time ? r.start_time.slice(0, 16).replace("T", " ") : "–"}</td>
                  <td className="left"><Link className="sm-link" to={`/experiments/${r.run_id}`}>{r.run_id}</Link></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Page>
  );
}
