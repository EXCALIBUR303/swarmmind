import type { LevelSummary } from "../../api/endpoints/analytics";
import { fmtInterval } from "../../lib/stats";
import { METHODS, METHOD_LABEL, METRICS, metricOf, modeStats, PRIMARY_KEYS } from "./model";

/** mean ± SD [95% CI], n seeds: copied from summary.json, never recomputed. */
export function ResultsTable({ level, split, mode, all }: { level: LevelSummary; split: string; mode: string; all?: boolean }) {
  const metrics = METRICS.filter((m) => all || PRIMARY_KEYS.includes(m.key));
  return (
    <div className="sm-table-wrap" tabIndex={0} role="region" aria-label={`Results table, ${split} ${mode}`}>
      <table className="sm-table">
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">n seeds</th>
            {metrics.map((m) => (
              <th key={m.key} scope="col" title={m.unit}>
                {m.label}
                <div className="sm-section-note" style={{ margin: 0 }}>{m.better === "higher" ? "higher is better" : "lower is better"}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METHODS.map((method) => {
            const ms = modeStats(level, method, split, mode);
            return (
              <tr key={method}>
                <th scope="row" className="left">{METHOD_LABEL[method]}</th>
                <td>{ms ? ms.n_seeds : "–"}</td>
                {metrics.map((m) => {
                  const st = metricOf(ms, m.key);
                  return <td key={m.key}>{st ? fmtInterval(st.interval, m.digits) : "–"}</td>;
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
