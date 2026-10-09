import { useId, useState, type ReactNode } from "react";
import "./charts.css";

export interface DataTable {
  columns: string[];
  rows: (string | number | null)[][];
}

export type ChartState = "loading" | "empty" | "error" | "ready";

interface Props {
  title: string;
  caption?: string; // states windows, units, "illustrative episode, not a statistic", ...
  state: ChartState;
  error?: string | null;
  emptyText?: string;
  legend?: { label: string; color: string; dash?: boolean; shape?: "line" | "dot" | "box" }[];
  table?: DataTable; // required for "ready" charts: the accessible alternative (§30.8)
  children?: ReactNode;
}

/** Every chart: title, optional caption, legend (>1 series), explicit loading/empty/error, and a "View data" table. */
export function ChartFrame({ title, caption, state, error, emptyText = "No data for this selection.", legend, table, children }: Props) {
  const [showTable, setShowTable] = useState(false);
  const id = useId();
  return (
    <figure className="sm-chart" aria-labelledby={`${id}-t`}>
      <figcaption>
        <div className="sm-chart-head">
          <h3 id={`${id}-t`} className="sm-chart-title">{title}</h3>
          {state === "ready" && table && (
            <button className="sm-chart-toggle" aria-pressed={showTable} onClick={() => setShowTable((v) => !v)}>
              {showTable ? "View chart" : "View data"}
            </button>
          )}
        </div>
        {caption && <div className="sm-chart-caption">{caption}</div>}
        {state === "ready" && legend && legend.length > 1 && (
          <ul className="sm-chart-legend" aria-label="Legend">
            {legend.map((l) => (
              <li key={l.label}>
                <svg width="22" height="10" aria-hidden="true">
                  {l.shape === "box" ? (
                    <rect x="4" y="1" width="14" height="8" fill={l.color} />
                  ) : l.shape === "dot" ? (
                    <circle cx="11" cy="5" r="4" fill={l.color} />
                  ) : (
                    <line x1="1" y1="5" x2="21" y2="5" stroke={l.color} strokeWidth="2" strokeDasharray={l.dash ? "4 3" : undefined} />
                  )}
                </svg>
                {l.label}
              </li>
            ))}
          </ul>
        )}
      </figcaption>
      <div className="sm-chart-body">
        {state === "loading" && <div className="sm-chart-skeleton" role="status" aria-label="Loading chart" />}
        {state === "error" && (
          <div className="sm-chart-msg error" role="alert">
            Could not load this chart{error ? `: ${error}` : "."}
          </div>
        )}
        {state === "empty" && <div className="sm-chart-msg">{emptyText}</div>}
        {state === "ready" && (showTable && table ? <DataTableView table={table} title={title} /> : children)}
      </div>
    </figure>
  );
}

function DataTableView({ table, title }: { table: DataTable; title: string }) {
  return (
    <div className="sm-chart-tablewrap">
      <table className="sm-chart-table" aria-label={`${title}: data`}>
        <thead>
          <tr>{table.columns.map((c) => <th key={c} scope="col">{c}</th>)}</tr>
        </thead>
        <tbody>
          {table.rows.map((r, i) => (
            <tr key={i}>{r.map((v, j) => <td key={j}>{v ?? "–"}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
