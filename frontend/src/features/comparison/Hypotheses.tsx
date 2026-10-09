import type { Difference, Hypothesis, LevelSummary } from "../../api/endpoints/analytics";
import { fmtNum } from "../../lib/stats";
import { badgeClass } from "./model";

const label = (v: string) => v.replace("_", " ").toUpperCase();

function Badge({ verdict }: { verdict: string }) {
  const icon = verdict === "supported" ? "✓" : verdict === "not_supported" ? "✗" : "•"; // icon + text, never colour alone
  return <span className={`sm-badge ${badgeClass(verdict)}`}><span aria-hidden="true">{icon}</span>{label(verdict)}</span>;
}

/** The pre-registered rule (from docs/EXPERIMENTS.md) and the computed statement, both quoted verbatim. No editorial text. */
export function HypothesisPanel({ level, rules, source }: { level: LevelSummary; rules: { id: string; text: string }[]; source: string }) {
  const entries: { id: string; verdict: string; statement: string; extra?: Difference }[] = [];
  for (const [id, h] of Object.entries(level.hypotheses)) {
    const hh = h as Hypothesis & Record<string, Hypothesis>;
    if (typeof hh.verdict === "string") entries.push({ id, verdict: hh.verdict, statement: hh.statement, extra: (hh as unknown as { difference?: Difference }).difference });
    else for (const [sub, s] of Object.entries(hh)) entries.push({ id: `${id} · ${sub.toUpperCase()}`, verdict: (s as Hypothesis).verdict, statement: (s as Hypothesis).statement, extra: (s as unknown as { difference?: Difference }).difference });
  }
  return (
    <div style={{ display: "grid", gap: "var(--sm-space-4)" }}>
      {entries.map((e) => {
        const rule = rules.find((r) => r.id === e.id.slice(0, 2));
        return (
          <article className="sm-hyp" key={e.id} aria-label={`Hypothesis ${e.id}`}>
            <div className="sm-hyp-head"><strong>{e.id}</strong><Badge verdict={e.verdict} /></div>
            <blockquote className="sm-quote" cite={source}>{e.statement}</blockquote>
            {e.extra && e.extra.lo != null && (
              <div className="sm-section-note" style={{ margin: 0 }}>
                difference {fmtNum(e.extra.diff, 3)}, 95% Welch CI [{fmtNum(e.extra.lo, 3)}, {fmtNum(e.extra.hi, 3)}], Hedges' g {fmtNum(e.extra.hedges_g, 2)}, p {fmtNum(e.extra.p_value, 4)}, n {e.extra.n_a} vs {e.extra.n_b}
              </div>
            )}
            {rule && (
              <details>
                <summary className="sm-section-note" style={{ cursor: "pointer", margin: 0 }}>Pre-registered rule ({source})</summary>
                <blockquote className="sm-quote">{rule.text}</blockquote>
              </details>
            )}
          </article>
        );
      })}
    </div>
  );
}

export function PairwiseTable({ rows }: { rows: { pair: string; metric: string; d: Difference }[] }) {
  return (
    <div className="sm-table-wrap" tabIndex={0} role="region" aria-label="Pairwise differences">
      <table className="sm-table">
        <thead>
          <tr><th scope="col">Comparison (A − B)</th><th scope="col">Metric</th><th scope="col">Difference</th><th scope="col">95% Welch CI</th><th scope="col">Hedges' g</th><th scope="col">p</th><th scope="col">CI excludes 0</th></tr>
        </thead>
        <tbody>
          {rows.map(({ pair, metric, d }) => (
            <tr key={`${pair}-${metric}`}>
              <td className="left">{pair.replace("-", " − ").toUpperCase()}</td>
              <td className="left">{metric}</td>
              <td>{fmtNum(d.diff, 3)}</td>
              <td>{d.lo == null ? "undefined" : `[${fmtNum(d.lo, 3)}, ${fmtNum(d.hi, 3)}]`}</td>
              <td>{fmtNum(d.hedges_g, 2)}</td>
              <td>{fmtNum(d.p_value, 4)}</td>
              <td>{d.ci_excludes_zero ? "yes" : "no"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
