import { useState } from "react";
import { fmtNum, linear, niceTicks } from "../../lib/stats";
import { useWidth } from "../../lib/useWidth";

export interface DotRow {
  id: string;
  label: string;
  color: string;
  perSeed: number[];
  mean: number;
  lo: number | null;
  hi: number | null;
}

interface Props {
  rows: DotRow[];
  reference?: { label: string; value: number } | null; // the heuristic: a labelled dashed line, not a series
  xLabel: string;
  format?: (v: number) => string;
  domain?: [number, number];
  ariaLabel: string;
}

const ROW = 34;
const M = { l: 84, r: 20, t: 14, b: 34 };

/** Per-seed dots + mean (black diamond) + CI whisker, matching figures.plot_heldout_dots. */
export function DotWhisker({ rows, reference, xLabel, format = (v) => fmtNum(v, 3), domain, ariaLabel }: Props) {
  const [wrapRef, W] = useWidth<HTMLDivElement>(440);
  const [tip, setTip] = useState<{ x: number; y: number; row: DotRow } | null>(null);
  const all = [...rows.flatMap((r) => [...r.perSeed, r.mean, ...(r.lo != null && r.hi != null ? [r.lo, r.hi] : [])]), ...(reference ? [reference.value] : [])];
  const lo = domain?.[0] ?? Math.min(...all);
  const hi = domain?.[1] ?? Math.max(...all);
  const pad = domain ? 0 : (hi - lo) * 0.08 || 0.5;
  const dom: [number, number] = [lo - pad, hi + pad];
  const H = M.t + rows.length * ROW + M.b;
  const sx = linear(dom, [M.l, W - M.r]);
  const ticks = niceTicks(dom[0], dom[1], 5);
  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel} onPointerLeave={() => setTip(null)}>
        <g className="sm-chart-axis">
          {ticks.map((t) => (
            <g key={t}>
              <line className="sm-chart-grid" x1={sx(t)} x2={sx(t)} y1={M.t} y2={H - M.b} />
              <text x={sx(t)} y={H - M.b + 16} textAnchor="middle">{format(t)}</text>
            </g>
          ))}
        </g>
        <text className="sm-chart-axis-title" x={(M.l + W - M.r) / 2} y={H - 3} textAnchor="middle">{xLabel}{!domain && dom[0] > 0 ? ` · axis does not start at 0 (starts at ${format(dom[0])})` : ""}</text>
        {reference && (
          <g>
            <line x1={sx(reference.value)} x2={sx(reference.value)} y1={M.t} y2={H - M.b} stroke="var(--sm-algo-heuristic)" strokeWidth="1.5" strokeDasharray="5 4" />
            <text className="sm-chart-axis-title" x={sx(reference.value)} y={M.t - 1} textAnchor="middle" style={{ fill: "var(--sm-text)" }}>{reference.label}</text>
          </g>
        )}
        {rows.map((r, i) => {
          const cy = M.t + i * ROW + ROW / 2;
          return (
            <g key={r.id} onPointerMove={(e) => { const b = e.currentTarget.ownerSVGElement!.getBoundingClientRect(); setTip({ x: ((e.clientX - b.left) / b.width) * 100, y: ((e.clientY - b.top) / b.height) * 100, row: r }); }}>
              <rect x={0} y={cy - ROW / 2} width={W} height={ROW} fill="transparent" />
              <text x={M.l - 10} y={cy + 4} textAnchor="end" style={{ font: "var(--sm-font-sm)", fill: "var(--sm-text)" }}>{r.label}</text>
              {r.lo != null && r.hi != null && <line x1={sx(r.lo)} x2={sx(r.hi)} y1={cy} y2={cy} stroke="var(--sm-text)" strokeWidth="2" />}
              {r.lo != null && r.hi != null && [r.lo, r.hi].map((v, ci) => <line key={ci} x1={sx(v)} x2={sx(v)} y1={cy - 5} y2={cy + 5} stroke="var(--sm-text)" strokeWidth="2" />)}
              {r.perSeed.map((v, k) => (
                <circle key={k} cx={sx(v)} cy={cy + (k % 2 ? 6 : -6)} r="4" fill={r.color} fillOpacity="0.85" stroke="var(--sm-bg-panel)" strokeWidth="1.5" />
              ))}
              <path d={`M${sx(r.mean)} ${cy - 6} l6 6 l-6 6 l-6 -6 Z`} fill="var(--sm-text)" stroke="var(--sm-bg-panel)" strokeWidth="1.5" />
            </g>
          );
        })}
      </svg>
      {tip && (
        <div className="sm-chart-tip" role="status" style={{ left: `${Math.min(tip.x, 70)}%`, top: `${tip.y}%`, transform: "translate(var(--sm-tip-offset), -50%)" }}>
          <div className="sm-chart-tip-head">{tip.row.label} · n={tip.row.perSeed.length}</div>
          <div className="sm-chart-tip-row"><span>mean</span><b>{format(tip.row.mean)}</b></div>
          {tip.row.lo != null && tip.row.hi != null && <div className="sm-chart-tip-row"><span>95% CI</span><b>[{format(tip.row.lo)}, {format(tip.row.hi)}]</b></div>}
          <div className="sm-chart-tip-row"><span>per seed</span><b>{tip.row.perSeed.map((v) => format(v)).join(" ")}</b></div>
        </div>
      )}
    </div>
  );
}
