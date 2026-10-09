import { fmtNum, linear } from "../../lib/stats";

export interface BarDatum {
  label: string;
  value: number;
  color: string;
}

interface Props {
  data: BarDatum[];
  domain: [number, number]; // untruncated: bars start at 0
  format?: (v: number) => string;
  ariaLabel: string;
  xLabel: string;
  annotation?: string; // e.g. "Gini 0.014"
}

const W = 520;
const ROW = 30;
const M = { l: 56, r: 56, t: 6, b: 30 };

/** Horizontal bars from a zero baseline, value labels at the bar end (the table view carries the same numbers). */
export function Bars({ data, domain, format = (v) => fmtNum(v, 2), ariaLabel, xLabel, annotation }: Props) {
  const H = M.t + data.length * ROW + M.b;
  const sx = linear(domain, [M.l, W - M.r]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
      <g className="sm-chart-axis">
        <line x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} stroke="var(--sm-text-muted)" />
        {[domain[0], (domain[0] + domain[1]) / 2, domain[1]].map((t) => (
          <text key={t} x={sx(t)} y={H - M.b + 16} textAnchor="middle">{format(t)}</text>
        ))}
      </g>
      <text className="sm-chart-axis-title" x={(M.l + W - M.r) / 2} y={H - 3} textAnchor="middle">{xLabel}</text>
      {data.map((d, i) => {
        const y = M.t + i * ROW + 5;
        const w = Math.max(sx(Math.max(d.value, domain[0])) - sx(domain[0]), 0);
        return (
          <g key={d.label}>
            <text x={M.l - 8} y={y + 12} textAnchor="end" style={{ font: "var(--sm-font-sm)", fill: "var(--sm-text)" }}>{d.label}</text>
            <rect x={M.l} y={y} width={w} height={ROW - 12} fill={d.color} rx="0" />
            <text x={M.l + w + 6} y={y + 12} style={{ font: "var(--sm-font-xs)", fill: "var(--sm-text)", fontFamily: "var(--sm-font-mono)" }}>{format(d.value)}</text>
          </g>
        );
      })}
      {annotation && <text x={W - M.r} y={M.t + 10} textAnchor="end" className="sm-chart-axis-title" style={{ fill: "var(--sm-text)" }}>{annotation}</text>}
    </svg>
  );
}
