import { linear, niceTicks } from "../../lib/stats";
import { useWidth } from "../../lib/useWidth";

export interface DiscoveryRow {
  target: number;
  t_discovered: number | null;
  t_rescued: number | null;
  rescued_by: number | null;
}

interface Props {
  rows: DiscoveryRow[];
  maxT: number;
  ariaLabel: string;
}

const ROW = 34;
const M = { l: 80, r: 24, t: 8, b: 34 };

/** One row per target: ◆ discovered, ★ rescued (shape carries the meaning), connecting bar = time to rescue. */
export function DiscoveryTimeline({ rows, maxT, ariaLabel }: Props) {
  const [wrapRef, W] = useWidth<HTMLDivElement>(560, 280, 720);
  const H = M.t + rows.length * ROW + M.b;
  const sx = linear([0, maxT], [M.l, W - M.r]);
  return (
    <div ref={wrapRef}>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel}>
      <g className="sm-chart-axis">
        {niceTicks(0, maxT, 5).map((t) => (
          <g key={t}>
            <line className="sm-chart-grid" x1={sx(t)} x2={sx(t)} y1={M.t} y2={H - M.b} />
            <text x={sx(t)} y={H - M.b + 16} textAnchor="middle">{t}</text>
          </g>
        ))}
      </g>
      <text className="sm-chart-axis-title" x={(M.l + W - M.r) / 2} y={H - 3} textAnchor="middle">step t</text>
      {rows.map((r, i) => {
        const cy = M.t + i * ROW + ROW / 2;
        return (
          <g key={r.target}>
            <text x={M.l - 8} y={cy + 4} textAnchor="end" style={{ font: "var(--sm-font-sm)", fill: "var(--sm-text)" }}>Target {r.target + 1}</text>
            {r.t_discovered != null && r.t_rescued != null && <line x1={sx(r.t_discovered)} x2={sx(r.t_rescued)} y1={cy} y2={cy} stroke="var(--sm-text-muted)" strokeWidth="2" />}
            {r.t_discovered != null && (
              <g>
                <title>{`discovered at t=${r.t_discovered}`}</title>
                <text x={sx(r.t_discovered)} y={cy + 5} textAnchor="middle" fontSize="15" fill="var(--sm-target-outline)">◆</text>
              </g>
            )}
            {r.t_rescued != null && (
              <g>
                <title>{`rescued at t=${r.t_rescued}${r.rescued_by != null ? ` by robot ${r.rescued_by + 1}` : ""}`}</title>
                <text x={sx(r.t_rescued)} y={cy + 5} textAnchor="middle" fontSize="15" fill="var(--sm-success)">★</text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
    </div>
  );
}
