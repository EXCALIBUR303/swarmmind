import { useState } from "react";
import { fmtNum, sequential } from "../../lib/stats";

interface Props {
  counts: number[][]; // [y][x]
  obstacles: boolean[][];
  ariaLabel: string;
  unit: string; // "visits", "collisions"
  max?: number; // shared colour-scale maximum, so panels are comparable
}

/** Single-hue sequential scale, obstacle cells hatched out (not a colour), numeric colour bar. */
export function Heatmap({ counts, obstacles, ariaLabel, unit, max: maxProp }: Props) {
  const [hover, setHover] = useState<{ y: number; x: number } | null>(null);
  const h = counts.length;
  const w = counts[0]?.length ?? 0;
  const cell = 36;
  const max = Math.max(1, maxProp ?? 0, ...counts.flatMap((r, y) => r.filter((_, x) => !obstacles[y][x])));
  const pad = 4;
  const barW = 14;
  const gridW = w * cell;
  const vbW = gridW + pad * 2 + 70;
  const vbH = h * cell + pad * 2;
  return (
    <div style={{ position: "relative", maxWidth: 520 }}>
      <svg viewBox={`0 0 ${vbW} ${vbH}`} role="img" aria-label={ariaLabel} onPointerLeave={() => setHover(null)}>
        <defs>
          <pattern id="sm-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill="var(--sm-cell-obstacle)" />
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--sm-bg-panel)" strokeWidth="1.5" />
          </pattern>
        </defs>
        {counts.map((row, y) =>
          row.map((v, x) => {
            const obs = obstacles[y][x];
            return (
              <rect
                key={`${y}-${x}`}
                x={pad + x * cell + 1}
                y={pad + y * cell + 1}
                width={cell - 2}
                height={cell - 2}
                fill={obs ? "url(#sm-hatch)" : sequential(v / max)}
                stroke={hover && hover.x === x && hover.y === y ? "var(--sm-text)" : "none"}
                strokeWidth="2"
                onPointerMove={() => setHover({ y, x })}
              />
            );
          }),
        )}
        <g className="sm-chart-axis">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <rect x={gridW + pad + 18} y={pad + (1 - t) * (h * cell - 1) - (t === 1 ? 0 : 0)} width="0" height="0" />
              <text x={gridW + pad + 18 + barW + 4} y={pad + (1 - t) * (h * cell) + (t === 0 ? 0 : t === 1 ? 10 : 4)}>{fmtNum(t * max, max < 10 ? 1 : 0)}</text>
            </g>
          ))}
        </g>
        {Array.from({ length: 40 }, (_, i) => (
          <rect key={i} x={gridW + pad + 18} y={pad + (i / 40) * h * cell} width={barW} height={(h * cell) / 40 + 0.5} fill={sequential(1 - i / 39)} />
        ))}
      </svg>
      {hover && (
        <div className="sm-chart-tip" role="status" style={{ left: `${((pad + hover.x * cell) / vbW) * 100}%`, top: `${((pad + hover.y * cell) / vbH) * 100}%`, transform: "translate(40px, 0)" }}>
          <div className="sm-chart-tip-head">row {hover.y}, col {hover.x}</div>
          <div className="sm-chart-tip-row">
            <span>{obstacles[hover.y][hover.x] ? "obstacle" : unit}</span>
            {!obstacles[hover.y][hover.x] && <b>{counts[hover.y][hover.x]}</b>}
          </div>
        </div>
      )}
    </div>
  );
}
