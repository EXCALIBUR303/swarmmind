import { useMemo, useState } from "react";
import { fmtNum, linear, niceTicks } from "../../lib/stats";

export interface Pt {
  x: number;
  y: number;
}
export interface LineSeries {
  id: string;
  label: string;
  color: string;
  dash?: boolean;
  width?: number; // default 2
  opacity?: number;
  points: Pt[];
  band?: { x: number; lo: number; hi: number }[]; // e.g. 95% CI
  thin?: Pt[][]; // per-seed lines, drawn faint and 1px
  directLabel?: boolean;
}

interface Props {
  series: LineSeries[];
  xLabel: string;
  yLabel: string;
  xFormat?: (x: number) => string;
  yFormat?: (y: number) => string;
  xDomain?: [number, number];
  yDomain?: [number, number]; // fixed domains for bounded quantities (0-1 success, 0-100% coverage)
  height?: number;
  ariaLabel: string;
  compact?: boolean;
}

const W = 640;
const M = { l: 64, r: 96, t: 10, b: 38 };

/** One y-axis, thin marks (2px lines, 1px per-seed), CI bands, crosshair + one tooltip listing every series (§30.8). */
export function LineChart({ series, xLabel, yLabel, xFormat = (x) => fmtNum(x, 0), yFormat = (y) => fmtNum(y, 3), xDomain, yDomain, height = 240, ariaLabel, compact }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const H = height;
  const dom = useMemo(() => {
    const xs = series.flatMap((s) => s.points.map((p) => p.x));
    const ys = series.flatMap((s) => [...s.points.map((p) => p.y), ...(s.band ?? []).flatMap((b) => [b.lo, b.hi]), ...(s.thin ?? []).flat().map((p) => p.y)]);
    const x: [number, number] = xDomain ?? [Math.min(...xs), Math.max(...xs)];
    let y: [number, number] = yDomain ?? [Math.min(...ys), Math.max(...ys)];
    if (!yDomain) {
      const pad = (y[1] - y[0]) * 0.06 || 1;
      y = [y[0] - pad, y[1] + pad];
    }
    return { x, y };
  }, [series, xDomain, yDomain]);

  const sx = linear(dom.x, [M.l, W - M.r]);
  const sy = linear(dom.y, [H - M.b, M.t]);
  const xTicks = niceTicks(dom.x[0], dom.x[1], compact ? 3 : 5);
  const yTicks = niceTicks(dom.y[0], dom.y[1], compact ? 3 : 4);
  const xsAll = useMemo(() => [...new Set(series.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b), [series]);
  // Direct labels at the right end, pushed apart so they never overprint each other.
  const labelY = (() => {
    const items = series
      .filter((s) => s.directLabel !== false && series.length <= 4 && s.points.length > 0)
      .map((s) => ({ id: s.id, y: sy(s.points[s.points.length - 1].y) + 4 }))
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + 14);
    return new Map(items.map((i) => [i.id, i.y]));
  })();
  const line = (pts: Pt[]) => pts.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join(" ");

  const idxFromEvent = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const vx = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    xsAll.forEach((x, i) => {
      if (Math.abs(sx(x) - vx) < Math.abs(sx(xsAll[best]) - vx)) best = i;
    });
    return best;
  };
  const hx = hover === null ? null : xsAll[hover];
  const valueAt = (s: LineSeries, x: number) => s.points.reduce((a, p) => (Math.abs(p.x - x) < Math.abs(a.x - x) ? p : a), s.points[0]);

  return (
    <div className="sm-chart-plot" style={{ position: "relative", maxWidth: 820 }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={ariaLabel}
        tabIndex={0}
        onPointerMove={(e) => setHover(idxFromEvent(e))}
        onPointerLeave={() => setHover(null)}
        onFocus={() => setHover((h) => h ?? xsAll.length - 1)}
        onBlur={() => setHover(null)}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") setHover((h) => Math.min((h ?? -1) + 1, xsAll.length - 1));
          if (e.key === "ArrowLeft") setHover((h) => Math.max((h ?? xsAll.length) - 1, 0));
          if (e.key === "Escape") setHover(null);
        }}
      >
        <g className="sm-chart-axis">
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line className="sm-chart-grid" x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} />
              <text x={M.l - 8} y={sy(t) + 4} textAnchor="end">{yFormat(t)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={`x${t}`} x={sx(t)} y={H - M.b + 16} textAnchor="middle">{xFormat(t)}</text>
          ))}
          <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} stroke="var(--sm-text-muted)" strokeWidth="1" />
        </g>
        <text className="sm-chart-axis-title" x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
        <text className="sm-chart-axis-title" transform={`translate(11 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>

        {series.map((s) => (
          <g key={s.id}>
            {s.band && s.band.length > 1 && (
              <path
                d={`${s.band.map((b, i) => `${i ? "L" : "M"}${sx(b.x).toFixed(1)} ${sy(b.hi).toFixed(1)}`).join(" ")} ${[...s.band].reverse().map((b) => `L${sx(b.x).toFixed(1)} ${sy(b.lo).toFixed(1)}`).join(" ")} Z`}
                fill={s.color}
                fillOpacity={0.14}
              />
            )}
            {s.thin?.map((pts, i) => (
              <path key={i} d={line(pts)} fill="none" stroke={s.color} strokeWidth={1} strokeOpacity={0.3} />
            ))}
            <path d={line(s.points)} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash ? "6 4" : undefined} strokeOpacity={s.opacity ?? 1} strokeLinejoin="round" />
            {s.directLabel !== false && series.length <= 4 && s.points.length > 0 && (
              <text className="sm-chart-axis-title" x={sx(s.points[s.points.length - 1].x) + 6} y={labelY.get(s.id) ?? 0} style={{ fill: "var(--sm-text)" }}>
                {s.label}
              </text>
            )}
          </g>
        ))}

        {hx !== null && (
          <g>
            <line x1={sx(hx)} x2={sx(hx)} y1={M.t} y2={H - M.b} stroke="var(--sm-text-muted)" strokeWidth="1" />
            {series.map((s) => {
              const p = valueAt(s, hx);
              return <circle key={s.id} cx={sx(p.x)} cy={sy(p.y)} r="4" fill={s.color} stroke="var(--sm-bg-panel)" strokeWidth="2" />;
            })}
          </g>
        )}
      </svg>
      {hx !== null && (
        <div className="sm-chart-tip" role="status" style={{ left: `${(sx(hx) / W) * 100}%`, top: 0, transform: sx(hx) > W * 0.6 ? "translateX(-105%)" : "translateX(8px)" }}>
          <div className="sm-chart-tip-head">{xLabel}: {xFormat(hx)}</div>
          {series.map((s) => (
            <div className="sm-chart-tip-row" key={s.id}>
              <span>
                <svg width="14" height="8" aria-hidden="true"><line x1="0" y1="4" x2="14" y2="4" stroke={s.color} strokeWidth="2" strokeDasharray={s.dash ? "3 2" : undefined} /></svg>
                {s.label}
              </span>
              <b>{yFormat(valueAt(s, hx).y)}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
