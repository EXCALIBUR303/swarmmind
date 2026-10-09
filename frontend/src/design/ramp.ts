/**
 * Raw values the DOM cannot read through CSS variables: the sequential heatmap ramp (an SVG fill
 * computed per cell) and canvas fallbacks used only when a token is unreadable (e.g. jsdom).
 * `design/design.test.ts` fails if any of these drift from tokens.css.
 */
export const RAMP_START: [number, number, number] = [239, 244, 252]; // --sm-ramp-start #eff4fc
export const RAMP_END: [number, number, number] = [29, 78, 216]; // --sm-ramp-end   #1d4ed8

export const CANVAS_FALLBACK: Record<string, string> = {
  "--sm-robot-1": "#0072b2",
  "--sm-cell-free": "#ffffff",
  "--sm-cell-unknown": "#cbd2db",
  "--sm-cell-unknown-hatch": "#bcc4cf",
  "--sm-cell-obstacle": "#374151",
  "--sm-cell-visited-team": "#e3edfb",
  "--sm-grid-line": "#e5e7eb",
  "--sm-target-fill": "#f2b600",
  "--sm-target-outline": "#7a5a00",
  "--sm-comm-link": "#64748b",
  "--sm-success": "#166534",
  "--sm-danger": "#b91c1c",
  "--sm-accent": "#2563eb",
  "--sm-bg-sunken": "#eef1f5",
};

/** Sequential single-hue ramp: --sm-ramp-start -> --sm-ramp-end, t in [0,1] (lightness decreases monotonically). */
export function sequential(t: number): string {
  const c = Math.min(Math.max(t, 0), 1);
  return `rgb(${RAMP_START.map((f, i) => Math.round(f + (RAMP_END[i] - f) * c)).join(",")})`;
}
