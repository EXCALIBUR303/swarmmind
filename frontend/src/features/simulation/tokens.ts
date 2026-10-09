/** Canvas can't use CSS variables, so resolve the design tokens once (spec §30.3:
 * tokens.css stays the single source of raw colours). */
export interface RenderTokens {
  robots: string[];
  free: string;
  unknown: string;
  unknownHatch: string;
  obstacle: string;
  visitedTeam: string;
  gridLine: string;
  targetFill: string;
  targetOutline: string;
  commLink: string;
  success: string;
  danger: string;
  accent: string;
  surface: string;
  fontUi: string;
}

export function readTokens(el: Element = document.documentElement): RenderTokens {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    robots: [1, 2, 3, 4, 5, 6].map((i) => v(`--sm-robot-${i}`, "#0072b2")),
    free: v("--sm-cell-free", "#ffffff"),
    unknown: v("--sm-cell-unknown", "#cbd2db"),
    unknownHatch: v("--sm-cell-unknown-hatch", "#bcc4cf"),
    obstacle: v("--sm-cell-obstacle", "#374151"),
    visitedTeam: v("--sm-cell-visited-team", "#e3edfb"),
    gridLine: v("--sm-grid-line", "#e5e7eb"),
    targetFill: v("--sm-target-fill", "#f2b600"),
    targetOutline: v("--sm-target-outline", "#7a5a00"),
    commLink: v("--sm-comm-link", "#64748b"),
    success: v("--sm-success", "#15803d"),
    danger: v("--sm-danger", "#b91c1c"),
    accent: v("--sm-accent", "#2563eb"),
    surface: v("--sm-bg-sunken", "#eef1f5"),
    fontUi: v("--sm-font-ui", "system-ui, sans-serif"),
  };
}
