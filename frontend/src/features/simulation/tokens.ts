/** Canvas can't use CSS variables, so resolve the design tokens once (spec §30.3:
 * tokens.css stays the single source of raw colours). */
import { CANVAS_FALLBACK } from "../../design/ramp";

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
  const v = (name: string) => cs.getPropertyValue(name).trim() || (CANVAS_FALLBACK[name] ?? CANVAS_FALLBACK["--sm-robot-1"]);
  return {
    robots: [1, 2, 3, 4, 5, 6].map((i) => v(`--sm-robot-${i}`)),
    free: v("--sm-cell-free"),
    unknown: v("--sm-cell-unknown"),
    unknownHatch: v("--sm-cell-unknown-hatch"),
    obstacle: v("--sm-cell-obstacle"),
    visitedTeam: v("--sm-cell-visited-team"),
    gridLine: v("--sm-grid-line"),
    targetFill: v("--sm-target-fill"),
    targetOutline: v("--sm-target-outline"),
    commLink: v("--sm-comm-link"),
    success: v("--sm-success"),
    danger: v("--sm-danger"),
    accent: v("--sm-accent"),
    surface: v("--sm-bg-sunken"),
    fontUi: cs.getPropertyValue("--sm-font-ui").trim() || "system-ui, sans-serif",
  };
}
