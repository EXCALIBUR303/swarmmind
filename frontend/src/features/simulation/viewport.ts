/**
 * Grid <-> pixel mapping. The ONLY place where `[y, x]` (row, column) becomes
 * `(px, py)`; every renderer and the hit-tester go through these functions so
 * the classic y/x transposition bug can live in one tested spot.
 */
import type { Cell } from "./model";

export interface Viewport {
  cssWidth: number;
  cssHeight: number;
  dpr: number;
  cols: number;
  rows: number;
  cell: number; // CSS px per cell
  offsetX: number; // CSS px, centres the grid in the canvas
  offsetY: number;
}

export function computeViewport(cssWidth: number, cssHeight: number, dpr: number, cols: number, rows: number): Viewport {
  const cell = Math.max(1, Math.min(cssWidth / cols, cssHeight / rows));
  return {
    cssWidth,
    cssHeight,
    dpr,
    cols,
    rows,
    cell,
    offsetX: (cssWidth - cell * cols) / 2,
    offsetY: (cssHeight - cell * rows) / 2,
  };
}

/** Backing-store size in device pixels (rounded so the canvas is never blurry-stretched). */
export function backingSize(vp: Viewport): { width: number; height: number } {
  return { width: Math.max(1, Math.round(vp.cssWidth * vp.dpr)), height: Math.max(1, Math.round(vp.cssHeight * vp.dpr)) };
}

/** Top-left CSS-pixel corner of cell `(y, x)`; fractional cells (interpolated robots) are fine. */
export function cellOrigin(vp: Viewport, y: number, x: number): { px: number; py: number } {
  return { px: vp.offsetX + x * vp.cell, py: vp.offsetY + y * vp.cell };
}

export function cellCenter(vp: Viewport, y: number, x: number): { px: number; py: number } {
  return { px: vp.offsetX + (x + 0.5) * vp.cell, py: vp.offsetY + (y + 0.5) * vp.cell };
}

/** Grid cell under a CSS-pixel point, or null outside the grid. */
export function cellAt(vp: Viewport, px: number, py: number): Cell | null {
  const x = Math.floor((px - vp.offsetX) / vp.cell);
  const y = Math.floor((py - vp.offsetY) / vp.cell);
  return x < 0 || y < 0 || x >= vp.cols || y >= vp.rows ? null : [y, x];
}

/** Nearest robot whose disc contains the point (radius in cells), else null. */
export function robotAt(vp: Viewport, drawPos: readonly Cell[], px: number, py: number, radiusCells = 0.42): number | null {
  let best: number | null = null;
  let bestD = Infinity;
  drawPos.forEach(([y, x], i) => {
    const c = cellCenter(vp, y, x);
    const d = Math.hypot(c.px - px, c.py - py);
    if (d <= radiusCells * vp.cell && d < bestD) {
      best = i;
      bestD = d;
    }
  });
  return best;
}
