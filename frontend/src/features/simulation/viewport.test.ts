import { describe, expect, it } from "vitest";
import { interpolatePositions } from "./interpolation";
import { backingSize, cellAt, cellCenter, cellOrigin, computeViewport, robotAt } from "./viewport";

describe("viewport: [y, x] <-> pixels (no transposition)", () => {
  // Deliberately NON-square: 12 columns x 6 rows, so swapping x/y can't hide.
  const vp = computeViewport(600, 600, 2, 12, 6);

  it("cell size fits the limiting axis and the grid is centred", () => {
    expect(vp.cell).toBe(50); // 600/12 limits, not 600/6
    expect(vp.offsetX).toBe(0);
    expect(vp.offsetY).toBe((600 - 50 * 6) / 2);
  });

  it("x (column) maps to horizontal pixels, y (row) to vertical", () => {
    const o = cellOrigin(vp, 2, 7); // row 2, column 7
    expect(o.px).toBe(7 * 50);
    expect(o.py).toBe(vp.offsetY + 2 * 50);
    const c = cellCenter(vp, 2, 7);
    expect(c.px).toBe(7 * 50 + 25);
  });

  it("cellAt is the inverse of cellCenter for every cell, and null outside", () => {
    for (let y = 0; y < 6; y++)
      for (let x = 0; x < 12; x++) {
        const c = cellCenter(vp, y, x);
        expect(cellAt(vp, c.px, c.py)).toEqual([y, x]);
      }
    expect(cellAt(vp, 5, 5)).toBeNull(); // letterbox above the grid
    expect(cellAt(vp, 601, 300)).toBeNull();
    expect(cellAt(vp, -1, 300)).toBeNull();
  });

  it("robotAt picks the nearest robot disc and ignores empty space", () => {
    const pos: [number, number][] = [[1, 1], [1, 2], [4, 10]];
    const c = cellCenter(vp, 4, 10);
    expect(robotAt(vp, pos, c.px + 3, c.py - 3)).toBe(2);
    const between = cellCenter(vp, 1, 1.5);
    expect(robotAt(vp, pos, between.px, between.py, 0.6)).not.toBeNull();
    expect(robotAt(vp, pos, cellCenter(vp, 3, 5).px, cellCenter(vp, 3, 5).py)).toBeNull();
  });

  it("the backing store is dpr-scaled and never zero", () => {
    expect(backingSize(vp)).toEqual({ width: 1200, height: 1200 });
    expect(backingSize(computeViewport(0, 0, 1, 4, 4))).toEqual({ width: 1, height: 1 });
  });
});

describe("interpolation", () => {
  const prev: [number, number][] = [[2, 2], [5, 5], [0, 0]];
  const cur: [number, number][] = [[2, 3], [5, 5], [3, 3]];
  it("moves between cells, holds still robots, and never smears a teleport", () => {
    const mid = interpolatePositions(prev, cur, 0.5);
    expect(mid[0][0]).toBe(2);
    expect(mid[0][1]).toBeGreaterThan(2);
    expect(mid[0][1]).toBeLessThan(3);
    expect(mid[1]).toEqual([5, 5]);
    expect(mid[2]).toEqual([3, 3]); // 6 cells away: a seek/reset, drawn at the destination
  });
  it("is exact at the endpoints and instant under reduced motion", () => {
    expect(interpolatePositions(prev, cur, 0)[0]).toEqual([2, 2]);
    expect(interpolatePositions(prev, cur, 1)[0]).toEqual([2, 3]);
    expect(interpolatePositions(prev, cur, 0.1, true)[0]).toEqual([2, 3]);
  });
  it("eases out: more than half the distance covered by half the time", () => {
    expect(interpolatePositions(prev, cur, 0.5)[0][1] - 2).toBeGreaterThan(0.5);
  });
});
