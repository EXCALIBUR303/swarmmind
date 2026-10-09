/// <reference types="node" />
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { contrast } from "./contrast";

const css = readFileSync(`${process.cwd()}/src/design/tokens.css`, "utf-8");
const t = (name: string): string => css.match(new RegExp(`${name}:\\s*(#[0-9a-fA-F]{3,6})`))?.[1] ?? "";

const SURFACES = ["--sm-bg-app", "--sm-bg-panel", "--sm-bg-sunken"];
// [foreground token, text role] pairs actually used as TEXT in the app (AA normal text >= 4.5:1)
const TEXT_ON_SURFACES = ["--sm-text", "--sm-text-muted", "--sm-accent", "--sm-success", "--sm-danger", "--sm-warning", "--sm-target-outline"];

describe("WCAG 2.2 AA contrast of the design tokens (text >= 4.5:1, graphics >= 3:1)", () => {
  for (const fg of TEXT_ON_SURFACES)
    for (const bg of SURFACES) it(`${fg} on ${bg}`, () => expect(contrast(t(fg), t(bg)), `${fg} on ${bg}`).toBeGreaterThanOrEqual(4.5));

  it("white text on the accent (primary buttons, active segments) and on its hover state", () => {
    expect(contrast("#ffffff", t("--sm-accent"))).toBeGreaterThanOrEqual(4.5);
    expect(contrast("#ffffff", t("--sm-accent-hover"))).toBeGreaterThanOrEqual(4.5);
  });
  it("navigation text on the dark nav surface and its deep gradient end", () => {
    for (const bg of ["--sm-bg-nav", "--sm-bg-nav-deep"]) expect(contrast(t("--sm-text-on-dark"), t(bg))).toBeGreaterThanOrEqual(4.5);
  });
  it("algorithm colours as graphics against the white panel (>= 3:1)", () => {
    for (const k of ["--sm-algo-ippo", "--sm-algo-mappo", "--sm-algo-random", "--sm-algo-heuristic"]) expect(contrast(t(k), t("--sm-bg-panel")), k).toBeGreaterThanOrEqual(3);
  });
  it("robot colours 1-5 as graphics against a free cell (robot 4 is the documented 2.98:1 case)", () => {
    for (const i of [1, 2, 3, 5]) expect(contrast(t(`--sm-robot-${i}`), t("--sm-cell-free")), `robot ${i}`).toBeGreaterThanOrEqual(3);
    expect(contrast(t("--sm-robot-4"), t("--sm-cell-free"))).toBeGreaterThan(2.9); // WARN band, always labelled (DESIGN_SYSTEM §3.2)
  });
  it("target outline vs free cell (the graphical-contrast mechanism, §3.3)", () => {
    expect(contrast(t("--sm-target-outline"), t("--sm-cell-free"))).toBeGreaterThanOrEqual(3);
  });
});
