/// <reference types="node" />
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { CANVAS_FALLBACK, RAMP_END, RAMP_START } from "./ramp";

const css = readFileSync(`${process.cwd()}/src/design/tokens.css`, "utf-8");
const token = (name: string): string => css.match(new RegExp(`${name}:\\s*([^;]+);`))?.[1].trim() ?? "";
const hex = (rgb: number[]) => `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;

describe("design/ramp.ts mirrors tokens.css (no silent drift)", () => {
  it("canvas fallbacks equal the CSS tokens", () => {
    for (const [name, value] of Object.entries(CANVAS_FALLBACK)) expect(value.toLowerCase(), name).toBe(token(name).toLowerCase());
  });
  it("heatmap ramp endpoints equal the ramp tokens, and the end is the accent-hover colour", () => {
    expect(hex(RAMP_START)).toBe(token("--sm-ramp-start").toLowerCase());
    expect(hex(RAMP_END)).toBe(token("--sm-ramp-end").toLowerCase());
    expect(token("--sm-ramp-end").toLowerCase()).toBe(token("--sm-accent-hover").toLowerCase());
  });
});
