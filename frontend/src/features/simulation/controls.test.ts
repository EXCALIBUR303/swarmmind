import { describe, expect, it } from "vitest";
import { controlsEnabled } from "./controls";

describe("controlsEnabled (regression: Play was silently dropped while the live socket was connecting)", () => {
  it("is off with no scenario or while loading", () => {
    expect(controlsEnabled("none", "ready", undefined)).toBe(false);
    expect(controlsEnabled("replay", "loading", undefined)).toBe(false);
  });
  it("replay needs no socket", () => {
    expect(controlsEnabled("replay", "ready", undefined)).toBe(true);
  });
  it("live is off until the socket is open, and off again when it drops", () => {
    expect(controlsEnabled("live", "ready", "connecting")).toBe(false);
    expect(controlsEnabled("live", "ready", "open")).toBe(true);
    expect(controlsEnabled("live", "ready", "reconnecting")).toBe(false);
    expect(controlsEnabled("live", "ready", "closed")).toBe(false);
  });
});
