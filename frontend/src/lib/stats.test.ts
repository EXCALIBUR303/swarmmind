import { describe, expect, it } from "vitest";
import { fmtInterval, fmtNum, fmtSigned, fmtSteps, linear, niceTicks, rollingMean, sequential } from "./stats";

describe("formatting", () => {
  it("steps axis reads 0, 500k, 1M (never raw integers)", () => {
    expect([0, 500_000, 1_000_000, 1_500_000, 245_760, 999].map(fmtSteps)).toEqual(["0", "500k", "1M", "1.5M", "245.76k", "999"]);
  });
  it("numbers and signs, with a dash for missing values", () => {
    expect(fmtNum(0.5, 2)).toBe("0.50");
    expect(fmtNum(null)).toBe("–");
    expect(fmtNum(NaN)).toBe("–");
    expect(fmtSigned(-0.315)).toBe("-0.315");
    expect(fmtSigned(0.564)).toBe("+0.564");
  });
  it("intervals state n=1 instead of inventing a CI", () => {
    expect(fmtInterval({ n: 5, mean: 0.999, std: 0.002, lo: 0.996, hi: 1.002, conf: 0.95 })).toBe("0.999 ± 0.002 [0.996, 1.002]");
    expect(fmtInterval({ n: 1, mean: 1, std: null, lo: null, hi: null, conf: 0.95 })).toBe("1.000 (n=1, no CI)");
  });
});

describe("scales", () => {
  it("niceTicks are round, ordered and cover the range", () => {
    const t = niceTicks(0, 1, 5);
    expect(t[0]).toBe(0);
    expect(t.at(-1)).toBe(1);
    expect(t.length).toBeGreaterThanOrEqual(4);
    expect(niceTicks(3, 3)).toEqual([3]);
    for (const v of niceTicks(0, 245_760, 5)) expect(v % 10_000).toBe(0);
  });
  it("linear maps domain to range, including reversed ranges (SVG y axes)", () => {
    expect(linear([0, 10], [100, 200])(5)).toBe(150);
    expect(linear([0, 1], [50, 10])(1)).toBe(10);
    expect(linear([2, 2], [0, 100])(2)).toBe(0);
  });
  it("rollingMean is a trailing window", () => {
    expect(rollingMean([1, 2, 3, 4], 2)).toEqual([1, 1.5, 2.5, 3.5]);
  });
  it("the sequential ramp gets monotonically darker (one hue, light to dark)", () => {
    const lum = (c: string) => c.match(/\d+/g)!.map(Number).reduce((a, b) => a + b, 0);
    let prev = Infinity;
    for (let i = 0; i <= 10; i++) {
      const l = lum(sequential(i / 10));
      expect(l).toBeLessThan(prev);
      prev = l;
    }
    expect(sequential(-5)).toBe(sequential(0));
    expect(sequential(9)).toBe(sequential(1));
  });
});
