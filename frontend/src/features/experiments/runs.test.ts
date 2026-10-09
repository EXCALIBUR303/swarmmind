import { describe, expect, it } from "vitest";
import { RUNS as REAL } from "../fixtures";
import type { RunSummary } from "../../api/endpoints/analytics";

// The real results contain no unreadable run, so one hand-made row represents what the backend reports for a corrupt manifest.
const RUNS: RunSummary[] = [...REAL, { run_id: "s9-corrupt", study: "x", algo: "ippo", level: "L1", status: "invalid" }];
import { distinct, EMPTY_FILTERS, filterRuns, finalVal, sortRuns } from "./runs";

describe("runs table logic on the real run list", () => {
  it("filters by every column and combines filters", () => {
    expect(filterRuns(RUNS, EMPTY_FILTERS)).toHaveLength(RUNS.length);
    const mappo = filterRuns(RUNS, { ...EMPTY_FILTERS, algo: "mappo" });
    expect(mappo.length).toBeGreaterThan(0);
    expect(mappo.every((r) => r.algo === "mappo")).toBe(true);
    expect(filterRuns(RUNS, { ...EMPTY_FILTERS, algo: "mappo", status: "no-such" })).toEqual([]);
  });
  it("sorts numerically and lexically in both directions, missing values always last", () => {
    const up = sortRuns(RUNS, "final_val", "asc");
    const down = sortRuns(RUNS, "final_val", "desc");
    const vals = (rs: typeof RUNS) => rs.map(finalVal).filter((v) => v !== null) as number[];
    expect(vals(up)).toEqual([...vals(up)].sort((a, b) => a - b));
    expect(vals(down)).toEqual([...vals(down)].sort((a, b) => b - a));
    expect(finalVal(up.at(-1)!)).toBeNull();
    expect(finalVal(down.at(-1)!)).toBeNull();
    const algos = sortRuns(RUNS, "algo", "asc").map((r) => r.algo);
    expect(algos).toEqual([...algos].sort());
  });
  it("is stable and does not mutate its input", () => {
    const copy = [...RUNS];
    sortRuns(RUNS, "seed", "asc");
    expect(RUNS).toEqual(copy);
  });
  it("lists an unreadable run as invalid rather than dropping it", () => {
    expect(distinct(RUNS, "status")).toContain("invalid");
  });
});
