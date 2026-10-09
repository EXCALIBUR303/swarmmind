import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { PREREG, STUDY } from "../fixtures";
import { DotPlots } from "./DotPlots";
import { HypothesisPanel, PairwiseTable } from "./Hypotheses";
import { LearningCurves } from "./LearningCurves";
import { availableModes, availableSplits, comparisonRows, METRICS, metricOf, modeStats } from "./model";
import { ResultsTable } from "./ResultsTable";
import { fmtInterval } from "../../lib/stats";

afterEach(cleanup);
const level = STUDY.summary.levels.L2;

describe("model selectors", () => {
  it("finds the real splits and modes, tolerating absent ones", () => {
    expect(availableSplits(level)).toEqual(["validation", "test", "structured"]);
    expect(availableModes(level, "test")).toEqual(["stochastic", "greedy"]);
    expect(availableModes(level, "nope")).toEqual([]);
    expect(modeStats(level, "ippo", "nope", "stochastic")).toBeNull();
    expect(metricOf(null, "success_rate")).toBeNull();
  });
  it("every metric the table can show exists in the study", () => {
    const ms = modeStats(level, "mappo", "test", "stochastic");
    for (const m of METRICS) expect(metricOf(ms, m.key), m.key).not.toBeNull();
  });
});

describe("ResultsTable", () => {
  it("prints each cell exactly as fmtInterval of summary.json's interval (nothing recomputed)", () => {
    render(<ResultsTable level={level} split="test" mode="stochastic" />);
    const row = screen.getByRole("row", { name: /MAPPO/ });
    const iv = metricOf(modeStats(level, "mappo", "test", "stochastic"), "censored_steps")!.interval;
    expect(within(row).getByText(fmtInterval(iv, 2))).toBeInTheDocument();
    expect(within(row).getAllByRole("cell")[0]).toHaveTextContent("5"); // n seeds
  });
  it("a single-seed reference says so instead of showing a CI", () => {
    render(<ResultsTable level={level} split="test" mode="stochastic" />);
    expect(within(screen.getByRole("row", { name: /Frontier heuristic/ })).getAllByText(/n=1, no CI/).length).toBeGreaterThan(0);
  });
  it("shows a dash for an arm that is missing in this mode", () => {
    render(<ResultsTable level={level} split="test" mode="greedy" />);
    expect(within(screen.getByRole("row", { name: /Random/ })).getAllByText("–").length).toBeGreaterThan(3);
  });
});

describe("DotPlots", () => {
  it("renders the six primary metrics with legends and a data table each", () => {
    render(<DotPlots level={level} split="test" mode="stochastic" />);
    expect(screen.getAllByRole("figure")).toHaveLength(6);
    expect(screen.getAllByRole("button", { name: "View data" })).toHaveLength(6);
    expect(screen.getAllByRole("list", { name: "Legend" })).toHaveLength(6);
  });
});

describe("LearningCurves", () => {
  it("draws IPPO and MAPPO (categorical) plus Random/heuristic as dashed neutral references", () => {
    const { container } = render(<LearningCurves level={level} />);
    expect(within(screen.getByRole("list", { name: "Legend" })).getAllByRole("listitem").map((l) => l.textContent)).toEqual(["IPPO", "MAPPO", "Random", "Frontier heuristic"]);
    expect(container.querySelectorAll("path[stroke-dasharray='6 4']")).toHaveLength(2);
    expect(screen.getByText(/245.76k steps/)).toBeInTheDocument(); // the AUC window is stated
  });
  it("an empty state when the study has no validation curves", () => {
    render(<LearningCurves level={{ ...level, methods: { ippo: { splits: {} }, mappo: { splits: {} } } }} />);
    expect(screen.getByText("This study has no validation curves.")).toBeInTheDocument();
  });
});

describe("HypothesisPanel quotes, never editorialises", () => {
  it("every verdict statement appears verbatim, with the frozen rule behind a disclosure", () => {
    render(<HypothesisPanel level={level} rules={PREREG.rules} source={PREREG.source} />);
    const h2 = level.hypotheses.H2 as unknown as { statement: string; verdict: string };
    expect(screen.getByText(h2.statement)).toBeInTheDocument();
    const h3 = level.hypotheses.H3 as unknown as { statement: string };
    expect(screen.getByText(h3.statement)).toBeInTheDocument();
    const h1 = level.hypotheses.H1 as unknown as Record<string, { statement: string }>;
    expect(screen.getByText(h1.ippo.statement)).toBeInTheDocument();
    expect(screen.getByText(h1.mappo.statement)).toBeInTheDocument();
    const art = screen.getByRole("article", { name: "Hypothesis H2" });
    expect(within(art).getByText(/NOT SUPPORTED/)).toBeInTheDocument();
    expect(art.querySelector("details blockquote")?.textContent).toBe(PREREG.rules.find((r) => r.id === "H2")!.text);
  });
});

describe("PairwiseTable", () => {
  it("lists every comparison with CI and Hedges' g, undefined CIs stated as such", () => {
    const rows = comparisonRows(level, "test", "stochastic");
    expect(rows.length).toBeGreaterThan(20);
    render(<PairwiseTable rows={rows} />);
    expect(screen.getAllByRole("row").length).toBe(rows.length + 1);
    render(<PairwiseTable rows={[{ pair: "a-b", metric: "m", d: { n_a: 5, n_b: 5, diff: 0, lo: null, hi: null, df: null, p_value: null, hedges_g: null, conf: 0.95, ci_excludes_zero: false } }]} />);
    expect(screen.getByText("undefined")).toBeInTheDocument();
  });
});
