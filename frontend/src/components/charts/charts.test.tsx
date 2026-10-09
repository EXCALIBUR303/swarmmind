import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EP_MAPPO, HM, STUDY } from "../../features/fixtures";
import { Bars } from "./Bars";
import { ChartFrame } from "./ChartFrame";
import { DiscoveryTimeline } from "./DiscoveryTimeline";
import { DotWhisker } from "./DotWhisker";
import { Heatmap } from "./Heatmap";
import { LineChart } from "./LineChart";

afterEach(cleanup);
const table = { columns: ["a", "b"], rows: [[1, 2]] };

describe("ChartFrame", () => {
  it("renders explicit loading, empty and error states, never a blank box", () => {
    const { rerender } = render(<ChartFrame title="T" state="loading" />);
    expect(screen.getByRole("status", { name: /loading/i })).toBeInTheDocument();
    rerender(<ChartFrame title="T" state="empty" emptyText="Nothing here" />);
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
    rerender(<ChartFrame title="T" state="error" error="HTTP 500" />);
    expect(screen.getByRole("alert")).toHaveTextContent("HTTP 500");
  });
  it("View data swaps the chart for an accessible table and back", () => {
    render(<ChartFrame title="Success" state="ready" table={table}><div data-testid="plot" /></ChartFrame>);
    expect(screen.getByTestId("plot")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "View data" }));
    expect(screen.queryByTestId("plot")).toBeNull();
    expect(screen.getByRole("table", { name: "Success: data" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "View chart" }));
    expect(screen.getByTestId("plot")).toBeInTheDocument();
  });
  it("shows a legend only for two or more series", () => {
    const { rerender } = render(<ChartFrame title="T" state="ready" legend={[{ label: "A", color: "red" }]}><i /></ChartFrame>);
    expect(screen.queryByRole("list", { name: "Legend" })).toBeNull();
    rerender(<ChartFrame title="T" state="ready" legend={[{ label: "A", color: "red" }, { label: "B", color: "blue" }]}><i /></ChartFrame>);
    expect(within(screen.getByRole("list", { name: "Legend" })).getAllByRole("listitem")).toHaveLength(2);
  });
});

describe("LineChart on the real learning curves", () => {
  const ippo = STUDY.summary.levels.L2.methods.ippo.validation_curve!;
  const series = [{ id: "ippo", label: "IPPO", color: "var(--sm-algo-ippo)", points: ippo.grid_env_steps.map((x, i) => ({ x, y: ippo.mean[i] })), band: ippo.grid_env_steps.map((x, i) => ({ x, lo: ippo.lo[i], hi: ippo.hi[i] })), thin: ippo.per_seed.map((ys) => ippo.grid_env_steps.map((x, i) => ({ x, y: ys[i] }))) },
    { id: "ref", label: "Random", color: "grey", dash: true, points: [{ x: 0, y: 0.5 }, { x: 1_000_000, y: 0.5 }] }];

  it("draws a 2px mean line, a CI band, and a thin line per seed", () => {
    const { container } = render(<LineChart series={series} xLabel="steps" yLabel="success" ariaLabel="curves" />);
    const paths = [...container.querySelectorAll("path")];
    expect(paths.filter((p) => p.getAttribute("stroke-width") === "1")).toHaveLength(ippo.per_seed.length);
    expect(paths.filter((p) => p.getAttribute("fill-opacity"))).toHaveLength(1);
    expect(paths.some((p) => p.getAttribute("stroke-dasharray") === "6 4")).toBe(true);
  });
  it("one tooltip lists every series at the hovered x, and the keyboard reaches it too", () => {
    render(<LineChart series={series} xLabel="steps" yLabel="success" ariaLabel="curves" />);
    const svg = screen.getByRole("img", { name: "curves" });
    fireEvent.focus(svg);
    fireEvent.keyDown(svg, { key: "ArrowLeft" });
    const tip = screen.getByRole("status");
    expect(within(tip).getByText("IPPO")).toBeInTheDocument();
    expect(within(tip).getByText("Random")).toBeInTheDocument();
    fireEvent.keyDown(svg, { key: "Escape" });
    expect(screen.queryByRole("status")).toBeNull();
  });
  it("uses a fixed domain for bounded quantities (no auto-zoom of a 0-1 axis)", () => {
    const { container } = render(<LineChart series={series} xLabel="x" yLabel="y" yDomain={[0, 1]} ariaLabel="c" yFormat={(v) => v.toFixed(2)} />);
    const labels = [...container.querySelectorAll(".sm-chart-axis text")].map((t) => t.textContent);
    expect(labels).toContain("0.00");
    expect(labels).toContain("1.00");
  });
});

describe("DotWhisker on the real per-seed results", () => {
  it("one dot per seed, a mean diamond, a CI whisker, and the heuristic as a labelled dashed line", () => {
    const st = STUDY.summary.levels.L2.methods.ippo.splits.test as unknown as { stochastic: { success_rate: { per_seed: number[]; interval: { mean: number; lo: number; hi: number } } } };
    const sr = st.stochastic.success_rate;
    const { container } = render(<DotWhisker rows={[{ id: "ippo", label: "IPPO", color: "c", perSeed: sr.per_seed, mean: sr.interval.mean, lo: sr.interval.lo, hi: sr.interval.hi }]} reference={{ label: "heuristic 1.000", value: 1 }} xLabel="success" ariaLabel="dots" />);
    expect(container.querySelectorAll("circle")).toHaveLength(5);
    expect(container.querySelectorAll("path")).toHaveLength(1);
    expect(screen.getByText("heuristic 1.000")).toBeInTheDocument();
    expect(container.querySelector("line[stroke-dasharray]")).not.toBeNull();
  });
  it("hover shows mean, CI and the per-seed values", () => {
    render(<DotWhisker rows={[{ id: "a", label: "Alg", color: "c", perSeed: [0.1, 0.2], mean: 0.15, lo: 0.0, hi: 0.3 }]} xLabel="x" ariaLabel="d" />);
    fireEvent.pointerMove(screen.getByText("Alg").parentElement!);
    expect(screen.getByRole("status")).toHaveTextContent("0.150");
    expect(screen.getByRole("status")).toHaveTextContent("[0.000, 0.300]");
  });
});

describe("Heatmap on the real visit counts", () => {
  it("one rect per cell, obstacles hatched, a numeric colour bar, tooltip per cell", () => {
    const d = HM.mappo;
    const { container } = render(<Heatmap counts={d.counts} obstacles={d.obstacles} unit="visits" ariaLabel="map" />);
    const cells = [...container.querySelectorAll("rect")].filter((r) => r.getAttribute("width") === "34");
    expect(cells).toHaveLength(d.counts.length * d.counts[0].length);
    const nObs = d.obstacles.flat().filter(Boolean).length;
    expect(cells.filter((r) => r.getAttribute("fill") === "url(#sm-hatch)")).toHaveLength(nObs);
    expect([...container.querySelectorAll(".sm-chart-axis text")].length).toBe(5);
    fireEvent.pointerMove(cells[0]);
    expect(screen.getByRole("status")).toHaveTextContent("row 0, col 0");
  });
  it("sequential colour darkens with the count", () => {
    const { container } = render(<Heatmap counts={[[0, 10]]} obstacles={[[false, false]]} unit="visits" ariaLabel="m" />);
    const [a, b] = [...container.querySelectorAll("rect")].filter((r) => r.getAttribute("width") === "34").map((r) => r.getAttribute("fill")!);
    const sum = (c: string) => c.match(/\d+/g)!.map(Number).reduce((x, y) => x + y, 0);
    expect(sum(b)).toBeLessThan(sum(a));
  });
});

describe("Bars and DiscoveryTimeline on the real example episode", () => {
  it("bars start at zero (untruncated) and annotate the Gini", () => {
    const { container } = render(<Bars data={EP_MAPPO.workload_share_per_robot.map((v, i) => ({ label: `R${i + 1}`, value: v, color: "c" }))} domain={[0, 1]} xLabel="share" annotation="Gini 0.014" ariaLabel="b" />);
    const bars = [...container.querySelectorAll("rect")];
    expect(bars).toHaveLength(3);
    for (const b of bars) expect(Number(b.getAttribute("x"))).toBe(56); // shared zero baseline
    expect(screen.getByText("Gini 0.014")).toBeInTheDocument();
  });
  it("timeline marks ◆ and ★ per target with titles", () => {
    render(<DiscoveryTimeline rows={EP_MAPPO.discovery_timeline} maxT={EP_MAPPO.final_t} ariaLabel="t" />);
    expect(screen.getAllByText("◆")).toHaveLength(EP_MAPPO.discovery_timeline.length);
    expect(screen.getAllByText("★")).toHaveLength(EP_MAPPO.discovery_timeline.length);
    expect(screen.getByText(/rescued at t=21 by robot 2/)).toBeInTheDocument();
  });
});

describe("degenerate intervals", () => {
  it("a zero-width CI (lo == hi, e.g. success 1.000 in every seed) renders without duplicate-key warnings", () => {
    const errs: unknown[] = [];
    const spy = vi.spyOn(console, "error").mockImplementation((...a) => void errs.push(a));
    render(<DotWhisker rows={[{ id: "a", label: "A", color: "c", perSeed: [1, 1, 1], mean: 1, lo: 1, hi: 1 }]} xLabel="x" ariaLabel="d" />);
    expect(errs).toEqual([]);
    spy.mockRestore();
  });
});

describe("zoomed dot plots say so", () => {
  it("states that the axis does not start at zero when the domain is auto-zoomed, and not when it is fixed", () => {
    const rows = [{ id: "a", label: "A", color: "c", perSeed: [120, 125], mean: 122, lo: 118, hi: 126 }];
    const { rerender } = render(<DotWhisker rows={rows} xLabel="steps" ariaLabel="d" />);
    expect(screen.getByText(/axis does not start at 0/)).toBeInTheDocument();
    rerender(<DotWhisker rows={rows} xLabel="steps" domain={[0, 150]} ariaLabel="d" />);
    expect(screen.queryByText(/axis does not start at 0/)).toBeNull();
  });
});
