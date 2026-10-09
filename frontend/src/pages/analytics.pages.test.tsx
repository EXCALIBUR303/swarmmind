import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { EP_IPPO, EP_MAPPO, HM, PREREG, RUN_DETAIL, RUN_EVAL, RUN_METRICS, RUNS as REAL_RUNS, STUDIES, STUDY, TRAIN_SCHEMA } from "../features/fixtures";
import { REAL_FAIL, REAL_SUCCESS } from "../features/simulation/fixtures";
import type { Job } from "../api/endpoints/analytics";
import type { ReplayIndexEntry } from "../api/endpoints/catalog";
import { AlgorithmComparison } from "./AlgorithmComparison";
import { ExperimentHistory } from "./ExperimentHistory";
import { RunDetail } from "./RunDetail";
import { SwarmAnalysis } from "./SwarmAnalysis";
import { TrainSwarm } from "./TrainSwarm";

const RUNS = [...REAL_RUNS, { run_id: "s9-corrupt", study: "x", algo: "ippo", level: "L1", status: "invalid" }];
const json = (b: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(b), { status, headers: { "Content-Type": "application/json" } }));
type Routes_ = Record<string, (url: URL, init?: RequestInit) => unknown>;
let calls: { path: string; method: string }[] = [];

function stub(routes: Routes_, fallback: "404" | "down" = "404") {
  calls = [];
  vi.stubGlobal("fetch", vi.fn((url: string, init?: RequestInit) => {
    const u = new URL(url);
    calls.push({ path: u.pathname + u.search, method: init?.method ?? "GET" });
    const hit = Object.entries(routes).find(([k]) => (k.endsWith("*") ? u.pathname.startsWith(k.slice(0, -1)) : u.pathname === k));
    if (hit) {
      const r = hit[1](u, init);
      return r instanceof Promise ? r : json(r);
    }
    return fallback === "down" ? Promise.reject(new TypeError("Failed to fetch")) : json({ detail: "not found" }, 404);
  }));
}
const media = (matches: boolean) =>
  vi.stubGlobal("matchMedia", (q: string) => ({ matches, media: q, addEventListener() {}, removeEventListener() {} }));

beforeEach(() => {
  media(false);
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} unobserve() {} });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  vi.useRealTimers();
});
const wrap = (ui: React.ReactNode, at = "/") => render(<MemoryRouter initialEntries={[at]}>{ui}</MemoryRouter>);

describe("Algorithm Comparison page", () => {
  const ok: Routes_ = { "/api/studies": () => STUDIES, "/api/studies/main_L2": () => STUDY, "/api/preregistration": () => PREREG };
  it("shows the real results, hypotheses and curves for the main study", async () => {
    stub(ok);
    wrap(<AlgorithmComparison />);
    expect(await screen.findByRole("heading", { name: "Results" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /Results table, test stochastic/ })).toBeInTheDocument();
    expect(await screen.findByText(/H2 not supported/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Learning curves" })).toBeInTheDocument();
    expect(screen.getByText(/None: every planned run completed/)).toBeInTheDocument();
  });
  it("switching the evaluation set re-renders every table from the same summary", async () => {
    stub(ok);
    wrap(<AlgorithmComparison />);
    await screen.findByRole("heading", { name: "Results" });
    fireEvent.click(screen.getByRole("button", { name: "structured" }));
    expect(screen.getByRole("region", { name: /Results table, structured stochastic/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /greedy/ }));
    expect(screen.getByRole("region", { name: /structured greedy/ })).toBeInTheDocument();
  });
  it("only smoke studies: they are offered, flagged as not a result", async () => {
    stub({ ...ok, "/api/studies": () => [{ ...STUDIES[0], smoke: true }], "/api/studies/main_L2": () => ({ ...STUDY, summary: { ...STUDY.summary, smoke: true } }) });
    wrap(<AlgorithmComparison />);
    expect(await screen.findByText(/smoke-test study/)).toBeInTheDocument();
  });
  it("no study yet: an actionable empty state naming the command", async () => {
    stub({ "/api/studies": () => [] });
    wrap(<AlgorithmComparison />);
    expect(await screen.findByText("No completed study yet")).toBeInTheDocument();
    expect(screen.getByText(/scripts\/run_study.py/)).toBeInTheDocument();
  });
  it("missing runs are listed, not hidden", async () => {
    stub({ ...ok, "/api/studies/main_L2": () => ({ ...STUDY, summary: { ...STUDY.summary, missing: [{ algo: "mappo", seed: 4, reason: "crashed" }] } }) });
    wrap(<AlgorithmComparison />);
    expect(await screen.findByText(/crashed/)).toBeInTheDocument();
  });
  it("backend down and a failing study fetch are explicit errors with retry", async () => {
    stub({}, "down");
    wrap(<AlgorithmComparison />);
    expect(await screen.findByText("Backend unreachable")).toBeInTheDocument();
    cleanup();
    stub({ "/api/studies": () => STUDIES });
    wrap(<AlgorithmComparison />);
    expect(await screen.findByText("Could not load the study")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});

describe("Swarm Analysis page", () => {
  const entry = (id: string, frac: number, r: typeof REAL_SUCCESS, ck: number): ReplayIndexEntry => ({ replay_id: id, map_id: "L2-struct-m", algo: "mappo", seed: 0, fraction: frac, checkpoint_step: ck, success: r.outcome.success, steps: r.outcome.steps, study: "main_L2", run_id: "run-a" });
  const index = [entry("rp0", 0, REAL_FAIL, 0), entry("rp1", 1, REAL_SUCCESS, 999424)];
  const files = ["L2_mappo_example_episode.json", "L2_ippo_example_episode.json", "L2_mappo_L2-struct-bottleneck_vertical_heatmap.json", "L2_ippo_L2-struct-bottleneck_vertical_heatmap.json", "L2_coordination.json"];
  const routes: Routes_ = {
    "/api/studies": () => STUDIES,
    "/api/studies/main_L2": () => ({ ...STUDY, analysis_files: files }),
    "/api/studies/main_L2/analysis/L2_mappo_example_episode": () => EP_MAPPO,
    "/api/studies/main_L2/analysis/L2_ippo_example_episode": () => EP_IPPO,
    "/api/analysis/main_L2/heatmap": (u) => HM[u.searchParams.get("algo") as "ippo" | "mappo"],
    "/api/replays": () => index,
    "/api/replays/rp0": () => ({ ...REAL_FAIL, replay_id: "rp0" }),
    "/api/replays/rp1": () => ({ ...REAL_SUCCESS, replay_id: "rp1" }),
    "/api/replays/rp0/initial": () => ({ known_team: REAL_FAIL.initial!.known_team }),
    "/api/replays/rp1/initial": () => ({ known_team: REAL_SUCCESS.initial!.known_team }),
  };
  it("renders heatmaps, the illustrative-episode charts and labels them as not a statistic", async () => {
    stub(routes);
    wrap(<SwarmAnalysis />);
    expect(await screen.findByRole("img", { name: /MAPPO visits heatmap on bottleneck_vertical/ })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /IPPO visits heatmap/ })).toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /Coverage over time for the illustrative episode/ })).toBeInTheDocument();
    expect(screen.getAllByText(/illustrative episode, not a statistic/i).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByRole("img", { name: "Discovery and rescue times per target" })).toBeInTheDocument();
    expect(screen.getByText("Gini 0.014")).toBeInTheDocument();
  });
  it("switching to collisions refetches the heatmaps; switching algorithm loads the other episode", async () => {
    stub(routes);
    wrap(<SwarmAnalysis />);
    await screen.findByRole("img", { name: /MAPPO visits heatmap/ });
    fireEvent.click(screen.getByRole("button", { name: "collisions" }));
    await waitFor(() => expect(calls.some((c) => c.path.includes("kind=collisions"))).toBe(true));
    fireEvent.click(screen.getByRole("button", { name: "IPPO" }));
    await waitFor(() => expect(calls.some((c) => c.path.endsWith("L2_ippo_example_episode"))).toBe(true));
  });
  it("Swarm Evolution plays all checkpoints on ONE shared clock with a mission strip per panel", async () => {
    stub(routes);
    wrap(<SwarmAnalysis />);
    const slider = await screen.findByRole("slider", { name: "Shared timeline" });
    expect(await screen.findAllByText(/% of training/)).toHaveLength(2);
    expect(slider).toHaveAttribute("max", "150"); // the longest replay sets the shared length
    fireEvent.change(slider, { target: { value: "30" } });
    const panels = screen.getAllByRole("figure").filter((f) => /of training/.test(f.textContent ?? ""));
    expect(panels).toHaveLength(2);
    expect(within(panels[0]).getByText("running")).toBeInTheDocument(); // the failing checkpoint is still going
    expect(within(panels[1]).getByText("✓ SUCCESS")).toBeInTheDocument(); // the trained one finished at 25
    expect(within(panels[1]).getByText(/rescued 2\/2/)).toBeInTheDocument();
  });
  it("on narrow screens Evolution shows one canvas and a checkpoint selector", async () => {
    media(true);
    stub(routes);
    wrap(<SwarmAnalysis />);
    await screen.findByRole("slider", { name: "Shared timeline" });
    expect(screen.getAllByRole("figure").filter((f) => /of training/.test(f.textContent ?? ""))).toHaveLength(1);
    fireEvent.change(screen.getByLabelText("Checkpoint"), { target: { value: "1" } });
    expect(screen.getByText(/100% of training/)).toBeInTheDocument();
  });
  it("a study with no structured maps says so instead of drawing nothing", async () => {
    stub({ ...routes, "/api/studies/main_L2": () => ({ ...STUDY, analysis_files: ["L2_coordination.json"] }) });
    wrap(<SwarmAnalysis />);
    expect(await screen.findByText(/No structured \(G2\) maps/)).toBeInTheDocument();
    expect(screen.getByText("No example episode for this algorithm")).toBeInTheDocument();
  });
  it("no checkpoint replays: an actionable empty state", async () => {
    stub({ ...routes, "/api/replays": () => [] });
    wrap(<SwarmAnalysis />);
    expect(await screen.findByText("No checkpoint replays")).toBeInTheDocument();
  });
});

describe("Experiment History and run detail", () => {
  it("lists real runs; sorting and filtering work; an unreadable manifest shows as invalid", async () => {
    stub({ "/api/experiments": () => RUNS });
    wrap(<ExperimentHistory />);
    expect(await screen.findByRole("region", { name: "Runs" })).toBeInTheDocument();
    expect(within(screen.getByRole("region", { name: "Runs" })).getByText("invalid")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(`${RUNS.length} of ${RUNS.length} runs`);
    fireEvent.click(screen.getByRole("button", { name: /Seed/ }));
    const seedHeader = screen.getByRole("columnheader", { name: /Seed/ });
    expect(seedHeader).toHaveAttribute("aria-sort", "ascending");
    fireEvent.click(screen.getByRole("button", { name: /Seed/ }));
    expect(seedHeader).toHaveAttribute("aria-sort", "descending");
    fireEvent.change(screen.getByLabelText("algo"), { target: { value: "mappo" } });
    const n = RUNS.filter((r) => r.algo === "mappo").length;
    expect(screen.getByRole("status")).toHaveTextContent(`${n} of ${RUNS.length} runs`);
    fireEvent.change(screen.getByLabelText("status"), { target: { value: "invalid" } }); // the only invalid row is IPPO, so MAPPO + invalid matches nothing
    expect(screen.getByText("No run matches these filters")).toBeInTheDocument();
  });
  it("small screens get cards with the same fields", async () => {
    media(true);
    stub({ "/api/experiments": () => RUNS });
    wrap(<ExperimentHistory />);
    expect(await screen.findAllByRole("listitem")).toHaveLength(RUNS.length);
    expect(screen.queryByRole("table")).toBeNull();
  });
  it("empty and error states", async () => {
    stub({ "/api/experiments": () => [] });
    wrap(<ExperimentHistory />);
    expect(await screen.findByText("No runs yet")).toBeInTheDocument();
    cleanup();
    stub({}, "down");
    wrap(<ExperimentHistory />);
    expect(await screen.findByText("Backend unreachable")).toBeInTheDocument();
  });
  it("run detail shows the manifest, real training curves and links recorded replays into Swarm Lab", async () => {
    const rid = RUN_DETAIL.summary.run_id;
    stub({
      [`/api/experiments/${rid}`]: () => RUN_DETAIL,
      [`/api/experiments/${rid}/metrics`]: () => RUN_METRICS,
      [`/api/experiments/${rid}/eval`]: () => RUN_EVAL,
      "/api/replays": () => [{ replay_id: "rp1", map_id: "L2-struct-bottleneck_vertical", algo: "mappo", seed: 0, fraction: 1, checkpoint_step: 999424, success: true, steps: 25, study: "main_L2", run_id: rid }],
    });
    render(<MemoryRouter initialEntries={[`/experiments/${rid}`]}><Routes><Route path="/experiments/:runId" element={<RunDetail />} /></Routes></MemoryRouter>);
    expect(await screen.findByRole("heading", { name: /MAPPO · L2 · seed 0/ })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Policy loss" })).toBeInTheDocument();
    expect(screen.getByText("config_hash")).toBeInTheDocument();
    const link = await screen.findByRole("link", { name: /bottleneck_vertical/ });
    expect(link).toHaveAttribute("href", "/?replay=rp1");
    expect(JSON.stringify(RUN_DETAIL)).not.toMatch(/\/Users\//);
  });
  it("an unknown run is a clear error with a way back", async () => {
    stub({});
    render(<MemoryRouter initialEntries={["/experiments/nope"]}><Routes><Route path="/experiments/:runId" element={<RunDetail />} /></Routes></MemoryRouter>);
    expect(await screen.findByText("Could not load this run")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /All runs/ })).toBeInTheDocument();
  });
});

describe("Train Swarm page", () => {
  const job = (over: Partial<Job> = {}): Job => ({ job_id: "j1", status: "running", run_id: "run-x", request: { level: "L1", algo: "ippo", seed: 0, total_env_steps: 20000, overrides: {} }, created_at: "2026-01-01T00:00:00Z", finished_at: null, exit_code: null, run_status: "running", progress: { update: 3, env_steps: 6000, total_env_steps: 20000 }, log_tail: null, ...over });
  const base: Routes_ = { "/api/training/schema": () => TRAIN_SCHEMA };
  it("no job yet: the form works and the right side explains what happens next", async () => {
    stub({ ...base, "/api/training/jobs": () => [] });
    wrap(<TrainSwarm />);
    expect(await screen.findByRole("form", { name: "Training configuration" })).toBeInTheDocument();
    expect(await screen.findByText("No training job yet")).toBeInTheDocument();
  });
  it("launching posts the validated request and shows the job card; the form locks while it runs", async () => {
    stub({ ...base, "/api/training/jobs": (_u, init) => (init?.method === "POST" ? job({ status: "starting", progress: null, run_id: null }) : []) });
    wrap(<TrainSwarm />);
    await screen.findByText("No training job yet");
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    expect(await screen.findByRole("region", { name: "Job status" })).toHaveTextContent("starting");
    const post = calls.find((c) => c.method === "POST")!;
    expect(post.path).toBe("/api/training/jobs");
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
  });
  it("adopts a job already running (page reload), shows progress and offers Cancel", async () => {
    stub({ ...base, "/api/training/jobs": () => [job()], "/api/training/jobs/j1/cancel": () => job({ status: "cancelled", run_status: "interrupted" }), "/api/experiments/*": () => [], "/api/experiments/run-x": () => RUN_DETAIL });
    wrap(<TrainSwarm />);
    const card = await screen.findByRole("region", { name: "Job status" });
    expect(card).toHaveTextContent("6k / 20k env steps");
    expect(card).toHaveTextContent("measuring speed…");
    fireEvent.click(screen.getByRole("button", { name: "Cancel job" }));
    await waitFor(() => expect(screen.getByRole("region", { name: "Job status" })).toHaveTextContent("cancelled"));
  });
  it("a failed job shows the tail of its real output, not a generic message", async () => {
    stub({ ...base, "/api/training/jobs": () => [job({ status: "failed", exit_code: 1, run_status: "failed", log_tail: ["Traceback (most recent call last):", "ValueError: boom"] })], "/api/experiments/*": () => [] });
    wrap(<TrainSwarm />);
    expect(await screen.findByText(/ValueError: boom/)).toBeInTheDocument();
    expect(screen.getByText(/exit code 1/)).toBeInTheDocument();
  });
  it("a server rejection (e.g. one job at a time) is shown next to the form", async () => {
    stub({ ...base, "/api/training/jobs": (_u, init) => (init?.method === "POST" ? json({ detail: "at most 1 training job(s) may run at once" }, 409) : []) });
    wrap(<TrainSwarm />);
    await screen.findByText("No training job yet");
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    expect(await screen.findByText(/at most 1 training job/)).toBeInTheDocument();
  });
  it("backend unreachable is an explicit error", async () => {
    stub({}, "down");
    wrap(<TrainSwarm />);
    expect(await screen.findByText("Backend unreachable")).toBeInTheDocument();
  });
});
