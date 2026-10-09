import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { REAL_FAIL, REAL_SUCCESS } from "../features/simulation/fixtures";
import { DEFAULT_SCENARIO, useScenario } from "../stores/scenarioStore";
import { useSimStore } from "../stores/simStore";
import { SwarmLab } from "./SwarmLab";

const entry = (r: typeof REAL_SUCCESS, run: string, step: number, frac: number) => ({
  replay_id: r.replay_id, map_id: r.map.id, algo: r.policy.kind, seed: 0, fraction: frac, checkpoint_step: step,
  success: r.outcome.success, steps: r.outcome.steps, study: "main_L2", run_id: run,
});
const INDEX = [entry(REAL_SUCCESS, "run-a", 999424, 1), entry(REAL_FAIL, "run-a", 999424, 1)];
const REPLAYS = { [REAL_SUCCESS.replay_id]: REAL_SUCCESS, [REAL_FAIL.replay_id]: REAL_FAIL };

class FakeWS {
  static last: FakeWS | null = null;
  readyState = 1;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: ((e: { code: number; reason: string }) => void) | null = null;
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeWS.last = this;
    queueMicrotask(() => this.onopen?.());
  }
  send(d: string) { this.sent.push(d); }
  close() { this.onclose?.({ code: 1000, reason: "" }); }
}

function json(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

let fetchMock: ReturnType<typeof vi.fn>;
function installFetch(opts: { down?: boolean } = {}) {
  fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (opts.down) return Promise.reject(new TypeError("Failed to fetch"));
    const path = new URL(url).pathname;
    if (path === "/api/replays") return json(INDEX);
    const m = path.match(/^\/api\/replays\/([^/]+)(\/initial)?$/);
    if (m) return m[2] ? json({ known_team: REPLAYS[m[1]].initial!.known_team, source: "derived" }) : json(REPLAYS[m[1]]);
    if (path === "/api/levels") return json([{ level: "L2", config: REAL_SUCCESS.env.config }]);
    if (path === "/api/models") return json([{ model_id: "m1", algo: "mappo", level: "L2", seed: 0, trained_env_steps: 999424, study: "main_L2" }]);
    if (path === "/api/sessions" && init?.method === "POST") {
      return json({
        session_id: "sess1", step_index: 0, done: false, stochastic: true, seed: 0, env: { level: "L2", config: REAL_SUCCESS.env.config },
        map: REAL_SUCCESS.map, policy: { kind: "heuristic", stochastic: true, eval_seed: 0 }, initial: { pos: REAL_SUCCESS.map.robot_starts, known_team: REAL_SUCCESS.initial!.known_team },
        current_pos: REAL_SUCCESS.map.robot_starts, current_known_team: REAL_SUCCESS.initial!.known_team, target_status: [0, 0], last_frame: null,
        metrics: { t: 0, coverage: 0, n_discovered: 0, n_rescued: 0, n_collisions: 0, return_mean: 0, success: false, done: false },
      }, 201);
    }
    if (init?.method === "DELETE") return Promise.resolve(new Response(null, { status: 204 }));
    return json({ detail: "not found" }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
}

beforeEach(() => {
  FakeWS.last = null;
  localStorage.clear();
  useScenario.setState({ ...DEFAULT_SCENARIO });
  useSimStore.getState().clear();
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} unobserve() {} });
  vi.stubGlobal("WebSocket", FakeWS as unknown as typeof WebSocket);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const mission = () => document.querySelector(".sm-lab-mission")!.textContent!;

describe("SwarmLab page (real replay fixtures behind a stubbed API)", () => {
  it("auto-loads the best real replay and shows the mission strip, canvas and timeline", async () => {
    installFetch();
    render(<SwarmLab />);
    await waitFor(() => expect(mission()).toContain("0/150"));
    expect(screen.getByRole("img", { name: /Simulation grid 10 by 10, 3 robots/ })).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: /scrub timeline/i })).toHaveAttribute("max", "25");
    expect(useSimStore.getState().replayInfo?.id).toBe(REAL_SUCCESS.replay_id); // success + final checkpoint preferred
  });

  it("Space plays/pauses, arrows step, R resets, 1-3 focus robots, L opens Layers", async () => {
    installFetch();
    render(<SwarmLab />);
    await waitFor(() => expect(mission()).toContain("0/150"));
    fireEvent.keyDown(window, { key: " " });
    expect(useSimStore.getState().playing).toBe(true);
    fireEvent.keyDown(window, { key: " " });
    expect(useSimStore.getState().playing).toBe(false);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    await waitFor(() => expect(mission()).toContain("2/150"));
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    await waitFor(() => expect(mission()).toContain("1/150"));
    fireEvent.keyDown(window, { key: "2" });
    expect(useSimStore.getState().focusedRobot).toBe(1);
    fireEvent.keyDown(window, { key: "6" }); // only 3 robots: ignored
    expect(useSimStore.getState().focusedRobot).toBe(1);
    fireEvent.keyDown(window, { key: "r" });
    await waitFor(() => expect(mission()).toContain("0/150"));
    fireEvent.keyDown(window, { key: "l" });
    expect(screen.getByRole("tab", { name: "Layers" })).toHaveAttribute("aria-selected", "true");
  });

  it("keys are ignored while typing in a control", async () => {
    installFetch();
    render(<SwarmLab />);
    await waitFor(() => expect(mission()).toContain("0/150"));
    fireEvent.keyDown(screen.getByLabelText("Playback speed"), { key: " " });
    expect(useSimStore.getState().playing).toBe(false);
  });

  it("picking another map loads that replay (and its own initial known cells)", async () => {
    installFetch();
    render(<SwarmLab />);
    await waitFor(() => expect(mission()).toContain("0/150"));
    fireEvent.change(screen.getByLabelText("Map"), { target: { value: REAL_FAIL.replay_id } });
    await waitFor(() => expect(useSimStore.getState().replayInfo?.id).toBe(REAL_FAIL.replay_id));
    expect(useSimStore.getState().timeline!.maxT).toBe(150);
    const urls = fetchMock.mock.calls.map((c) => String(c[0]));
    expect(urls.some((u) => u.endsWith(`/api/replays/${REAL_FAIL.replay_id}/initial`))).toBe(true);
  });

  it("switching to Live creates a session and the server drives the clock over the socket", async () => {
    installFetch();
    render(<SwarmLab />);
    await waitFor(() => expect(mission()).toContain("0/150"));
    fireEvent.click(screen.getByRole("button", { name: "Live" }));
    await waitFor(() => expect(FakeWS.last).not.toBeNull());
    const post = fetchMock.mock.calls.find((c) => (c[1] as RequestInit | undefined)?.method === "POST")!;
    expect(JSON.parse((post[1] as RequestInit).body as string)).toMatchObject({ level: "L2", policy: { kind: "heuristic" }, map: { kind: "procedural" } });
    expect(FakeWS.last!.url).toContain("/ws/sessions/sess1");
    expect(useSimStore.getState().mode).toBe("live");

    fireEvent.keyDown(window, { key: " " });
    expect(JSON.parse(FakeWS.last!.sent.at(-1)!)).toEqual({ cmd: "play" }); // a command, not a local clock
    expect(useSimStore.getState().playing).toBe(false); // until the server confirms

    const ws = FakeWS.last!;
    act(() => {
      ws.onmessage?.({ data: JSON.stringify({ type: "frame", frame: REAL_SUCCESS.frames[0], step_index: 1, metrics: {} }) });
    });
    await waitFor(() => expect(mission()).toContain("1/150"));
    fireEvent.click(screen.getByRole("button", { name: "Replay" }));
    await waitFor(() => expect(useSimStore.getState().mode).toBe("replay"));
    expect(fetchMock.mock.calls.some((c) => (c[1] as RequestInit | undefined)?.method === "DELETE")).toBe(true); // session closed
  });

  it("reconnecting and expired-session states are visible", async () => {
    installFetch();
    useScenario.setState({ mode: "live" });
    render(<SwarmLab />);
    await waitFor(() => expect(FakeWS.last).not.toBeNull());
    act(() => FakeWS.last!.onclose?.({ code: 1006, reason: "" }));
    expect(await screen.findByText(/Reconnecting/)).toBeInTheDocument();
    act(() => useSimStore.getState().setConnection("closed", "The session expired or was closed. Start a new one."));
    expect(await screen.findByText(/session expired/i)).toBeInTheDocument();
    const before = fetchMock.mock.calls.filter((c) => (c[1] as RequestInit | undefined)?.method === "POST").length;
    fireEvent.click(screen.getByRole("button", { name: /start a new session/i }));
    await waitFor(() => expect(fetchMock.mock.calls.filter((c) => (c[1] as RequestInit | undefined)?.method === "POST").length).toBe(before + 1));
  });

  it("backend unreachable: explicit error state with a retry, not a blank page", async () => {
    installFetch({ down: true });
    render(<SwarmLab />);
    expect(await screen.findByText("Backend unreachable")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    expect(mission()).toContain("NO SESSION");
  });

  it("no replays recorded: an explanatory empty state", async () => {
    installFetch();
    fetchMock.mockImplementation((url: string) => (new URL(url).pathname === "/api/replays" ? json([]) : json([])));
    render(<SwarmLab />);
    expect(await screen.findByText("No session")).toBeInTheDocument();
  });

  it("a failing replay fetch shows the error and keeps controls disabled", async () => {
    installFetch();
    const orig = fetchMock.getMockImplementation() as (u: string, i?: RequestInit) => Promise<Response>;
    fetchMock.mockImplementation((url: string, init?: RequestInit) => (/\/api\/replays\/[^/]+$/.test(url) ? json({ detail: "replay 'x' not found" }, 404) : orig(url, init)));
    render(<SwarmLab />);
    expect(await screen.findByText("Could not load the simulation")).toBeInTheDocument();
    expect(screen.getByText(/not found/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Play/ })).toBeDisabled();
  });
});
