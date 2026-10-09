import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Timeline } from "../replay/Timeline";
import { useSimStore } from "../../stores/simStore";
import { DEFAULT_LAYERS, useUiStore } from "../../stores/uiStore";
import { MissionStrip } from "./MissionStrip";
import { RightRail } from "./RightRail";
import { ScenarioDrawer } from "./ScenarioDrawer";
import { Toolbar, type ToolbarProps } from "./Toolbar";
import { REAL_FAIL, REAL_SUCCESS } from "./fixtures";
import { DEFAULT_SCENARIO } from "../../stores/scenarioStore";

afterEach(cleanup);
beforeEach(() => {
  useSimStore.getState().clear();
  useUiStore.setState({ layers: { ...DEFAULT_LAYERS }, railTab: "robots" });
});

const load = (r = REAL_SUCCESS) => act(() => useSimStore.getState().loadReplay(r, r.initial!.known_team));

describe("MissionStrip", () => {
  it("shows no-session state before anything loads", () => {
    render(<MissionStrip />);
    expect(screen.getByRole("status")).toHaveTextContent("NO SESSION");
  });

  it("reports t, rescued, coverage, collisions straight from the reducer state", () => {
    load(REAL_FAIL);
    act(() => useSimStore.getState().seek(150));
    render(<MissionStrip />);
    const text = document.body.textContent!;
    expect(text).toContain("150/150");
    expect(text).toContain("Collisions 15");
    expect(text).toContain("Rescued 0/2");
    expect(text).toContain("68%"); // coverage_final 0.681 from the real outcome
    expect(screen.getByRole("status")).toHaveTextContent("FAILED");
  });

  it("a successful replay ends as SUCCESS and playing shows RUNNING", () => {
    load();
    const { rerender } = render(<MissionStrip />);
    expect(screen.getByRole("status")).toHaveTextContent("PAUSED");
    act(() => useSimStore.getState().play());
    rerender(<MissionStrip />);
    expect(screen.getByRole("status")).toHaveTextContent("RUNNING");
    act(() => useSimStore.getState().seek(25));
    rerender(<MissionStrip />);
    expect(screen.getByRole("status")).toHaveTextContent("SUCCESS");
  });
});

describe("MissionStrip while live", () => {
  it("never claims RUNNING when the connection is down", () => {
    const snap = {
      session_id: "s", step_index: 0, done: false, stochastic: true, seed: 0, env: { level: "L2", config: REAL_SUCCESS.env.config }, map: REAL_SUCCESS.map,
      policy: REAL_SUCCESS.policy, initial: { pos: [], known_team: REAL_SUCCESS.initial!.known_team }, current_pos: REAL_SUCCESS.map.robot_starts,
      current_known_team: REAL_SUCCESS.initial!.known_team, target_status: [0, 0], last_frame: null,
      metrics: { t: 0, coverage: 0, n_discovered: 0, n_rescued: 0, n_collisions: 0, return_mean: 0, success: false, done: false },
    };
    act(() => useSimStore.getState().startLive(snap as never));
    act(() => useSimStore.setState({ playing: true }));
    act(() => useSimStore.getState().setConnection("open"));
    const { rerender } = render(<MissionStrip />);
    expect(screen.getByRole("status")).toHaveTextContent("RUNNING");
    act(() => useSimStore.getState().setConnection("reconnecting"));
    rerender(<MissionStrip />);
    expect(screen.getByRole("status")).toHaveTextContent("DISCONNECTED");
  });
});

describe("Timeline", () => {
  it("renders nothing without a replay", () => {
    const { container } = render(<Timeline />);
    expect(container).toBeEmptyDOMElement();
  });

  it("scrubbing seeks the playhead and pauses; markers use shapes ◆ ★ ✕", () => {
    load();
    render(<Timeline />);
    const scrub = screen.getByRole("slider", { name: /scrub timeline/i });
    expect(scrub).toHaveAttribute("max", "25");
    act(() => useSimStore.getState().play());
    fireEvent.change(scrub, { target: { value: "13" } });
    expect(useSimStore.getState().playhead).toBe(13);
    expect(useSimStore.getState().playing).toBe(false);
    const glyphs = [...document.querySelectorAll(".sm-lab-marker")].map((n) => n.textContent);
    expect(glyphs.filter((g) => g === "◆")).toHaveLength(2);
    expect(glyphs.filter((g) => g === "★")).toHaveLength(2);
    expect(screen.getByRole("list", { name: "Events" }).children.length).toBe(4);
  });

  it("collision markers appear for a failing episode", () => {
    load(REAL_FAIL);
    render(<Timeline />);
    expect([...document.querySelectorAll(".sm-lab-marker.collision")]).toHaveLength(15);
  });
});

describe("RightRail", () => {
  it("lists robots with observable facts and focuses on click", () => {
    load();
    act(() => useSimStore.getState().seek(8));
    render(<RightRail />);
    const rows = screen.getAllByRole("button", { name: /R\d/ });
    expect(rows).toHaveLength(3);
    expect(rows[0]).toHaveTextContent(/Moving|Waiting|Blocked|Rescued/);
    fireEvent.click(rows[1]);
    expect(useSimStore.getState().focusedRobot).toBe(1);
    expect(screen.getAllByRole("button", { name: /R\d/ })[1]).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getAllByRole("button", { name: /R\d/ })[1]);
    expect(useSimStore.getState().focusedRobot).toBeNull();
  });

  it("layer toggles drive the ui store; ground truth is off by default and labelled", () => {
    render(<RightRail />);
    fireEvent.click(screen.getByRole("tab", { name: "Layers" }));
    const gt = screen.getByRole("checkbox", { name: /ground truth/i });
    expect(gt).not.toBeChecked();
    expect(screen.getByText("Not visible to robots")).toBeInTheDocument();
    fireEvent.click(gt);
    expect(useUiStore.getState().layers.groundTruth).toBe(true);
    fireEvent.click(screen.getByRole("checkbox", { name: /sensor radius/i }));
    expect(useUiStore.getState().layers.sensor).toBe(false);
  });

  it("Live tab shows the coverage series and the frame's real reward components", () => {
    load();
    act(() => useSimStore.getState().seek(10));
    render(<RightRail />);
    fireEvent.click(screen.getByRole("tab", { name: "Live" }));
    expect(screen.getByRole("img", { name: /Coverage over time/ })).toBeInTheDocument();
    const comps = Object.keys(REAL_SUCCESS.frames[9].reward_components);
    for (const k of comps) expect(screen.getByText(k)).toBeInTheDocument();
  });

  it("empty states instead of blank panels", () => {
    render(<RightRail />);
    expect(screen.getByText(/No session/)).toBeInTheDocument();
  });
});

describe("Toolbar", () => {
  const base = (over: Partial<ToolbarProps> = {}): ToolbarProps => ({
    mode: "replay",
    onMode: vi.fn(),
    selectors: [{ label: "Policy", value: "a", options: [{ value: "a", label: "A" }, { value: "b", label: "B", group: "G" }], onChange: vi.fn() }],
    playback: { playing: false, speed: 1, enabled: true, canStep: true, canStepBack: false, onToggle: vi.fn(), onStep: vi.fn(), onStepBack: vi.fn(), onReset: vi.fn(), onSpeed: vi.fn() },
    onOpenDrawer: vi.fn(),
    ...over,
  });

  it("fires every control callback", () => {
    const p = base();
    render(<Toolbar {...p} />);
    fireEvent.click(screen.getByRole("button", { name: /Play/ }));
    fireEvent.click(screen.getByRole("button", { name: /^Step \(/ }));
    fireEvent.click(screen.getByRole("button", { name: /Reset/ }));
    fireEvent.click(screen.getByRole("button", { name: "Live" }));
    fireEvent.click(screen.getByRole("button", { name: /Scenario settings/ }));
    fireEvent.change(screen.getByLabelText("Playback speed"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("Policy"), { target: { value: "b" } });
    expect(p.playback.onToggle).toHaveBeenCalled();
    expect(p.playback.onStep).toHaveBeenCalled();
    expect(p.playback.onReset).toHaveBeenCalled();
    expect(p.onMode).toHaveBeenCalledWith("live");
    expect(p.onOpenDrawer).toHaveBeenCalled();
    expect(p.playback.onSpeed).toHaveBeenCalledWith(4);
    expect(p.selectors[0].onChange).toHaveBeenCalledWith("b");
  });

  it("disables stepping back when it can't, everything when no session; labels the toggle by state", () => {
    const { rerender } = render(<Toolbar {...base()} />);
    expect(screen.getByRole("button", { name: /Step back/ })).toBeDisabled();
    rerender(<Toolbar {...base({ playback: { ...base().playback, enabled: false } })} />);
    expect(screen.getByRole("button", { name: /Play/ })).toBeDisabled();
    rerender(<Toolbar {...base({ playback: { ...base().playback, playing: true } })} />);
    expect(screen.getByRole("button", { name: /Pause/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Replay" })).toHaveAttribute("aria-pressed", "true");
  });

  it("groups options and shows a placeholder for an unknown value", () => {
    render(<Toolbar {...base({ selectors: [{ label: "Map", value: "zzz", options: [{ value: "a", label: "A", group: "G" }], onChange: vi.fn(), placeholder: "Pick" }] })} />);
    const sel = screen.getByLabelText("Map") as HTMLSelectElement;
    expect(sel.value).toBe("");
    expect(within(sel).getByText("Pick")).toBeInTheDocument();
    expect(sel.querySelector("optgroup")?.getAttribute("label")).toBe("G");
  });
});

describe("ScenarioDrawer", () => {
  const levels = [{ level: "L1", config: { width: 8, height: 8, n_robots: 2, n_targets: 1, obstacle_density: 0.1, comm_mode: "shared_map", max_steps: 100 } }, { level: "L2", config: { width: 10, height: 10, n_robots: 3, n_targets: 2, obstacle_density: 0.15, comm_mode: "shared_map", max_steps: 150 } }];

  it("is a dialog over the rail, closed by Escape, showing the real level facts", () => {
    const onClose = vi.fn();
    render(<ScenarioDrawer open onClose={onClose} scenario={DEFAULT_SCENARIO} levels={levels} onApply={vi.fn()} />);
    expect(screen.getByRole("dialog", { name: /scenario/i })).toBeInTheDocument();
    expect(screen.getByText("10×10")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });

  it("renders nothing when closed; applies seed and sampling on submit", () => {
    const { rerender } = render(<ScenarioDrawer open={false} onClose={vi.fn()} scenario={DEFAULT_SCENARIO} levels={levels} onApply={vi.fn()} />);
    expect(screen.queryByRole("dialog")).toBeNull();
    const onApply = vi.fn();
    const onClose = vi.fn();
    rerender(<ScenarioDrawer open onClose={onClose} scenario={DEFAULT_SCENARIO} levels={levels} onApply={onApply} />);
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "7" } });
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: /apply and restart/i }));
    expect(onApply).toHaveBeenCalledWith({ seed: 7, stochastic: false });
    expect(onClose).toHaveBeenCalled();
  });

  it("changing level resets the policy and map choice", () => {
    const onApply = vi.fn();
    render(<ScenarioDrawer open onClose={vi.fn()} scenario={{ ...DEFAULT_SCENARIO, policyKey: "mappo:x", mapSet: "test", mapId: "m" }} levels={levels} onApply={onApply} />);
    fireEvent.change(screen.getByLabelText("Level"), { target: { value: "L1" } });
    expect(onApply).toHaveBeenCalledWith({ level: "L1", policyKey: "heuristic", mapSet: "procedural", mapId: null });
  });
});
