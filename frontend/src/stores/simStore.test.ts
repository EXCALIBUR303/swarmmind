import { beforeEach, describe, expect, it, vi } from "vitest";
import { REAL_FAIL, REAL_SUCCESS } from "../features/simulation/fixtures";
import { useSimStore, type LiveCommand } from "./simStore";
import type { SessionSnapshot, WsEpisodeEnd, WsFrameMessage, WsStateMessage } from "../api/wire.gen";

const S = () => useSimStore.getState();
const initial = (r: typeof REAL_SUCCESS) => r.initial!.known_team;

beforeEach(() => {
  S().clear();
  S().setCommandSink(null);
  useSimStore.setState({ speed: 1 });
});

describe("replay playback", () => {
  beforeEach(() => S().loadReplay(REAL_SUCCESS, initial(REAL_SUCCESS)));

  it("loads paused at t=0 with the real policy/outcome", () => {
    expect(S().mode).toBe("replay");
    expect(S().playhead).toBe(0);
    expect(S().playing).toBe(false);
    expect(S().timeline!.maxT).toBe(25);
    expect(S().policy?.kind).toBe("mappo");
    expect(S().replayInfo?.outcome.success).toBe(true);
  });

  it("step forward/back clamp at the ends and pause", () => {
    S().stepBack();
    expect(S().playhead).toBe(0);
    S().stepForward();
    S().stepForward();
    expect(S().playhead).toBe(2);
    S().stepBack();
    expect(S().playhead).toBe(1);
    S().seek(999);
    expect(S().playhead).toBe(25);
    S().stepForward();
    expect(S().playhead).toBe(25);
  });

  it("advance moves the playhead at 4 steps/s x speed and stops at the end", () => {
    S().play();
    S().advance(0.5);
    expect(S().playhead).toBeCloseTo(2, 5);
    S().setSpeed(4);
    S().advance(0.25); // 4 * 4 * 0.25
    expect(S().playhead).toBeCloseTo(6, 5);
    S().advance(100);
    expect(S().playhead).toBe(25);
    expect(S().playing).toBe(false);
  });

  it("does not advance while paused, and play at the end restarts", () => {
    S().advance(5);
    expect(S().playhead).toBe(0);
    S().seek(25);
    S().play();
    expect(S().playhead).toBe(0);
    expect(S().playing).toBe(true);
  });

  it("seek pauses; reset returns to 0; version bumps only when a whole step is crossed", () => {
    S().play();
    S().seek(10);
    expect(S().playing).toBe(false);
    S().reset();
    expect(S().playhead).toBe(0);
    S().play();
    const v0 = S().version;
    S().advance(0.05); // 0.2 of a step
    expect(S().version).toBe(v0);
    S().advance(0.2); // crosses 1
    expect(S().version).toBe(v0 + 1);
  });

  it("focus toggles and clears on load", () => {
    S().toggleFocus(1);
    expect(S().focusedRobot).toBe(1);
    S().toggleFocus(1);
    expect(S().focusedRobot).toBeNull();
    S().focus(2);
    S().loadReplay(REAL_FAIL, initial(REAL_FAIL));
    expect(S().focusedRobot).toBeNull();
    expect(S().timeline!.maxT).toBe(150);
  });
});

describe("live session", () => {
  const frames = REAL_SUCCESS.frames;
  const snap = (over: Partial<SessionSnapshot> = {}): SessionSnapshot =>
    ({
      session_id: "sess1",
      step_index: 0,
      done: false,
      stochastic: true,
      seed: 0,
      env: { level: "L2", config: REAL_SUCCESS.env.config },
      map: REAL_SUCCESS.map,
      policy: REAL_SUCCESS.policy,
      initial: { pos: REAL_SUCCESS.map.robot_starts, known_team: initial(REAL_SUCCESS) },
      current_pos: REAL_SUCCESS.map.robot_starts,
      current_known_team: initial(REAL_SUCCESS),
      target_status: [0, 0],
      last_frame: null,
      metrics: { t: 0, coverage: 0, n_discovered: 0, n_rescued: 0, n_collisions: 0, return_mean: 0, success: false, done: false },
      ...over,
    }) as SessionSnapshot;
  const frameMsg = (i: number): WsFrameMessage => ({ type: "frame", frame: frames[i], step_index: i + 1, metrics: snap().metrics });
  const stateMsg = (playing: boolean, s: SessionSnapshot): WsStateMessage => ({ type: "state", playing, speed: 1, snapshot: s });

  it("starts at t=0 with the initial known cells and appends frames in order", () => {
    S().startLive(snap());
    expect(S().mode).toBe("live");
    expect(S().timeline!.stateAt(0).knownFree).toBeGreaterThan(0);
    for (let i = 0; i < 5; i++) S().pushLiveFrame(frameMsg(i), 1000 + i);
    expect(S().timeline!.maxT).toBe(5);
    expect(S().playhead).toBe(5);
    expect(S().live!.lastFrameAt).toBe(1004);
  });

  it("ignores an out-of-order frame rather than corrupting the timeline", () => {
    S().startLive(snap());
    S().pushLiveFrame(frameMsg(3));
    expect(S().timeline!.maxT).toBe(0);
  });

  it("a state message whose step matches the timeline just syncs play/speed", () => {
    S().startLive(snap());
    const tl = S().timeline;
    S().applyLiveState(stateMsg(true, snap()));
    expect(S().playing).toBe(true);
    expect(S().timeline).toBe(tl); // not rebuilt
  });

  it("a state message from further ahead (reconnect after missed frames) rebuilds the timeline from the snapshot", () => {
    S().startLive(snap());
    S().pushLiveFrame(frameMsg(0));
    const ref = (() => {
      const { timeline } = S();
      return timeline!;
    })();
    void ref;
    // the server is now at step 12 and we missed 11 frames
    const full = REAL_SUCCESS.frames[11];
    const resync = snap({
      step_index: 12,
      current_pos: full.pos,
      current_known_team: initial(REAL_SUCCESS),
      target_status: [1, 0],
      metrics: { t: 12, coverage: 0.5, n_discovered: 1, n_rescued: 0, n_collisions: 0, return_mean: 0, success: false, done: false },
    });
    S().applyLiveState(stateMsg(true, resync));
    expect(S().timeline!.startT).toBe(12);
    expect(S().playhead).toBe(12);
    expect(S().timeline!.stateAt(12).pos).toEqual(full.pos);
    S().pushLiveFrame({ type: "frame", frame: frames[12], step_index: 13, metrics: resync.metrics });
    expect(S().timeline!.maxT).toBe(13);
  });

  it("a reset (server step < ours) starts the timeline over", () => {
    S().startLive(snap());
    for (let i = 0; i < 4; i++) S().pushLiveFrame(frameMsg(i));
    S().applyLiveState(stateMsg(false, snap()));
    expect(S().timeline!.maxT).toBe(0);
    expect(S().timeline!.events).toHaveLength(0);
  });

  it("ignores state for another session", () => {
    S().startLive(snap());
    S().applyLiveState(stateMsg(true, snap({ session_id: "other" })));
    expect(S().playing).toBe(false);
  });

  it("episode_end records the outcome and stops play", () => {
    S().startLive(snap());
    S().applyLiveState(stateMsg(true, snap()));
    const end: WsEpisodeEnd = { type: "episode_end", metrics: { n_rescued: 2 }, success: true, steps: 25 };
    S().endLive(end);
    expect(S().playing).toBe(false);
    expect(S().live!.done).toBe(true);
    expect(S().live!.success).toBe(true);
  });

  it("controls are commands to the server; the local clock never advances live", () => {
    const sent: LiveCommand[] = [];
    S().setCommandSink((c) => (sent.push(c), true));
    S().startLive(snap());
    S().play();
    S().pause();
    S().stepForward();
    S().stepBack(); // replay only: no command
    S().reset();
    S().setSpeed(4);
    expect(sent).toEqual([{ cmd: "play" }, { cmd: "pause" }, { cmd: "step" }, { cmd: "reset" }, { cmd: "speed", value: 4 }]);
    S().advance(10);
    expect(S().playhead).toBe(0);
    expect(S().playing).toBe(false); // `playing` only changes when the server says so
  });

  it("connection status and notices", () => {
    S().startLive(snap());
    S().setConnection("reconnecting");
    expect(S().live!.connection).toBe("reconnecting");
    S().setConnection("closed", "The session expired.");
    expect(S().live!.notice).toBe("The session expired.");
    const spy = vi.fn();
    S().setCommandSink(spy);
    S().setCommandSink(null);
    S().play();
    expect(spy).not.toHaveBeenCalled();
  });
});
