/**
 * Playback + live-session state (spec §29.2). High-frequency values (the
 * playhead, the timeline) are read by the canvas loop via `getState()` outside
 * React, so a 60 fps animation never triggers a React render. React components
 * subscribe to slow-changing slices (`version` bumps once per step).
 */
import { create } from "zustand";
import { loadPref } from "../lib/prefs";
import type { ReplayInfoLite } from "./types";
import { createInitialState, stateFromSnapshot } from "../features/simulation/frameReducer";
import { envMeta, normalizeFrame, type Frame, type SimMeta } from "../features/simulation/model";
import { Timeline } from "../features/simulation/timeline";
import type { Connection } from "../api/ws";
import type { Replay, ReplayFrame, ReplayOutcome, ReplayPolicyInfo, SessionSnapshot, WsEpisodeEnd, WsFrameMessage, WsStateMessage } from "../api/wire.gen";

export const BASE_STEPS_PER_SECOND = 4; // 1x; mirrors the server's ws_base_steps_per_second
export const MAX_STEPS_PER_SECOND = 30;
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8] as const;

export type Mode = "none" | "replay" | "live";
export type Load = "idle" | "loading" | "ready" | "error";

export interface LiveInfo {
  sessionId: string;
  connection: Connection;
  done: boolean;
  success: boolean | null;
  finalMetrics: Record<string, unknown> | null;
  lastFrameAt: number;
  seed: number;
  stochastic: boolean;
  notice: string | null;
}

export type LiveCommand = { cmd: "play" | "pause" | "step" | "reset" | "speed"; value?: number | Record<string, unknown> | null };

interface SimStore {
  mode: Mode;
  load: Load;
  error: string | null;
  meta: SimMeta | null;
  timeline: Timeline | null;
  policy: ReplayPolicyInfo | null;
  replayInfo: ReplayInfoLite | null;
  version: number; // bumps per step / structural change
  playhead: number;
  playing: boolean;
  speed: number;
  focusedRobot: number | null;
  live: LiveInfo | null;

  setLoading: () => void;
  setError: (message: string) => void;
  clear: () => void;
  loadReplay: (replay: Replay, initialKnown: number[][], initialDiscovered?: number[]) => void;
  startLive: (snap: SessionSnapshot) => void;
  applyLiveState: (msg: WsStateMessage) => void;
  pushLiveFrame: (msg: WsFrameMessage, now?: number) => void;
  endLive: (msg: WsEpisodeEnd) => void;
  setConnection: (c: Connection, notice?: string | null) => void;
  setNotice: (n: string | null) => void;

  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  stepForward: () => void;
  stepBack: () => void;
  seek: (t: number) => void;
  reset: () => void;
  setSpeed: (s: number) => void;
  toggleFocus: (robot: number) => void;
  focus: (robot: number | null) => void;
  advance: (dtSeconds: number) => void;
  setCommandSink: (sink: ((c: LiveCommand) => boolean) | null) => void;
}

let commandSink: ((c: LiveCommand) => boolean) | null = null;

export const useSimStore = create<SimStore>((set, get) => {
  const liveSend = (c: LiveCommand): boolean => (commandSink ? commandSink(c) : false);
  const clampT = (t: number): number => {
    const tl = get().timeline;
    return tl ? Math.min(Math.max(t, tl.startT), tl.maxT) : 0;
  };

  function buildLive(snap: SessionSnapshot, prev: LiveInfo | null): Partial<SimStore> {
    const meta = envMeta({
      mapId: snap.map.id,
      width: snap.map.width,
      height: snap.map.height,
      terrain: snap.map.terrain,
      robotStarts: snap.map.robot_starts,
      targets: snap.map.targets,
      config: (snap.env as { config: Record<string, unknown> }).config,
    });
    const base =
      snap.step_index === 0 ? createInitialState(meta, snap.initial.known_team, snap.initial.discovered ?? []) : stateFromSnapshot(meta, snap as never);
    const timeline = new Timeline(meta, base);
    return {
      mode: "live",
      load: "ready",
      error: null,
      meta,
      timeline,
      policy: snap.policy,
      replayInfo: null,
      playhead: timeline.maxT,
      version: get().version + 1,
      live: {
        sessionId: snap.session_id,
        connection: prev?.connection ?? "connecting",
        done: snap.done,
        success: snap.done ? snap.metrics.success : null,
        finalMetrics: (snap.final_metrics as Record<string, unknown> | null | undefined) ?? null,
        lastFrameAt: 0,
        seed: snap.seed,
        stochastic: snap.stochastic,
        notice: null,
      },
    };
  }

  return {
    mode: "none",
    load: "idle",
    error: null,
    meta: null,
    timeline: null,
    policy: null,
    replayInfo: null,
    version: 0,
    playhead: 0,
    playing: false,
    speed: loadPref<number>("defaultSpeed", 1),
    focusedRobot: null,
    live: null,

    setLoading: () => set({ load: "loading", error: null }),
    setError: (message) => set({ load: "error", error: message, playing: false }),
    clear: () =>
      set({ mode: "none", load: "idle", error: null, meta: null, timeline: null, policy: null, replayInfo: null, playhead: 0, playing: false, focusedRobot: null, live: null, version: get().version + 1 }),

    loadReplay: (replay, initialKnown, initialDiscovered = []) => {
      const meta = envMeta({
        mapId: replay.map.id,
        width: replay.map.width,
        height: replay.map.height,
        terrain: replay.map.terrain,
        robotStarts: replay.map.robot_starts,
        targets: replay.map.targets,
        config: replay.env.config,
      });
      const frames: Frame[] = replay.frames.map((f: ReplayFrame) => normalizeFrame(f));
      const timeline = new Timeline(meta, createInitialState(meta, initialKnown, initialDiscovered), frames);
      set({
        mode: "replay",
        load: "ready",
        error: null,
        meta,
        timeline,
        policy: replay.policy,
        replayInfo: { id: replay.replay_id, outcome: replay.outcome as ReplayOutcome },
        playhead: 0,
        playing: false,
        focusedRobot: null,
        live: null,
        version: get().version + 1,
      });
    },

    startLive: (snap) => set({ ...buildLive(snap, null), playing: false, focusedRobot: null } as Partial<SimStore>),

    applyLiveState: (msg) => {
      const { timeline, live } = get();
      if (!live || msg.snapshot.session_id !== live.sessionId) return;
      const snap = msg.snapshot;
      const mismatch = !timeline || snap.step_index !== timeline.maxT;
      const patch: Partial<SimStore> = mismatch ? buildLive(snap, live) : {};
      set({
        ...patch,
        playing: msg.playing,
        speed: msg.speed,
        live: { ...((patch.live as LiveInfo | undefined) ?? live), done: snap.done, success: snap.done ? snap.metrics.success : null },
      });
    },

    pushLiveFrame: (msg, now = performance.now()) => {
      const { timeline, live } = get();
      if (!timeline || !live) return;
      try {
        timeline.append(normalizeFrame(msg.frame));
      } catch {
        return; // out-of-order: the next `state` message resynchronises us
      }
      set({ playhead: timeline.maxT, version: get().version + 1, live: { ...live, lastFrameAt: now } });
    },

    endLive: (msg) => {
      const { live } = get();
      if (!live) return;
      set({ playing: false, live: { ...live, done: true, success: msg.success, finalMetrics: msg.metrics }, version: get().version + 1 });
    },

    setConnection: (connection, notice = null) => {
      const { live } = get();
      if (live) set({ live: { ...live, connection, notice: notice ?? live.notice } });
    },
    setNotice: (notice) => {
      const { live } = get();
      if (live) set({ live: { ...live, notice } });
    },

    play: () => {
      const { mode, timeline, playhead } = get();
      if (mode === "live") {
        liveSend({ cmd: "play" });
      } else if (mode === "replay" && timeline) {
        // Pressing play at the end restarts, as video players do.
        set({ playing: true, playhead: playhead >= timeline.maxT ? timeline.startT : playhead });
      }
    },
    pause: () => {
      if (get().mode === "live") liveSend({ cmd: "pause" });
      else set({ playing: false });
    },
    togglePlay: () => (get().playing ? get().pause() : get().play()),
    stepForward: () => {
      const { mode, playhead } = get();
      if (mode === "live") liveSend({ cmd: "step" });
      else if (mode === "replay") set({ playing: false, playhead: clampT(Math.floor(playhead) + 1), version: get().version + 1 });
    },
    stepBack: () => {
      const { mode, playhead } = get();
      if (mode === "replay") set({ playing: false, playhead: clampT(Math.ceil(playhead) - 1), version: get().version + 1 });
    },
    seek: (t) => {
      if (get().mode === "replay") set({ playing: false, playhead: clampT(t), version: get().version + 1 });
    },
    reset: () => {
      const { mode, timeline } = get();
      if (mode === "live") liveSend({ cmd: "reset" });
      else if (mode === "replay" && timeline) set({ playing: false, playhead: timeline.startT, version: get().version + 1 });
    },
    setSpeed: (speed) => {
      set({ speed });
      if (get().mode === "live") liveSend({ cmd: "speed", value: speed });
    },
    toggleFocus: (robot) => set({ focusedRobot: get().focusedRobot === robot ? null : robot }),
    focus: (robot) => set({ focusedRobot: robot }),

    advance: (dt) => {
      const { mode, playing, timeline, playhead, speed } = get();
      if (mode !== "replay" || !playing || !timeline) return;
      const next = playhead + dt * Math.min(MAX_STEPS_PER_SECOND, BASE_STEPS_PER_SECOND * speed);
      if (next >= timeline.maxT) set({ playhead: timeline.maxT, playing: false, version: get().version + 1 });
      else {
        const crossed = Math.floor(next) !== Math.floor(playhead);
        set(crossed ? { playhead: next, version: get().version + 1 } : { playhead: next });
      }
    },

    setCommandSink: (sink) => {
      commandSink = sink;
    },
  };
});
