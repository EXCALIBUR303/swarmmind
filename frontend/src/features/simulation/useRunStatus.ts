import { useSimStore } from "../../stores/simStore";

export type RunStatus = "running" | "paused" | "success" | "failure" | "idle" | "offline";
export const STATUS_LABEL: Record<RunStatus, string> = { running: "RUNNING", paused: "PAUSED", success: "SUCCESS", failure: "FAILED", idle: "NO SESSION", offline: "DISCONNECTED" };

export function useRunStatus(): RunStatus {
  const mode = useSimStore((s) => s.mode);
  const playing = useSimStore((s) => s.playing);
  const live = useSimStore((s) => s.live);
  const info = useSimStore((s) => s.replayInfo);
  const atEnd = useSimStore((s) => (s.timeline ? s.playhead >= s.timeline.maxT : false));
  if (mode === "none") return "idle";
  if (mode === "live" && live && live.connection !== "open" && !live.done) return "offline";
  if (mode === "live") return live?.done ? (live.success ? "success" : "failure") : playing ? "running" : "paused";
  if (atEnd && info) return info.outcome.success ? "success" : "failure";
  return playing ? "running" : "paused";
}

