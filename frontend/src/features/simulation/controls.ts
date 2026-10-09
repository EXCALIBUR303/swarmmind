import type { Connection } from "../../api/ws";

/**
 * Whether the playback controls act. In live mode every command goes over the session socket, and
 * `SimSocket.send` drops commands while it is not open; so until the socket is open the controls
 * must be disabled rather than look available and ignore the click (found by the Phase 14 browser run).
 */
export function controlsEnabled(mode: "none" | "live" | "replay", load: string, connection: Connection | undefined): boolean {
  if (mode === "none" || load !== "ready") return false;
  return mode !== "live" || connection === "open";
}
