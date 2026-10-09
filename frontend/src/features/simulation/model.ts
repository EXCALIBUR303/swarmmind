/**
 * Simulation view-model types shared by the reducer, renderers and store.
 * Coordinates are always `[y, x]` (row, column), exactly as the backend and
 * replay schema send them. Pixel conversion happens only in `viewport.ts`.
 */
import type { ReplayFrame } from "../../api/wire.gen";

export type Cell = [number, number]; // [y, x]
/** A frame with every field present. The wire type marks defaulted list fields
 * optional; `normalizeFrame` fills them once at the API boundary. */
export type Frame = Required<ReplayFrame>;

export function normalizeFrame(f: ReplayFrame): Frame {
  return {
    ...f,
    newly_known: f.newly_known ?? [],
    discovered: f.discovered ?? [],
    rescued: f.rescued ?? [],
    comm_edges: f.comm_edges ?? [],
  };
}

export type TargetStatus = 0 | 1 | 2; // hidden | discovered | rescued (backend constants)
export const HIDDEN = 0;
export const DISCOVERED = 1;
export const RESCUED = 2;

/** Everything about the episode that does not change from frame to frame. */
export interface SimMeta {
  mapId: string;
  width: number;
  height: number;
  terrain: string[]; // '.' free, '#' obstacle
  robotStarts: Cell[];
  targets: Cell[];
  nRobots: number;
  obsRadius: number;
  sensorRadius: number;
  maxSteps: number;
  commMode: string;
  nFree: number; // free cells (coverage denominator)
}

/** The state after `t` steps. Owned by a `Timeline`: treat as read-only. */
export interface SimState {
  t: number;
  pos: Cell[];
  known: Uint8Array; // H*W, 1 = known to the team (free or obstacle), index y*W+x
  visited: Uint16Array; // H*W, robot-visits per cell
  targetStatus: Int8Array;
  rescuedBy: Int8Array; // -1 = none
  knownFree: number;
  collisions: number; // cumulative blocked_by_robot (same definition as the backend metric)
  discovered: number;
  rescued: number;
  distance: number[]; // per-robot cells moved
  lastFrame: Frame | null;
}

export type LayerKey = "sensor" | "paths" | "visited" | "unknown" | "commLinks" | "groundTruth";
export type Layers = Record<LayerKey, boolean>;

export function envMeta(args: {
  mapId: string;
  width: number;
  height: number;
  terrain: string[];
  robotStarts: number[][];
  targets: number[][];
  config: Record<string, unknown>;
}): SimMeta {
  const c = args.config as Record<string, number | string>;
  const nFree = args.terrain.reduce((n, row) => n + [...row].filter((ch) => ch === ".").length, 0);
  return {
    mapId: args.mapId,
    width: args.width,
    height: args.height,
    terrain: args.terrain,
    robotStarts: args.robotStarts.map(([y, x]) => [y, x] as Cell),
    targets: args.targets.map(([y, x]) => [y, x] as Cell),
    nRobots: Number(c.n_robots),
    obsRadius: Number(c.obs_radius),
    sensorRadius: Number(c.sensor_radius),
    maxSteps: Number(c.max_steps),
    commMode: String(c.comm_mode ?? "none"),
    nFree,
  };
}
