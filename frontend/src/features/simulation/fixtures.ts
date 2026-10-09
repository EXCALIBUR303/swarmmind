/** Test helpers over the REAL recorded replays in `__fixtures__/` (produced by the
 * backend recorder; `initial` is what GET /api/replays/{id}/initial returns). */
import mappoSuccess from "./__fixtures__/mappo-success.json";
import untrainedFail from "./__fixtures__/untrained-fail.json";
import { createInitialState } from "./frameReducer";
import { envMeta, normalizeFrame, type Frame, type SimMeta } from "./model";
import { Timeline } from "./timeline";
import type { Replay, ReplayFrame } from "../../api/wire.gen";

export const REAL_SUCCESS = mappoSuccess as unknown as Replay;
export const REAL_FAIL = untrainedFail as unknown as Replay;

export function metaOf(r: Replay): SimMeta {
  return envMeta({
    mapId: r.map.id,
    width: r.map.width,
    height: r.map.height,
    terrain: r.map.terrain,
    robotStarts: r.map.robot_starts,
    targets: r.map.targets,
    config: r.env.config,
  });
}

export function framesOf(r: Replay): Frame[] {
  return r.frames.map((f: ReplayFrame) => normalizeFrame(f));
}

export function timelineOf(r: Replay): Timeline {
  const meta = metaOf(r);
  return new Timeline(meta, createInitialState(meta, r.initial?.known_team ?? []), framesOf(r));
}
