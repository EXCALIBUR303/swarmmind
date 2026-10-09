/**
 * Delta accumulation (spec §27.4): the frontend never re-simulates. The
 * known map at step t is `initial.known_team ∪ ⋃_{i<=t} frames[i].newly_known`;
 * everything else is a running sum over the frames' own fields.
 *
 * The reset-time sensing pass is in NO frame's `newly_known`, so the initial
 * known cells must come from `replay.initial` / `GET /replays/{id}/initial` /
 * `session.initial` — skipping them leaves the start area fogged forever.
 *
 * Limitation (documented, not hidden): a target discovered by the reset-time
 * sensing pass would not appear in any frame's `discovered`. Every shipped
 * level has target_min_dist (3) > sensor_radius (2), so it cannot occur.
 */
import { DISCOVERED, HIDDEN, RESCUED, type Cell, type Frame, type SimMeta, type SimState } from "./model";

export const idx = (meta: Pick<SimMeta, "width">, y: number, x: number): number => y * meta.width + x;

function markKnown(meta: SimMeta, state: SimState, cells: ArrayLike<number>[]): void {
  for (const c of cells) {
    const i = idx(meta, c[0], c[1]);
    if (!state.known[i]) {
      state.known[i] = 1;
      if (meta.terrain[c[0]][c[1]] === ".") state.knownFree++;
    }
  }
}

export function createInitialState(meta: SimMeta, initialKnown: ArrayLike<number>[]): SimState {
  const state: SimState = {
    t: 0,
    pos: meta.robotStarts.map(([y, x]) => [y, x] as Cell),
    known: new Uint8Array(meta.width * meta.height),
    visited: new Uint16Array(meta.width * meta.height),
    targetStatus: new Int8Array(meta.targets.length).fill(HIDDEN),
    rescuedBy: new Int8Array(meta.targets.length).fill(-1),
    knownFree: 0,
    collisions: 0,
    discovered: 0,
    rescued: 0,
    distance: new Array(meta.nRobots).fill(0),
    lastFrame: null,
  };
  markKnown(meta, state, initialKnown);
  for (const [y, x] of state.pos) state.visited[idx(meta, y, x)]++; // world.reset counts the start cell
  return state;
}

export function cloneState(s: SimState): SimState {
  return {
    ...s,
    pos: s.pos.map(([y, x]) => [y, x] as Cell),
    known: s.known.slice(),
    visited: s.visited.slice(),
    targetStatus: s.targetStatus.slice(),
    rescuedBy: s.rescuedBy.slice(),
    distance: s.distance.slice(),
  };
}

/** Apply one step's frame in place. */
export function applyFrame(meta: SimMeta, state: SimState, frame: Frame): void {
  state.t = frame.t;
  state.pos = frame.pos.map(([y, x]) => [y, x] as Cell);
  markKnown(meta, state, frame.newly_known);
  for (let r = 0; r < state.pos.length; r++) {
    const [y, x] = state.pos[r];
    state.visited[idx(meta, y, x)]++;
    if (frame.moved[r]) state.distance[r]++;
    if (frame.blocked_by_robot[r]) state.collisions++;
  }
  for (const k of frame.discovered) {
    if (state.targetStatus[k] === HIDDEN) {
      state.targetStatus[k] = DISCOVERED;
      state.discovered++;
    }
  }
  for (const [k, r] of frame.rescued) {
    if (state.targetStatus[k] !== RESCUED) {
      if (state.targetStatus[k] === HIDDEN) state.discovered++; // rescued implies discovered
      state.targetStatus[k] = RESCUED;
      state.rescued++;
    }
    state.rescuedBy[k] = r;
  }
  state.lastFrame = frame;
}

export function coverage(meta: SimMeta, state: SimState): number {
  return meta.nFree === 0 ? 0 : state.knownFree / meta.nFree;
}

/** Observable per-robot fact for the Robots tab — never a fabricated intent (§30.1). */
export function robotFact(state: SimState, robot: number): string {
  const f = state.lastFrame;
  if (!f) return "Waiting";
  const rescue = f.rescued.find(([, r]) => r === robot);
  if (rescue) return `Rescued T${rescue[0] + 1} @ t=${f.t}`;
  if (f.blocked_by_robot[robot] || f.bumped_wall[robot]) return "Blocked";
  return f.moved[robot] ? "Moving" : "Waiting";
}

/** State of a live session captured mid-episode (reconnect after missed frames). Distances and
 * per-robot history before this point are unknown, so they restart from 0; counts come from the
 * server's own metrics. */
export function stateFromSnapshot(
  meta: SimMeta,
  snap: {
    step_index: number;
    current_pos: number[][];
    current_known_team: number[][];
    target_status: number[];
    metrics: { n_collisions: number; n_discovered: number; n_rescued: number };
  },
): SimState {
  const s = createInitialState(meta, snap.current_known_team);
  s.t = snap.step_index;
  s.pos = snap.current_pos.map(([y, x]) => [y, x] as Cell);
  s.visited.fill(0);
  for (const [y, x] of s.pos) s.visited[idx(meta, y, x)]++;
  s.targetStatus = Int8Array.from(snap.target_status);
  s.collisions = snap.metrics.n_collisions;
  s.discovered = snap.metrics.n_discovered;
  s.rescued = snap.metrics.n_rescued;
  return s;
}
