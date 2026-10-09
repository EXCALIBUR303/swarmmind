/**
 * Opt-in integration check against a RUNNING backend with real study results:
 *   VITE_E2E_API=http://127.0.0.1:8000 npx vitest run src/features/simulation/realReplays.e2e.test.ts
 * Every recorded replay must reduce to exactly its own recorded outcome.
 */
import { describe, expect, it } from "vitest";
import { coverage, createInitialState } from "./frameReducer";
import { envMeta, normalizeFrame } from "./model";
import { Timeline } from "./timeline";
import type { Replay } from "../../api/wire.gen";

const API = import.meta.env.VITE_E2E_API as string | undefined;

describe.skipIf(!API)("every real replay served by the backend", () => {
  it("reduces to its recorded outcome (rescued, discovered, collisions, coverage, steps)", async () => {
    const index = (await (await fetch(`${API}/api/replays`)).json()) as { replay_id: string }[];
    expect(index.length).toBeGreaterThan(0);
    let checked = 0;
    for (const { replay_id } of index) {
      const replay = (await (await fetch(`${API}/api/replays/${replay_id}`)).json()) as Replay;
      const init = (await (await fetch(`${API}/api/replays/${replay_id}/initial`)).json()) as { known_team: number[][] };
      const meta = envMeta({ mapId: replay.map.id, width: replay.map.width, height: replay.map.height, terrain: replay.map.terrain, robotStarts: replay.map.robot_starts, targets: replay.map.targets, config: replay.env.config });
      const tl = new Timeline(meta, createInitialState(meta, init.known_team), replay.frames.map(normalizeFrame));
      const s = tl.stateAt(tl.maxT);
      const m = replay.outcome.metrics as Record<string, number>;
      expect([replay_id, s.t]).toEqual([replay_id, replay.outcome.steps]);
      expect([replay_id, s.rescued]).toEqual([replay_id, m.n_rescued]);
      expect([replay_id, s.discovered]).toEqual([replay_id, m.n_discovered]);
      expect([replay_id, s.collisions]).toEqual([replay_id, m.n_collisions]);
      expect(coverage(meta, s)).toBeCloseTo(m.coverage_final, 10);
      checked++;
    }
    console.log(`verified ${checked} real replays`);
  }, 120_000);
});
