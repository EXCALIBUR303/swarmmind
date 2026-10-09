import { create } from "zustand";
import type { SessionCreate } from "../api/types.gen";
import { loadPref, savePref } from "../lib/prefs";

export type MapSet = "procedural" | "validation" | "test" | "structured";

export interface Scenario {
  mode: "live" | "replay";
  level: string;
  policyKey: string; // see parsePolicyKey
  mapSet: MapSet;
  mapId: string | null;
  seed: number;
  stochastic: boolean;
  replayId: string | null;
}

export const DEFAULT_SCENARIO: Scenario = {
  mode: "replay",
  level: "L2",
  policyKey: "heuristic",
  mapSet: "procedural",
  mapId: null,
  seed: 0,
  stochastic: true,
  replayId: null,
};

interface ScenarioStore extends Scenario {
  patch: (p: Partial<Scenario>) => void;
}

export const useScenario = create<ScenarioStore>((set, get) => ({
  ...DEFAULT_SCENARIO,
  ...loadPref<Partial<Scenario>>("scenario", {}),
  patch: (p) => {
    set(p);
    const { patch: _p, ...rest } = get();
    savePref("scenario", rest);
  },
}));

/** policyKey is "random" | "heuristic" | "ippo:<model_id>" | "mappo:<model_id>". */
export function parsePolicyKey(key: string): { kind: "random" | "heuristic" | "ippo" | "mappo"; model_id?: string } {
  const [kind, ...rest] = key.split(":");
  if (kind === "ippo" || kind === "mappo") return { kind, model_id: rest.join(":") };
  return { kind: kind === "random" ? "random" : "heuristic" };
}

/** The REST body for the current live scenario. */
export function toSessionCreate(s: Scenario): SessionCreate {
  return {
    level: s.level,
    map: s.mapSet === "procedural" || !s.mapId ? { kind: "procedural" } : { kind: "fixed", set: s.mapSet, id: s.mapId },
    policy: parsePolicyKey(s.policyKey),
    stochastic: s.stochastic,
    seed: s.seed,
  } as SessionCreate;
}
