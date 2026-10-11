import type { Replay } from "../wire.gen";
import { request } from "../client";

export interface LevelInfo {
  level: string;
  config: Record<string, unknown>;
}

export interface MapInfo {
  id: string;
  set: "validation" | "test" | "structured";
  level: string;
  hash: string;
  width: number;
  height: number;
  n_robots: number;
  n_targets: number;
  terrain: string[];
  robot_starts: number[][];
  targets: number[][];
}

export interface ModelInfo {
  model_id: string;
  source: string;
  run_id?: string;
  study?: string | null;
  algo?: string | null;
  level?: string | null;
  seed?: number | null;
  trained_env_steps?: number | null;
  final_val_metrics?: Record<string, number> | null;
  test_metrics?: Record<string, number> | null;
}

export interface ReplayIndexEntry {
  replay_id: string;
  map_id: string;
  algo: string;
  seed: number;
  fraction: number;
  checkpoint_step: number;
  success: boolean;
  steps: number;
  study: string;
  run_id: string;
}

export const catalogApi = {
  levels: () => request<LevelInfo[]>("/api/levels"),
  maps: (set: MapInfo["set"], level?: string) =>
    request<MapInfo[]>(`/api/maps?set=${set}${level ? `&level=${encodeURIComponent(level)}` : ""}`),
  models: (level?: string) => request<ModelInfo[]>(`/api/models${level ? `?level=${encodeURIComponent(level)}` : ""}`),
  replays: () => request<ReplayIndexEntry[]>("/api/replays"),
  /** The browser transparently gunzips the stored `.json.gz` (Content-Encoding: gzip). */
  replay: (id: string) => request<Replay>(`/api/replays/${encodeURIComponent(id)}`),
  replayInitial: (id: string) =>
    request<{ known_team: number[][]; discovered?: number[]; source: "recorded" | "derived" }>(`/api/replays/${encodeURIComponent(id)}/initial`),
};
