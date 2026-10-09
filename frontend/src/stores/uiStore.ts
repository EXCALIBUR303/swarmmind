import { create } from "zustand";
import type { LayerKey, Layers } from "../features/simulation/model";
import { loadPref, savePref } from "../lib/prefs";

export type RailTab = "robots" | "live" | "layers";

export const DEFAULT_LAYERS: Layers = {
  sensor: true,
  paths: true,
  visited: true,
  unknown: true,
  commLinks: false,
  groundTruth: false, // debug only, off by default (§30.4)
};

interface UiState {
  layers: Layers;
  railTab: RailTab;
  railOpen: boolean;
  drawerOpen: boolean;
  showCoords: boolean;
  setShowCoords: (v: boolean) => void;
  toggleLayer: (k: LayerKey) => void;
  setRailTab: (t: RailTab) => void;
  setRailOpen: (open: boolean) => void;
  setDrawerOpen: (open: boolean) => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  layers: { ...DEFAULT_LAYERS, ...loadPref<Partial<Layers>>("layers", {}) },
  railTab: "robots",
  railOpen: true,
  drawerOpen: false,
  showCoords: loadPref<boolean>("showCoords", false),
  setShowCoords: (showCoords) => {
    savePref("showCoords", showCoords);
    set({ showCoords });
  },
  toggleLayer: (k) => {
    const layers = { ...get().layers, [k]: !get().layers[k] };
    savePref("layers", layers);
    set({ layers });
  },
  setRailTab: (railTab) => set({ railTab }),
  setRailOpen: (railOpen) => set({ railOpen }),
  setDrawerOpen: (drawerOpen) => set({ drawerOpen }),
}));
