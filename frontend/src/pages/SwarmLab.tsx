import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { catalogApi, type ReplayIndexEntry } from "../api/endpoints/catalog";
import { EmptyState } from "../components/EmptyState";
import { ErrorState, LoadingState } from "../components/StateViews";
import type { SelectOption } from "../components/Select";
import { Timeline } from "../features/replay/Timeline";
import { MissionStrip } from "../features/simulation/MissionStrip";
import { RightRail } from "../features/simulation/RightRail";
import { ScenarioDrawer } from "../features/simulation/ScenarioDrawer";
import { SimulationCanvas } from "../features/simulation/SimulationCanvas";
import { Toolbar } from "../features/simulation/Toolbar";
import { startLive, stopLive } from "../features/simulation/liveController";
import { PerfOverlay } from "../features/simulation/PerfOverlay";
import { algoLabel, kSteps } from "../lib/format";
import { useAsync } from "../lib/useAsync";
import { useScenario, toSessionCreate, type MapSet, type Scenario } from "../stores/scenarioStore";
import { useSimStore } from "../stores/simStore";
import { useUiStore } from "../stores/uiStore";
import "./SwarmLab.css";

const shortMap = (id: string) => id.replace(/^L\d-(struct-)?/, "");

/** Default replay: the most instructive real one: a final-checkpoint success if any exists. */
function defaultReplay(list: ReplayIndexEntry[]): ReplayIndexEntry | undefined {
  return list.find((r) => r.fraction === 1 && r.success) ?? list[0];
}

export function SwarmLab() {
  const sc = useScenario();
  const [params] = useSearchParams();
  const deepLink = params.get("replay");
  const mode = useSimStore((s) => s.mode);
  const load = useSimStore((s) => s.load);
  const storeError = useSimStore((s) => s.error);
  const live = useSimStore((s) => s.live);
  const playing = useSimStore((s) => s.playing);
  const speed = useSimStore((s) => s.speed);
  useSimStore((s) => s.version);
  const drawerOpen = useUiStore((s) => s.drawerOpen);
  const setDrawerOpen = useUiStore((s) => s.setDrawerOpen);
  const drawerBtn = useRef<HTMLButtonElement>(null);
  const loadToken = useRef(0);
  const [bootstrapped, setBootstrapped] = useState(false);

  const levels = useAsync(() => catalogApi.levels(), []);
  const replays = useAsync(() => catalogApi.replays(), []);
  const models = useAsync(() => catalogApi.models(sc.level), [sc.level]);
  const maps = useAsync(() => catalogApi.maps(sc.mapSet as Exclude<MapSet, "procedural">, sc.level), [sc.mapSet, sc.level], sc.mapSet !== "procedural");

  const replayList = useMemo(() => replays.data ?? [], [replays.data]);
  const current = replayList.find((r) => r.replay_id === sc.replayId);

  // ---- loading ---------------------------------------------------------
  const loadReplayById = useCallback(async (id: string) => {
    const token = ++loadToken.current;
    const st = useSimStore.getState();
    st.setLoading();
    try {
      const [replay, initial] = await Promise.all([catalogApi.replay(id), catalogApi.replayInitial(id)]);
      if (token !== loadToken.current) return; // a newer selection won
      useSimStore.getState().loadReplay(replay, initial.known_team);
      useScenario.getState().patch({ replayId: id });
    } catch (err) {
      if (token === loadToken.current) useSimStore.getState().setError(err instanceof Error ? err.message : "could not load the replay");
    }
  }, []);

  const startLiveNow = useCallback(async () => {
    loadToken.current++;
    const s = useScenario.getState();
    await startLive(toSessionCreate(s));
  }, []);

  // First load, once the catalog is known.
  useEffect(() => {
    if (bootstrapped || replays.loading) return;
    setBootstrapped(true);
    if (replays.error) return;
    const s = useScenario.getState();
    if (deepLink && replayList.some((r) => r.replay_id === deepLink)) {
      s.patch({ mode: "replay", replayId: deepLink });
      void loadReplayById(deepLink); // /?replay=<id>, e.g. from Experiment History
      return;
    }
    if (s.mode === "live") {
      void startLiveNow();
      return;
    }
    const pick = replayList.find((r) => r.replay_id === s.replayId) ?? defaultReplay(replayList);
    if (pick) void loadReplayById(pick.replay_id);
  }, [bootstrapped, replays.loading, replays.error, replayList, loadReplayById, startLiveNow, deepLink]);

  useEffect(
    () => () => {
      void stopLive();
      useSimStore.getState().clear();
    },
    [],
  );

  // ---- keyboard (§30.4) ---------------------------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target;
      if (el instanceof HTMLElement && (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(el.tagName) || el.getAttribute("role") === "tab" || el.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const st = useSimStore.getState();
      if (e.key === " ") {
        e.preventDefault();
        st.togglePlay();
      } else if (e.key === "ArrowRight") st.stepForward();
      else if (e.key === "ArrowLeft") st.stepBack();
      else if (e.key === "r" || e.key === "R") st.reset();
      else if (e.key === "l" || e.key === "L") useUiStore.getState().setRailTab("layers");
      else if (/^[1-6]$/.test(e.key) && st.meta && Number(e.key) <= st.meta.nRobots) st.toggleFocus(Number(e.key) - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ---- selectors ----------------------------------------------------------
  const onMode = (m: "live" | "replay") => {
    if (m === sc.mode && useSimStore.getState().mode === m) return;
    sc.patch({ mode: m });
    if (m === "live") void startLiveNow();
    else {
      void stopLive();
      const pick = current ?? defaultReplay(replayList);
      if (pick) void loadReplayById(pick.replay_id);
      else useSimStore.getState().clear();
    }
  };

  const selectors = useMemo(() => {
    if (sc.mode === "replay") {
      const runs = new Map<string, ReplayIndexEntry>();
      for (const r of replayList) if (!runs.has(r.run_id)) runs.set(r.run_id, r);
      const runOptions: SelectOption[] = [...runs.values()]
        .sort((a, b) => a.algo.localeCompare(b.algo) || a.seed - b.seed)
        .map((r) => ({ value: r.run_id, label: `${algoLabel(r.algo)} · s${r.seed} · ${r.study}`, group: algoLabel(r.algo) }));
      const ofRun = replayList.filter((r) => r.run_id === current?.run_id);
      const steps = [...new Map(ofRun.map((r) => [r.checkpoint_step, r])).values()].sort((a, b) => a.checkpoint_step - b.checkpoint_step);
      const ofStep = ofRun.filter((r) => r.checkpoint_step === current?.checkpoint_step);
      const pickFrom = (cands: ReplayIndexEntry[], prefer?: ReplayIndexEntry) =>
        cands.find((r) => r.map_id === prefer?.map_id) ?? cands[0];
      return [
        {
          label: "Policy",
          value: current?.run_id ?? "",
          options: runOptions,
          placeholder: "No replays",
          onChange: (runId: string) => {
            const cands = replayList.filter((r) => r.run_id === runId);
            const finalStep = Math.max(...cands.map((r) => r.checkpoint_step));
            const pick = pickFrom(cands.filter((r) => r.checkpoint_step === finalStep), current);
            if (pick) void loadReplayById(pick.replay_id);
          },
        },
        {
          label: "Step",
          value: String(current?.checkpoint_step ?? ""),
          options: steps.map((r) => ({ value: String(r.checkpoint_step), label: `${kSteps(r.checkpoint_step)} (${Math.round(r.fraction * 100)}%)` })),
          placeholder: "—",
          onChange: (v: string) => {
            const pick = pickFrom(ofRun.filter((r) => r.checkpoint_step === Number(v)), current);
            if (pick) void loadReplayById(pick.replay_id);
          },
        },
        {
          label: "Map",
          value: current?.replay_id ?? "",
          options: ofStep.map((r) => ({ value: r.replay_id, label: `${shortMap(r.map_id)} · ${r.success ? "✓" : "✗"} ${r.steps} steps` })),
          placeholder: "—",
          onChange: (id: string) => void loadReplayById(id),
        },
      ];
    }
    const modelOpts: SelectOption[] = (models.data ?? [])
      .filter((m) => m.algo === "ippo" || m.algo === "mappo")
      .sort((a, b) => String(a.algo).localeCompare(String(b.algo)) || (a.seed ?? 0) - (b.seed ?? 0))
      .map((m) => ({ value: `${m.algo}:${m.model_id}`, label: `${algoLabel(m.algo)} · s${m.seed ?? "?"} · ${kSteps(m.trained_env_steps)} · ${m.study ?? "export"}`, group: `Learned (${sc.level})` }));
    const mapOptions: SelectOption[] =
      sc.mapSet === "procedural"
        ? [{ value: "", label: `Procedural · seed ${sc.seed}` }]
        : (maps.data ?? []).map((m) => ({ value: m.id, label: shortMap(m.id) }));
    return [
      {
        label: "Policy",
        value: sc.policyKey,
        options: [{ value: "random", label: "Random", group: "Reference" }, { value: "heuristic", label: "Frontier heuristic", group: "Reference" }, ...modelOpts],
        onChange: (v: string) => {
          sc.patch({ policyKey: v });
          void startLiveNow();
        },
      },
      {
        label: "Map",
        value: sc.mapSet === "procedural" ? "" : (sc.mapId ?? ""),
        options: mapOptions,
        placeholder: maps.loading ? "Loading…" : "Select a map",
        onChange: (v: string) => {
          sc.patch({ mapId: v || null });
          void startLiveNow();
        },
      },
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sc.mode, sc.policyKey, sc.mapSet, sc.mapId, sc.seed, sc.level, replayList, current, models.data, maps.data, maps.loading]);

  const applyScenario = (patch: Partial<Scenario>) => {
    sc.patch(patch);
    const restart = "seed" in patch || "stochastic" in patch || "level" in patch || "mapSet" in patch;
    if (restart && useScenario.getState().mode === "live" && !("mapSet" in patch && patch.mapSet !== "procedural")) void startLiveNow();
  };

  const st = useSimStore.getState();
  const tl = st.timeline;
  const replayMode = mode === "replay";
  const playback = {
    playing,
    speed,
    enabled: mode !== "none" && load === "ready",
    canStep: !!tl && (replayMode ? st.playhead < tl.maxT : !live?.done),
    canStepBack: replayMode && !!tl && st.playhead > tl.startT,
    onToggle: st.togglePlay,
    onStep: st.stepForward,
    onStepBack: st.stepBack,
    onReset: st.reset,
    onSpeed: st.setSpeed,
  };

  const backendDown = !!(replays.error && levels.error);
  let stage: React.ReactNode;
  if (backendDown) {
    stage = <ErrorState title="Backend unreachable" detail={`${replays.error}. Start it with: uvicorn backend.app.main:app --port 8000`} action={{ label: "Retry", onClick: () => { replays.reload(); levels.reload(); setBootstrapped(false); } }} />;
  } else if (load === "error") {
    stage = <ErrorState title="Could not load the simulation" detail={storeError ?? undefined} action={{ label: "Retry", onClick: () => (sc.mode === "live" ? void startLiveNow() : sc.replayId ? void loadReplayById(sc.replayId) : undefined) }} />;
  } else if (load === "loading" || (!bootstrapped && replays.loading)) {
    stage = <LoadingState title="Loading simulation…" />;
  } else if (mode === "none") {
    stage = <EmptyState title="No session" detail={sc.mode === "replay" ? "No recorded replays were found. Switch to Live to start a session." : "Pick a policy and map to start a live session."} />;
  } else {
    stage = <SimulationCanvas />;
  }

  return (
    <div className="sm-lab">
      <div className="sm-lab-main">
        <Toolbar
          mode={sc.mode}
          onMode={onMode}
          selectors={selectors}
          playback={playback}
          onOpenDrawer={() => setDrawerOpen(true)}
          drawerButtonRef={drawerBtn}
        />
        <MissionStrip />
        {live && live.connection !== "open" && (
          <div className="sm-lab-banner" role="status">
            {live.connection === "closed" ? (live.notice ?? "Disconnected.") : live.connection === "reconnecting" ? "Connection lost. Reconnecting…" : "Connecting…"}
            {live.connection === "closed" && (
              <button className="sm-lab-btn" style={{ marginLeft: "var(--sm-space-4)" }} onClick={() => void startLiveNow()}>
                Start a new session
              </button>
            )}
          </div>
        )}
        {live?.notice && live.connection === "open" && (
          <div className="sm-lab-banner" role="status">{live.notice}</div>
        )}
        <div className="sm-lab-stage">
          <div className="sm-lab-canvas-frame">{stage}</div>
        </div>
        <Timeline />
      </div>
      <RightRail />
      <ScenarioDrawer
        open={drawerOpen}
        onClose={() => {
          setDrawerOpen(false);
          drawerBtn.current?.focus();
        }}
        scenario={sc}
        levels={levels.data ?? []}
        onApply={applyScenario}
      />
      <PerfOverlay />
    </div>
  );
}
