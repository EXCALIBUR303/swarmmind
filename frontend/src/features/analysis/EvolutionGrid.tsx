import { useEffect, useMemo, useState } from "react";
import { catalogApi, type ReplayIndexEntry } from "../../api/endpoints/catalog";
import { coverage, createInitialState } from "../simulation/frameReducer";
import { envMeta, normalizeFrame } from "../simulation/model";
import { Timeline } from "../simulation/timeline";
import { ErrorState, LoadingState } from "../../components/StateViews";
import { Select } from "../../components/Select";
import { useAsync } from "../../lib/useAsync";
import { useMedia } from "../../lib/useMedia";
import { evolutionSet, fmtSteps, pctText } from "./util";
import { ReplayCanvas } from "./ReplayCanvas";

const BASE_SPS = 4;

interface Loaded {
  entry: ReplayIndexEntry;
  timeline: Timeline;
}

async function loadAll(entries: ReplayIndexEntry[]): Promise<Loaded[]> {
  return Promise.all(
    entries.map(async (entry) => {
      const [replay, initial] = await Promise.all([catalogApi.replay(entry.replay_id), catalogApi.replayInitial(entry.replay_id)]);
      const meta = envMeta({ mapId: replay.map.id, width: replay.map.width, height: replay.map.height, terrain: replay.map.terrain, robotStarts: replay.map.robot_starts, targets: replay.map.targets, config: replay.env.config });
      return { entry, timeline: new Timeline(meta, createInitialState(meta, initial.known_team, initial.discovered ?? []), replay.frames.map(normalizeFrame)) };
    }),
  );
}

/**
 * Swarm Evolution: the SAME map and evaluation seed at several checkpoints of one run, played on ONE shared clock
 * so the only thing that differs between panels is training progress. Replays are recorded, never re-simulated.
 */
export function EvolutionGrid({ index, runId, mapId }: { index: ReplayIndexEntry[]; runId: string; mapId: string }) {
  const entries = useMemo(() => evolutionSet(index, runId, mapId), [index, runId, mapId]);
  // Remounting on a new selection resets the shared clock without an effect.
  return <EvolutionInner key={entries.map((e) => e.replay_id).join("|")} entries={entries} />;
}

function EvolutionInner({ entries }: { entries: ReplayIndexEntry[] }) {
  const key = entries.map((e) => e.replay_id).join("|");
  const loaded = useAsync(() => loadAll(entries), [key], entries.length > 0);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [pick, setPick] = useState(0);
  const compact = useMedia("(max-width: 1023px)");
  const maxT = Math.max(0, ...(loaded.data ?? []).map((l) => l.timeline.maxT));
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(
      () =>
        setT((cur) => {
          if (cur + 1 >= maxT) setPlaying(false);
          return Math.min(cur + 1, maxT);
        }),
      1000 / (BASE_SPS * speed),
    );
    return () => clearInterval(id);
  }, [playing, speed, maxT]);

  if (entries.length < 2) return <p className="sm-section-note">This run/map has fewer than two recorded checkpoints, so there is nothing to compare. Record them with scripts/record_replays.py --evolution.</p>;
  if (loaded.loading) return <LoadingState title="Loading checkpoint replays…" />;
  if (loaded.error) return <ErrorState title="Could not load the checkpoint replays" detail={loaded.error} action={{ label: "Retry", onClick: loaded.reload }} />;
  const panels = compact ? (loaded.data ?? []).filter((_, i) => i === Math.min(pick, (loaded.data ?? []).length - 1)) : (loaded.data ?? []);
  return (
    <div>
      <div className="sm-evo-controls" role="group" aria-label="Swarm Evolution playback">
        <button className="sm-btn primary" onClick={() => (t >= maxT ? (setT(0), setPlaying(true)) : setPlaying((p) => !p))} aria-label={playing ? "Pause all" : "Play all"}>{playing ? "Pause" : "Play"}</button>
        <button className="sm-btn" onClick={() => { setPlaying(false); setT(0); }}>Reset</button>
        <input type="range" min={0} max={maxT} value={Math.min(t, maxT)} onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} aria-label="Shared timeline" aria-valuetext={`step ${t} of ${maxT}`} style={{ flex: 1, minWidth: 120 }} />
        <span className="sm-section-note" style={{ margin: 0 }}>t {Math.min(t, maxT)}/{maxT}</span>
        <Select label="Playback speed" className="" value={String(speed)} options={[0.5, 1, 2, 4].map((s) => ({ value: String(s), label: `${s}×` }))} onChange={(v) => setSpeed(Number(v))} />
        {compact && <Select label="Checkpoint" className="" value={String(pick)} options={(loaded.data ?? []).map((l, i) => ({ value: String(i), label: `${Math.round(l.entry.fraction * 100)}% · ${fmtSteps(l.entry.checkpoint_step)} steps` }))} onChange={(v) => setPick(Number(v))} />}
      </div>
      <div className={`sm-evo-grid${compact ? " single" : ""}`}>
        {panels.map(({ entry, timeline }) => {
          const st = timeline.stateAt(Math.min(t, timeline.maxT));
          const done = t >= timeline.maxT;
          return (
            <figure key={entry.replay_id} className="sm-chart sm-evo-panel">
              <figcaption>
                <div className="sm-chart-title">{Math.round(entry.fraction * 100)}% of training</div>
                <div className="sm-chart-caption">{fmtSteps(entry.checkpoint_step)} env steps</div>
              </figcaption>
              <ReplayCanvas timeline={timeline} t={t} label={`Checkpoint ${Math.round(entry.fraction * 100)} percent, step ${Math.min(t, timeline.maxT)}`} />
              <div className="sm-evo-mission" aria-live="off">
                <span>t {st.t}</span><span>rescued {st.rescued}/{timeline.meta.targets.length}</span><span>coverage {pctText(coverage(timeline.meta, st))}</span>
                <span>{done ? (entry.success ? "✓ SUCCESS" : "✗ FAILED") : "running"}</span>
              </div>
            </figure>
          );
        })}
      </div>
    </div>
  );
}
