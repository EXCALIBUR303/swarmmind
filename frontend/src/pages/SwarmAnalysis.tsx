import { useMemo, useState } from "react";
import { studiesApi, type ExampleEpisode, type HeatmapData } from "../api/endpoints/analytics";
import { catalogApi } from "../api/endpoints/catalog";
import { Bars } from "../components/charts/Bars";
import { ChartFrame } from "../components/charts/ChartFrame";
import { DiscoveryTimeline } from "../components/charts/DiscoveryTimeline";
import { Heatmap } from "../components/charts/Heatmap";
import { LineChart } from "../components/charts/LineChart";
import { EmptyState } from "../components/EmptyState";
import { Field, FieldGroup, Page, Section } from "../components/PageLayout";
import { Select } from "../components/Select";
import { ErrorState, LoadingState } from "../components/StateViews";
import { EvolutionGrid } from "../features/analysis/EvolutionGrid";
import { parseAnalysisFiles } from "../features/analysis/parse";
import { METHOD_COLOR, METHOD_LABEL, type Method } from "../features/comparison/model";
import { fmtNum } from "../lib/stats";
import { useAsync } from "../lib/useAsync";
import "./pages.css";

const ROBOT = (i: number) => `var(--sm-robot-${i + 1})`;
const shortMap = (id: string) => id.replace(/^L\d-(struct-)?/, "");

export function SwarmAnalysis() {
  const studies = useAsync(() => studiesApi.list(), []);
  const [picked, setStudyId] = useState<string | null>(null);
  const options = (studies.data ?? []).filter((s) => !s.smoke);
  const list = options.length ? options : (studies.data ?? []);
  const studyId = picked ?? list[0]?.study ?? null;
  const detail = useAsync(() => studiesApi.get(studyId!), [studyId], !!studyId);
  const idx = useMemo(() => parseAnalysisFiles(detail.data?.analysis_files ?? []), [detail.data]);
  const levels = [...new Set([...idx.heatmaps.map((h) => h.level), ...idx.episodes.map((e) => e.level)])];
  const level = levels[0] ?? Object.keys(detail.data?.summary.levels ?? {})[0] ?? "L2";

  const [kind, setKind] = useState<"visits" | "collisions">("visits");
  const maps = [...new Set(idx.heatmaps.filter((h) => h.level === level).map((h) => h.mapId))];
  const [mapId, setMapId] = useState<string | null>(null);
  const effMap = mapId && maps.includes(mapId) ? mapId : maps[0];
  const [algo, setAlgo] = useState<"ippo" | "mappo">("mappo");

  const heat = useAsync(
    () => Promise.all((["ippo", "mappo"] as const).map((a) => studiesApi.heatmap(studyId!, level, a, effMap!, kind))),
    [studyId, level, effMap, kind],
    !!studyId && !!effMap,
  );
  const epMeta = idx.episodes.find((e) => e.algo === algo && e.level === level);
  const episode = useAsync(() => studiesApi.analysis<ExampleEpisode>(studyId!, epMeta!.name), [studyId, epMeta?.name], !!studyId && !!epMeta);
  const replays = useAsync(() => catalogApi.replays(), []);

  const evoRuns = useMemo(() => {
    const byRun = new Map<string, { run_id: string; algo: string; seed: number; maps: Set<string>; fracs: Set<number> }>();
    for (const r of replays.data ?? []) {
      if (r.study !== studyId) continue;
      const e = byRun.get(r.run_id) ?? { run_id: r.run_id, algo: r.algo, seed: r.seed, maps: new Set(), fracs: new Set() };
      e.maps.add(r.map_id);
      e.fracs.add(r.fraction);
      byRun.set(r.run_id, e);
    }
    return [...byRun.values()].filter((r) => r.fracs.size >= 2).sort((a, b) => a.algo.localeCompare(b.algo) || a.seed - b.seed);
  }, [replays.data, studyId]);
  const [evoRun, setEvoRun] = useState<string | null>(null);
  const [evoMap, setEvoMap] = useState<string | null>(null);
  const effRun = evoRuns.find((r) => r.run_id === evoRun) ?? evoRuns[0];
  const evoMaps = effRun ? [...effRun.maps].sort() : [];
  const effEvoMap = evoMap && evoMaps.includes(evoMap) ? evoMap : evoMaps[0];

  if (studies.loading) return <Page title="Swarm Analysis"><LoadingState title="Loading studies…" /></Page>;
  if (studies.error) return <Page title="Swarm Analysis"><ErrorState title="Backend unreachable" detail={studies.error} action={{ label: "Retry", onClick: studies.reload }} /></Page>;
  if (!list.length) return <Page title="Swarm Analysis"><EmptyState title="No completed study yet" detail="Run scripts/run_study.py, then scripts/analyze_study.py." /></Page>;

  const sharedMax = Math.max(0, ...(heat.data ?? []).flatMap((d) => d.counts.flatMap((r, y) => r.filter((_, x) => !d.obstacles[y][x]))));
  const ep = episode.data;
  const color = METHOD_COLOR[algo as Method];
  const robots = ep ? ep.workload_share_per_robot.map((_, i) => i) : [];
  const steps = ep ? ep.coverage_over_time.map((_, i) => i) : [];
  return (
    <Page
      title="Swarm Analysis"
      subtitle="Where the robots go, how they divide the work, and how behaviour changes with training. Heatmaps aggregate held-out structured maps; the episode charts show one illustrative episode, not a statistic."
      filters={
        <>
          <Field label="Study"><Select label="Study" className="" value={studyId ?? ""} options={list.map((s) => ({ value: s.study, label: s.study }))} onChange={setStudyId} /></Field>
          <FieldGroup label="Algorithm (episode charts)"><div className="sm-seg">{(["ippo", "mappo"] as const).map((a) => <button key={a} aria-pressed={algo === a} onClick={() => setAlgo(a)}>{METHOD_LABEL[a]}</button>)}</div></FieldGroup>
        </>
      }
    >
      <Section title="Spatial heatmaps" note="Cumulative over the study's evaluation episodes on this map. Hatched cells are obstacles. Both panels share one colour scale, so shades are comparable across algorithms.">
        {maps.length === 0 ? (
          <EmptyState title={`No structured (G2) maps for ${level}`} detail="Heatmaps are only produced for levels with a structured evaluation set; nothing is fabricated for the others." />
        ) : (
          <>
            <div className="sm-filters" role="group" aria-label="Heatmap options">
              <Field label="Map"><Select label="Map" className="" value={effMap ?? ""} options={maps.map((m) => ({ value: m, label: shortMap(m) }))} onChange={setMapId} /></Field>
              <FieldGroup label="Quantity"><div className="sm-seg">{(["visits", "collisions"] as const).map((k) => <button key={k} aria-pressed={kind === k} onClick={() => setKind(k)}>{k}</button>)}</div></FieldGroup>
            </div>
            <div className="sm-grid wide">
              {(["ippo", "mappo"] as const).map((a, i) => {
                const d: HeatmapData | undefined = heat.data?.[i];
                return (
                  <ChartFrame
                    key={a}
                    title={`${METHOD_LABEL[a]}: ${kind} per cell`}
                    caption={d ? `${shortMap(d.map_id)} · ${d.n_episodes} episodes` : undefined}
                    state={heat.loading ? "loading" : heat.error ? "error" : d ? "ready" : "empty"}
                    error={heat.error}
                    table={d ? { columns: ["row", "col", kind], rows: d.counts.flatMap((r, y) => r.map((v, x) => [y, x, d.obstacles[y][x] ? "obstacle" : v])) } : undefined}
                  >
                    {d && <Heatmap counts={d.counts} obstacles={d.obstacles} unit={kind} max={sharedMax} ariaLabel={`${METHOD_LABEL[a]} ${kind} heatmap on ${shortMap(d.map_id)}`} />}
                  </ChartFrame>
                );
              })}
            </div>
          </>
        )}
      </Section>

      <Section title="One illustrative episode" note={ep ? `${METHOD_LABEL[algo]} on ${shortMap(ep.map_id)}: ${ep.success ? "success" : "failure"} after ${ep.final_t} steps. Illustrative episode, not a statistic: these curves have no confidence interval.` : undefined}>
        {!epMeta ? (
          <EmptyState title="No example episode for this algorithm" detail="scripts/analyze_study.py writes one per learned algorithm." />
        ) : (
          <div className="sm-grid wide">
            <ChartFrame title="Coverage over time" caption="fraction of free cells known to the team (fixed 0–100% axis)" state={episode.loading ? "loading" : episode.error ? "error" : ep ? "ready" : "empty"} error={episode.error} table={ep ? { columns: ["t", "coverage"], rows: ep.coverage_over_time.map((v, t) => [t, fmtNum(v, 3)]) } : undefined}>
              {ep && <LineChart series={[{ id: "cov", label: METHOD_LABEL[algo], color, points: ep.coverage_over_time.map((y, x) => ({ x, y })) }]} xLabel="step t" yLabel="coverage" yDomain={[0, 1]} yFormat={(v) => `${Math.round(v * 100)}%`} ariaLabel="Coverage over time for the illustrative episode" />}
            </ChartFrame>
            <ChartFrame title="Division of exploration effort (DER) vs coverage" caption="illustrative episode, not a statistic: lower = robots explore different cells; DER is defined in spec §24.1" state={ep ? "ready" : "empty"} table={ep ? { columns: ["t", "DER"], rows: ep.der_over_time.map((v, t) => [t + 1, fmtNum(v, 3)]) } : undefined}>
              {ep && <LineChart series={[{ id: "der", label: "DER", color, points: ep.der_over_time.map((y, i) => ({ x: i + 1, y })) }]} xLabel="step t" yLabel="DER" yDomain={[0, 1]} ariaLabel="DER over time for the illustrative episode" />}
            </ChartFrame>
            <ChartFrame
              title="Inter-robot distance"
              caption="pairwise distance between robots, grid cells: mean (solid) and closest pair (dashed)"
              state={ep ? "ready" : "empty"}
              legend={[{ label: "mean", color }, { label: "closest pair", color: "var(--sm-text-muted)", dash: true }]}
              table={ep ? { columns: ["t", "mean", "closest pair"], rows: ep.inter_robot_distance_over_time.map((d, t) => [t, fmtNum(d.mean, 2), fmtNum(d.min, 2)]) } : undefined}
            >
              {ep && (
                <LineChart
                  series={[
                    { id: "mean", label: "mean", color, points: ep.inter_robot_distance_over_time.map((d, x) => ({ x, y: d.mean })) },
                    { id: "min", label: "closest", color: "var(--sm-text-muted)", dash: true, width: 1.5, points: ep.inter_robot_distance_over_time.map((d, x) => ({ x, y: d.min })) },
                  ]}
                  xLabel="step t"
                  yLabel="cells"
                  yFormat={(v) => fmtNum(v, 1)}
                  ariaLabel="Inter-robot distance over time"
                />
              )}
            </ChartFrame>
            <ChartFrame title="Discovery and rescue timeline" caption="◆ discovered · ★ rescued; one row per target" state={ep ? "ready" : "empty"} table={ep ? { columns: ["target", "discovered", "rescued", "rescued by"], rows: ep.discovery_timeline.map((r) => [r.target + 1, r.t_discovered, r.t_rescued, r.rescued_by == null ? null : r.rescued_by + 1]) } : undefined}>
              {ep && <DiscoveryTimeline rows={ep.discovery_timeline} maxT={Math.max(ep.final_t, 1)} ariaLabel="Discovery and rescue times per target" />}
            </ChartFrame>
            <ChartFrame title="Workload share per robot" caption="share of total distance travelled; even split = 1/n" state={ep ? "ready" : "empty"} table={ep ? { columns: ["robot", "share", "distance"], rows: robots.map((i) => [`R${i + 1}`, fmtNum(ep.workload_share_per_robot[i], 3), ep.distance_travelled_per_robot[i]]) } : undefined}>
              {ep && <Bars data={robots.map((i) => ({ label: `R${i + 1}`, value: ep.workload_share_per_robot[i], color: ROBOT(i) }))} domain={[0, 1]} xLabel="share of distance travelled" annotation={`Gini ${fmtNum(ep.workload_gini, 3)}`} ariaLabel="Workload share per robot" />}
            </ChartFrame>
            <ChartFrame title="Idle fraction per robot" caption="fraction of steps spent not moving" state={ep ? "ready" : "empty"} table={ep ? { columns: ["robot", "idle fraction"], rows: robots.map((i) => [`R${i + 1}`, fmtNum(ep.idle_fraction_per_robot[i], 3)]) } : undefined}>
              {ep && <Bars data={robots.map((i) => ({ label: `R${i + 1}`, value: ep.idle_fraction_per_robot[i], color: ROBOT(i) }))} domain={[0, 1]} xLabel="idle fraction" ariaLabel="Idle fraction per robot" />}
            </ChartFrame>
          </div>
        )}
        <span className="sm-section-note" hidden>{steps.length}</span>
      </Section>

      <Section title="Swarm Evolution" note="The same map and evaluation seed at checkpoints through training, played on one shared clock. These are recorded replays; nothing is re-simulated.">
        {replays.loading ? <LoadingState title="Loading replay index…" /> : !effRun ? (
          <EmptyState title="No checkpoint replays" detail="Record them with: python scripts/record_replays.py --study <study> --evolution" />
        ) : (
          <>
            <div className="sm-filters" role="group" aria-label="Evolution options">
              <Field label="Run"><Select label="Run" className="" value={effRun.run_id} options={evoRuns.map((r) => ({ value: r.run_id, label: `${METHOD_LABEL[r.algo as Method] ?? r.algo} · seed ${r.seed}` }))} onChange={setEvoRun} /></Field>
              <Field label="Map"><Select label="Map" className="" value={effEvoMap ?? ""} options={evoMaps.map((m) => ({ value: m, label: shortMap(m) }))} onChange={setEvoMap} /></Field>
            </div>
            {effEvoMap && <EvolutionGrid index={replays.data ?? []} runId={effRun.run_id} mapId={effEvoMap} />}
          </>
        )}
      </Section>
    </Page>
  );
}
