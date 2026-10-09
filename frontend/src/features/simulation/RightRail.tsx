import { robotFact } from "./frameReducer";
import { useSimStore } from "../../stores/simStore";
import { useUiStore, type RailTab } from "../../stores/uiStore";
import type { LayerKey } from "./model";
import { pct } from "../../lib/format";

const TABS: [RailTab, string][] = [
  ["robots", "Robots"],
  ["live", "Live"],
  ["layers", "Layers"],
];

export function RightRail() {
  const tab = useUiStore((s) => s.railTab);
  const setTab = useUiStore((s) => s.setRailTab);
  return (
    <aside className="sm-lab-rail" aria-label="Details">
      <div className="sm-lab-rail-tabs" role="tablist">
        {TABS.map(([key, label]) => (
          <button key={key} id={`rail-tab-${key}`} role="tab" aria-selected={tab === key} aria-controls="rail-panel" className={`sm-lab-rail-tab${tab === key ? " active" : ""}`} onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </div>
      <div className="sm-lab-rail-body" id="rail-panel" role="tabpanel" aria-labelledby={`rail-tab-${tab}`}>
        {tab === "robots" && <RobotsTab />}
        {tab === "live" && <LiveTab />}
        {tab === "layers" && <LayersTab />}
      </div>
    </aside>
  );
}

function currentT(): number {
  const { timeline, playhead, mode } = useSimStore.getState();
  if (!timeline) return 0;
  return mode === "live" ? timeline.maxT : Math.min(Math.max(Math.floor(playhead), timeline.startT), timeline.maxT);
}

function Empty({ children }: { children: string }) {
  return <p className="sm-lab-hint">{children}</p>;
}

function RobotsTab() {
  useSimStore((s) => s.version);
  const focused = useSimStore((s) => s.focusedRobot);
  const toggle = useSimStore((s) => s.toggleFocus);
  const { timeline, meta } = useSimStore.getState();
  if (!timeline || !meta) return <Empty>No session. Load a replay or start a live session.</Empty>;
  const t = currentT();
  const st = timeline.stateAt(t);
  const steps = Math.max(t - timeline.startT, 1);
  return (
    <div>
      {st.pos.map(([y, x], id) => {
        const idle = Math.round((1 - st.distance[id] / steps) * 100);
        return (
          <button key={id} className={`sm-lab-robot-row${focused === id ? " focused" : ""}`} onClick={() => toggle(id)} aria-pressed={focused === id}>
            <span className="sm-lab-swatch" style={{ background: `var(--sm-robot-${id + 1})` }} aria-hidden="true" />
            <span>
              <strong>R{id + 1}</strong>{" "}
              <span className="sm-lab-muted">
                ({y},{x}) · {robotFact(st, id)}
              </span>
            </span>
            <span className="sm-lab-muted xs">
              d={st.distance[id]} idle={idle}%
            </span>
          </button>
        );
      })}
      <p className="sm-lab-hint">Robot states are observable facts (Moving / Blocked / Waiting / Rescued T# @ t=#), never an invented intent. Click a robot to see its own {2 * meta.obsRadius + 1}×{2 * meta.obsRadius + 1} observation window.</p>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  const w = 260;
  const h = 60;
  const d = values.map((v, i) => `${i === 0 ? "M" : "L"} ${(i / Math.max(values.length - 1, 1)) * w} ${h - v * h}`).join(" ");
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Coverage over time, now ${pct(values[values.length - 1] ?? 0)}`} className="sm-lab-spark">
      <rect x="0" y="0" width={w} height={h} fill="none" stroke="var(--sm-border)" />
      <path d={d || `M 0 ${h} L ${w} ${h}`} stroke="var(--sm-algo-mappo)" strokeWidth={2} fill="none" />
    </svg>
  );
}

function LiveTab() {
  useSimStore((s) => s.version);
  const { timeline } = useSimStore.getState();
  if (!timeline) return <Empty>No session.</Empty>;
  const t = currentT();
  const series = timeline.coverageSeries.slice(0, t - timeline.startT + 1);
  const frame = timeline.stateAt(t).lastFrame;
  return (
    <div>
      <div className="sm-lab-muted">Coverage C(t), fixed 0–100% axis</div>
      <Sparkline values={series} />
      <div className="sm-lab-muted" style={{ marginTop: "var(--sm-space-5)" }}>
        Reward components (this step, per robot)
      </div>
      {frame ? (
        <table className="sm-lab-table">
          <tbody>
            {Object.entries(frame.reward_components).map(([k, v]) => (
              <tr key={k}>
                <td className="sm-lab-muted">{k}</td>
                <td className="mono">{v.map((x) => x.toFixed(2)).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Empty>No step taken yet.</Empty>
      )}
    </div>
  );
}

const LAYER_ROWS: [LayerKey, string][] = [
  ["sensor", "Sensor radius"],
  ["paths", "Paths"],
  ["visited", "Visited cells"],
  ["unknown", "Unknown (fog)"],
  ["commLinks", "Communication links"],
  ["groundTruth", "Ground truth (debug)"],
];

function LayersTab() {
  const layers = useUiStore((s) => s.layers);
  const toggle = useUiStore((s) => s.toggleLayer);
  const commMode = useSimStore((s) => s.meta?.commMode);
  return (
    <div>
      {LAYER_ROWS.map(([key, label]) => (
        <label key={key} className="sm-lab-layer-row">
          <span>
            {label}
            {key === "groundTruth" && <span className="sm-lab-muted xs block">Not visible to robots</span>}
            {key === "commLinks" && commMode === "shared_map" && <span className="sm-lab-muted xs block">Links are only recorded for limited-range communication</span>}
          </span>
          <input type="checkbox" checked={layers[key]} onChange={() => toggle(key)} className="sm-lab-check" />
        </label>
      ))}
    </div>
  );
}
