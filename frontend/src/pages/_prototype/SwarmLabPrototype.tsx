/**
 * Static Swarm Lab design prototype (Phase 9, spec §30.4/§30.10).
 *
 * NOT a production route — not registered in `app/routes.tsx`. Built from
 * a REAL recorded replay (Phase 7's main_L2 study, MAPPO seed 0's
 * near-final checkpoint, a successful 25-step episode on the frozen
 * structured map `L2-struct-bottleneck_vertical`), never mock data. Local
 * playback only (no network/backend calls) — that arrives in Phase 11
 * once the session API (Phase 10) exists.
 *
 * This file exists to be screenshotted at the four breakpoints and
 * reviewed for the Phase 9 design approval gate; it is not meant to be
 * imported anywhere else.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { animate, createScope, stagger, type Scope } from "animejs";
import {
  LayoutDashboard,
  LineChart,
  GitCompare,
  Activity,
  History,
  Settings as SettingsIcon,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Settings2,
} from "lucide-react";
import "../../design/tokens.css";
import "./SwarmLabPrototype.css";
import replayJson from "./replay-fixture.json";

/** Matches the recorded Replay schema (spec §27.2) — declared explicitly
 * because per-frame arrays that are empty in some frames (e.g. no
 * discoveries that step) would otherwise infer as `never[]` from the raw
 * JSON literal. */
interface Frame {
  t: number;
  pos: [number, number][];
  action: number[];
  moved: boolean[];
  blocked_by_robot: boolean[];
  bumped_wall: boolean[];
  newly_known: [number, number][];
  discovered: number[];
  rescued: [number, number][];
  reward: number[];
  reward_components: Record<string, number[]>;
  comm_edges: [number, number][];
}

const replay = replayJson as unknown as {
  env: { config: { n_robots: number; obs_radius: number; sensor_radius: number; max_steps: number } };
  map: { id: string; targets: [number, number][]; terrain: string[]; height: number; width: number; robot_starts: [number, number][] };
  outcome: { success: boolean; steps: number; metrics: Record<string, number> };
  frames: Frame[];
};

type RailTab = "robots" | "live" | "layers";
type LayerState = { sensor: boolean; paths: boolean; visited: boolean; unknown: boolean; commLinks: boolean; groundTruth: boolean };

const ROBOT_COLORS = ["var(--sm-robot-1)", "var(--sm-robot-2)", "var(--sm-robot-3)"];
const N_ROBOTS = replay.env.config.n_robots;
const TARGETS: [number, number][] = replay.map.targets as [number, number][];
const TERRAIN: string[] = replay.map.terrain;
const HEIGHT = replay.map.height;
const WIDTH = replay.map.width;
const OBS_RADIUS = replay.env.config.obs_radius;
const SENSOR_RADIUS = replay.env.config.sensor_radius;
const MAX_STEPS = replay.env.config.max_steps;
const CELL = 44;

const NAV_ITEMS = [
  { label: "Swarm Lab", icon: LayoutDashboard, active: true },
  { label: "Train Swarm", icon: LineChart, active: false },
  { label: "Algorithm Comparison", icon: GitCompare, active: false },
  { label: "Swarm Analysis", icon: Activity, active: false },
  { label: "Experiment History", icon: History, active: false },
  { label: "Settings", icon: SettingsIcon, active: false },
];

/** Reconstruct known-free cells up to frame index t by accumulating
 * `newly_known` deltas — exactly §27.4's rule for the real replay viewer,
 * so the fog-of-war rendering here matches what Phase 11 must do. */
function knownFreeCellsUpTo(frameIndex: number): Set<string> {
  const known = new Set<string>();
  for (let i = 0; i <= frameIndex && i < replay.frames.length; i++) {
    for (const [y, x] of (replay.frames[i] as Frame).newly_known) known.add(`${y},${x}`);
  }
  return known;
}

function targetStatusAt(frameIndex: number, k: number): "hidden" | "discovered" | "rescued" {
  let status: "hidden" | "discovered" | "rescued" = "hidden";
  for (let i = 0; i <= frameIndex && i < replay.frames.length; i++) {
    const f = replay.frames[i] as Frame;
    if (f.discovered.includes(k)) status = "discovered";
    if (f.rescued.some(([rk]) => rk === k)) status = "rescued";
  }
  return status;
}

function robotStateAt(frameIndex: number, robotId: number): string {
  if (frameIndex < 0) return "Waiting";
  const f = replay.frames[frameIndex] as Frame;
  const rescue = f.rescued.find(([, r]) => r === robotId);
  if (rescue) return `Rescued T${rescue[0]} @ t=${f.t}`;
  if (f.blocked_by_robot[robotId]) return "Blocked";
  if (f.bumped_wall[robotId]) return "Blocked";
  return f.moved[robotId] ? "Moving" : "Waiting";
}

export function SwarmLabPrototype() {
  const [frameIndex, setFrameIndex] = useState(-1); // -1 = initial state, before frame 0
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [focusedRobot, setFocusedRobot] = useState<number | null>(null);
  const [railTab, setRailTab] = useState<RailTab>("robots");
  const [layers, setLayers] = useState({
    sensor: true,
    paths: true,
    visited: true,
    unknown: true,
    commLinks: false,
    groundTruth: false,
  });
  const [railOpenMobile, setRailOpenMobile] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const navRef = useRef<HTMLElement>(null);

  // Nav entrance (anime.js, §30.7-style micro-interaction: communicates
  // "the app is ready", once, on mount — never looping). `createScope`
  // auto-cleans its animations on unmount, so this is safe under React's
  // effect re-run rules. Honours prefers-reduced-motion (§30.9) by
  // skipping straight to the settled state instead of animating.
  useEffect(() => {
    if (!navRef.current) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const items = navRef.current.querySelectorAll(".sm-proto-nav-item");
    if (reduced) {
      animate(items, { opacity: 1, translateX: 0, duration: 0 });
      return;
    }
    const scope: Scope = createScope({ root: navRef }).add(() => {
      animate(".sm-proto-nav-title", { opacity: [0, 1], translateY: [-6, 0], duration: 500, easing: "easeOutQuad" });
      animate(items, {
        opacity: [0, 1],
        translateX: [-16, 0],
        duration: 420,
        delay: stagger(60, { start: 150 }),
        easing: "easeOutCubic",
      });
    });
    return () => scope.revert();
  }, []);

  const lastIndex = replay.frames.length - 1;
  const clampedIndex = Math.min(Math.max(frameIndex, -1), lastIndex);
  const positions = useMemo(
    () => (clampedIndex === -1 ? (replay.map.robot_starts as [number, number][]) : (replay.frames[clampedIndex] as Frame).pos),
    [clampedIndex],
  ) as [number, number][];

  useEffect(() => {
    if (!playing || clampedIndex >= lastIndex) return;
    // Advance and, in the same updater, stop at the end — avoids a
    // separate setPlaying(false) call synchronously inside the effect.
    const id = setTimeout(
      () =>
        setFrameIndex((i) => {
          const next = Math.min(i + 1, lastIndex);
          if (next >= lastIndex) setPlaying(false);
          return next;
        }),
      500 / speed,
    );
    return () => clearTimeout(id);
  }, [playing, clampedIndex, lastIndex, speed]);

  const stepForward = useCallback(() => setFrameIndex((i) => Math.min(i + 1, lastIndex)), [lastIndex]);
  const stepBack = useCallback(() => setFrameIndex((i) => Math.max(i - 1, -1)), []);
  const reset = useCallback(() => {
    setFrameIndex(-1);
    setPlaying(false);
  }, []);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.target instanceof HTMLSelectElement || e.target instanceof HTMLInputElement) return;
      switch (e.key) {
        case " ":
          e.preventDefault();
          setPlaying((p) => !p);
          break;
        case "ArrowRight":
          stepForward();
          break;
        case "ArrowLeft":
          stepBack();
          break;
        case "r":
        case "R":
          reset();
          break;
        case "l":
        case "L":
          setRailTab("layers");
          setRailOpenMobile(true);
          break;
        default:
          if (/^[1-6]$/.test(e.key)) {
            const id = Number(e.key) - 1;
            if (id < N_ROBOTS) setFocusedRobot((f) => (f === id ? null : id));
          }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [stepForward, stepBack, reset]);

  const knownFree = layers.unknown ? knownFreeCellsUpTo(clampedIndex) : null;
  const events = useMemo(() => {
    const out: { t: number; kind: "discover" | "rescue"; robotId?: number }[] = [];
    replay.frames.forEach((f) => {
      f.discovered.forEach(() => out.push({ t: f.t, kind: "discover" }));
      f.rescued.forEach(([, r]) => out.push({ t: f.t, kind: "rescue", robotId: r }));
    });
    return out;
  }, []);

  const coveragePct = clampedIndex === -1 ? 0 : Math.round((replay.frames[clampedIndex] as Frame).t >= 0 ? coverageAt(clampedIndex) * 100 : 0);
  const rescuedCount = TARGETS.filter((_, k) => targetStatusAt(clampedIndex, k) === "rescued").length;
  const collisionsSoFar = replay.frames.slice(0, clampedIndex + 1).reduce((sum, f) => sum + f.blocked_by_robot.filter(Boolean).length, 0);
  const isDone = clampedIndex >= lastIndex;
  const status = isDone ? (replay.outcome.success ? "success" : "failure") : playing ? "running" : "paused";

  return (
    <div className="sm-proto" ref={containerRef}>
      <nav className="sm-proto-nav" aria-label="Primary" ref={navRef}>
        <div className="sm-proto-nav-title">SWARMMIND</div>
        {NAV_ITEMS.map(({ label, icon: Icon, active }) => (
          <div key={label} className={`sm-proto-nav-item${active ? " active" : ""}`}>
            <Icon size={20} />
            <span className="label">{label}</span>
          </div>
        ))}
      </nav>

      <div className="sm-proto-main">
        <div className="sm-proto-toolbar" role="toolbar" aria-label="Simulation controls">
          <div className="sm-proto-toolbar-group">
            Policy
            <select className="sm-proto-select" defaultValue="mappo-l2-s0" aria-label="Policy checkpoint">
              <option value="mappo-l2-s0">MAPPO · L2 · s0 · 999k</option>
            </select>
          </div>
          <div className="sm-proto-toolbar-group">
            Map
            <select className="sm-proto-select" defaultValue="bv" aria-label="Map">
              <option value="bv">{replay.map.id}</option>
            </select>
          </div>
          <div className="sm-proto-toolbar-group" style={{ marginLeft: "auto" }}>
            <button className="sm-proto-btn" onClick={reset} aria-label="Reset (R)">
              <RotateCcw size={16} />
            </button>
            <button className="sm-proto-btn" onClick={stepBack} aria-label="Step back (←)" disabled={clampedIndex <= -1}>
              <SkipBack size={16} />
            </button>
            <button className="sm-proto-btn primary" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause (Space)" : "Play (Space)"}>
              {playing ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button className="sm-proto-btn" onClick={stepForward} aria-label="Step (→)" disabled={isDone}>
              <SkipForward size={16} />
            </button>
            <select
              className="sm-proto-select"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              aria-label="Playback speed"
            >
              {[0.25, 0.5, 1, 2, 4, 8].map((s) => (
                <option key={s} value={s}>
                  {s}×
                </option>
              ))}
            </select>
            <span className="sm-proto-toolbar-group">Mode: Replay</span>
            <button
              className="sm-proto-btn"
              aria-label="Scenario settings"
              onClick={() => setRailOpenMobile((v) => !v)}
            >
              <Settings2 size={16} />
            </button>
          </div>
        </div>

        <div className="sm-proto-mission" aria-live="polite">
          <span className="sm-proto-mission-stat">
            t <span className="value">{Math.max(clampedIndex + 1, 0)}/{MAX_STEPS}</span>
          </span>
          <span className="sm-proto-mission-stat">
            Rescued <span className="value">{rescuedCount}/{TARGETS.length}</span>
          </span>
          <span className="sm-proto-mission-stat">
            Coverage <span className="value">{coveragePct}%</span>
          </span>
          <span className="sm-proto-mission-stat">
            Collisions <span className="value">{collisionsSoFar}</span>
          </span>
          <span className="sm-proto-mission-stat" style={{ marginLeft: "auto" }}>
            <span className={`sm-proto-status-dot ${status}`} />
            {status === "running" ? "RUNNING" : status === "success" ? "SUCCESS" : status === "failure" ? "FAILED" : "PAUSED"}
          </span>
        </div>

        <div className="sm-proto-canvas-wrap">
          <div className="sm-proto-canvas-frame">
            <SimulationCanvas
              positions={positions}
              frameIndex={clampedIndex}
              knownFree={knownFree}
              layers={layers}
              focusedRobot={focusedRobot}
              onRobotClick={(id) => setFocusedRobot((f) => (f === id ? null : id))}
            />
          </div>
        </div>

        <div className="sm-proto-timeline">
          <input
            className="sm-proto-scrub"
            type="range"
            min={-1}
            max={lastIndex}
            value={clampedIndex}
            onChange={(e) => setFrameIndex(Number(e.target.value))}
            aria-label="Scrub timeline"
          />
          <svg width="100%" height="20" style={{ display: "block", marginTop: 2 }}>
            {events.map((e, i) => (
              <text
                key={i}
                x={`${((e.t - 1) / (lastIndex + 1)) * 100}%`}
                y={14}
                fontSize={12}
                fill={e.kind === "discover" ? "var(--sm-target-fill)" : "var(--sm-success)"}
              >
                {e.kind === "discover" ? "◆" : "★"}
              </text>
            ))}
          </svg>
        </div>
      </div>

      {railOpenMobile && (
        <aside className="sm-proto-rail" aria-label="Details">
          <div className="sm-proto-rail-tabs" role="tablist">
            {(["robots", "live", "layers"] as RailTab[]).map((tab) => (
              <button
                key={tab}
                role="tab"
                aria-selected={railTab === tab}
                className={`sm-proto-rail-tab${railTab === tab ? " active" : ""}`}
                onClick={() => setRailTab(tab)}
              >
                {tab === "robots" ? "Robots" : tab === "live" ? "Live" : "Layers"}
              </button>
            ))}
          </div>
          <div className="sm-proto-rail-body">
            {railTab === "robots" && (
              <RobotsTab
                positions={positions}
                frameIndex={clampedIndex}
                focusedRobot={focusedRobot}
                onFocus={(id) => setFocusedRobot((f) => (f === id ? null : id))}
              />
            )}
            {railTab === "live" && <LiveTab frameIndex={clampedIndex} />}
            {railTab === "layers" && <LayersTab layers={layers} setLayers={setLayers} />}
          </div>
        </aside>
      )}
    </div>
  );
}

function coverageAt(frameIndex: number): number {
  const nFree = TERRAIN.join("").split("").filter((c) => c === ".").length;
  return knownFreeCellsUpTo(frameIndex).size / nFree;
}

function RobotsTab({
  positions,
  frameIndex,
  focusedRobot,
  onFocus,
}: {
  positions: [number, number][];
  frameIndex: number;
  focusedRobot: number | null;
  onFocus: (id: number) => void;
}) {
  return (
    <div>
      {positions.map((pos, id) => {
        const distance = replay.frames.slice(0, frameIndex + 1).filter((f) => f.moved[id]).length;
        const steps = Math.max(frameIndex + 1, 1);
        const idlePct = Math.round((1 - distance / steps) * 100);
        return (
          <button key={id} className={`sm-proto-robot-row${focusedRobot === id ? " focused" : ""}`} onClick={() => onFocus(id)}>
            <span className="sm-proto-swatch" style={{ background: ROBOT_COLORS[id] }} />
            <span>
              <strong>R{id + 1}</strong>{" "}
              <span style={{ color: "var(--sm-text-muted)", font: "var(--sm-font-sm)" }}>
                ({pos[0]},{pos[1]}) · {robotStateAt(frameIndex, id)}
              </span>
            </span>
            <span style={{ font: "var(--sm-font-xs)", color: "var(--sm-text-muted)" }}>
              d={distance} idle={idlePct}%
            </span>
          </button>
        );
      })}
      <p style={{ font: "var(--sm-font-xs)", color: "var(--sm-text-muted)", marginTop: "var(--sm-space-4)" }}>
        States are observable facts only (Moving / Blocked / Waiting / Rescued T# @ t=#) — never a fabricated
        intent label (§30.1).
      </p>
    </div>
  );
}

function LiveTab({ frameIndex }: { frameIndex: number }) {
  const points = replay.frames.slice(0, frameIndex + 1).map((_, i) => coverageAt(i) * 100);
  const w = 260;
  const h = 60;
  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${(i / Math.max(points.length - 1, 1)) * w} ${h - (p / 100) * h}`)
    .join(" ");
  return (
    <div>
      <div style={{ font: "var(--sm-font-sm)", color: "var(--sm-text-muted)", marginBottom: "var(--sm-space-2)" }}>
        Coverage C(t)
      </div>
      <svg width={w} height={h} role="img" aria-label="Coverage over time sparkline">
        <path d={path || `M 0 ${h} L ${w} ${h}`} stroke="var(--sm-algo-mappo)" strokeWidth={2} fill="none" />
      </svg>
      <div style={{ font: "var(--sm-font-sm)", color: "var(--sm-text-muted)", marginTop: "var(--sm-space-5)" }}>
        Reward components (this frame)
      </div>
      {frameIndex >= 0 && (
        <table style={{ width: "100%", font: "var(--sm-font-xs)", marginTop: "var(--sm-space-2)" }}>
          <tbody>
            {Object.entries((replay.frames[frameIndex] as Frame).reward_components).map(([k, v]) => (
              <tr key={k}>
                <td style={{ color: "var(--sm-text-muted)" }}>{k}</td>
                <td style={{ fontFamily: "var(--sm-font-mono)", textAlign: "right" }}>
                  {(v as number[]).map((x) => x.toFixed(2)).join(", ")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function LayersTab({
  layers,
  setLayers,
}: {
  layers: LayerState;
  setLayers: (fn: (l: LayerState) => LayerState) => void;
}) {
  const rows: [keyof LayerState, string][] = [
    ["sensor", "Sensor radius"],
    ["paths", "Paths"],
    ["visited", "Visited cells"],
    ["unknown", "Unknown (fog)"],
    ["commLinks", "Communication links"],
    ["groundTruth", "Ground truth (debug)"],
  ];
  return (
    <div>
      {rows.map(([key, label]) => (
        <label key={key} className="sm-proto-layer-row">
          <span>
            {label}
            {key === "groundTruth" && (
              <div style={{ font: "var(--sm-font-xs)", color: "var(--sm-text-muted)" }}>Not visible to robots</div>
            )}
          </span>
          <input
            type="checkbox"
            checked={layers[key]}
            onChange={() => setLayers((l) => ({ ...l, [key]: !l[key] }))}
            style={{ width: 18, height: 18, accentColor: "var(--sm-accent)" }}
          />
        </label>
      ))}
    </div>
  );
}

function SimulationCanvas({
  positions,
  frameIndex,
  knownFree,
  layers,
  focusedRobot,
  onRobotClick,
}: {
  positions: [number, number][];
  frameIndex: number;
  knownFree: Set<string> | null;
  layers: LayerState;
  focusedRobot: number | null;
  onRobotClick: (id: number) => void;
}) {
  const size = CELL * WIDTH;
  const pathHistory = layers.paths
    ? Array.from({ length: N_ROBOTS }, (_, id) =>
        (replay.map.robot_starts as [number, number][])[id]
          ? [
              (replay.map.robot_starts as [number, number][])[id],
              ...replay.frames.slice(0, frameIndex + 1).map((f) => f.pos[id] as [number, number]),
            ]
          : [],
      )
    : [];

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label="Simulation grid"
      style={{ width: "100%", height: "auto", maxWidth: size, display: "block" }}
    >
      {Array.from({ length: HEIGHT }).map((_, y) =>
        Array.from({ length: WIDTH }).map((_, x) => {
          const isObstacle = TERRAIN[y][x] === "#";
          const known = knownFree ? knownFree.has(`${y},${x}`) : true;
          const fill = isObstacle
            ? "var(--sm-cell-obstacle)"
            : !layers.unknown || known
              ? "var(--sm-cell-free)"
              : "var(--sm-cell-unknown)";
          return (
            <rect
              key={`${y}-${x}`}
              x={x * CELL}
              y={y * CELL}
              width={CELL}
              height={CELL}
              fill={fill}
              stroke={CELL >= 14 ? "var(--sm-grid-line)" : "none"}
            />
          );
        }),
      )}

      {focusedRobot !== null && (
        <rect
          x={(positions[focusedRobot][1] - OBS_RADIUS) * CELL}
          y={(positions[focusedRobot][0] - OBS_RADIUS) * CELL}
          width={(2 * OBS_RADIUS + 1) * CELL}
          height={(2 * OBS_RADIUS + 1) * CELL}
          fill="none"
          stroke={ROBOT_COLORS[focusedRobot]}
          strokeWidth={2}
          strokeDasharray="6 4"
        />
      )}

      {layers.paths &&
        pathHistory.map((pts, id) => (
          <polyline
            key={id}
            points={pts.map(([y, x]) => `${x * CELL + CELL / 2},${y * CELL + CELL / 2}`).join(" ")}
            fill="none"
            stroke={ROBOT_COLORS[id]}
            strokeWidth={2}
            opacity={focusedRobot === null || focusedRobot === id ? 0.5 : 0.15}
          />
        ))}

      {TARGETS.map(([y, x], k) => {
        const status = targetStatusAt(frameIndex, k);
        if (status === "hidden") return null;
        const cx = x * CELL + CELL / 2;
        const cy = y * CELL + CELL / 2;
        return (
          <g key={k} opacity={status === "rescued" ? 0.3 : 1}>
            <StarShape cx={cx} cy={cy} r={CELL * 0.28} fill={status === "rescued" ? "none" : "var(--sm-target-fill)"} />
            <StarShape cx={cx} cy={cy} r={CELL * 0.28} fill="none" stroke="var(--sm-target-outline)" strokeWidth={1.5} />
          </g>
        );
      })}

      {positions.map(([y, x], id) => {
        const cx = x * CELL + CELL / 2;
        const cy = y * CELL + CELL / 2;
        const dimmed = focusedRobot !== null && focusedRobot !== id;
        return (
          <g key={id} opacity={dimmed ? 0.4 : 1} style={{ cursor: "pointer" }} onClick={() => onRobotClick(id)}>
            {layers.sensor && (
              <circle cx={cx} cy={cy} r={SENSOR_RADIUS * CELL} fill={ROBOT_COLORS[id]} fillOpacity={0.08} stroke={ROBOT_COLORS[id]} strokeOpacity={0.35} strokeWidth={1} strokeDasharray="3 3" />
            )}
            <circle cx={cx} cy={cy} r={CELL * 0.3} fill={ROBOT_COLORS[id]} stroke="#fff" strokeWidth={2} />
            <text x={cx} y={cy + 4} textAnchor="middle" fontSize={11} fontWeight={600} fill="#fff">
              {id + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function StarShape({ cx, cy, r, fill, stroke, strokeWidth }: { cx: number; cy: number; r: number; fill: string; stroke?: string; strokeWidth?: number }) {
  const points = Array.from({ length: 10 }, (_, i) => {
    const angle = (Math.PI / 5) * i - Math.PI / 2;
    const radius = i % 2 === 0 ? r : r * 0.45;
    return `${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`;
  }).join(" ");
  return <polygon points={points} fill={fill} stroke={stroke} strokeWidth={strokeWidth} />;
}
