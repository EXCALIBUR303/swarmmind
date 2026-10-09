import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { Select } from "../../components/Select";
import type { LevelInfo } from "../../api/endpoints/catalog";
import type { MapSet, Scenario } from "../../stores/scenarioStore";

interface Props {
  open: boolean;
  onClose: () => void;
  scenario: Scenario;
  levels: LevelInfo[];
  onApply: (patch: Partial<Scenario>) => void;
}

const MAP_SETS: { value: MapSet; label: string }[] = [
  { value: "procedural", label: "Procedural (from seed)" },
  { value: "validation", label: "Validation maps" },
  { value: "structured", label: "Structured maps (held-out)" },
  { value: "test", label: "Test maps (held-out)" },
];

/** Scenario drawer: opens OVER the right rail; it never reflows or shrinks the canvas (§30.4). */
export function ScenarioDrawer({ open, onClose, scenario, levels, onApply }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const firstRef = useRef<HTMLSelectElement>(null);

  useEffect(() => {
    if (!open) return;
    firstRef.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  const cfg = (levels.find((l) => l.level === scenario.level)?.config ?? {}) as Record<string, number | string>;
  return (
    <div className="sm-lab-drawer" role="dialog" aria-label="Scenario settings" aria-modal="false" ref={ref}>
      <div className="sm-lab-drawer-head">
        <strong>Scenario</strong>
        <button className="sm-lab-btn" onClick={onClose} aria-label="Close scenario settings">
          <X size={16} />
        </button>
      </div>
      <form
        className="sm-lab-drawer-body"
        onSubmit={(e) => {
          e.preventDefault();
          const f = new FormData(e.currentTarget);
          onApply({ seed: Math.max(0, Math.trunc(Number(f.get("seed")) || 0)), stochastic: f.get("stochastic") === "on" });
          onClose();
        }}
      >
        <label className="sm-lab-field">
          <span>Level</span>
          <select ref={firstRef} className="sm-lab-select" value={scenario.level} onChange={(e) => onApply({ level: e.target.value, policyKey: "heuristic", mapSet: "procedural", mapId: null })}>
            {levels.map((l) => (
              <option key={l.level} value={l.level}>{l.level}</option>
            ))}
          </select>
        </label>
        <dl className="sm-lab-facts">
          <dt>Grid</dt><dd>{cfg.width}×{cfg.height}</dd>
          <dt>Robots</dt><dd>{cfg.n_robots}</dd>
          <dt>Targets</dt><dd>{cfg.n_targets}</dd>
          <dt>Obstacle density</dt><dd>{cfg.obstacle_density}</dd>
          <dt>Communication</dt><dd>{String(cfg.comm_mode ?? "").replace("_", " ")}</dd>
          <dt>Max steps</dt><dd>{cfg.max_steps}</dd>
        </dl>
        <p className="sm-lab-hint">Levels are fixed (L1–L4); the backend can take a custom grid, but it is not exposed here.</p>
        <label className="sm-lab-field">
          <span>Map source</span>
          <Select label="Map source" value={scenario.mapSet} options={MAP_SETS} onChange={(v) => onApply({ mapSet: v as MapSet, mapId: null })} />
        </label>
        <label className="sm-lab-field">
          <span>Seed</span>
          <input name="seed" type="number" min={0} step={1} defaultValue={scenario.seed} className="sm-lab-input" />
        </label>
        <label className="sm-lab-field row">
          <input name="stochastic" type="checkbox" defaultChecked={scenario.stochastic} className="sm-lab-check" />
          <span>Sample actions (stochastic policy; the evaluation default)</span>
        </label>
        <p className="sm-lab-hint">The seed generates a procedural map and seeds the policy's sampling, exactly like the batch evaluator, so a session is reproducible.</p>
        <button type="submit" className="sm-lab-btn primary wide">Apply and restart</button>
      </form>
    </div>
  );
}
