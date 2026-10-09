import { Pause, Play, RotateCcw, SkipBack, SkipForward, Settings2 } from "lucide-react";
import { Select, type SelectOption } from "../../components/Select";
import { SPEEDS } from "../../stores/simStore";

export interface ToolbarProps {
  mode: "live" | "replay";
  onMode: (m: "live" | "replay") => void;
  selectors: { label: string; value: string; options: SelectOption[]; onChange: (v: string) => void; placeholder?: string }[];
  playback: {
    playing: boolean;
    speed: number;
    enabled: boolean;
    canStep: boolean;
    canStepBack: boolean;
    onToggle: () => void;
    onStep: () => void;
    onStepBack: () => void;
    onReset: () => void;
    onSpeed: (s: number) => void;
  };
  onOpenDrawer: () => void;
  drawerButtonRef?: React.Ref<HTMLButtonElement>;
}

export function Toolbar({ mode, onMode, selectors, playback: p, onOpenDrawer, drawerButtonRef }: ToolbarProps) {
  return (
    <div className="sm-lab-toolbar" role="toolbar" aria-label="Simulation controls">
      <div className="sm-lab-seg" role="group" aria-label="Mode">
        {(["replay", "live"] as const).map((m) => (
          <button key={m} className={`sm-lab-seg-btn${mode === m ? " active" : ""}`} aria-pressed={mode === m} onClick={() => onMode(m)}>
            {m === "live" ? "Live" : "Replay"}
          </button>
        ))}
      </div>
      {selectors.map((s) => (
        <div className="sm-lab-toolbar-group" key={s.label}>
          <span className="sm-lab-toolbar-label">{s.label}</span>
          <Select label={s.label} value={s.value} options={s.options} onChange={s.onChange} placeholder={s.placeholder} />
        </div>
      ))}
      <div className="sm-lab-toolbar-group sm-lab-toolbar-playback">
        <button className="sm-lab-btn" onClick={p.onReset} aria-label="Reset (R)" title="Reset (R)" disabled={!p.enabled}>
          <RotateCcw size={16} />
        </button>
        <button className="sm-lab-btn" onClick={p.onStepBack} aria-label="Step back (←)" title="Step back (←, replay only)" disabled={!p.enabled || !p.canStepBack}>
          <SkipBack size={16} />
        </button>
        <button className="sm-lab-btn primary" onClick={p.onToggle} aria-label={p.playing ? "Pause (Space)" : "Play (Space)"} title={p.playing ? "Pause (Space)" : "Play (Space)"} disabled={!p.enabled}>
          {p.playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button className="sm-lab-btn" onClick={p.onStep} aria-label="Step (→)" title="Step (→)" disabled={!p.enabled || !p.canStep}>
          <SkipForward size={16} />
        </button>
        <Select label="Playback speed" value={String(p.speed)} options={SPEEDS.map((s) => ({ value: String(s), label: `${s}×` }))} onChange={(v) => p.onSpeed(Number(v))} disabled={!p.enabled} />
        <button className="sm-lab-btn" ref={drawerButtonRef} aria-label="Scenario settings" title="Scenario settings" aria-haspopup="dialog" onClick={onOpenDrawer}>
          <Settings2 size={16} />
        </button>
      </div>
    </div>
  );
}
