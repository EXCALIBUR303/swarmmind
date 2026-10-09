import { useSimStore } from "../../stores/simStore";

const GLYPH = { discover: "◆", rescue: "★", collision: "✕" } as const;
const LABEL = { discover: "Target discovered", rescue: "Target rescued", collision: "Collision" } as const;

/** Scrubber with event markers: ◆ discovery, ★ rescue, ✕ collision (shape carries the meaning, not colour). */
export function Timeline() {
  useSimStore((s) => s.version);
  const { timeline, playhead, seek, mode } = useSimStore.getState();
  if (!timeline || mode !== "replay") return null;
  const span = Math.max(timeline.maxT - timeline.startT, 1);
  const t = Math.min(Math.max(Math.floor(playhead), timeline.startT), timeline.maxT);
  return (
    <div className="sm-lab-timeline">
      <input
        className="sm-lab-scrub"
        type="range"
        min={timeline.startT}
        max={timeline.maxT}
        value={t}
        onChange={(e) => seek(Number(e.target.value))}
        aria-label="Scrub timeline"
        aria-valuetext={`step ${t} of ${timeline.maxT}`}
      />
      <div className="sm-lab-markers" aria-hidden="true">
        {timeline.events.map((e, i) => (
          <span
            key={i}
            className={`sm-lab-marker ${e.kind}`}
            style={{ left: `${((e.t - timeline.startT) / span) * 100}%` }}
            title={`${LABEL[e.kind]} · t=${e.t}`}
          >
            {GLYPH[e.kind]}
          </span>
        ))}
      </div>
      <ul className="sm-lab-sr-only" aria-label="Events">
        {timeline.events.map((e, i) => (
          <li key={i}>{`${LABEL[e.kind]} at step ${e.t}`}</li>
        ))}
      </ul>
    </div>
  );
}
