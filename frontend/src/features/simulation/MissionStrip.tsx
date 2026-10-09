import { useSimStore } from "../../stores/simStore";
import { pct } from "../../lib/format";
import { coverage } from "./frameReducer";

import { STATUS_LABEL, useRunStatus } from "./useRunStatus";

/** One strip instead of an overlay card plus a duplicate panel (§30.1). `aria-live` announces
 * state changes only: the throttled `status` word, not every step. */
export function MissionStrip() {
  useSimStore((s) => s.version); // re-render once per step, not per animation frame
  const { timeline, meta, playhead, mode } = useSimStore.getState();
  const status = useRunStatus();
  if (!timeline || !meta) {
    return (
      <div className="sm-lab-mission" aria-live="polite">
        <span className="sm-lab-mission-stat" style={{ marginLeft: "auto" }} role="status">
          <span className="sm-lab-status-dot idle" aria-hidden="true" />
          {STATUS_LABEL.idle}
        </span>
      </div>
    );
  }
  const t = mode === "live" ? timeline.maxT : Math.min(Math.max(Math.floor(playhead), timeline.startT), timeline.maxT);
  const st = timeline.stateAt(t);
  return (
    <div className="sm-lab-mission">
      <span className="sm-lab-mission-stat">
        t <span className="value">{t}/{meta.maxSteps}</span>
      </span>
      <span className="sm-lab-mission-stat">
        Rescued <span className="value">{st.rescued}/{meta.targets.length}</span>
      </span>
      <span className="sm-lab-mission-stat">
        Coverage <span className="value">{pct(coverage(meta, st))}</span>
      </span>
      <span className="sm-lab-mission-stat">
        Collisions <span className="value">{st.collisions}</span>
      </span>
      <span className="sm-lab-mission-stat" style={{ marginLeft: "auto" }} role="status" aria-live="polite">
        <span className={`sm-lab-status-dot ${status}`} aria-hidden="true" />
        {STATUS_LABEL[status]}
      </span>
    </div>
  );
}
