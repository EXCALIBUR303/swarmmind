import { useEffect, useState } from "react";
import { getPerfStats, type PerfStats } from "./perf";

/** Developer overlay, shown only with `?perf=1`: measured frame rate while animating. */
export function PerfOverlay() {
  const enabled = typeof window !== "undefined" && new URLSearchParams(window.location.search).has("perf");
  const [stats, setStats] = useState<PerfStats | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => setStats(getPerfStats()), 500);
    return () => clearInterval(id);
  }, [enabled]);
  if (!enabled || !stats) return null;
  return (
    <div className="sm-perf" role="status" aria-label="Rendering performance">
      {stats.fps ? `${stats.fps.toFixed(1)} fps` : "idle"} · draw {stats.drawMsMean?.toFixed(2) ?? "–"} ms · p95 frame {stats.frameMsP95?.toFixed(1) ?? "–"} ms
    </div>
  );
}
