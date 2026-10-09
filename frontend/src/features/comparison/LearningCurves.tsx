import type { LevelSummary } from "../../api/endpoints/analytics";
import { ChartFrame } from "../../components/charts/ChartFrame";
import { LineChart, type LineSeries } from "../../components/charts/LineChart";
import { fmtNum, fmtSteps } from "../../lib/stats";
import { METHOD_COLOR, METHOD_LABEL, metricOf, modeStats } from "./model";

/** Validation success vs training env steps: mean ± 95% CI band, thin per-seed lines, reference lines for the baselines. */
export function LearningCurves({ level }: { level: LevelSummary }) {
  const series: LineSeries[] = [];
  for (const algo of ["ippo", "mappo"] as const) {
    const c = level.methods[algo]?.validation_curve;
    if (!c) continue;
    const x = c.grid_env_steps;
    series.push({
      id: algo,
      label: METHOD_LABEL[algo],
      color: METHOD_COLOR[algo],
      points: x.map((xv, i) => ({ x: xv, y: c.mean[i] })),
      band: x.map((xv, i) => ({ x: xv, lo: c.lo[i], hi: c.hi[i] })),
      thin: c.per_seed.map((ys) => x.map((xv, i) => ({ x: xv, y: ys[i] }))),
    });
  }
  const xs = series.flatMap((s) => s.points.map((p) => p.x));
  const xMax = xs.length ? Math.max(...xs) : 1;
  for (const ref of ["random", "heuristic"] as const) {
    const st = metricOf(modeStats(level, ref, "validation", "stochastic"), "success_rate");
    if (st && xs.length) series.push({ id: ref, label: METHOD_LABEL[ref], color: METHOD_COLOR[ref], dash: true, width: 1.5, points: [{ x: 0, y: st.interval.mean }, { x: xMax, y: st.interval.mean }] });
  }
  const learned = series.filter((s) => s.id === "ippo" || s.id === "mappo");
  const aucMax = level.methods.ippo?.validation_curve?.auc_x_max ?? level.methods.mappo?.validation_curve?.auc_x_max;
  return (
    <ChartFrame
      title="Learning curves: validation success vs training steps"
      caption={`Mean over training seeds with 95% CI band; thin lines are individual seeds; dashed lines are the Random and heuristic references on the same validation maps.${aucMax ? ` The sample-efficiency AUC uses the first ${fmtSteps(aucMax)} steps.` : ""}`}
      state={learned.length ? "ready" : "empty"}
      emptyText="This study has no validation curves."
      legend={series.map((s) => ({ label: s.label, color: s.color, dash: s.dash }))}
      table={{
        columns: ["env steps", ...learned.flatMap((s) => [`${s.label} mean`, `${s.label} CI low`, `${s.label} CI high`])],
        rows: (learned[0]?.points ?? []).map((p, i) => [fmtSteps(p.x), ...learned.flatMap((s) => [fmtNum(s.points[i]?.y, 3), fmtNum(s.band?.[i]?.lo, 3), fmtNum(s.band?.[i]?.hi, 3)])]),
      }}
    >
      <LineChart series={series} xLabel="training env steps" yLabel="validation success rate" xFormat={fmtSteps} yFormat={(v) => fmtNum(v, 2)} yDomain={[0, 1.05]} height={280} ariaLabel="Validation success rate versus training steps for IPPO and MAPPO" />
    </ChartFrame>
  );
}
