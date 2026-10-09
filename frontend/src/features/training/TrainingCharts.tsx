import { ChartFrame } from "../../components/charts/ChartFrame";
import { LineChart, type LineSeries } from "../../components/charts/LineChart";
import type { MetricRow } from "../../api/endpoints/analytics";
import { fmtNum, fmtSteps, rollingMean } from "../../lib/stats";

interface Def {
  key: string;
  title: string;
  unit: string;
  smooth?: boolean;
  fixed?: [number, number];
}

/** Only metrics that actually exist in metrics.jsonl are drawn (§30.6); absent ones render nothing. */
const TRAIN_DEFS: Def[] = [
  { key: "mean_reward", title: "Mean reward", unit: "reward per agent-step", smooth: true },
  { key: "policy_loss", title: "Policy loss", unit: "PPO surrogate" },
  { key: "value_loss", title: "Value loss", unit: "MSE (normalised returns)" },
  { key: "entropy", title: "Policy entropy", unit: "nats" },
  { key: "approx_kl", title: "Approximate KL", unit: "nats per update" },
  { key: "explained_variance", title: "Explained variance", unit: "1 = perfect critic" },
  { key: "clip_fraction", title: "Clip fraction", unit: "fraction of samples", fixed: [0, 1] },
  { key: "actor_grad_norm", title: "Actor gradient norm", unit: "L2 norm" },
  { key: "critic_grad_norm", title: "Critic gradient norm", unit: "L2 norm" },
];
const VAL_DEFS: Def[] = [
  { key: "success_rate", title: "Validation success", unit: "fraction of episodes", fixed: [0, 1] },
  { key: "mean_coverage", title: "Validation coverage", unit: "fraction of free cells", fixed: [0, 1] },
  { key: "mean_return", title: "Validation return", unit: "mean per robot" },
  { key: "censored_mean_steps", title: "Validation steps to completion", unit: "steps (failures count max_steps)" },
];

const MAX_POINTS = 400;
const WINDOW = 20;
const stride = <T,>(a: T[]): T[] => (a.length <= MAX_POINTS ? a : a.filter((_, i) => i % Math.ceil(a.length / MAX_POINTS) === 0 || i === a.length - 1));

export function TrainingCharts({ metrics, val, color = "var(--sm-accent)", live }: { metrics: MetricRow[]; val: MetricRow[]; color?: string; live?: boolean }) {
  const build = (rows: MetricRow[], d: Def): { series: LineSeries[]; shown: MetricRow[]; total: number } | null => {
    const have = rows.filter((r) => typeof r[d.key] === "number" && Number.isFinite(r[d.key]));
    if (have.length === 0) return null;
    const shown = stride(have);
    const pts = shown.map((r) => ({ x: r.env_steps, y: r[d.key] }));
    if (d.smooth) {
      const smooth = rollingMean(have.map((r) => r[d.key]), WINDOW);
      const sm = shown.map((r) => ({ x: r.env_steps, y: smooth[have.indexOf(r)] }));
      return { shown, total: have.length, series: [{ id: "raw", label: "raw", color, width: 1, opacity: 0.35, points: pts, directLabel: false }, { id: "mean", label: `${WINDOW}-update mean`, color, points: sm, directLabel: false }] };
    }
    return { shown, total: have.length, series: [{ id: d.key, label: d.title, color, points: pts, directLabel: false }] };
  };
  const frames = [...TRAIN_DEFS.map((d) => ({ d, rows: metrics })), ...VAL_DEFS.map((d) => ({ d, rows: val }))]
    .map(({ d, rows }) => ({ d, b: build(rows, d) }))
    .filter((x): x is { d: Def; b: NonNullable<ReturnType<typeof build>> } => x.b !== null);
  if (frames.length === 0) return <ChartFrame title="Training metrics" state="empty" emptyText={live ? "Waiting for the first update…" : "No metrics were logged for this run."}>{null}</ChartFrame>;
  return (
    <div className="sm-grid wide">
      {frames.map(({ d, b }) => (
        <ChartFrame
          key={d.key}
          title={d.title}
          caption={`${d.unit}${d.smooth ? ` · faint line = raw, solid = ${WINDOW}-update trailing mean` : ""}${b.shown.length < b.total ? ` · every ${Math.ceil(b.total / MAX_POINTS)}th point shown` : ""}`}
          state="ready"
          table={{ columns: ["env steps", d.key], rows: b.shown.map((r) => [fmtSteps(r.env_steps), fmtNum(r[d.key], 4)]) }}
        >
          <LineChart series={b.series} xLabel="training env steps" yLabel={d.title} xFormat={fmtSteps} yFormat={(v) => fmtNum(v, Math.abs(v) < 0.01 ? 4 : 3)} yDomain={d.fixed} height={200} compact ariaLabel={`${d.title} over training`} />
        </ChartFrame>
      ))}
    </div>
  );
}
