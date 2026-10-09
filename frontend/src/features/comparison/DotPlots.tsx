import type { LevelSummary } from "../../api/endpoints/analytics";
import { ChartFrame } from "../../components/charts/ChartFrame";
import { DotWhisker, type DotRow } from "../../components/charts/DotWhisker";
import { fmtNum } from "../../lib/stats";
import { METHOD_COLOR, METHOD_LABEL, METRICS, metricOf, modeStats, PRIMARY_KEYS, type Method } from "./model";

const ROWS: Method[] = ["random", "ippo", "mappo"];

export function DotPlots({ level, split, mode }: { level: LevelSummary; split: string; mode: string }) {
  return (
    <div className="sm-grid">
      {METRICS.filter((m) => PRIMARY_KEYS.includes(m.key)).map((def) => {
        const rows: DotRow[] = ROWS.flatMap((method) => {
          const st = metricOf(modeStats(level, method, split, mode), def.key);
          return st ? [{ id: method, label: METHOD_LABEL[method], color: METHOD_COLOR[method], perSeed: st.per_seed, mean: st.interval.mean, lo: st.interval.lo, hi: st.interval.hi }] : [];
        });
        const heur = metricOf(modeStats(level, "heuristic", split, mode), def.key);
        const ref = heur ? { label: `heuristic ${fmtNum(heur.interval.mean, def.digits)}`, value: heur.interval.mean } : null;
        return (
          <ChartFrame
            key={def.key}
            title={def.label}
            caption={`${def.unit} · ${split}, ${mode} · each dot is one training seed (Random: one evaluation seed); ◆ mean, bar 95% CI`}
            state={rows.length ? "ready" : "empty"}
            legend={[
              { label: "IPPO", color: METHOD_COLOR.ippo, shape: "dot" },
              { label: "MAPPO", color: METHOD_COLOR.mappo, shape: "dot" },
              { label: "Random", color: METHOD_COLOR.random, shape: "dot" },
              { label: "Heuristic (reference)", color: METHOD_COLOR.heuristic, dash: true },
            ]}
            table={{
              columns: ["Method", "mean", "CI low", "CI high", "per seed"],
              rows: rows.map((r) => [r.label, fmtNum(r.mean, def.digits), fmtNum(r.lo, def.digits), fmtNum(r.hi, def.digits), r.perSeed.map((v) => fmtNum(v, def.digits)).join(", ")]),
            }}
          >
            <DotWhisker rows={rows} reference={ref} xLabel={def.unit} format={(v) => fmtNum(v, def.digits > 2 ? 2 : def.digits)} domain={def.domain} ariaLabel={`${def.label} by method, ${split} ${mode}`} />
          </ChartFrame>
        );
      })}
    </div>
  );
}
