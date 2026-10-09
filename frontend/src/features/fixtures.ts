/** Real API payloads captured from the running backend (study main_L2, run s0 MAPPO). See __fixtures__/. */
import study from "./__fixtures__/study.json";
import studies from "./__fixtures__/studies.json";
import prereg from "./__fixtures__/prereg.json";
import epMappo from "./__fixtures__/analysis_L2_mappo_example_episode.json";
import epIppo from "./__fixtures__/analysis_L2_ippo_example_episode.json";
import hmMappo from "./__fixtures__/analysis_L2_mappo_L2-struct-bottleneck_vertical_heatmap.json";
import hmIppo from "./__fixtures__/analysis_L2_ippo_L2-struct-bottleneck_vertical_heatmap.json";
import runs from "./__fixtures__/runs.json";
import runDetail from "./__fixtures__/run_detail.json";
import runMetrics from "./__fixtures__/run_metrics.json";
import runEval from "./__fixtures__/run_eval.json";
import trainSchema from "./__fixtures__/train_schema.json";
import replaysIndex from "./__fixtures__/replays_index.json";
import type { ExampleEpisode, MetricRow, RunDetail, RunSummary, StudyDetail, StudyListItem, TrainSchema } from "../api/endpoints/analytics";
import type { ReplayIndexEntry } from "../api/endpoints/catalog";

export const STUDY = study as unknown as StudyDetail;
export const STUDIES = studies as unknown as StudyListItem[];
export const PREREG = prereg as { source: string; rules: { id: string; text: string }[] };
export const EP_MAPPO = epMappo as unknown as ExampleEpisode;
export const EP_IPPO = epIppo as unknown as ExampleEpisode;
export const HM = { mappo: { ...hmMappo, study: "main_L2", level: "L2", algo: "mappo", kind: "visits", counts: (hmMappo as { visit_counts: number[][] }).visit_counts }, ippo: { ...hmIppo, study: "main_L2", level: "L2", algo: "ippo", kind: "visits", counts: (hmIppo as { visit_counts: number[][] }).visit_counts } };
export const RUNS = runs as unknown as RunSummary[];
export const RUN_DETAIL = runDetail as unknown as RunDetail;
export const RUN_METRICS = runMetrics as unknown as MetricRow[];
export const RUN_EVAL = runEval as unknown as { validation_history: MetricRow[] };
export const TRAIN_SCHEMA = trainSchema as unknown as TrainSchema;
export const REPLAYS_INDEX = replaysIndex as unknown as ReplayIndexEntry[];
