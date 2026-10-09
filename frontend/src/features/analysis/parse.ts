/** The backend lists analysis files by stem; the naming convention is fixed by analyze_study.py. */
export interface AnalysisIndex {
  heatmaps: { algo: string; mapId: string; level: string }[];
  episodes: { algo: string; level: string; name: string }[];
}

export function parseAnalysisFiles(files: string[]): AnalysisIndex {
  const out: AnalysisIndex = { heatmaps: [], episodes: [] };
  for (const f of files) {
    const stem = f.replace(/\.json$/, "");
    let m = stem.match(/^(L\d)_(ippo|mappo)_(.+)_heatmap$/);
    if (m) out.heatmaps.push({ level: m[1], algo: m[2], mapId: m[3] });
    m = stem.match(/^(L\d)_(ippo|mappo)_example_episode$/);
    if (m) out.episodes.push({ level: m[1], algo: m[2], name: stem });
  }
  return out;
}
