export const pct = (x: number): string => `${Math.round(x * 100)}%`;
export const kSteps = (n: number | null | undefined): string => (n == null ? "?" : n >= 1000 ? `${Math.round(n / 1000)}k` : String(n));
export const ALGO_LABEL: Record<string, string> = { mappo: "MAPPO", ippo: "IPPO", random: "Random", heuristic: "Heuristic" };
export const algoLabel = (a: string | null | undefined): string => (a ? (ALGO_LABEL[a] ?? a) : "?");
