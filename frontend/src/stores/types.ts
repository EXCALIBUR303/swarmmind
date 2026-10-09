import type { ReplayOutcome } from "../api/wire.gen";

export interface ReplayInfoLite {
  id: string;
  outcome: ReplayOutcome;
}
