import { EmptyState } from "../components/EmptyState";

/** App preferences (spec §30.6). Implemented in Phase 12/13. */
export function Settings() {
  return (
    <EmptyState
      title="Settings"
      detail="API base URL, playback defaults and accessibility preferences arrive alongside the analytics UI."
    />
  );
}
