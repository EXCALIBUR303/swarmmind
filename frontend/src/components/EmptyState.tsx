interface EmptyStateProps {
  title: string;
  detail?: string;
}

/** Shared empty-state used by placeholder pages until their real data
 * sources exist (Phases 11-12). See spec §30.8 (every view needs an
 * explicit empty state, never a blank screen). */
export function EmptyState({ title, detail }: EmptyStateProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        minHeight: 240,
        color: "var(--sm-text-muted)",
        gap: "var(--sm-space-3)",
      }}
    >
      <div style={{ font: "var(--sm-font-lg)", color: "var(--sm-text)" }}>{title}</div>
      {detail && <div style={{ font: "var(--sm-font-sm)", maxWidth: 420, textAlign: "center" }}>{detail}</div>}
    </div>
  );
}
