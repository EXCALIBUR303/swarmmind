const CLASS: Record<string, string> = { completed: "ok", running: "warn", failed: "bad", invalid: "bad" };

/** Status = icon + text + colour, never colour alone. */
export function StatusBadge({ status }: { status: string }) {
  const icon = status === "completed" ? "✓" : status === "failed" || status === "invalid" ? "✗" : "•";
  return <span className={`sm-badge ${CLASS[status] ?? "warn"}`}><span aria-hidden="true">{icon}</span>{status}</span>;
}
