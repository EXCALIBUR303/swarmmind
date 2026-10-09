interface Props {
  title: string;
  detail?: string;
  action?: { label: string; onClick: () => void };
}

export function LoadingState({ title = "Loading…" }: { title?: string }) {
  return (
    <div className="sm-state" role="status" aria-live="polite">
      <div className="sm-state-title">{title}</div>
    </div>
  );
}

export function ErrorState({ title, detail, action }: Props) {
  return (
    <div className="sm-state error" role="alert">
      <div className="sm-state-title">{title}</div>
      {detail && <div className="sm-state-detail">{detail}</div>}
      {action && (
        <button className="sm-lab-btn" onClick={action.onClick}>
          {action.label}
        </button>
      )}
    </div>
  );
}
