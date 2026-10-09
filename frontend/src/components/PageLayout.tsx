import type { ReactNode } from "react";
import "./PageLayout.css";

export function Page({ title, subtitle, filters, children }: { title: string; subtitle?: string; filters?: ReactNode; children: ReactNode }) {
  return (
    <div className="sm-page">
      <header className="sm-page-head">
        <h1>{title}</h1>
        {subtitle && <p>{subtitle}</p>}
      </header>
      {filters && <div className="sm-filters" role="group" aria-label="Filters">{filters}</div>}
      <div className="sm-page-body">{children}</div>
    </div>
  );
}

export function Section({ title, note, children }: { title: string; note?: ReactNode; children: ReactNode }) {
  return (
    <section className="sm-section">
      <h2>{title}</h2>
      {note && <p className="sm-section-note">{note}</p>}
      {children}
    </section>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="sm-field">
      <span>{label}</span>
      {children}
    </label>
  );
}

/** A labelled group of buttons (segmented control). Not a <label>: a label would lend its whole text to the first button's accessible name. */
export function FieldGroup({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="sm-field" role="group" aria-label={label}>
      <span aria-hidden="true">{label}</span>
      {children}
    </div>
  );
}
