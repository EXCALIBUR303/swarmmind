import { useState } from "react";
import type { TrainRequestBody, TrainSchema } from "../../api/endpoints/analytics";
import { fmtSteps } from "../../lib/stats";
import { OVERRIDE_LABELS, validateTrain, type FormValues } from "./validate";

const DEFAULTS: FormValues = { level: "L1", algo: "ippo", seed: "0", total_env_steps: "20000", overrides: {} };

interface Props {
  schema: TrainSchema;
  busy: boolean;
  disabledReason?: string | null;
  serverError?: string | null;
  onStart: (body: TrainRequestBody) => void;
}

/**
 * Bounds come from GET /api/training/schema (the backend's own validator constants), not from copies here.
 * Enter never launches: only the explicit Start button does (prevents accidental multi-hour jobs).
 */
export function TrainForm({ schema, busy, disabledReason, serverError, onStart }: Props) {
  const [v, setV] = useState<FormValues>(DEFAULTS);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const set = (patch: Partial<FormValues>) => setV((cur) => ({ ...cur, ...patch }));
  const start = () => {
    const { errors: e, body } = validateTrain(v, schema);
    setErrors(e);
    if (body) onStart(body);
  };
  const err = (k: string) => errors[k] && <div className="sm-field-error" role="alert">{errors[k]}</div>;
  return (
    <form className="sm-train-form" onSubmit={(e) => e.preventDefault()} aria-label="Training configuration" noValidate>
      <label className="sm-field"><span>Level</span>
        <select value={v.level} onChange={(e) => set({ level: e.target.value })}>{schema.levels.map((l) => <option key={l}>{l}</option>)}</select>{err("level")}
      </label>
      <label className="sm-field"><span>Algorithm</span>
        <select value={v.algo} onChange={(e) => set({ algo: e.target.value })}>{schema.algos.map((a) => <option key={a} value={a}>{a.toUpperCase()}</option>)}</select>{err("algo")}
      </label>
      <label className="sm-field"><span>Seed</span>
        <input inputMode="numeric" value={v.seed} onChange={(e) => set({ seed: e.target.value })} aria-invalid={!!errors.seed} />
        <small>{schema.seed.min} to {schema.seed.max}</small>{err("seed")}
      </label>
      <label className="sm-field"><span>Total env steps</span>
        <input inputMode="numeric" value={v.total_env_steps} onChange={(e) => set({ total_env_steps: e.target.value })} aria-invalid={!!errors.total_env_steps} />
        <small>{fmtSteps(schema.total_env_steps.min)} to {fmtSteps(schema.total_env_steps.max)}. Throughput on this machine is roughly 50–80 steps/s per run.</small>{err("total_env_steps")}
      </label>
      <fieldset className="sm-train-adv">
        <legend>Optional overrides (blank = the algorithm default)</legend>
        {Object.entries(schema.overrides).map(([key, spec]) => (
          <label className="sm-field" key={key}>
            <span>{OVERRIDE_LABELS[key]?.label ?? key}</span>
            <input inputMode="decimal" value={v.overrides[key] ?? ""} placeholder="default" onChange={(e) => set({ overrides: { ...v.overrides, [key]: e.target.value } })} aria-invalid={!!errors[key]} />
            <small>{OVERRIDE_LABELS[key]?.help} · {spec.min} to {spec.max}{spec.type === "int" ? " (whole number)" : ""}</small>{err(key)}
          </label>
        ))}
      </fieldset>
      {serverError && <div className="sm-field-error" role="alert">{serverError}</div>}
      <button type="button" className="sm-btn primary" onClick={start} disabled={busy || !!disabledReason} title={disabledReason ?? undefined}>Start training</button>
      {disabledReason && <small className="sm-section-note">{disabledReason}</small>}
    </form>
  );
}
