export interface SelectOption {
  value: string;
  label: string;
  group?: string;
  disabled?: boolean;
}

interface SelectProps {
  label: string; // accessible name (also shown if `showLabel`)
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  disabled?: boolean;
  className?: string;
  placeholder?: string;
}

/** Native <select> (keyboard and screen-reader support for free), with optional optgroups. */
export function Select({ label, value, options, onChange, disabled, className = "sm-lab-select", placeholder }: SelectProps) {
  const groups = new Map<string, SelectOption[]>();
  for (const o of options) groups.set(o.group ?? "", [...(groups.get(o.group ?? "") ?? []), o]);
  const hasValue = options.some((o) => o.value === value);
  return (
    <select className={className} aria-label={label} value={hasValue ? value : ""} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
      {!hasValue && <option value="" disabled>{placeholder ?? "Select…"}</option>}
      {[...groups.entries()].map(([group, opts]) =>
        group ? (
          <optgroup key={group} label={group}>
            {opts.map((o) => (
              <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>
            ))}
          </optgroup>
        ) : (
          opts.map((o) => (
            <option key={o.value} value={o.value} disabled={o.disabled}>{o.label}</option>
          ))
        ),
      )}
    </select>
  );
}
