import { useState } from "react";
import { DEFAULT_API_BASE_URL, apiBase } from "../api/client";
import { Field, Page, Section } from "../components/PageLayout";
import { applyMotionPref, loadPref, savePref, type MotionPref } from "../lib/prefs";
import { SPEEDS, useSimStore } from "../stores/simStore";
import { useUiStore } from "../stores/uiStore";
import "./pages.css";

/** Local preferences only (spec §30.6): nothing here is stored on the server. */
export function Settings() {
  const [url, setUrl] = useState(apiBase());
  const [urlStatus, setUrlStatus] = useState<{ kind: "idle" | "checking" | "ok" | "error"; message?: string }>({ kind: "idle" });
  const [speed, setSpeed] = useState(loadPref<number>("defaultSpeed", 1));
  const [motion, setMotion] = useState<MotionPref>(loadPref<MotionPref>("motion", "system"));
  const showCoords = useUiStore((s) => s.showCoords);
  const setShowCoords = useUiStore((s) => s.setShowCoords);

  const saveUrl = async () => {
    const candidate = url.trim().replace(/\/+$/, "");
    setUrlStatus({ kind: "checking" });
    try {
      const res = await fetch(`${candidate}/api/health`);
      const body = (await res.json()) as { status?: string };
      if (!res.ok || body.status !== "ok") throw new Error("not a SWARMMIND backend");
      savePref("apiBase", candidate);
      setUrlStatus({ kind: "ok", message: "Saved. Reload the page to use it everywhere." });
    } catch {
      setUrlStatus({ kind: "error", message: `Could not reach a SWARMMIND backend at ${candidate}. The setting was not saved.` });
    }
  };

  return (
    <Page title="Settings" subtitle="Preferences are stored in this browser only; nothing is sent to the server.">
      <Section title="Backend">
        <div className="sm-settings-row">
          <Field label="API base URL">
            <input value={url} onChange={(e) => { setUrl(e.target.value); setUrlStatus({ kind: "idle" }); }} aria-invalid={urlStatus.kind === "error"} aria-describedby="api-url-status" inputMode="url" />
          </Field>
          <button className="sm-btn primary" onClick={() => void saveUrl()} disabled={urlStatus.kind === "checking"}>Check and save</button>
          <button className="sm-btn" onClick={() => { setUrl(DEFAULT_API_BASE_URL); setUrlStatus({ kind: "idle" }); savePref("apiBase", DEFAULT_API_BASE_URL); }}>Use default</button>
        </div>
        <div id="api-url-status" role="status" className={urlStatus.kind === "error" ? "sm-field-error" : "sm-section-note"}>
          {urlStatus.kind === "checking" ? "Checking…" : (urlStatus.message ?? `Default: ${DEFAULT_API_BASE_URL}. The address is checked with GET /api/health before it is saved.`)}
        </div>
      </Section>
      <Section title="Swarm Lab">
        <div className="sm-settings-row">
          <Field label="Default playback speed">
            <select value={String(speed)} onChange={(e) => { const v = Number(e.target.value); setSpeed(v); savePref("defaultSpeed", v); useSimStore.setState({ speed: v }); }}>
              {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
            </select>
          </Field>
          <label className="sm-field row">
            <input type="checkbox" checked={showCoords} onChange={(e) => setShowCoords(e.target.checked)} />
            <span>Show the cell coordinate under the pointer</span>
          </label>
        </div>
      </Section>
      <Section title="Accessibility">
        <div className="sm-settings-row">
          <Field label="Motion">
            <select value={motion} onChange={(e) => { const v = e.target.value as MotionPref; setMotion(v); savePref("motion", v); applyMotionPref(v); }}>
              <option value="system">Follow the system setting</option>
              <option value="reduce">Always reduce motion</option>
              <option value="full">Always allow motion</option>
            </select>
          </Field>
          <Field label="Theme">
            <select value="light" disabled title="Light is the only theme in v1" aria-describedby="theme-note"><option value="light">Light</option></select>
          </Field>
        </div>
        <div id="theme-note" className="sm-section-note">Dark mode is not part of v1; the control is shown so it is discoverable.</div>
      </Section>
    </Page>
  );
}
