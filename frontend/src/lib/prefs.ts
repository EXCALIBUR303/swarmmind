/** localStorage for per-viewer conveniences only; every access is guarded (private mode, blocked storage). */
export function loadPref<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(`swarmmind:${key}`);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

export function savePref(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(`swarmmind:${key}`, JSON.stringify(value));
  } catch {
    /* storage unavailable: preference simply isn't remembered */
  }
}

export type MotionPref = "system" | "reduce" | "full";

/** The user's override wins; "system" follows the OS `prefers-reduced-motion` (spec §30.9). */
export function prefersReducedMotion(): boolean {
  const pref = loadPref<MotionPref>("motion", "system");
  if (pref === "reduce") return true;
  if (pref === "full") return false;
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}

/** Mirror the override onto <html> so CSS animations/transitions obey it too. */
export function applyMotionPref(pref: MotionPref = loadPref<MotionPref>("motion", "system")): void {
  const el = document.documentElement;
  if (pref === "reduce") el.setAttribute("data-motion", "reduce");
  else el.removeAttribute("data-motion");
}
