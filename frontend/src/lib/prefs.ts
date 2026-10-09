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

export function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return false;
  }
}
