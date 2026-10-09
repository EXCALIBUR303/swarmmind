import { useEffect, useRef, useState } from "react";

/** Measured CSS width of an element, so SVG charts draw at 1:1 scale (readable text) instead of shrinking a fixed viewBox. */
export function useWidth<T extends HTMLElement>(fallback: number, min = 300, max = 820): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => {
      const cw = el.getBoundingClientRect().width;
      if (cw > 0) setW(Math.round(Math.min(Math.max(cw, min), max)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [min, max]);
  return [ref, w];
}
