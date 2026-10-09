/** WCAG 2.2 relative luminance and contrast ratio. */
export function parseHex(h: string): [number, number, number] {
  const m = h.trim().replace("#", "");
  const full = m.length === 3 ? [...m].map((c) => c + c).join("") : m;
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16)) as [number, number, number];
}
export function luminance([r, g, b]: [number, number, number]): number {
  const f = (v: number) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
}
export function contrast(a: string, b: string): number {
  const [la, lb] = [luminance(parseHex(a)), luminance(parseHex(b))];
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}
