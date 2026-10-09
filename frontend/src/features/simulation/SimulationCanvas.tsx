import { useCallback, useEffect, useMemo, useRef } from "react";
import { cellAt } from "./viewport";
import { prefersReducedMotion } from "../../lib/prefs";
import { useSimStore, BASE_STEPS_PER_SECOND, MAX_STEPS_PER_SECOND } from "../../stores/simStore";
import { useUiStore } from "../../stores/uiStore";
import { MOTION, moveDurationMs } from "../../design/motion";
import { recordAnimatedFrame } from "./perf";
import { RENDER_ORDER } from "./renderers";
import { readTokens, type RenderTokens } from "./tokens";
import { buildRenderView } from "./view";
import { backingSize, computeViewport, robotAt, type Viewport } from "./viewport";

/**
 * The canvas lives OUTSIDE React state: one rAF loop reads the stores with
 * `getState()` and draws, so a 60 fps animation never causes a React render.
 * It draws only when something changed or is animating.
 */
export function SimulationCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const vpRef = useRef<Viewport | null>(null);
  const tokensRef = useRef<RenderTokens | null>(null);
  const lastDrawPos = useRef<[number, number][]>([]);
  const dirty = useRef(true);
  const coordRef = useRef<HTMLDivElement>(null);
  const showCoords = useUiStore((s) => s.showCoords);

  const meta = useSimStore((s) => s.meta);
  const nRobots = meta?.nRobots ?? 0;
  const label = useMemo(
    () => (meta ? `Simulation grid ${meta.width} by ${meta.height}, ${nRobots} robots, map ${meta.mapId}` : "Simulation grid"),
    [meta, nRobots],
  );

  // Resize: CSS size from the wrapper, backing store scaled by devicePixelRatio (no blur).
  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas || !meta) return;
    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      const vp = computeViewport(Math.max(1, rect.width), Math.max(1, rect.height), window.devicePixelRatio || 1, meta.width, meta.height);
      const { width, height } = backingSize(vp);
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      canvas.style.width = `${vp.cssWidth}px`;
      canvas.style.height = `${vp.cssHeight}px`;
      vpRef.current = vp;
      dirty.current = true;
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);
    window.addEventListener("resize", resize); // dpr changes (zoom / display move)
    return () => {
      ro.disconnect();
      window.removeEventListener("resize", resize);
    };
  }, [meta]);

  // The draw loop.
  useEffect(() => {
    if (!meta) return;
    tokensRef.current = readTokens();
    const reduced = prefersReducedMotion();
    let raf = 0;
    let prevNow = performance.now();
    let lastSig = "";
    let shownT = -1;
    let shownTl: unknown = null;
    let changedAt = performance.now();
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const dt = Math.min((now - prevNow) / 1000, 0.1);
      prevNow = now;
      const sim = useSimStore.getState();
      const ui = useUiStore.getState();
      const canvas = canvasRef.current;
      const vp = vpRef.current;
      const tl = sim.timeline;
      if (!canvas || !vp || !tl || !tokensRef.current) return;

      sim.advance(dt);
      const live = sim.mode === "live";
      const rate = Math.min(MAX_STEPS_PER_SECOND, BASE_STEPS_PER_SECOND * sim.speed);
      // One clock for every effect: ms since the displayed step became current (spec §30.7 durations).
      const curT = live ? tl.maxT : Math.min(Math.max(Math.floor(sim.playhead), tl.startT), tl.maxT);
      if (curT !== shownT || tl !== shownTl) {
        changedAt = now;
        shownT = curT;
        shownTl = tl;
      }
      const rawSince = now - changedAt;
      const sinceMs = reduced ? Infinity : rawSince;
      const animating = !reduced && rawSince < MOTION.rescueMs; // the longest effect
      const moveMs = moveDurationMs(1000 / rate);
      const alpha = animating ? Math.min(rawSince / moveMs, 1) : 1;
      const sig = `${sim.version}|${curT}|${animating ? Math.round(rawSince / 16) : "s"}|${JSON.stringify(ui.layers)}|${sim.focusedRobot}|${vp.cssWidth}x${vp.cssHeight}`;
      if (!dirty.current && sig === lastSig && !animating) return;
      lastSig = sig;
      dirty.current = false;

      const t0 = performance.now();
      const view = buildRenderView({ timeline: tl, playhead: sim.playhead, live, alpha, sinceMs: animating ? sinceMs : Infinity, layers: ui.layers, focused: sim.focusedRobot, reducedMotion: reduced });
      lastDrawPos.current = view.drawPos;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(vp.dpr, 0, 0, vp.dpr, 0, 0);
      for (const draw of RENDER_ORDER) draw(ctx, view, vp, tokensRef.current);
      if (animating) recordAnimatedFrame(now, performance.now() - t0);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [meta]);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const vp = vpRef.current;
    if (!vp) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const hit = robotAt(vp, lastDrawPos.current, e.clientX - rect.left, e.clientY - rect.top);
    const st = useSimStore.getState();
    if (hit !== null) st.toggleFocus(hit);
    else if (st.focusedRobot !== null) st.focus(null);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLCanvasElement>) => {
    const vp = vpRef.current;
    const el = coordRef.current;
    if (!vp || !el) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const c = cellAt(vp, e.clientX - rect.left, e.clientY - rect.top);
    el.textContent = c ? `row ${c[0]}, col ${c[1]}` : "";
  }, []);

  return (
    <div ref={wrapRef} className="sm-canvas-wrap">
      <canvas ref={canvasRef} role="img" aria-label={label} onPointerDown={onPointerDown} onPointerMove={showCoords ? onPointerMove : undefined} onPointerLeave={() => { if (coordRef.current) coordRef.current.textContent = ""; }} className="sm-canvas" />
      {showCoords && <div ref={coordRef} className="sm-coord" aria-hidden="true" />}
    </div>
  );
}
