import { useCallback, useEffect, useRef, useState } from "react";
import { runsApi, type MetricRow } from "../../api/endpoints/analytics";
import { wsUrl } from "../../api/client";

export type StreamMode = "idle" | "websocket" | "polling";
export interface Stream {
  metrics: MetricRow[];
  val: MetricRow[];
  status: string | null;
  mode: StreamMode;
}

/**
 * Live training metrics for one run: WebSocket first (`/ws/training/{run_id}`), polling
 * `GET /experiments/{id}/metrics?from_update=` if the socket cannot be used. Rows are keyed by
 * `update`, so a reconnect that replays history never duplicates points.
 */
export function useTrainingStream(runId: string | null, pollMs = 2000, WS: typeof WebSocket | undefined = typeof WebSocket === "undefined" ? undefined : WebSocket): Stream {
  const [store, setStore] = useState<{ runId: string | null; s: Stream }>({ runId: null, s: { metrics: [], val: [], status: null, mode: "idle" } });
  // State is keyed by run: switching runs shows an empty stream immediately, with no reset-in-effect.
  const state = store.runId === runId ? store.s : { metrics: [], val: [], status: null, mode: "idle" as StreamMode };
  const setState = useCallback((f: (s: Stream) => Stream) => setStore((cur) => ({ runId, s: f(cur.runId === runId ? cur.s : { metrics: [], val: [], status: null, mode: "idle" }) })), [runId]);
  const seen = useRef<{ run: string | null; last: number }>({ run: null, last: -1 });

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setInterval> | null = null;
    seen.current = { run: runId, last: -1 };

    const merge = (kind: "metrics" | "val", rows: MetricRow[]) =>
      setState((s) => {
        const have = new Set(s[kind].map((r) => r.update));
        const fresh = rows.filter((r) => !have.has(r.update));
        return fresh.length ? { ...s, [kind]: [...s[kind], ...fresh] } : s;
      });

    const startPolling = () => {
      if (timer || cancelled) return;
      const tick = async () => {
        try {
          setState((s) => (s.mode === "polling" ? s : { ...s, mode: "polling" }));
          const rows = await runsApi.metrics(runId, Math.max(seen.current.last, 0));
          if (cancelled) return;
          if (rows.length) seen.current.last = Math.max(seen.current.last, ...rows.map((r) => r.update));
          merge("metrics", rows);
          const ev = await runsApi.valEval(runId);
          if (!cancelled) merge("val", ev.validation_history ?? []);
          const detail = await runsApi.get(runId);
          if (!cancelled) setState((s) => ({ ...s, status: detail.summary.status }));
        } catch {
          /* the next tick retries */
        }
      };
      void tick();
      timer = setInterval(tick, pollMs);
    };

    if (!WS) startPolling();
    else {
      try {
        socket = new WS(wsUrl(`/ws/training/${encodeURIComponent(runId)}`));
        socket.onopen = () => !cancelled && setState((s) => ({ ...s, mode: "websocket" }));
        let ended = false;
        socket.onmessage = (ev: MessageEvent) => {
          try {
            const m = JSON.parse(String(ev.data)) as { type: string } & MetricRow & { status?: string };
            if (m.type === "update") {
              seen.current.last = Math.max(seen.current.last, m.update);
              merge("metrics", [m]);
            } else if (m.type === "val_eval") merge("val", [m]);
            else if (m.type === "status") setState((s) => ({ ...s, status: m.status as unknown as string }));
            else if (m.type === "end") ended = true;
          } catch {
            /* ignore a malformed message */
          }
        };
        socket.onclose = () => {
          if (!cancelled && !ended) startPolling();
        };
        socket.onerror = () => {
          /* onclose follows */
        };
      } catch {
        startPolling();
      }
    }
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
      socket?.close();
    };
  }, [runId, pollMs, WS, setState]);

  return runId ? state : { metrics: [], val: [], status: null, mode: "idle" };
}
