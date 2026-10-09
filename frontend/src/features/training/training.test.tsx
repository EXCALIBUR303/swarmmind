import { act, cleanup, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RUN_EVAL, RUN_METRICS, TRAIN_SCHEMA } from "../fixtures";
import { formatEta, nextMeasure } from "./measure";
import { TrainForm } from "./TrainForm";
import { TrainingCharts } from "./TrainingCharts";
import { useTrainingStream } from "./useTrainingStream";
import { validateTrain, type FormValues } from "./validate";
import type { Job } from "../../api/endpoints/analytics";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const base: FormValues = { level: "L1", algo: "ippo", seed: "0", total_env_steps: "20000", overrides: {} };

describe("validateTrain mirrors the backend schema (bounds come from /api/training/schema)", () => {
  it("accepts a valid request and omits blank overrides", () => {
    const { body, errors } = validateTrain({ ...base, overrides: { "ppo.lr_actor": "0.0003", "ppo.n_envs": "" } }, TRAIN_SCHEMA);
    expect(errors).toEqual({});
    expect(body).toEqual({ level: "L1", algo: "ippo", seed: 0, total_env_steps: 20000, overrides: { "ppo.lr_actor": 0.0003 } });
  });
  it("rejects exactly what the backend rejects, at every advertised bound", () => {
    for (const [key, spec] of Object.entries(TRAIN_SCHEMA.overrides)) {
      const at = (v: number) => validateTrain({ ...base, overrides: { [key]: String(v) } }, TRAIN_SCHEMA);
      expect(at(spec.min).body, `${key} min`).not.toBeNull();
      expect(at(spec.max).body, `${key} max`).not.toBeNull();
      expect(at(spec.min - (spec.type === "int" ? 1 : 1e-9)).errors[key], `${key} below`).toBeTruthy();
      expect(at(spec.max + (spec.type === "int" ? 1 : 1e-3)).errors[key], `${key} above`).toBeTruthy();
      if (spec.type === "int") expect(at(spec.min + 0.5).errors[key], `${key} fractional`).toMatch(/whole number/);
    }
  });
  it("bounds on seed, steps, level and algo; non-numeric input", () => {
    const v = (p: Partial<FormValues>) => validateTrain({ ...base, ...p }, TRAIN_SCHEMA).errors;
    expect(v({ total_env_steps: "100" }).total_env_steps).toMatch(/between 2048/);
    expect(v({ total_env_steps: "10000001" }).total_env_steps).toBeTruthy();
    expect(v({ seed: "-1" }).seed).toBeTruthy();
    expect(v({ seed: "abc" }).seed).toMatch(/whole number/);
    expect(v({ level: "L9" }).level).toBeTruthy();
    expect(v({ algo: "a2c" }).algo).toBeTruthy();
    expect(v({ overrides: { device: "cuda" } }).device).toBeTruthy();
    expect(v({ overrides: { "ppo.lr_actor": "fast" } })["ppo.lr_actor"]).toMatch(/number/);
  });
});

describe("TrainForm", () => {
  it("Enter in a field never launches; only the Start button does", () => {
    const onStart = vi.fn();
    render(<TrainForm schema={TRAIN_SCHEMA} busy={false} onStart={onStart} />);
    fireEvent.keyDown(screen.getByLabelText(/^Seed/), { key: "Enter" });
    fireEvent.submit(screen.getByRole("form", { name: "Training configuration" }));
    expect(onStart).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    expect(onStart).toHaveBeenCalledWith(expect.objectContaining({ level: "L1", algo: "ippo", seed: 0 }));
  });
  it("shows field errors with their bounds and does not call onStart", () => {
    const onStart = vi.fn();
    render(<TrainForm schema={TRAIN_SCHEMA} busy={false} onStart={onStart} />);
    fireEvent.change(screen.getByLabelText(/^Total env steps/), { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/between 2048 and 10000000/);
    expect(onStart).not.toHaveBeenCalled();
  });
  it("is disabled with a stated reason while a job runs, and shows server errors", () => {
    render(<TrainForm schema={TRAIN_SCHEMA} busy={false} disabledReason="A job is running." serverError="at most 1 training job(s) may run at once" onStart={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
    expect(screen.getByText("A job is running.")).toBeInTheDocument();
    expect(screen.getByText(/at most 1 training/)).toBeInTheDocument();
  });
});

describe("TrainingCharts renders only metrics that exist", () => {
  it("real metrics.jsonl + eval_val.jsonl: one chart per logged metric, none for absent ones", () => {
    render(<TrainingCharts metrics={RUN_METRICS} val={RUN_EVAL.validation_history} />);
    const titles = screen.getAllByRole("heading", { level: 3 }).map((h) => h.textContent);
    for (const t of ["Mean reward", "Policy loss", "Value loss", "Policy entropy", "Approximate KL", "Explained variance", "Validation success"]) expect(titles).toContain(t);
    // success/coverage/collisions per update are NOT in metrics.jsonl, so there is no such training chart
    expect(titles).not.toContain("Collisions");
    expect(titles).not.toContain("Training success");
  });
  it("drops a chart when its key is missing from every row", () => {
    const trimmed = RUN_METRICS.map(({ entropy: _e, ...rest }) => rest);
    render(<TrainingCharts metrics={trimmed} val={[]} />);
    expect(screen.queryByRole("heading", { name: "Policy entropy" })).toBeNull();
    expect(screen.queryByRole("heading", { name: "Validation success" })).toBeNull();
    expect(screen.getByRole("heading", { name: "Policy loss" })).toBeInTheDocument();
  });
  it("states the smoothing window and shows raw + mean for reward", () => {
    render(<TrainingCharts metrics={RUN_METRICS} val={[]} />);
    expect(screen.getByText(/20-update trailing mean/)).toBeInTheDocument();
  });
  it("empty state differs for a live run vs a finished one", () => {
    const { rerender } = render(<TrainingCharts metrics={[]} val={[]} live />);
    expect(screen.getByText("Waiting for the first update…")).toBeInTheDocument();
    rerender(<TrainingCharts metrics={[]} val={[]} />);
    expect(screen.getByText("No metrics were logged for this run.")).toBeInTheDocument();
  });
});

class FakeWS {
  static last: FakeWS | null = null;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeWS.last = this;
  }
  push(m: object) { this.onmessage?.({ data: JSON.stringify(m) }); }
  close() { this.closed = true; }
}

describe("useTrainingStream", () => {
  beforeEach(() => { FakeWS.last = null; });
  it("streams updates over the WebSocket, de-duplicating a replayed history", async () => {
    const { result } = renderHook(() => useTrainingStream("run1", 2000, FakeWS as unknown as typeof WebSocket));
    act(() => FakeWS.last!.onopen?.());
    act(() => {
      FakeWS.last!.push({ type: "update", update: 0, env_steps: 2048, mean_reward: 0.1 });
      FakeWS.last!.push({ type: "update", update: 0, env_steps: 2048, mean_reward: 0.1 });
      FakeWS.last!.push({ type: "update", update: 1, env_steps: 4096, mean_reward: 0.2 });
      FakeWS.last!.push({ type: "val_eval", update: 1, env_steps: 4096, success_rate: 0.5 });
      FakeWS.last!.push({ type: "status", status: "running" });
    });
    expect(result.current.mode).toBe("websocket");
    expect(result.current.metrics.map((m) => m.update)).toEqual([0, 1]);
    expect(result.current.val).toHaveLength(1);
    expect(result.current.status).toBe("running");
    expect(FakeWS.last!.url).toContain("/ws/training/run1");
  });
  it("falls back to polling when the socket closes without an end message", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn((url: string) => {
      const body = url.includes("/metrics") ? [{ update: 3, env_steps: 8192, mean_reward: 0.3 }] : url.includes("/eval") ? { validation_history: [] } : { summary: { status: "running" }, manifest: null };
      return Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));
    });
    vi.stubGlobal("fetch", fetchMock);
    const { result } = renderHook(() => useTrainingStream("run1", 500, FakeWS as unknown as typeof WebSocket));
    act(() => FakeWS.last!.onclose?.());
    await act(async () => { await vi.advanceTimersByTimeAsync(10); });
    expect(result.current.mode).toBe("polling");
    expect(result.current.metrics.map((m) => m.update)).toEqual([3]);
    await act(async () => { await vi.advanceTimersByTimeAsync(600); });
    expect(fetchMock.mock.calls.some((c) => String(c[0]).includes("from_update=3"))).toBe(true);
  });
  it("does not poll after a clean end, and closes the socket on unmount", () => {
    const { unmount } = renderHook(() => useTrainingStream("run1", 500, FakeWS as unknown as typeof WebSocket));
    const sock = FakeWS.last!;
    act(() => { sock.push({ type: "end", status: "completed" }); sock.onclose?.(); });
    unmount();
    expect(sock.closed).toBe(true);
  });
  it("an idle stream when there is no run", () => {
    const { result } = renderHook(() => useTrainingStream(null));
    expect(result.current).toEqual({ metrics: [], val: [], status: null, mode: "idle" });
  });
});

describe("measured speed and ETA", () => {
  const job = (steps: number | null): Job => ({ job_id: "j", status: "running", run_id: "r", request: { level: "L1", algo: "ippo", seed: 0, total_env_steps: 10000, overrides: {} }, created_at: "", finished_at: null, exit_code: null, run_status: "running", progress: steps === null ? null : { update: 1, env_steps: steps, total_env_steps: 10000 }, log_tail: null });
  it("reports no speed until >= 3 s of observed progress, then steps/s from real observations", () => {
    let m = nextMeasure(null, job(null), 0);
    expect(m).toBeNull();
    m = nextMeasure(m, job(1000), 0);
    expect(m?.sps).toBeNull();
    m = nextMeasure(m, job(1500), 1000);
    expect(m?.sps).toBeNull();
    m = nextMeasure(m, job(2500), 5000);
    expect(m?.sps).toBe(300);
  });
  it("restarts for a different job and never divides by zero progress", () => {
    const m = nextMeasure(null, job(1000), 0)!;
    expect(nextMeasure(m, job(1000), 10_000)?.sps).toBeNull();
    expect(nextMeasure(m, { ...job(50), job_id: "other" }, 10_000)?.jobId).toBe("other");
  });
  it("formats ETAs", () => {
    expect(formatEta(30)).toBe("30 s");
    expect(formatEta(600)).toBe("10 min");
    expect(formatEta(7200)).toBe("2.0 h");
    expect(formatEta(NaN)).toBe("–");
  });
});
