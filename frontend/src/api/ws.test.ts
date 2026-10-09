import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SimSocket, type Connection, type ServerMessage } from "./ws";

class FakeWS {
  static instances: FakeWS[] = [];
  readyState = 0;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: ((e: { code: number; reason: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  url: string;
  constructor(url: string) {
    this.url = url;
    FakeWS.instances.push(this);
  }
  send(d: string) {
    this.sent.push(d);
  }
  close(code = 1000) {
    this.readyState = 3;
    this.onclose?.({ code, reason: "" });
  }
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  serverClose(code: number) {
    this.readyState = 3;
    this.onclose?.({ code, reason: "" });
  }
}

function make(opts: Partial<ConstructorParameters<typeof SimSocket>[0]> = {}) {
  const states: Connection[] = [];
  const msgs: ServerMessage[] = [];
  const sock = new SimSocket({
    url: "ws://x/ws/sessions/a",
    onMessage: (m) => msgs.push(m),
    onConnection: (s) => states.push(s),
    WebSocketImpl: FakeWS as unknown as typeof WebSocket,
    ...opts,
  });
  return { sock, states, msgs };
}

beforeEach(() => {
  FakeWS.instances = [];
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

describe("SimSocket", () => {
  it("connects, parses messages, sends commands only when open", () => {
    const { sock, states, msgs } = make();
    sock.connect();
    expect(states).toEqual(["connecting"]);
    expect(sock.send({ cmd: "play" })).toBe(false);
    FakeWS.instances[0].open();
    expect(states.at(-1)).toBe("open");
    expect(sock.send({ cmd: "step", value: 2 })).toBe(true);
    expect(JSON.parse(FakeWS.instances[0].sent[0])).toEqual({ cmd: "step", value: 2 });
    FakeWS.instances[0].onmessage?.({ data: JSON.stringify({ type: "error", code: "x", message: "m" }) });
    FakeWS.instances[0].onmessage?.({ data: "{broken" }); // ignored, stream continues
    expect(msgs).toHaveLength(1);
  });

  it("reconnects with exponential backoff after an abnormal close, and resets the counter once open", () => {
    const { sock, states } = make({ baseDelayMs: 100, maxDelayMs: 1000 });
    sock.connect();
    FakeWS.instances[0].open();
    FakeWS.instances[0].serverClose(1006);
    expect(states.at(-1)).toBe("reconnecting");
    vi.advanceTimersByTime(99);
    expect(FakeWS.instances).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(FakeWS.instances).toHaveLength(2);
    FakeWS.instances[1].serverClose(1006); // never opened: attempt 2 -> 200 ms
    vi.advanceTimersByTime(199);
    expect(FakeWS.instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeWS.instances).toHaveLength(3);
    FakeWS.instances[2].open(); // success resets the backoff
    FakeWS.instances[2].serverClose(1006);
    vi.advanceTimersByTime(100);
    expect(FakeWS.instances).toHaveLength(4);
  });

  it.each([4403, 4404, 4409])("does not retry after close code %i", (code) => {
    const { sock, states } = make();
    sock.connect();
    FakeWS.instances[0].open();
    FakeWS.instances[0].serverClose(code);
    expect(states.at(-1)).toBe("closed");
    vi.advanceTimersByTime(60_000);
    expect(FakeWS.instances).toHaveLength(1);
  });

  it("gives up after maxAttempts", () => {
    const { sock, states } = make({ maxAttempts: 2, baseDelayMs: 10 });
    sock.connect();
    for (let i = 0; i < 5; i++) {
      FakeWS.instances.at(-1)!.serverClose(1006);
      vi.advanceTimersByTime(1000);
    }
    expect(FakeWS.instances).toHaveLength(3); // initial + 2 retries
    expect(states.at(-1)).toBe("closed");
  });

  it("close() is final: no reconnect, pending timers cancelled", () => {
    const { sock, states } = make({ baseDelayMs: 50 });
    sock.connect();
    FakeWS.instances[0].open();
    FakeWS.instances[0].serverClose(1006);
    sock.close();
    vi.advanceTimersByTime(10_000);
    expect(FakeWS.instances).toHaveLength(1);
    expect(states.at(-1)).toBe("closed");
    expect(sock.send({ cmd: "play" })).toBe(false);
  });
});
