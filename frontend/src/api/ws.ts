/**
 * Simulation WebSocket client with reconnect (§28.3, Phase 11).
 * On every (re)connect the server sends a `state` message, which is how a
 * reconnecting client resynchronises; this class only manages the transport.
 */
import type { WsCommand, WsEpisodeEnd, WsError, WsFrameMessage, WsStateMessage } from "./wire.gen";

export type ServerMessage = WsStateMessage | WsFrameMessage | WsEpisodeEnd | WsError;
export type Connection = "connecting" | "open" | "reconnecting" | "closed";

/** Close codes after which retrying is pointless (server told us why). */
const FATAL_CODES = new Set([4403, 4404, 4409]);

export interface SimSocketOptions {
  url: string;
  onMessage: (msg: ServerMessage) => void;
  onConnection: (state: Connection, info?: { code?: number; reason?: string }) => void;
  WebSocketImpl?: typeof WebSocket;
  maxAttempts?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
}

export class SimSocket {
  private ws: WebSocket | null = null;
  private attempts = 0;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private closedByUs = false;
  private readonly o: Required<Omit<SimSocketOptions, "WebSocketImpl">> & { WebSocketImpl: typeof WebSocket };

  constructor(opts: SimSocketOptions) {
    this.o = {
      maxAttempts: 8,
      baseDelayMs: 500,
      maxDelayMs: 8000,
      WebSocketImpl: opts.WebSocketImpl ?? WebSocket,
      ...opts,
    } as SimSocket["o"];
  }

  connect(): void {
    this.closedByUs = false;
    this.o.onConnection(this.attempts === 0 ? "connecting" : "reconnecting");
    const ws = new this.o.WebSocketImpl(this.o.url);
    this.ws = ws;
    ws.onopen = () => {
      this.attempts = 0;
      this.o.onConnection("open");
    };
    ws.onmessage = (ev: MessageEvent) => {
      try {
        this.o.onMessage(JSON.parse(String(ev.data)) as ServerMessage);
      } catch {
        /* ignore a malformed message; the stream continues */
      }
    };
    ws.onclose = (ev: CloseEvent) => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.closedByUs || FATAL_CODES.has(ev.code) || this.attempts >= this.o.maxAttempts) {
        this.o.onConnection("closed", { code: ev.code, reason: ev.reason });
        return;
      }
      const delay = Math.min(this.o.baseDelayMs * 2 ** this.attempts, this.o.maxDelayMs);
      this.attempts++;
      this.o.onConnection("reconnecting", { code: ev.code });
      this.timer = setTimeout(() => this.connect(), delay);
    };
    ws.onerror = () => {
      /* onclose follows and carries the code */
    };
  }

  send(cmd: WsCommand): boolean {
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(JSON.stringify(cmd));
    return true;
  }

  close(): void {
    this.closedByUs = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    const ws = this.ws;
    this.ws = null;
    ws?.close(1000);
    this.o.onConnection("closed", { code: 1000 });
  }
}
