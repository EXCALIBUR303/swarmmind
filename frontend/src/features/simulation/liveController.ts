/** Owns the live session lifecycle: REST create -> WebSocket -> store. The server owns the clock. */
import { ApiError, wsUrl } from "../../api/client";
import { sessionsApi } from "../../api/endpoints/sessions";
import type { SessionCreate } from "../../api/types.gen";
import { SimSocket, type Connection, type ServerMessage } from "../../api/ws";
import { useSimStore } from "../../stores/simStore";

let socket: SimSocket | null = null;
let sessionId: string | null = null;

const NOTICES: Record<number, string> = {
  4404: "The session expired or was closed. Start a new one.",
  4409: "Another window took over this session.",
  4403: "The backend refused this page's origin.",
};

function route(msg: ServerMessage): void {
  const s = useSimStore.getState();
  switch (msg.type) {
    case "state":
      s.applyLiveState(msg);
      break;
    case "frame":
      s.pushLiveFrame(msg);
      break;
    case "episode_end":
      s.endLive(msg);
      break;
    case "error":
      s.setNotice(msg.message);
      break;
  }
}

function onConnection(state: Connection, info?: { code?: number }): void {
  const notice = state === "closed" && info?.code ? (NOTICES[info.code] ?? null) : null;
  useSimStore.getState().setConnection(state, notice);
}

export async function stopLive(): Promise<void> {
  const id = sessionId;
  socket?.close();
  socket = null;
  sessionId = null;
  useSimStore.getState().setCommandSink(null);
  if (id) await sessionsApi.close(id).catch(() => undefined); // best effort: the server's idle TTL reaps it anyway
}

export async function startLive(req: SessionCreate): Promise<void> {
  const store = useSimStore.getState();
  await stopLive();
  store.setLoading();
  try {
    const snap = await sessionsApi.create(req);
    sessionId = snap.session_id;
    useSimStore.getState().startLive(snap);
    socket = new SimSocket({ url: wsUrl(`/ws/sessions/${snap.session_id}`), onMessage: route, onConnection });
    useSimStore.getState().setCommandSink((c) => socket?.send(c as never) ?? false);
    socket.connect();
  } catch (err) {
    useSimStore.getState().setError(err instanceof ApiError ? err.message : "could not start a session");
  }
}
