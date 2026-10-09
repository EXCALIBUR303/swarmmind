import type { SessionCreate, SessionReset, SessionSnapshot } from "../types.gen";
import { postJson, request } from "../client";

export const sessionsApi = {
  create: (body: SessionCreate) => postJson<SessionSnapshot>("/api/sessions", body),
  get: (id: string) => request<SessionSnapshot>(`/api/sessions/${encodeURIComponent(id)}`),
  reset: (id: string, body?: SessionReset) => postJson<SessionSnapshot>(`/api/sessions/${encodeURIComponent(id)}/reset`, body),
  close: (id: string) => request<void>(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),
};
