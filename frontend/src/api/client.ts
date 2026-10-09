/**
 * Fetch wrapper (§29.2). All network access lives under `api/`.
 * Base URL defaults to the local backend; override with VITE_API_BASE_URL.
 */

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function wsUrl(path: string): string {
  return API_BASE_URL.replace(/^http/, "ws") + path;
}

async function errorMessage(res: Response, path: string): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail) && body.detail.length > 0) {
      const first = body.detail[0] as { msg?: string; loc?: unknown[] };
      return `${first.msg ?? "invalid request"}${first.loc ? ` (${first.loc.slice(1).join(".")})` : ""}`;
    }
  } catch {
    /* non-JSON error body */
  }
  return `request to ${path} failed (${res.status})`;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (err) {
    console.error("backend unreachable", err);
    throw new ApiError(`cannot reach the backend at ${API_BASE_URL}`, 0);
  }
  if (!res.ok) throw new ApiError(await errorMessage(res, path), res.status);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function postJson<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export interface HealthResponse {
  status: string;
  versions: Record<string, unknown>;
  device: string;
  counts: { runs: number; models: number; replays?: number; live_sessions?: number };
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
};
