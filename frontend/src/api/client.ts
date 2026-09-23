/**
 * Minimal fetch wrapper (§29.2). Grows in Phase 10 once real endpoints
 * exist; for now it only exposes the health check used by the app shell.
 */

const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    throw new ApiError(`request to ${path} failed: ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

export interface HealthResponse {
  status: string;
  versions: Record<string, unknown>;
  device: string;
  counts: { runs: number; models: number };
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
};
