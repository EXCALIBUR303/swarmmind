import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { apiBase, DEFAULT_API_BASE_URL } from "../api/client";
import { loadPref, prefersReducedMotion } from "../lib/prefs";
import { useSimStore } from "../stores/simStore";
import { useUiStore } from "../stores/uiStore";
import { Settings } from "./Settings";

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-motion");
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} }));
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Settings", () => {
  it("validates the API base URL against GET /api/health before saving it", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response(JSON.stringify({ status: "ok" }), { status: 200 })));
    vi.stubGlobal("fetch", fetchMock);
    render(<Settings />);
    fireEvent.change(screen.getByLabelText("API base URL"), { target: { value: "http://localhost:9000/" } });
    fireEvent.click(screen.getByRole("button", { name: "Check and save" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/Saved/));
    expect(fetchMock).toHaveBeenCalledWith("http://localhost:9000/api/health");
    expect(apiBase()).toBe("http://localhost:9000"); // trailing slash trimmed, used by every later request
  });

  it("an unreachable or non-SWARMMIND address is an inline error and is NOT saved", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    render(<Settings />);
    fireEvent.change(screen.getByLabelText("API base URL"), { target: { value: "http://nowhere:1" } });
    fireEvent.click(screen.getByRole("button", { name: "Check and save" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/was not saved/));
    expect(screen.getByLabelText("API base URL")).toHaveAttribute("aria-invalid", "true");
    expect(apiBase()).toBe(DEFAULT_API_BASE_URL);
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(JSON.stringify({ hello: "world" }), { status: 200 }))));
    fireEvent.click(screen.getByRole("button", { name: "Check and save" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/was not saved/));
  });

  it("motion override beats the OS setting and is mirrored onto <html> for CSS", () => {
    vi.stubGlobal("matchMedia", (q: string) => ({ matches: true, media: q, addEventListener() {}, removeEventListener() {} }));
    render(<Settings />);
    expect(prefersReducedMotion()).toBe(true); // system says reduce
    fireEvent.change(screen.getByLabelText("Motion"), { target: { value: "full" } });
    expect(prefersReducedMotion()).toBe(false); // user override wins
    expect(document.documentElement.hasAttribute("data-motion")).toBe(false);
    fireEvent.change(screen.getByLabelText("Motion"), { target: { value: "reduce" } });
    expect(prefersReducedMotion()).toBe(true);
    expect(document.documentElement.getAttribute("data-motion")).toBe("reduce");
  });

  it("default speed and the coordinate overlay persist; theme is shown disabled with a reason", () => {
    render(<Settings />);
    fireEvent.change(screen.getByLabelText("Default playback speed"), { target: { value: "4" } });
    expect(loadPref("defaultSpeed", 1)).toBe(4);
    expect(useSimStore.getState().speed).toBe(4);
    fireEvent.click(screen.getByLabelText(/cell coordinate/i));
    expect(useUiStore.getState().showCoords).toBe(true);
    expect(loadPref("showCoords", false)).toBe(true);
    const theme = screen.getByLabelText("Theme");
    expect(theme).toBeDisabled();
    expect(theme).toHaveAttribute("title", "Light is the only theme in v1");
  });
});
