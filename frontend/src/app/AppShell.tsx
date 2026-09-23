import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { LayoutDashboard, LineChart, GitCompare, Activity, History, Settings as SettingsIcon } from "lucide-react";
import { api, ApiError } from "../api/client";

const NAV_ITEMS = [
  { to: "/", label: "Swarm Lab", icon: LayoutDashboard, end: true },
  { to: "/train", label: "Train Swarm", icon: LineChart },
  { to: "/compare", label: "Algorithm Comparison", icon: GitCompare },
  { to: "/analysis", label: "Swarm Analysis", icon: Activity },
  { to: "/experiments", label: "Experiment History", icon: History },
  { to: "/settings", label: "Settings", icon: SettingsIcon },
];

type BackendStatus = "checking" | "ok" | "unreachable";

function useBackendHealth(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>("checking");

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then(() => {
        if (!cancelled) setStatus("ok");
      })
      .catch((err: unknown) => {
        if (!cancelled) setStatus("unreachable");
        if (!(err instanceof ApiError)) {
          // eslint-disable-next-line no-console
          console.error("health check failed", err);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return status;
}

function StatusDot({ status }: { status: BackendStatus }) {
  const color =
    status === "ok" ? "var(--sm-success)" : status === "unreachable" ? "var(--sm-danger)" : "var(--sm-text-muted)";
  const label = status === "ok" ? "backend: ok" : status === "unreachable" ? "backend: unreachable" : "backend: checking…";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "var(--sm-space-2)", font: "var(--sm-font-xs)", color: "var(--sm-text-on-dark)" }}>
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block" }} />
      {label}
    </div>
  );
}

export function AppShell() {
  const backendStatus = useBackendHealth();

  return (
    <div style={{ display: "flex", height: "100vh", width: "100vw" }}>
      <nav
        style={{
          width: "var(--sm-nav-width-expanded)",
          background: "var(--sm-bg-nav)",
          color: "var(--sm-text-on-dark)",
          display: "flex",
          flexDirection: "column",
          flexShrink: 0,
        }}
      >
        <div style={{ padding: "var(--sm-space-5)", font: "var(--sm-font-lg)" }}>SWARMMIND</div>
        <div style={{ display: "flex", flexDirection: "column", gap: "var(--sm-space-1)", padding: "var(--sm-space-3)", flex: 1 }}>
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              style={({ isActive }) => ({
                display: "flex",
                alignItems: "center",
                gap: "var(--sm-space-3)",
                padding: "var(--sm-space-3) var(--sm-space-4)",
                borderRadius: "var(--sm-radius-control)",
                color: isActive ? "#fff" : "var(--sm-text-on-dark)",
                background: isActive ? "rgba(255,255,255,0.08)" : "transparent",
                textDecoration: "none",
                font: "var(--sm-font-md)",
              })}
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </div>
        <div style={{ padding: "var(--sm-space-5)" }}>
          <StatusDot status={backendStatus} />
        </div>
      </nav>
      <main style={{ flex: 1, overflow: "auto", background: "var(--sm-bg-app)" }}>
        <Outlet />
      </main>
    </div>
  );
}
