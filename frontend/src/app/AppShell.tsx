import { useEffect, useRef, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { animate, createScope, stagger, type Scope } from "animejs";
import { LayoutDashboard, LineChart, GitCompare, Activity, History, Settings as SettingsIcon } from "lucide-react";
import { api } from "../api/client";
import { prefersReducedMotion } from "../lib/prefs";
import "../design/tokens.css";
import "./AppShell.css";

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
      .then(() => !cancelled && setStatus("ok"))
      .catch(() => !cancelled && setStatus("unreachable"));
    return () => {
      cancelled = true;
    };
  }, []);
  return status;
}

export function AppShell() {
  const backend = useBackendHealth();
  const navRef = useRef<HTMLElement>(null);

  // Entrance, once on mount (anime.js): the nav settles in. Skipped under reduced motion (§30.9);
  // the scope is reverted on unmount so nothing runs twice or leaks.
  useEffect(() => {
    const nav = navRef.current;
    if (!nav) return;
    const items = nav.querySelectorAll(".sm-nav-item");
    const title = nav.querySelector(".sm-nav-title");
    if (prefersReducedMotion()) {
      animate([...items, ...(title ? [title] : [])], { opacity: 1, duration: 0 });
      return;
    }
    const scope: Scope = createScope({ root: navRef }).add(() => {
      animate(".sm-nav-title", { opacity: [0, 1], translateY: [-6, 0], duration: 500, easing: "easeOutQuad" });
      animate(items, { opacity: [0, 1], translateX: [-16, 0], duration: 420, delay: stagger(60, { start: 150 }), easing: "easeOutCubic" });
    });
    return () => scope.revert();
  }, []);

  const dot = backend === "ok" ? "var(--sm-success)" : backend === "unreachable" ? "var(--sm-danger)" : "var(--sm-text-muted)";
  const text = backend === "ok" ? "backend: ok" : backend === "unreachable" ? "backend: unreachable" : "backend: checking…";
  return (
    <div className="sm-shell">
      <nav className="sm-nav" aria-label="Primary" ref={navRef}>
        <div className="sm-nav-title" aria-label="SWARMMIND">
          <span className="full">SWARMMIND</span>
          <span className="short" aria-hidden="true">SM</span>
        </div>
        <div className="sm-nav-links">
          {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => `sm-nav-item${isActive ? " active" : ""}`} title={label} aria-label={label}>
              <Icon size={20} aria-hidden="true" />
              <span className="label">{label}</span>
            </NavLink>
          ))}
        </div>
        <div className="sm-nav-status" role="status" title={text}>
          <i style={{ background: dot }} aria-hidden="true" />
          <span className="label">{text}</span>
        </div>
      </nav>
      <main className="sm-shell-main">
        <Outlet />
      </main>
    </div>
  );
}
