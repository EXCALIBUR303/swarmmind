import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "bootstrap/dist/css/bootstrap-reboot.min.css";
import "./index.css";
import { App } from "./app/App";
import { applyMotionPref } from "./lib/prefs";

applyMotionPref();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
