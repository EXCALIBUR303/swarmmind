import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "./AppShell";
import { SwarmLab } from "../pages/SwarmLab";
import { TrainSwarm } from "../pages/TrainSwarm";
import { AlgorithmComparison } from "../pages/AlgorithmComparison";
import { SwarmAnalysis } from "../pages/SwarmAnalysis";
import { ExperimentHistory } from "../pages/ExperimentHistory";
import { Settings } from "../pages/Settings";
import { SwarmLabPrototype } from "../pages/_prototype/SwarmLabPrototype";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <SwarmLab /> },
      { path: "train", element: <TrainSwarm /> },
      { path: "compare", element: <AlgorithmComparison /> },
      { path: "analysis", element: <SwarmAnalysis /> },
      { path: "experiments", element: <ExperimentHistory /> },
      { path: "settings", element: <Settings /> },
    ],
  },
  // Phase 9 design-review prototype only (spec §30.10) — renders its own
  // shell, outside AppShell, and is not part of the app's real IA. Remove
  // or promote into the real Swarm Lab route in Phase 11.
  { path: "/_prototype/swarm-lab", element: <SwarmLabPrototype /> },
]);
