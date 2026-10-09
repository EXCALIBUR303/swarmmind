import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "./AppShell";
import { SwarmLab } from "../pages/SwarmLab";
import { TrainSwarm } from "../pages/TrainSwarm";
import { AlgorithmComparison } from "../pages/AlgorithmComparison";
import { SwarmAnalysis } from "../pages/SwarmAnalysis";
import { ExperimentHistory } from "../pages/ExperimentHistory";
import { Settings } from "../pages/Settings";
import { RunDetail } from "../pages/RunDetail";

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
      { path: "experiments/:runId", element: <RunDetail /> },
      { path: "settings", element: <Settings /> },
    ],
  },
]);
