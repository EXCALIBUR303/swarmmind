import { createBrowserRouter } from "react-router-dom";
import { AppShell } from "./AppShell";
import { SwarmLab } from "../pages/SwarmLab";
import { TrainSwarm } from "../pages/TrainSwarm";
import { AlgorithmComparison } from "../pages/AlgorithmComparison";
import { SwarmAnalysis } from "../pages/SwarmAnalysis";
import { ExperimentHistory } from "../pages/ExperimentHistory";
import { Settings } from "../pages/Settings";

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
]);
