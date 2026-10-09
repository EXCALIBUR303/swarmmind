import { drawComm } from "./comm";
import { drawGroundTruth } from "./groundTruth";
import { drawPaths } from "./paths";
import { drawFocusWindow, drawRobots } from "./robots";
import { drawSensors } from "./sensors";
import { drawTargets } from "./targets";
import { drawTerrain } from "./terrain";
import type { Renderer } from "./types";
import { drawVisited } from "./visited";

/** Bottom to top. Each renderer decides for itself whether its layer toggle is on. */
export const RENDER_ORDER: Renderer[] = [
  drawTerrain,
  drawVisited,
  drawGroundTruth,
  drawComm,
  drawPaths,
  drawTargets,
  drawSensors,
  drawFocusWindow,
  drawRobots,
];

export type { RenderView, Renderer } from "./types";
