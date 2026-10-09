/** Micro-interaction timings (spec §30.7), in milliseconds. One place, so polish is a one-file change. */
export const MOTION = {
  moveMaxMs: 160, // robot tween = min(frame interval x 0.8, 160 ms), ease-out
  moveIntervalShare: 0.8,
  nudgeMs: 120, // blocked move: 2px nudge toward the attempted direction and back
  nudgePx: 2,
  collisionMs: 400, // danger ring 0 -> 0.6 cell, fading
  discoveryMs: 500, // star scales 0.6 -> 1.0 with one gold pulse ring
  rescueMs: 800, // star -> hollow grey
  commPulseMs: 300, // link opacity pulse once on a merge
} as const;

/** 0..1 progress of an effect that started `sinceMs` ago; settled (1) when reduced motion or long past. */
export const progress = (sinceMs: number, durationMs: number): number => Math.min(Math.max(sinceMs / durationMs, 0), 1);

export const moveDurationMs = (intervalMs: number): number => Math.min(intervalMs * MOTION.moveIntervalShare, MOTION.moveMaxMs);
