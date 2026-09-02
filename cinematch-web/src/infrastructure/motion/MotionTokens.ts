/**
 * MotionTokens — Encapsulated wrapper around framer-motion primitives.
 * Single place that owns spring configs, easings, durations.
 * If framer-motion is replaced, only this file's exports change.
 *
 * Composition over inheritance: expose small, focused tokens, not a deep class hierarchy.
 */

export const spring = {
  gentle: { type: "spring" as const, stiffness: 300, damping: 22 },
  snappy: { type: "spring" as const, stiffness: 440, damping: 32, mass: 0.5 },
  bouncy: { type: "spring" as const, stiffness: 350, damping: 18 },
} as const;

export const easing = {
  appleOut: [0.22, 1, 0.36, 1] as const,
  appleInOut: [0.65, 0, 0.35, 1] as const,
  linear: "linear" as const,
} as const;

export const duration = {
  fast: 0.14,
  base: 0.22,
  slow: 0.36,
} as const;
