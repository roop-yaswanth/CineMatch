/**
 * Universal Haptics Engine for CineMatch
 * Uses the Web Vibration API with graceful fallback on unsupported platforms / iOS / desktop.
 */

export type HapticType =
  | "tap"           // Light crisp tick (tabs, micro buttons, dismissals)
  | "selection"     // Medium affirmative tick (genre chips, filters, dropdown items)
  | "impact"        // Firm thud (modal opening, big CTA buttons)
  | "love"          // Strong warm vibration (Love rating / super-like)
  | "like"          // Crisp positive pulse (Like rating / heart quick action)
  | "dislike"       // Double buzz (Dislike rating / negative feedback)
  | "skip"          // Subtle swipe-away tick (Haven't seen / skip recommendation)
  | "watchlist"     // Distinct lock-in double-tap (Saved to Watchlist)
  | "undo"          // Reversal pattern (Undo / revert action)
  | "remove"        // Deletion buzz (Remove from list)
  | "error"         // Warning triple-buzz (Network error / invalid action)
  | "success";      // Harmonious double-pulse (Saved preferences / onboarding finish)

const PATTERNS: Record<HapticType, number | number[]> = {
  tap: 10,
  selection: 16,
  impact: 28,
  love: [30, 20, 20],
  like: 18,
  dislike: [22, 12, 22],
  skip: 10,
  watchlist: [12, 25, 20],
  undo: [10, 30, 10],
  remove: [20, 15, 20],
  error: [35, 30, 35, 30, 45],
  success: [15, 35, 25],
};

export function triggerHaptic(type: HapticType | string = "tap"): void {
  if (typeof window === "undefined" || typeof navigator === "undefined") return;
  const nav = navigator as Navigator & { vibrate?: (p: number | number[]) => boolean };
  if (typeof nav.vibrate !== "function") return;

  try {
    const pattern = PATTERNS[type as HapticType] ?? 12;
    nav.vibrate(pattern);
  } catch {
    // Silently ignore if blocked by browser policy
  }
}

// Convenience export helpers
export const hapticTap = () => triggerHaptic("tap");
export const hapticSelection = () => triggerHaptic("selection");
export const hapticImpact = () => triggerHaptic("impact");
export const hapticLove = () => triggerHaptic("love");
export const hapticLike = () => triggerHaptic("like");
export const hapticDislike = () => triggerHaptic("dislike");
export const hapticSkip = () => triggerHaptic("skip");
export const hapticWatchlist = () => triggerHaptic("watchlist");
export const hapticUndo = () => triggerHaptic("undo");
export const hapticRemove = () => triggerHaptic("remove");
export const hapticError = () => triggerHaptic("error");
export const hapticSuccess = () => triggerHaptic("success");
