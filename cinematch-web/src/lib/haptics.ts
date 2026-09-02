/**
 * Universal Haptics Engine for CineMatch
 * Supports:
 * - Standard Web Vibration API (Android, Chrome, Firefox)
 * - iOS Safari / WebKit native switch hack (iOS 17.4+ & iOS 18+ Taptic Engine)
 * - Graceful degradation on unsupported platforms & desktop
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
  | "success"       // Harmonious double-pulse (Saved preferences / onboarding finish)
  | "light"         // Alias for tap
  | "medium"        // Alias for selection
  | "heavy";        // Alias for impact

const PATTERNS: Record<string, number | number[]> = {
  tap: 10,
  light: 10,
  selection: 16,
  medium: 20,
  impact: 28,
  heavy: 30,
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

// Pulse counts for iOS Safari switch hack
const IOS_PULSES: Record<string, number> = {
  tap: 1,
  light: 1,
  selection: 1,
  medium: 1,
  like: 1,
  skip: 1,
  impact: 1,
  heavy: 1,
  love: 2,
  dislike: 2,
  watchlist: 2,
  undo: 2,
  remove: 2,
  success: 2,
  error: 3,
};

let iosSwitchLabel: HTMLLabelElement | null = null;
let isIosCached: boolean | null = null;

function isIosPlatform(): boolean {
  if (isIosCached !== null) return isIosCached;
  if (typeof window === "undefined" || typeof navigator === "undefined") return false;

  const ua = navigator.userAgent || "";
  const isIos =
    /iPad|iPhone|iPod/.test(ua) ||
    (navigator.platform === "MacIntel" && (navigator.maxTouchPoints ?? 0) > 1);

  isIosCached = isIos;
  return isIos;
}

/**
 * Lazily creates and returns the hidden HTML switch label used to trigger iOS native haptics.
 */
function getIosSwitchLabel(): HTMLLabelElement | null {
  if (typeof document === "undefined" || typeof document.body === "undefined") return null;

  if (iosSwitchLabel && document.body.contains(iosSwitchLabel)) {
    return iosSwitchLabel;
  }

  try {
    const existingLabel = document.getElementById("cm-ios-haptic-label") as HTMLLabelElement | null;
    if (existingLabel) {
      iosSwitchLabel = existingLabel;
      return existingLabel;
    }

    const switchInput = document.createElement("input");
    switchInput.type = "checkbox";
    switchInput.setAttribute("switch", "");
    switchInput.id = "cm-ios-haptic-switch";
    switchInput.setAttribute("aria-hidden", "true");
    switchInput.tabIndex = -1;

    // Invisible but remains in WebKit layout tree for system haptic dispatch
    Object.assign(switchInput.style, {
      position: "fixed",
      top: "-9999px",
      left: "-9999px",
      width: "1px",
      height: "1px",
      opacity: "0.001",
      pointerEvents: "none",
      zIndex: "-9999",
      clipPath: "inset(50%)",
    });

    const label = document.createElement("label");
    label.htmlFor = "cm-ios-haptic-switch";
    label.id = "cm-ios-haptic-label";
    label.setAttribute("aria-hidden", "true");
    label.tabIndex = -1;
    Object.assign(label.style, {
      position: "fixed",
      top: "-9999px",
      left: "-9999px",
      width: "1px",
      height: "1px",
      opacity: "0.001",
      pointerEvents: "none",
      zIndex: "-9999",
      clipPath: "inset(50%)",
    });

    document.body.appendChild(switchInput);
    document.body.appendChild(label);
    iosSwitchLabel = label;
    return label;
  } catch {
    return null;
  }
}

/**
 * Triggers iOS native switch haptic feedback.
 * Initial pulse is synchronous in the user gesture event loop.
 */
function triggerIosHaptic(pulseCount = 1): void {
  const label = getIosSwitchLabel();
  if (!label) return;

  try {
    label.click();
  } catch {}

  if (pulseCount > 1) {
    for (let i = 1; i < pulseCount; i++) {
      setTimeout(() => {
        try {
          label.click();
        } catch {}
      }, i * 85);
    }
  }
}

/**
 * Universal haptic trigger.
 * Automatically selects Web Vibration API or iOS WebKit Switch Taptic.
 */
export function triggerHaptic(type: HapticType | string = "tap"): void {
  if (typeof window === "undefined" || typeof navigator === "undefined") return;

  const key = String(type).toLowerCase();
  const isIos = isIosPlatform();

  // 1. On iOS / WebKit, fire the native Switch haptic
  if (isIos) {
    const pulses = IOS_PULSES[key] ?? 1;
    triggerIosHaptic(pulses);
    return;
  }

  // 2. On standard platforms (Android, etc.), use the Web Vibration API
  const nav = navigator as Navigator & { vibrate?: (p: number | number[]) => boolean };
  if (typeof nav.vibrate === "function") {
    try {
      const pattern = PATTERNS[key] ?? 12;
      nav.vibrate(pattern);
      return;
    } catch {
      // Ignore if blocked by browser permission
    }
  }

  // 3. Fallback to Switch trigger if navigator.vibrate was unavailable
  triggerIosHaptic(IOS_PULSES[key] ?? 1);
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

