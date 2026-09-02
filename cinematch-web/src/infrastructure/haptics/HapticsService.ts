/**
 * HapticsService — Encapsulated wrapper around the Web Vibration API / haptics lib.
 * Components call `haptics.light()` etc., not `navigator.vibrate` directly.
 * If the haptics library is removed, only this file changes.
 */

export type HapticKind = "light" | "medium" | "heavy" | "success" | "selection" | "watchlist";

export interface HapticsPort {
  trigger(kind: HapticKind): void;
  light(): void;
  medium(): void;
  heavy(): void;
  success(): void;
  selection(): void;
}

class WebHaptics implements HapticsPort {
  trigger(kind: HapticKind): void {
    try {
      const map: Record<HapticKind, number | number[]> = {
        light: 10, medium: 20, heavy: 30, success: [10, 30, 10], selection: 8, watchlist: 15,
      };
      const pattern = map[kind];
      if (typeof navigator !== "undefined" && "vibrate" in navigator) {
        navigator.vibrate(pattern as number);
      }
    } catch {}
  }
  light() { this.trigger("light"); }
  medium() { this.trigger("medium"); }
  heavy() { this.trigger("heavy"); }
  success() { this.trigger("success"); }
  selection() { this.trigger("selection"); }
}

export const haptics: HapticsPort = new WebHaptics();

// Backward-compat helpers used by older components (will be phased out)
export const triggerHaptic = (action: string) => {
  const m: Record<string, HapticKind> = { love: "heavy", like: "light", dislike: "light", watchlist: "watchlist", skip: "selection" };
  haptics.trigger(m[action] ?? "light");
};
