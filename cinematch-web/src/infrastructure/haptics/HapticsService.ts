import { triggerHaptic, type HapticType } from "@/lib/haptics";

export type HapticKind =
  | "light"
  | "medium"
  | "heavy"
  | "success"
  | "selection"
  | "watchlist"
  | "tap"
  | "love"
  | "like"
  | "dislike"
  | "skip"
  | "undo"
  | "remove"
  | "error";

export interface HapticsPort {
  trigger(kind: HapticKind | string): void;
  light(): void;
  medium(): void;
  heavy(): void;
  success(): void;
  selection(): void;
}

class WebHaptics implements HapticsPort {
  trigger(kind: HapticKind | string): void {
    triggerHaptic(kind as HapticType);
  }
  light() { this.trigger("light"); }
  medium() { this.trigger("medium"); }
  heavy() { this.trigger("heavy"); }
  success() { this.trigger("success"); }
  selection() { this.trigger("selection"); }
}

export const haptics: HapticsPort = new WebHaptics();

// Backward-compat helpers used by older components
export const triggerHapticCompat = (action: string) => {
  haptics.trigger(action);
};

