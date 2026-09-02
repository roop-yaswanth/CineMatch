export type RatingValue = "love" | "like" | "dislike" | "skip" | "watchlist" | "not_watched";

export interface RatingDisplayConfig {
  label: string;
  color: string;
  emoji?: string;
  icon?: "heart" | "smile" | "frown" | "skip" | "bookmark";
  shortcut?: string;
}

export const RATING_DISPLAY: Record<RatingValue, RatingDisplayConfig> = {
  love: { label: "Loved", color: "var(--color-love)", emoji: "😍", icon: "heart", shortcut: "O" },
  like: { label: "Liked", color: "var(--color-like)", emoji: "😀", icon: "smile", shortcut: "L" },
  dislike: { label: "Disliked", color: "var(--color-dislike)", emoji: "🙁", icon: "frown", shortcut: "D" },
  skip: { label: "Skipped", color: "var(--color-text-muted)", icon: "skip", shortcut: "S" },
  not_watched: { label: "Skipped", color: "var(--color-text-muted)", icon: "skip", shortcut: "S" },
  watchlist: { label: "Watchlist", color: "var(--color-accent)", icon: "bookmark" },
};

export function ratingLabel(rating: string): string {
  return RATING_DISPLAY[rating as RatingValue]?.label ?? rating;
}

export function ratingColor(rating: string): string {
  return RATING_DISPLAY[rating as RatingValue]?.color ?? "var(--color-text-muted)";
}
