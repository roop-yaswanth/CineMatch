"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { Button } from "./Button";
import { triggerHaptic } from "@/lib/haptics";
import { RatingButton, Badge } from "@/design-system";
import type { Movie, Recommendation, ExploreMovie } from "@/lib/api";

export type RatingType = "love" | "like" | "dislike" | "skip";
export type ActionType = "love" | "like" | "dislike" | "watchlist";

export interface RatingConfig {
  type: RatingType;
  label: string;
  emoji: string;
  shortcut?: string;
  color: string;
  rgb: string;
}

export const RATING_CONFIGS: Record<RatingType, RatingConfig> = {
  love: { type: "love", label: "Love", emoji: "😍", shortcut: "O", color: "var(--color-love)", rgb: "var(--rgb-love)" },
  like: { type: "like", label: "Like", emoji: "😀", shortcut: "L", color: "var(--color-like)", rgb: "var(--rgb-like)" },
  dislike: { type: "dislike", label: "Dislike", emoji: "🙁", shortcut: "D", color: "var(--color-dislike)", rgb: "var(--rgb-dislike)" },
  skip: { type: "skip", label: "Skip", emoji: "", shortcut: "S", color: "var(--color-skip)", rgb: "var(--rgb-skip)" },
};

type MovieLike = Movie | Recommendation | ExploreMovie;

export interface MovieActionsProps {
  movie: MovieLike;
  userRating?: RatingType | null;
  isWatchlist?: boolean;
  onAction: (movie: MovieLike, action: ActionType) => void;
  variant?: "poster-overlay" | "poster-hover" | "bottom-sheet" | "inline" | "compact";
  showLabels?: boolean;
  compact?: boolean;
}

const variantStyles = {
  "poster-overlay": {
    container: { position: "absolute", bottom: 8, left: 8, right: 8, display: "flex", flexDirection: "column", gap: 8, zIndex: 10 },
    ratingRow: { display: "flex", gap: 6, justifyContent: "center" },
    watchlistRow: { display: "flex", justifyContent: "center" },
    button: { flex: 1, padding: "10px 8px", fontSize: "12px", fontWeight: 600 },
  },
  "poster-hover": {
    container: { position: "absolute", inset: 0, display: "flex", flexDirection: "column", justifyContent: "space-between", padding: 8, zIndex: 12, pointerEvents: "none" },
    topRow: { display: "flex", justifyContent: "space-between" },
    bottomRow: { display: "flex", justifyContent: "space-between", alignItems: "flex-end" },
    button: { pointerEvents: "auto", width: 44, height: 44, padding: 0, display: "flex", alignItems: "center", justifyContent: "center" },
    ratingGroup: { display: "flex", flexDirection: "column", gap: 4, pointerEvents: "auto" },
  },
  "bottom-sheet": {
    container: { display: "flex", flexDirection: "column", gap: 12, padding: "var(--s-5)" },
    ratingRow: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8 },
    watchlistRow: { display: "flex", justifyContent: "center" },
    button: { padding: "14px 8px", fontSize: "13px", fontWeight: 600 },
  },
  inline: {
    container: { display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" },
    ratingRow: { display: "flex", gap: 6 },
    watchlistRow: { display: "flex", alignItems: "center" },
    button: { padding: "8px 14px", fontSize: "var(--fs-sm)" },
  },
  compact: {
    container: { display: "flex", alignItems: "center", gap: 4 },
    ratingRow: { display: "flex", gap: 4 },
    watchlistRow: { display: "flex" },
    button: { width: 36, height: 36, padding: 0, display: "flex", alignItems: "center", justifyContent: "center" },
  },
};

interface SplashOverlayProps {
  feedbackSplash: ActionType | null;
}

function SplashOverlayComponent({ feedbackSplash }: SplashOverlayProps) {
  if (!feedbackSplash) return null;

  const config = feedbackSplash === "watchlist" 
    ? { emoji: "🔖", color: "var(--color-accent)", rgb: "var(--rgb-accent)" }
    : (RATING_CONFIGS[feedbackSplash as RatingType] || RATING_CONFIGS.like);

  return (
    <motion.div
      key="feedback-splash"
      initial={{ opacity: 0, scale: 0.55 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ type: "spring", stiffness: 450, damping: 22 }}
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "8px",
        background: "rgba(8, 9, 14, 0.78)",
        backdropFilter: "blur(6px)",
        WebkitBackdropFilter: "blur(6px)",
        zIndex: 25,
        pointerEvents: "none",
        borderRadius: "var(--radius-poster)",
      }}
    >
      <motion.span
        initial={{ scale: 0.5, rotate: -15 }}
        animate={{ scale: [0.5, 1.28, 1], rotate: [0, 8, 0] }}
        transition={{ duration: 0.35 }}
        style={{ fontSize: "38px", filter: "drop-shadow(0 6px 16px rgba(0,0,0,0.7))" }}
      >
        {config.emoji}
      </motion.span>
      <span
        style={{
          fontSize: "12px",
          fontWeight: 700,
          color: "#ffffff",
          background: `rgba(${config.rgb}, 0.5)`,
          padding: "4px 12px",
          borderRadius: "999px",
          border: "1px solid rgba(255, 255, 255, 0.35)",
          boxShadow: "0 4px 14px rgba(0,0,0,0.5)",
          letterSpacing: "-0.01em",
        }}
      >
        {feedbackSplash === "love" ? "Loved!" : feedbackSplash === "like" ? "Liked!" : feedbackSplash === "dislike" ? "Not for me" : "Added"}
      </span>
    </motion.div>
  );
}

interface WatchlistButtonProps {
  compact: boolean;
  isWatchlisted: boolean;
  showLabels: boolean;
  onClick: (e: React.MouseEvent) => void;
  className?: string;
}

function WatchlistButtonComponent({ compact, isWatchlisted, showLabels, onClick, className }: WatchlistButtonProps) {
  return (
    <Button
      variant="glass"
      size={compact ? "sm" : "md"}
      onClick={onClick}
      style={{ width: compact ? 36 : undefined, height: compact ? 36 : undefined }}
      aria-label={isWatchlisted ? "Remove from watchlist" : "Add to watchlist"}
      aria-pressed={isWatchlisted}
      className={className}
    >
      <svg width={compact ? 14 : 16} height={compact ? 14 : 16} viewBox="0 0 24 24" fill={isWatchlisted ? "currentColor" : "none"} stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
      </svg>
      {showLabels && !compact && <span>Watchlist</span>}
    </Button>
  );
}

export function MovieActions({
  movie,
  userRating,
  isWatchlist = false,
  onAction,
  variant = "poster-hover",
  showLabels = true,
  compact = false,
}: MovieActionsProps) {
  const [feedbackSplash, setFeedbackSplash] = useState<ActionType | null>(null);
  const splashTimerRef = useRef<NodeJS.Timeout | null>(null);
  const styles = variantStyles[variant];

  const ratingAction = userRating;
  const isWatchlisted = isWatchlist;

  const handleActionClick = useCallback(
    (e: React.MouseEvent, action: ActionType) => {
      e.stopPropagation();
      (document.activeElement as HTMLElement)?.blur();
      triggerHaptic(action);

      setFeedbackSplash(action);
      if (splashTimerRef.current) clearTimeout(splashTimerRef.current);
      splashTimerRef.current = setTimeout(() => setFeedbackSplash(null), 1700);

      onAction(movie, action);
    },
    [movie, onAction]
  );

  { useEffect(() => { return () => { if (splashTimerRef.current) clearTimeout(splashTimerRef.current); } }, []); }



  const containerStyle = {
      ...styles.container,
      opacity: variant === "poster-hover" ? 0 : 1,
      transition: "opacity var(--dur-base) var(--ease-out)",
    } as React.CSSProperties;

  return (
    <>
      <SplashOverlayComponent feedbackSplash={feedbackSplash} />
      <div style={containerStyle} className={`${variant}-actions`}>
        {variant === "poster-hover" && (
          <>
            <div style={(styles as Record<string, React.CSSProperties>).topRow}>
              <div style={{ display: "flex", gap: 4, opacity: ratingAction ? 1 : 0, transition: "opacity var(--dur-base) var(--ease-out)", pointerEvents: ratingAction ? "auto" : "none" }}>
                {["love", "like", "dislike"].map((type) => (
                  <RatingButton
                    key={type}
                    type={type as RatingType}
                    label=""
                    onClick={(e) => handleActionClick(e, type as ActionType)}
                    style={(({ pointerEvents, ...rest }) => { void pointerEvents; return rest; })(styles.button as React.CSSProperties) }
                  />
                ))}
              </div>
              {isWatchlisted && (
                <div style={{ opacity: 1, transition: "opacity var(--dur-base) var(--ease-out)" }}>
                  <WatchlistButtonComponent
                    compact={compact}
                    isWatchlisted={isWatchlisted}
                    showLabels={showLabels}
                    onClick={(e) => handleActionClick(e, "watchlist")}
                  />
                </div>
              )}
            </div>
            <div style={(styles as Record<string, React.CSSProperties>).bottomRow}>
              <div style={((styles as Record<string, React.CSSProperties>).ratingGroup)}>
                {["dislike", "like", "love"].map((type) => (
                  <RatingButton
                    key={type}
                    type={type as RatingType}
                    label={showLabels ? RATING_CONFIGS[type as RatingType].label : ""}
                    emoji={showLabels ? RATING_CONFIGS[type as RatingType].emoji : undefined}
                    onClick={(e) => handleActionClick(e, type as ActionType)}
                    style={(({ pointerEvents, ...rest }) => { void pointerEvents; return rest; })(styles.button as React.CSSProperties)}
                  />
                ))}
              </div>
              <div style={{ pointerEvents: "auto" }}>
                <WatchlistButtonComponent
                  compact={compact}
                  isWatchlisted={isWatchlisted}
                  showLabels={showLabels}
                  onClick={(e) => handleActionClick(e, "watchlist")}
                />
              </div>
            </div>
          </>
        )}

        {variant !== "poster-hover" && (
          <>
            <div style={((styles as Record<string, React.CSSProperties>).ratingRow)}>
              {(() => {
                const result: React.ReactElement[] = [];
                (["love", "like", "dislike"] as const).forEach((type: Exclude<RatingType, "skip">) => {
                  const config = RATING_CONFIGS[type];
                  const buttonStyles = styles.button as React.CSSProperties;
                  const { pointerEvents: _omit, ...restButtonStyles } = buttonStyles; void _omit;
                  result.push(
                    <RatingButton
                      key={type}
                      type={type}
                      label={showLabels ? config.label : ""}
                      emoji={showLabels ? config.emoji : undefined}
                      shortcut={config.shortcut}
                      onClick={(e) => handleActionClick(e, type as ActionType)}
                      disabled={false}
                      style={{
                        ...restButtonStyles,
                        opacity: ratingAction === type ? 1 : 0.6,
                        transform: ratingAction === type ? "scale(1.05)" : "none",
                      }}
                    />
                  );
                });
                return result;
              })()}
            </div>
            <div style={((styles as Record<string, React.CSSProperties>).watchlistRow)}>
              <WatchlistButtonComponent
                compact={compact}
                isWatchlisted={isWatchlisted}
                showLabels={showLabels}
                onClick={(e) => handleActionClick(e, "watchlist")}
              />
            </div>
          </>
        )}
      </div>
    </>
  );
}

export interface RatingChipProps {
  rating: RatingType | null;
  compact?: boolean;
  onClick?: (rating: RatingType) => void;
}

export function RatingChip({ rating, compact = false, onClick }: RatingChipProps) {
  if (!rating) return null;

  const config = RATING_CONFIGS[rating];

  return (
    <Badge
      variant="default"
      size={compact ? "xs" : "sm"}
      dot
      style={{
        background: `rgba(${config.rgb}, 0.15)`,
        color: config.color,
        borderColor: `rgba(${config.rgb}, 0.3)`,
        cursor: onClick ? "pointer" : "default",
      }}
      onClick={onClick ? () => onClick(rating) : undefined}
    >
      {config.emoji && <span style={{ fontSize: compact ? 10 : 12 }}>{config.emoji}</span>}
      <span style={{ fontWeight: 600, fontSize: compact ? "var(--fs-2xs)" : "var(--fs-xs)" }}>{config.label}</span>
    </Badge>
  );
}

export interface WatchlistIndicatorProps {
  isWatchlist: boolean;
  compact?: boolean;
  onClick?: () => void;
}

export function WatchlistIndicator({ isWatchlist, compact = false, onClick }: WatchlistIndicatorProps) {
  if (!isWatchlist) return null;

  return (
    <Button
      variant="glass"
      size={compact ? "sm" : "md"}
      onClick={onClick}
      style={{ padding: compact ? "4px 8px" : "6px 10px", gap: 4 }}
      aria-label="In watchlist"
    >
      <svg width={compact ? 12 : 14} height={compact ? 12 : 14} viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
      </svg>
      {!compact && <span style={{ fontSize: "var(--fs-xs)", fontWeight: 500 }}>Watchlist</span>}
    </Button>
  );
}

export interface QuickActionBarProps {
  movie: MovieLike;
  userRating?: RatingType | null;
  isWatchlist?: boolean;
  onAction: (movie: MovieLike, action: ActionType) => void;
}

export function QuickActionBar({ movie, userRating, isWatchlist = false, onAction }: QuickActionBarProps) {
  return <MovieActions movie={movie} userRating={userRating} isWatchlist={isWatchlist} onAction={onAction} variant="inline" showLabels={false} />;
}