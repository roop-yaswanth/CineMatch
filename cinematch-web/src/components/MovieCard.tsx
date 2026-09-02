"use client";

import { memo, useCallback, useRef } from "react";
import { motion, AnimatePresence, type Transition } from "framer-motion";
import { usePoster } from "@/lib/usePoster";
import { recommendationId, type Movie, type Recommendation, type ExploreMovie } from "@/lib/api";
import { StatusBadge, RatingBadge, PosterInfo } from "./MovieCard.parts";
import { MovieActions, type ActionType } from "@/design-system/components/RatingActions";

type MovieLike = Movie | Recommendation | ExploreMovie;

type MotionDivProps = {
  ref: React.RefObject<HTMLDivElement | null>;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
  onFocus: () => void;
  onBlur: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  tabIndex: number;
  role: "button" | undefined;
  "aria-label": string | undefined;
  whileHover: React.CSSProperties | undefined;
  transition: Transition;
  className: string;
  style: React.CSSProperties;
  children: React.ReactNode;
};

export interface MovieCardProps {
  movie: MovieLike;
  priority?: boolean;
  className?: string;
  compact?: boolean;
  overlay?: boolean;
  noLayout?: boolean;
  showFullDate?: boolean;
  userRating?: "love" | "like" | "dislike" | null;
  isWatchlist?: boolean;
  onQuickAction?: (movie: MovieLike, action: ActionType) => void;
}

const baseCardStyle: React.CSSProperties = {
  position: "relative",
};

const posterFrameStyle = (compact: boolean): React.CSSProperties => ({
  position: "relative",
  width: "100%",
  aspectRatio: "2 / 3",
  borderRadius: compact ? "14px" : "var(--radius-poster)",
  overflow: "hidden",
  background: "var(--color-surface)",
  isolation: "isolate",
});

const imageStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100%",
  height: "100%",
  objectFit: "cover",
  transition: "transform var(--dur-slow) var(--ease-spring)",
};

const hoverStyle: React.CSSProperties = {
  transform: "translateY(-6px) scale(1.02)",
  zIndex: 30,
};

const overlayCardStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  width: "100%",
  height: "100%",
};

const gradientOverlay: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  background: "linear-gradient(180deg, transparent 40%, rgba(0,0,0,0.6) 100%)",
  pointerEvents: "none",
  zIndex: 2,
};

export const MovieCard = memo(function MovieCard({
  movie,
  priority = false,
  className = "",
  compact = false,
  overlay = false,
  noLayout = false,
  showFullDate = false,
  userRating,
  isWatchlist = false,
  onQuickAction,
}: MovieCardProps) {
  const posterSize = compact ? "w342" : "w500";
  const poster = usePoster(movie.poster_path, recommendationId(movie), posterSize, movie.original_language);

  const imdb = ("imdb_rating" in movie && movie.imdb_rating)
    ? (movie.imdb_rating as number).toFixed(1)
    : null;
  const tmdbRating = movie.vote_average ? movie.vote_average.toFixed(1) : null;

  const cardRef = useRef<HTMLDivElement>(null);

  const handleHoverStart = useCallback(() => {
    if (cardRef.current) {
      cardRef.current.style.transform = compact ? "translateY(-4px) scale(1.03)" : "translateY(-6px) scale(1.02)";
      cardRef.current.style.zIndex = "30";
    }
  }, [compact]);

  const handleHoverEnd = useCallback(() => {
    if (cardRef.current) {
      cardRef.current.style.transform = "";
      cardRef.current.style.zIndex = "";
    }
  }, []);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if ((e.key === "Enter" || e.key === " ") && onQuickAction) {
      e.preventDefault();
      onQuickAction(movie, "like");
    }
  }, [movie, onQuickAction]);

  const cardContainerStyle: React.CSSProperties = overlay
    ? { ...overlayCardStyle, borderRadius: compact ? "14px" : "var(--radius-poster)" }
    : { ...baseCardStyle };

  const springTransition: Transition = { type: "spring", stiffness: 350, damping: 22 };
  const springTransitionOverlay: Transition = { type: "spring", stiffness: 300, damping: 20 };

  const cardMotionProps = overlay
    ? { whileHover: { scale: 1.03, y: -4 }, transition: springTransitionOverlay }
    : { whileHover: hoverStyle, transition: springTransition };

  const tabIndex = onQuickAction ? 0 : -1;
  const role = onQuickAction ? "button" : undefined;
  const ariaLabel = onQuickAction ? `View details for ${movie.title}` : undefined;

  const posterFrame = (
    <motion.div
      layout={!noLayout}
      className="card-poster-frame"
      style={posterFrameStyle(compact)}
      whileHover={overlay ? { scale: 1.03, y: -4 } : { scale: 1.02, y: -4 }}
      transition={springTransition}
    >
      <img
        src={poster}
        alt={movie.title}
        loading={priority ? "eager" : "lazy"}
        style={imageStyle}
      />
      <StatusBadge movie={movie} />
      {!overlay && (imdb || tmdbRating) && <RatingBadge score={imdb || tmdbRating!} />}

      {/* Persistent Action Badges on Poster */}
      {!overlay && userRating && (
        <span className={`poster-action-badge poster-action-badge--${userRating}`}>
          {userRating === "love" ? "😍 Loved" : userRating === "like" ? "😀 Liked" : "🙁 Not for me"}
        </span>
      )}
      {!overlay && isWatchlist && (
        <span
          className="poster-action-badge poster-action-badge--watchlist"
          style={{ bottom: userRating ? "34px" : "8px" }}
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 2 }}>
            <polyline points="20 6 9 17 4 12" />
          </svg>
          Watchlist
        </span>
      )}

      {onQuickAction && (
        <MovieActions
          movie={movie}
          userRating={userRating}
          isWatchlist={isWatchlist}
          onAction={onQuickAction}
          variant={overlay ? "poster-overlay" : "poster-hover"}
          compact={compact}
        />
      )}
      {overlay && (
        <>
          <div style={gradientOverlay} />
          <PosterInfo movie={movie} showFullDate={showFullDate} />
        </>
      )}
    </motion.div>
  );

  const renderCard = (props: MotionDivProps) => {
    const motionProps: Record<string, unknown> = {
      ref: props.ref,
      onMouseEnter: props.onMouseEnter,
      onMouseLeave: props.onMouseLeave,
      onFocus: props.onFocus,
      onBlur: props.onBlur,
      onKeyDown: props.onKeyDown,
      tabIndex: props.tabIndex,
      role: props.role,
      "aria-label": props["aria-label"],
      whileHover: props.whileHover,
      transition: props.transition,
      className: props.className,
      style: props.style,
    };
    return <motion.div {...motionProps}>{props.children}</motion.div>;
  };

  if (overlay) {
    return renderCard({
      ref: cardRef,
      onMouseEnter: handleHoverStart,
      onMouseLeave: handleHoverEnd,
      onFocus: handleHoverStart,
      onBlur: handleHoverEnd,
      onKeyDown: handleKeyDown,
      tabIndex,
      role,
      "aria-label": ariaLabel,
      whileHover: cardMotionProps.whileHover,
      transition: cardMotionProps.transition,
      className: `relative flex flex-col items-center justify-center no-select w-full h-full ${className}`,
      style: cardContainerStyle,
      children: (
        <>
          {posterFrame}
          <AnimatePresence>
            {false && null}
          </AnimatePresence>
        </>
      ),
    });
  }

  return renderCard({
    ref: cardRef,
    onMouseEnter: handleHoverStart,
    onMouseLeave: handleHoverEnd,
    onFocus: handleHoverStart,
    onBlur: handleHoverEnd,
    onKeyDown: handleKeyDown,
    tabIndex,
    role,
    "aria-label": ariaLabel,
    whileHover: cardMotionProps.whileHover,
    transition: cardMotionProps.transition,
    className: `card-hover relative flex flex-col items-center no-select ${className}`,
    style: cardContainerStyle,
    children: (
      <>
        {posterFrame}
        <PosterInfo movie={movie} showFullDate={showFullDate} />
      </>
    ),
  });
});

MovieCard.displayName = "MovieCard";

export default MovieCard;