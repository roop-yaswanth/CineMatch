"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { triggerHaptic } from "@/lib/haptics";
import {
  languageLabel,
  prefetchMovieDetails,
  recommendationId,
  type Movie,
  type Recommendation,
  type ExploreMovie,
} from "@/lib/api";
import { usePoster } from "@/lib/usePoster";

type MovieLike = Movie | Recommendation | ExploreMovie;

interface Props {
  movie: MovieLike;
  priority?: boolean;
  className?: string;
  compact?: boolean;
  overlay?: boolean;
  noLayout?: boolean;
  showFullDate?: boolean;
  userAction?: "love" | "like" | "dislike" | "watchlist" | null;
  onQuickAction?: (movie: MovieLike, action: "love" | "like" | "dislike" | "watchlist") => void;
}

const NOW_MS = new Date().getTime();
const CURRENT_YEAR = new Date().getFullYear();

export function formatRuntime(runtime?: number | string | null): string {
  if (!runtime) return "";
  const mins = typeof runtime === "string" ? parseInt(runtime, 10) : runtime;
  if (!mins || isNaN(mins) || mins <= 0) return "";
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

/**
 * Release status badge pinned to the poster's top-left corner (Upcoming vs In Theatres).
 */
export function PosterStatusBadge({ movie }: { movie: MovieLike }) {
  let statusText = "";
  let isUpcoming = false;

  if ("status" in movie && movie.status) {
    if (
      movie.status === "Upcoming" ||
      movie.status === "Post Production" ||
      movie.status === "In Production" ||
      movie.status === "Planned"
    ) {
      isUpcoming = true;
      statusText = "Upcoming";
    } else if (movie.status === "In Theatres" || movie.status === "Now Playing") {
      statusText = "In Theatres";
    }
  }

  if (!statusText && "release_date" in movie && movie.release_date) {
    const rDate = new Date(movie.release_date);
    if (!isNaN(rDate.getTime())) {
      const diff = NOW_MS - rDate.getTime();
      if (rDate.getTime() > NOW_MS) {
        isUpcoming = true;
        statusText = "Upcoming";
      } else if (diff >= 0 && diff < 60 * 24 * 3600 * 1000) {
        statusText = "In Theatres";
      }
    }
  }

  if (!statusText && movie.year && Number(movie.year) > CURRENT_YEAR) {
    isUpcoming = true;
    statusText = "Upcoming";
  }

  if (!statusText) return null;

  return (
    <span
      className={`poster-status-badge ${isUpcoming ? "status-upcoming" : "status-theatres"}`}
      style={{
        position: "absolute",
        top: "7px",
        left: "7px",
        zIndex: 2,
        padding: "3px 8px",
        borderRadius: "999px",
        fontSize: "9px",
        fontWeight: 700,
        letterSpacing: "0.02em",
        background: isUpcoming ? "rgba(99, 102, 241, 0.88)" : "rgba(16, 185, 129, 0.88)",
        color: "#ffffff",
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
      }}
    >
      {statusText}
    </span>
  );
}

/**
 * Gold score badge pinned to the poster's top-right corner (IMDb preferred,
 * TMDB ★ fallback). Lives on the poster — not the meta row — so the "year ·
 * language" line below never has to truncate the language to make room.
 */
export function PosterRatingBadge({ movie }: { movie: MovieLike }) {
  const imdb = ("imdb_rating" in movie && movie.imdb_rating)
    ? (movie.imdb_rating as number).toFixed(1)
    : null;
  const tmdbRating = movie.vote_average ? movie.vote_average.toFixed(1) : null;
  const score = imdb || tmdbRating;
  if (!score) return null;
  return (
    <span className="poster-rating-badge">
      ★ {score}
    </span>
  );
}

export function PosterInfo({ movie, showFullDate = false }: { movie: MovieLike; showFullDate?: boolean }) {
  let fullDate = "";
  if (showFullDate && "release_date" in movie && movie.release_date) {
    const date = new Date(movie.release_date);
    if (!isNaN(date.getTime())) {
      fullDate = date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
    }
  }
  const year = fullDate ? "" : (movie.year ? movie.year.toString() : "");
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const runtimeFormatted = "runtime" in movie && movie.runtime ? formatRuntime(movie.runtime) : "";
  const cert = "certification" in movie && movie.certification ? movie.certification : "";

  const metaParts = [year, lang, runtimeFormatted, cert].filter(Boolean);

  return (
    <div className="mt-3 w-full px-0.5">
      <h2 className="poster-info-title">{movie.title}</h2>
      {fullDate && (
        <div className="mt-1 text-[11px] font-light text-[var(--color-text-muted)]">
          {fullDate}
        </div>
      )}
      {metaParts.length > 0 && (
        <div className="poster-info-meta">
          <span className="poster-info-meta-text">
            {metaParts.join(" · ")}
          </span>
        </div>
      )}
    </div>
  );
}

export default function MovieCard({
  movie,
  priority = false,
  className = "",
  compact = false,
  overlay = false,
  noLayout = false,
  showFullDate = false,
  userAction,
  onQuickAction,
}: Props) {
  // Pick the smallest TMDB size that still looks crisp on the rendered card.
  // compact rails ≈ 130–140px, default cards ≈ 360px hero. Account for 2×/3×
  // DPR by going one size up.
  const posterSize = compact ? "w342" : "w500";
  // Non-English titles get TMDB's English poster variant when available.
  const poster = usePoster(movie.poster_path, recommendationId(movie), posterSize, movie.original_language);

  const year = movie.year ? movie.year.toString() : "";
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const imdb = ("imdb_rating" in movie && movie.imdb_rating)
    ? (movie.imdb_rating as number).toFixed(1)
    : null;
  const tmdbRating = movie.vote_average ? movie.vote_average.toFixed(1) : null;
  const runtimeFormatted = "runtime" in movie && movie.runtime ? formatRuntime(movie.runtime) : "";

  // ── Action State & Visual Feedback ──────────────────────────────────────────
  const [localRating, setLocalRating] = useState<"love" | "like" | "dislike" | null>(null);
  const [localWatchlist, setLocalWatchlist] = useState<boolean | null>(null);
  const [feedbackSplash, setFeedbackSplash] = useState<"love" | "like" | "dislike" | "watchlist" | null>(null);
  const splashTimerRef = useRef<NodeJS.Timeout | null>(null);

  const propRating = userAction === "love" || userAction === "like" || userAction === "dislike" ? userAction : null;
  const ratingAction = localRating ?? propRating;

  const propWatchlist = userAction === "watchlist";
  const isWatchlisted = localWatchlist !== null ? localWatchlist : propWatchlist;

  useEffect(() => {
    return () => {
      if (splashTimerRef.current) clearTimeout(splashTimerRef.current);
    };
  }, []);

  const handleActionClick = (e: React.MouseEvent, action: "love" | "like" | "dislike" | "watchlist") => {
    e.stopPropagation();
    (document.activeElement as HTMLElement)?.blur();
    triggerHaptic(action);

    if (action === "watchlist") {
      setLocalWatchlist((prev) => (prev !== null ? !prev : !propWatchlist));
    } else {
      setLocalRating(action);
    }

    setFeedbackSplash(action);
    if (splashTimerRef.current) clearTimeout(splashTimerRef.current);
    splashTimerRef.current = setTimeout(() => {
      setFeedbackSplash(null);
    }, 1700);

    onQuickAction?.(movie, action);
  };

  // Overlay mode: info displayed on poster
  if (overlay) {
    return (
      <motion.div
        layout={!noLayout}
        whileHover={{ scale: 1.03, y: -4 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        className={`relative flex flex-col items-center justify-center no-select w-full h-full ${className}`}
      >
        {/* Poster with overlay */}
        <div
          className="relative w-full h-full aspect-[2/3] overflow-hidden bg-[var(--color-surface)] group"
          style={{ borderRadius: compact ? "14px" : "var(--radius-poster)" }}
        >
          <img
            src={poster}
            alt={movie.title}
            loading={priority ? "eager" : "lazy"}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
          />

          <PosterStatusBadge movie={movie} />

          {/* Persistent Action Badges */}
          {ratingAction && (
            <span className={`poster-action-badge poster-action-badge--${ratingAction}`}>
              {ratingAction === "love" ? "😍 Loved" : ratingAction === "like" ? "😀 Liked" : "🙁 Not for me"}
            </span>
          )}
          {isWatchlisted && (
            <span
              className="poster-action-badge poster-action-badge--watchlist"
              style={{ bottom: ratingAction ? "34px" : "8px" }}
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 2 }}>
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Watchlist
            </span>
          )}

          {/* Gradient overlay at bottom */}
          <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-black/95 via-black/65 to-transparent pointer-events-none" />

          {/* Info overlay at bottom */}
          <div
            className="absolute inset-x-0 bottom-0 text-white"
            style={{ padding: "14px 18px 16px 18px" }}
          >
            {(imdb || tmdbRating) && (
              <div className="mb-1">
                <span className="text-[11px] font-bold text-[var(--color-rating)]">
                  ★ {imdb || tmdbRating}
                </span>
              </div>
            )}

            <h2 className="text-[13.5px] font-semibold leading-snug tracking-tight line-clamp-2 drop-shadow-lg">
              {movie.title}
            </h2>
            {(year || lang || runtimeFormatted) && (
              <div className="mt-1 text-[11px] text-white/70 font-light">
                {[year, lang, runtimeFormatted].filter(Boolean).join(" · ")}
              </div>
            )}
          </div>
        </div>
      </motion.div>
    );
  }

  // Default mode: info below poster
  return (
    <motion.div
      layout={!noLayout}
      whileHover={{ scale: 1.06, y: -6, zIndex: 30 }}
      transition={{ type: "spring", stiffness: 350, damping: 22 }}
      onHoverStart={() =>
        prefetchMovieDetails(recommendationId(movie), {
          title: movie.title,
          overview: movie.overview,
          genres: movie.genres,
          lang: movie.original_language,
          year: movie.year ? Number(movie.year) : undefined,
        })
      }
      className={`card-hover relative flex flex-col items-center no-select ${className}`}
      style={{ position: "relative" }}
    >
      {/* Poster */}
      <div
        className="card-poster-frame relative w-full aspect-[2/3] overflow-hidden bg-[var(--color-surface)]"
        style={{
          borderRadius: compact ? "14px" : "var(--radius-poster)",
          opacity: ratingAction === "dislike" ? 0.72 : 1,
          transition: "opacity 200ms ease",
        }}
      >
        <img
          src={poster}
          alt={movie.title}
          loading={priority ? "eager" : "lazy"}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
        />
        <PosterStatusBadge movie={movie} />
        <PosterRatingBadge movie={movie} />

        {/* Persistent Action Badges on Poster */}
        {ratingAction && (
          <span className={`poster-action-badge poster-action-badge--${ratingAction}`}>
            {ratingAction === "love" ? "😍 Loved" : ratingAction === "like" ? "😀 Liked" : "🙁 Not for me"}
          </span>
        )}
        {isWatchlisted && (
          <span
            className="poster-action-badge poster-action-badge--watchlist"
            style={{ bottom: ratingAction ? "34px" : "8px" }}
          >
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: 2 }}>
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Watchlist
          </span>
        )}

        {/* Interactive Splash Confirmation on Click */}
        <AnimatePresence>
          {feedbackSplash && (
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
                borderRadius: compact ? "14px" : "var(--radius-poster)",
              }}
            >
              <motion.span
                initial={{ scale: 0.5, rotate: -15 }}
                animate={{ scale: [0.5, 1.28, 1], rotate: [0, 8, 0] }}
                transition={{ duration: 0.35 }}
                style={{ fontSize: "38px", filter: "drop-shadow(0 6px 16px rgba(0,0,0,0.7))" }}
              >
                {feedbackSplash === "love" ? "😍" : feedbackSplash === "like" ? "😀" : feedbackSplash === "dislike" ? "🙁" : "🔖"}
              </motion.span>
              <span
                style={{
                  fontSize: "12px",
                  fontWeight: 700,
                  color: "#ffffff",
                  background: feedbackSplash === "love"
                    ? "rgba(236, 72, 153, 0.5)"
                    : feedbackSplash === "like"
                    ? "rgba(16, 185, 129, 0.5)"
                    : feedbackSplash === "dislike"
                    ? "rgba(239, 68, 68, 0.5)"
                    : "rgba(99, 102, 241, 0.5)",
                  padding: "4px 12px",
                  borderRadius: "999px",
                  border: "1px solid rgba(255, 255, 255, 0.35)",
                  boxShadow: "0 4px 14px rgba(0,0,0,0.5)",
                  letterSpacing: "-0.01em",
                }}
              >
                {feedbackSplash === "love" ? "Loved this!" : feedbackSplash === "like" ? "Liked!" : feedbackSplash === "dislike" ? "Not for me" : "Added to Watchlist"}
              </span>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Quick Actions on Poster on Hover */}
        {onQuickAction && (
          <div className="shelf-card-actions" style={{ zIndex: 12 }}>
            <div className="shelf-reaction-group">
              <div className="shelf-reaction-tray" role="group" aria-label="Reaction options">
                <button
                  type="button"
                  className="shelf-reaction-item"
                  aria-label={`Dislike ${movie.title}`}
                  onClick={(e) => handleActionClick(e, "dislike")}
                >
                  <span aria-hidden>🙁</span>
                  <span className="shelf-tooltip">Not for me</span>
                </button>
                <button
                  type="button"
                  className="shelf-reaction-item"
                  aria-label={`Like ${movie.title}`}
                  onClick={(e) => handleActionClick(e, "like")}
                >
                  <span aria-hidden>😀</span>
                  <span className="shelf-tooltip">I like this</span>
                </button>
                <button
                  type="button"
                  className="shelf-reaction-item"
                  aria-label={`Love ${movie.title}`}
                  onClick={(e) => handleActionClick(e, "love")}
                >
                  <span aria-hidden>😍</span>
                  <span className="shelf-tooltip">Love this!</span>
                </button>
              </div>

              <button
                type="button"
                className={`shelf-action-btn ${
                  ratingAction === "love"
                    ? "shelf-action-btn--loved"
                    : ratingAction === "like"
                    ? "shelf-action-btn--liked"
                    : ratingAction === "dislike"
                    ? "shelf-action-btn--disliked"
                    : "shelf-action-btn--reaction"
                }`}
                aria-label={`Rate ${movie.title}`}
                onClick={(e) => handleActionClick(e, "like")}
              >
                <span aria-hidden style={{ fontSize: 16 }}>
                  {ratingAction === "love" ? "😍" : ratingAction === "like" ? "😀" : ratingAction === "dislike" ? "🙁" : "😀"}
                </span>
                <span className="shelf-tooltip">
                  {ratingAction === "love" ? "Loved!" : ratingAction === "like" ? "Liked" : ratingAction === "dislike" ? "Not for me" : "Rate"}
                </span>
              </button>
            </div>
          </div>
        )}
      </div>

      <PosterInfo movie={movie} showFullDate={showFullDate} />
    </motion.div>
  );
}
