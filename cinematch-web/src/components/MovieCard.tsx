"use client";

import { motion } from "framer-motion";

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

          {/* Gradient overlay at bottom */}
          <div className="absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-black/95 via-black/65 to-transparent pointer-events-none" />

          {/* Info overlay at bottom — same vocabulary as PosterInfo:
              title + one "year · lang" line + gold rating. */}
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

  // Default mode: info below poster (the global PosterInfo treatment)
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
        style={{ borderRadius: compact ? "14px" : "var(--radius-poster)" }}
      >
        <img
          src={poster}
          alt={movie.title}
          loading={priority ? "eager" : "lazy"}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
        />
        <PosterStatusBadge movie={movie} />
        <PosterRatingBadge movie={movie} />

        {/* Quick Actions on Poster on Hover */}
        {onQuickAction && (
          <div className="shelf-card-actions" style={{ zIndex: 12 }}>
            <div className="shelf-reaction-group">
              <div className="shelf-reaction-tray" role="group" aria-label="Reaction options">
                <button
                  type="button"
                  className="shelf-reaction-item"
                  aria-label={`Dislike ${movie.title}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onQuickAction(movie, "dislike");
                  }}
                >
                  <span aria-hidden>🙁</span>
                  <span className="shelf-tooltip">Not for me</span>
                </button>
                <button
                  type="button"
                  className="shelf-reaction-item"
                  aria-label={`Like ${movie.title}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onQuickAction(movie, "like");
                  }}
                >
                  <span aria-hidden>😀</span>
                  <span className="shelf-tooltip">I like this</span>
                </button>
                <button
                  type="button"
                  className="shelf-reaction-item"
                  aria-label={`Love ${movie.title}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onQuickAction(movie, "love");
                  }}
                >
                  <span aria-hidden>😍</span>
                  <span className="shelf-tooltip">Love this!</span>
                </button>
              </div>

              <button
                type="button"
                className="shelf-action-btn shelf-action-btn--reaction"
                aria-label={`Rate ${movie.title}`}
                onClick={(e) => {
                  e.stopPropagation();
                  onQuickAction(movie, "like");
                }}
              >
                <span aria-hidden style={{ fontSize: 16 }}>😀</span>
                <span className="shelf-tooltip">Rate</span>
              </button>
            </div>

            <button
              type="button"
              className="shelf-action-btn shelf-action-btn--watchlist"
              aria-label={`Add ${movie.title} to watchlist`}
              onClick={(e) => {
                e.stopPropagation();
                onQuickAction(movie, "watchlist");
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
              </svg>
              <span className="shelf-tooltip">Add to Watchlist</span>
            </button>
          </div>
        )}
      </div>

      <PosterInfo movie={movie} showFullDate={showFullDate} />
    </motion.div>
  );
}
