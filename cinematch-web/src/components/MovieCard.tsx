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
  if (!imdb && !tmdbRating) return null;
  return (
    <span className="poster-rating-badge">
      {imdb ? `IMDb ${imdb}` : `★ ${tmdbRating}`}
    </span>
  );
}

/**
 * The one global "under the poster" block: two-line title with reserved
 * height, then a single "year · language" meta line at full width.
 * Every surface (explore grid, dashboard rails, onboarding slate, stack
 * detail) renders this, so cards never drift apart typographically.
 */
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

  return (
    <div className="mt-3 w-full px-0.5">
      <h2 className="poster-info-title">{movie.title}</h2>
      {fullDate && (
        <div className="mt-1 text-[11px] font-light text-[var(--color-text-muted)]">
          {fullDate}
        </div>
      )}
      {(year || lang) && (
        <div className="poster-info-meta">
          <span className="poster-info-meta-text">
            {[year, lang].filter(Boolean).join(" · ")}
          </span>
        </div>
      )}
    </div>
  );
}

export default function MovieCard({ movie, priority = false, className = "", compact = false, overlay = false, noLayout = false, showFullDate = false }: Props) {
  // Pick the smallest TMDB size that still looks crisp on the rendered card.
  // compact rails ≈ 130–140px, default cards ≈ 360px hero. Account for 2×/3×
  // DPR by going one size up.
  const posterSize = compact ? "w342" : "w500";
  const poster = usePoster(movie.poster_path, recommendationId(movie), posterSize);

  const year = movie.year ? movie.year.toString() : "";
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const imdb = ("imdb_rating" in movie && movie.imdb_rating)
    ? (movie.imdb_rating as number).toFixed(1)
    : null;
  const tmdbRating = movie.vote_average ? movie.vote_average.toFixed(1) : null;

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
                  {imdb ? `IMDb ${imdb}` : `★ ${tmdbRating}`}
                </span>
              </div>
            )}

            <h2 className="text-[13.5px] font-semibold leading-snug tracking-tight line-clamp-2 drop-shadow-lg">
              {movie.title}
            </h2>
            {(year || lang) && (
              <div className="mt-1 text-[11px] text-white/70 font-light">
                {[year, lang].filter(Boolean).join(" · ")}
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
      onHoverStart={() => prefetchMovieDetails(recommendationId(movie))}
      className={`relative flex flex-col items-center no-select ${className}`}
      style={{ position: "relative" }}
    >
      {/* Poster */}
      <div
        className="relative w-full aspect-[2/3] overflow-hidden bg-[var(--color-surface)]"
        style={{ borderRadius: compact ? "14px" : "var(--radius-poster)" }}
      >
        <img
          src={poster}
          alt={movie.title}
          loading={priority ? "eager" : "lazy"}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
        />
        <PosterRatingBadge movie={movie} />
      </div>

      <PosterInfo movie={movie} showFullDate={showFullDate} />
    </motion.div>
  );
}
