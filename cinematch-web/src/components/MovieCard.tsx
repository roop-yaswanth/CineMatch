"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import {
  languageLabel,
  recommendationId,
  type Movie,
  type Recommendation,
} from "@/lib/api";
import { usePoster } from "@/lib/usePoster";

type MovieLike = Movie | Recommendation;

interface Props {
  movie: MovieLike;
  priority?: boolean;
  className?: string;
  compact?: boolean;
  overlay?: boolean;
}

export default function MovieCard({ movie, priority = false, className = "", compact = false, overlay = false }: Props) {
  const poster = usePoster(movie.poster_path, recommendationId(movie), "w780");
  const year = movie.year || "";
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const genres = movie.genres?.slice(0, 2) || [];
  const primaryGenre = ("primary_genre" in movie && movie.primary_genre) ? movie.primary_genre as string : genres[0] || "";
  const imdb = ("imdb_rating" in movie && movie.imdb_rating)
    ? (movie.imdb_rating as number).toFixed(1)
    : null;
  const tmdbRating = movie.vote_average ? movie.vote_average.toFixed(1) : null;

  // Overlay mode: info displayed on poster
  if (overlay) {
    return (
      <motion.div
        layout
        whileHover={{ scale: 1.03, y: -4 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        className={`relative flex flex-col items-center no-select ${className}`}
      >
        {/* Poster with overlay */}
        <div
          className="relative w-full aspect-[2/3] overflow-hidden bg-[var(--color-surface)] group"
          style={{ borderRadius: compact ? "14px" : "var(--radius-poster)" }}
        >
          <Image
            src={poster}
            alt={movie.title}
            fill
            priority={priority}
            sizes={compact ? "(max-width: 640px) 45vw, 20vw" : "(max-width: 640px) 85vw, 360px"}
            className="object-cover"
            unoptimized
          />
          
          {/* Gradient overlay at bottom */}
          <div className="absolute inset-x-0 bottom-0 h-2/5 bg-gradient-to-t from-black/90 via-black/60 to-transparent pointer-events-none" />
          
          {/* Rating badge top-right */}
          {(imdb || tmdbRating) && (
            <div className="absolute top-2 right-2 px-2 py-1 rounded-md bg-black/70 backdrop-blur-sm">
              {imdb ? (
                <span className="text-xs font-semibold text-yellow-400">
                  IMDb {imdb}
                </span>
              ) : (
                <span className="text-xs font-semibold text-yellow-400">
                  ★ {tmdbRating}
                </span>
              )}
            </div>
          )}
          
          {/* Info overlay at bottom */}
          <div className="absolute inset-x-0 bottom-0 p-3 text-white">
            <h2 className="text-sm font-semibold leading-tight line-clamp-2 drop-shadow-lg">
              {movie.title}
            </h2>
            <div className="mt-1 flex items-center gap-1.5 text-[11px] text-white/80 font-light">
              {year && <span>{year}</span>}
              {year && lang && <span style={{ opacity: 0.5 }}>·</span>}
              {lang && <span>{lang}</span>}
            </div>
            {(primaryGenre || genres.length > 0) && (
              <div className="mt-0.5 text-[10px] text-white/70 font-light truncate">
                {primaryGenre || genres.join(", ")}
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
      layout
      whileHover={{ scale: 1.03, y: -4 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
      className={`relative flex flex-col items-center no-select ${className}`}
    >
      {/* Poster */}
      <div
        className="relative w-full aspect-[2/3] overflow-hidden bg-[var(--color-surface)]"
        style={{ borderRadius: compact ? "14px" : "var(--radius-poster)" }}
      >
        <Image
          src={poster}
          alt={movie.title}
          fill
          priority={priority}
          sizes={compact ? "(max-width: 640px) 45vw, 20vw" : "(max-width: 640px) 85vw, 360px"}
          className="object-cover"
          unoptimized
        />
      </div>

      {/* Info below poster */}
      <div className={compact ? "mt-3 w-full px-1" : "mt-2 w-full text-center px-2"}>
        <h2
          className={compact
            ? "text-xs font-medium tracking-[-0.01em] text-[var(--color-text-primary)] leading-tight truncate"
            : "text-base font-medium tracking-[-0.01em] text-[var(--color-text-primary)] leading-tight"
          }
        >
          {movie.title}
        </h2>

        {/* Metadata line: Year · Language · IMDb 7.0 */}
        <div className={compact
          ? "mt-1 flex items-center gap-1.5 text-[10px] text-[var(--color-text-muted)] font-light flex-wrap"
          : "mt-2 flex items-center justify-center gap-2 text-xs text-[var(--color-text-muted)] font-light flex-wrap"
        }>
          {year && <span>{year}</span>}
          {year && lang && <span style={{ opacity: 0.4 }}>·</span>}
          {lang && <span>{lang}</span>}
          {(year || lang) && (imdb || tmdbRating) && <span style={{ opacity: 0.4 }}>·</span>}
          {imdb ? (
            <span>IMDb {imdb}</span>
          ) : tmdbRating ? (
            <span>
              <span style={{ color: "var(--color-accent-warm)" }}>★</span> {tmdbRating}
            </span>
          ) : null}
        </div>

        {/* Genre line */}
        {(primaryGenre || genres.length > 0) && (
          <div className={compact
            ? "mt-1 text-[10px] text-[var(--color-text-muted)] font-light"
            : "mt-1.5 text-xs text-[var(--color-text-muted)] font-light"
          }>
            {primaryGenre || genres.join(", ")}
          </div>
        )}

      </div>
    </motion.div>
  );
}
