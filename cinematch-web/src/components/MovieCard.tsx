"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import { languageLabel, type Movie, type Recommendation } from "@/lib/api";
import { usePoster } from "@/lib/usePoster";

type MovieLike = Movie | Recommendation;

interface Props {
  movie: MovieLike;
  priority?: boolean;
  className?: string;
  compact?: boolean;
}

export default function MovieCard({ movie, priority = false, className = "", compact = false }: Props) {
  const poster = usePoster(movie.poster_path, movie.id, "w780");
  const year = movie.year || "";
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const genres = movie.genres?.slice(0, 2) || [];
  const primaryGenre = ("primary_genre" in movie && movie.primary_genre) ? movie.primary_genre as string : genres[0] || "";
  const imdb = ("imdb_rating" in movie && movie.imdb_rating)
    ? (movie.imdb_rating as number).toFixed(1)
    : null;
  const tmdbRating = movie.vote_average ? movie.vote_average.toFixed(1) : null;

  return (
    <motion.div
      layout
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
