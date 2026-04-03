"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import { posterUrl, type Movie, type Recommendation } from "@/lib/api";

type MovieLike = Movie | Recommendation;

interface Props {
  movie: MovieLike;
  priority?: boolean;
  className?: string;
}

export default function MovieCard({ movie, priority = false, className = "" }: Props) {
  const poster = posterUrl(movie.poster_path, "w780");
  const year = movie.year || "";
  const genres = movie.genres?.slice(0, 3) || [];
  const rating = movie.vote_average ? movie.vote_average.toFixed(1) : null;
  const imdb = ("imdb_rating" in movie && movie.imdb_rating)
    ? (movie.imdb_rating as number).toFixed(1)
    : null;

  return (
    <motion.div
      layout
      className={`relative flex flex-col items-center no-select ${className}`}
    >
      {/* Poster */}
      <div className="relative w-full aspect-[2/3] rounded-2xl overflow-hidden poster-shadow bg-[var(--color-surface)]">
        <Image
          src={poster}
          alt={movie.title}
          fill
          priority={priority}
          sizes="(max-width: 640px) 85vw, 360px"
          className="object-cover"
          unoptimized
        />
      </div>

      {/* Info below poster */}
      <div className="mt-5 w-full text-center px-2">
        <h2 className="text-lg md:text-xl font-medium tracking-[-0.01em] text-[var(--color-text-primary)] leading-tight">
          {movie.title}
        </h2>

        <div className="mt-2 flex items-center justify-center gap-3 text-xs text-[var(--color-text-muted)] font-light">
          {year && <span>{year}</span>}
          {rating && (
            <span className="flex items-center gap-1">
              <span className="text-[var(--color-accent-warm)]">★</span>
              {rating}
            </span>
          )}
          {imdb && (
            <span className="text-[var(--color-text-secondary)]">
              IMDb {imdb}
            </span>
          )}
        </div>

        {genres.length > 0 && (
          <div className="mt-3 flex items-center justify-center gap-2 flex-wrap">
            {genres.map((g) => (
              <span
                key={g}
                className="
                  px-3 py-1 rounded-full
                  text-[10px] font-medium tracking-wide uppercase
                  text-[var(--color-text-secondary)]
                  border border-[var(--color-border)]
                "
              >
                {g}
              </span>
            ))}
          </div>
        )}

        {movie.overview && (
          <p className="mt-4 text-xs text-[var(--color-text-muted)] font-light leading-relaxed line-clamp-3 max-w-sm mx-auto">
            {movie.overview}
          </p>
        )}
      </div>
    </motion.div>
  );
}
