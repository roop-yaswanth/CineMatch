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

export default function MovieCard({ movie, priority = false, className = "", compact = false, overlay = false, noLayout = false, showFullDate = false }: Props) {
  // Pick the smallest TMDB size that still looks crisp on the rendered card.
  // compact rails ≈ 130–140px, default cards ≈ 360px hero. Account for 2×/3×
  // DPR by going one size up.
  const posterSize = compact ? "w342" : "w500";
  const poster = usePoster(movie.poster_path, recommendationId(movie), posterSize);
  
  let fullDate = "";
  if (showFullDate && "release_date" in movie && movie.release_date) {
    const date = new Date(movie.release_date);
    if (!isNaN(date.getTime())) {
      fullDate = date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    }
  }
  const year = fullDate ? "" : (movie.year ? movie.year.toString() : "");
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
        layout={!noLayout}
        whileHover={{ scale: 1.03, y: -4 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        className={`relative flex flex-col items-center no-select ${className}`}
      >
        {/* Poster with overlay */}
        <div
          className="relative w-full aspect-[2/3] overflow-hidden bg-[var(--color-surface)] group"
          style={{ borderRadius: compact ? "14px" : "var(--radius-poster)" }}
        >
          <img
            src={poster}
            alt={movie.title}
            loading={priority ? "eager" : "lazy"}
            style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
          />
          
          {/* Gradient overlay at bottom */}
          <div className="absolute inset-x-0 bottom-0 h-2/5 bg-gradient-to-t from-black/90 via-black/60 to-transparent pointer-events-none" />

          {/* Info overlay at bottom */}
          <div className="absolute inset-x-0 bottom-0 p-3 text-white">
            {/* Rating on top of info */}
            {(imdb || tmdbRating) && (
              <div className="mb-2">
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
      </div>

      {/* Info below poster */}
      <div className={compact ? "mt-3 w-full px-1" : "mt-3 w-full text-left px-0.5"}>
        <h2
          className={compact
            // min-h reserves two lines so the metadata row lines up across every
            // card in the rail whether a title wraps to one line or two (fixes
            // the ragged look where short titles sat higher).
            ? "text-[13px] font-semibold tracking-tight text-white leading-snug line-clamp-2 min-h-[2.7em]"
            : "text-[13.5px] font-semibold tracking-tight text-white leading-snug line-clamp-2 min-h-[2.5em]"
          }
        >
          {movie.title}
        </h2>

        {compact ? (
          // One tidy row instead of wrapping pills: "year · lang" muted on the
          // left, a single gold rating chip pushed to the right. nowrap + a
          // fixed min-height keep every card's row the same height.
          <div className="mt-2 flex items-center gap-1.5 min-h-[22px]">
            {(year || lang) && (
              <span className="min-w-0 truncate text-[10.5px] font-medium text-white/55">
                {[year, lang].filter(Boolean).join(" · ")}
              </span>
            )}
            {imdb ? (
              <span className="ml-auto shrink-0 rounded bg-[#e8c84a]/15 px-1.5 py-0.5 text-[10px] font-bold text-[#e8c84a]">
                IMDb {imdb}
              </span>
            ) : tmdbRating ? (
              <span className="ml-auto shrink-0 rounded bg-[#e8c84a]/15 px-1.5 py-0.5 text-[10px] font-bold text-[#e8c84a]">
                ★ {tmdbRating}
              </span>
            ) : null}
          </div>
        ) : (
          <>
            {fullDate && (
              <div className="mt-1 text-[11px] text-[var(--color-text-muted)] font-light">
                {fullDate}
              </div>
            )}

            {/* Metadata line: Year badge, Language badge, IMDb/TMDB rating chip */}
            <div className="mt-2 flex items-center gap-1.5 flex-wrap min-h-[22px]">
              {year && (
                <span className="rounded bg-white/[0.07] border border-white/10 px-1.5 py-0.5 text-[10.5px] font-medium text-zinc-300 leading-none">
                  {year}
                </span>
              )}
              {lang && (
                <span className="rounded bg-white/[0.07] border border-white/10 px-1.5 py-0.5 text-[10.5px] font-medium text-zinc-300 leading-none">
                  {lang}
                </span>
              )}
              {imdb ? (
                <span className="ml-auto shrink-0 rounded bg-amber-400/15 border border-amber-400/30 px-1.5 py-0.5 text-[10.5px] font-bold text-amber-300 flex items-center gap-0.5 leading-none">
                  <span className="text-amber-400 text-[10px]">★</span> {imdb}
                </span>
              ) : tmdbRating ? (
                <span className="ml-auto shrink-0 rounded bg-amber-400/15 border border-amber-400/30 px-1.5 py-0.5 text-[10.5px] font-bold text-amber-300 flex items-center gap-0.5 leading-none">
                  <span className="text-amber-400 text-[10px]">★</span> {tmdbRating}
                </span>
              ) : null}
            </div>

            {/* Genre line: Accent Pill / Tag */}
            {(primaryGenre || genres.length > 0) && (
              <div className="mt-1.5 flex items-center gap-1 flex-wrap">
                <span className="rounded-full bg-purple-500/10 border border-purple-500/20 px-2 py-0.5 text-[10px] font-medium text-purple-300/90 truncate max-w-full leading-tight">
                  {primaryGenre || genres.join(", ")}
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}
