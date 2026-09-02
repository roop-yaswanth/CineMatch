import { languageLabel, type Movie, type Recommendation, type ExploreMovie } from "@/lib/api";

type MovieLike = Movie | Recommendation | ExploreMovie;

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

export function getStatusFromMovie(movie: MovieLike): { text: string; isUpcoming: boolean } | null {
  const NOW_MS = new Date().getTime();
  const CURRENT_YEAR = new Date().getFullYear();

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

  return { text: statusText, isUpcoming };
}

export function getRatingFromMovie(movie: MovieLike): { score: string; source: "imdb" | "tmdb" } | null {
  const imdb = ("imdb_rating" in movie && movie.imdb_rating)
    ? (movie.imdb_rating as number).toFixed(1)
    : null;
  const tmdbRating = movie.vote_average ? movie.vote_average.toFixed(1) : null;

  if (imdb) return { score: imdb, source: "imdb" };
  if (tmdbRating) return { score: tmdbRating, source: "tmdb" };
  return null;
}

export function getMetaParts(movie: MovieLike, showFullDate = false): string[] {
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

  return [year, lang, runtimeFormatted, cert].filter(Boolean);
}