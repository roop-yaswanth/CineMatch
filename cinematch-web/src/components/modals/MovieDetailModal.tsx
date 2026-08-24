import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState, useRef } from "react";

import { createPortal } from "react-dom";
import { posterUrl, languageLabel, apiSimilarMovies, apiCredits, apiImdbTitle, type Recommendation, type CastMember, type CrewMember, type ImdbTitle } from "@/lib/api";
import { PersonDetailOverlay } from "./PersonDetailOverlay";
import WatchProvidersPanel, { REGION_TO_COUNTRY, fetchWatchProviders } from "@/components/WatchProvidersPanel";
import { pushBackHandler } from "@/lib/backStack";

const NOW_MS = new Date().getTime();

export interface DetailMovie {
  id: number;
  tmdb_id?: number;
  title: string;
  poster_path?: string;
  backdrop_path?: string;
  year?: number | string;
  release_date?: string | null;
  status?: string | null;
  certification?: string | null;
  original_language?: string;
  imdb_id?: string;
  imdb_rating?: number;
  imdb_votes?: number;
  vote_average?: number;
  vote_count?: number;
  genres?: string[];
  primary_genre?: string;
  overview?: string;
  director?: string;
  runtime?: number | null;
  score?: number;
  reason?: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  movie: DetailMovie | null;
  onAction?: (action: "like" | "okay" | "dislike" | "watchlist" | "skip") => void;
  onMovieSelect?: (movie: DetailMovie) => void;
  sessionId?: string | null;
  userRegion?: string | null;
}

export default function MovieDetailModal({ isOpen, onClose, movie, onAction, onMovieSelect, sessionId, userRegion }: Props) {
  const [showWatchProviders, setShowWatchProviders] = useState(false);
  const [successAction, setSuccessAction] = useState<string | null>(null);
  const [similar, setSimilar] = useState<Recommendation[]>([]);
  const [similarLoading, setSimilarLoading] = useState(false);
  const [cast, setCast] = useState<CastMember[]>([]);
  const [directors, setDirectors] = useState<CrewMember[]>([]);
  const [writers, setWriters] = useState<CrewMember[]>([]);
  const [logoPath, setLogoPath] = useState<string | null>(null);
  const [englishPosterPath, setEnglishPosterPath] = useState<string | null>(null);
  const [runtimeLive, setRuntimeLive] = useState<number | null>(null);
  const [certificationLive, setCertificationLive] = useState<string | null>(null);
  const [statusLive, setStatusLive] = useState<string | null>(null);
  const [creditsLoading, setCreditsLoading] = useState(false);
  const [trailerKey, setTrailerKey] = useState<string | null>(null);
  const [trailerLanguages, setTrailerLanguages] = useState<Array<{ lang: string; label: string; key: string }>>([]);
  const [selectedTrailerLang, setSelectedTrailerLang] = useState<string | null>(null);
  const [trailerLoading, setTrailerLoading] = useState(false);
  const [trailerFetched, setTrailerFetched] = useState(false);
  const [showTrailerPlayer, setShowTrailerPlayer] = useState(false);
  const [activePersonId, setActivePersonId] = useState<number | null>(null);
  const [isMobile, setIsMobile] = useState(false);
  const [imdbLive, setImdbLive] = useState<ImdbTitle | null>(null);
  const similarRowRef = useRef<HTMLDivElement>(null);
  const castRowRef = useRef<HTMLDivElement>(null);

  // Treat phones + small tablets as compact layout
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 900);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Prevent body scroll when open; reset trailer & watch providers on close
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
      /* eslint-disable react-hooks/set-state-in-effect */
      setShowWatchProviders(false);
      setShowTrailerPlayer(false);
      setTrailerFetched(false);
      setTrailerKey(null);
      setSelectedTrailerLang(null);
      setTrailerLanguages([]);
      setActivePersonId(null);
      setEnglishPosterPath(null);
      setRuntimeLive(null);
      setCertificationLive(null);
      setStatusLive(null);
      /* eslint-enable react-hooks/set-state-in-effect */
    }
    return () => {
      document.body.style.overflow = "auto";
    };
  }, [isOpen]);

  // Keep a stable ref to onClose
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  // PWA back-gesture
  useEffect(() => {
    if (!isOpen) return;
    const cleanup = pushBackHandler(() => onCloseRef.current());
    return cleanup;
  }, [isOpen]);

  // Fetch similar movies whenever the movie changes
  const genresKey = Array.isArray(movie?.genres) ? movie.genres.join(",") : (movie?.primary_genre || "");

  useEffect(() => {
    const id = movie?.tmdb_id ?? movie?.id;
    if (!isOpen || !id) {
      /* eslint-disable react-hooks/set-state-in-effect */
      setSimilar([]);
      setSimilarLoading(false);
      /* eslint-enable react-hooks/set-state-in-effect */
      return;
    }

    setSimilarLoading(true);
    let cancelled = false;
    const seedYear = movie?.year != null ? Number(movie.year) || undefined : undefined;
    apiSimilarMovies(id, sessionId ?? null, 20, {
      title: movie?.title,
      overview: movie?.overview,
      genres: movie?.genres,
      lang: movie?.original_language,
      year: seedYear,
    })
      .then((results) => {
        if (!cancelled) setSimilar(results);
      })
      .catch((err) => {
        console.warn("Similar movies fetch failed:", err);
        if (!cancelled) setSimilar([]);
      })
      .finally(() => {
        if (!cancelled) setSimilarLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [movie?.id, movie?.tmdb_id, movie?.title, movie?.overview, movie?.genres, genresKey, movie?.original_language, movie?.year, isOpen, sessionId]);

  // Fetch cast & crew whenever the movie changes.
  useEffect(() => {
    const id = movie?.tmdb_id ?? movie?.id;
    if (!isOpen || !id) {
      /* eslint-disable react-hooks/set-state-in-effect */
      setCast([]); setDirectors([]); setWriters([]);
      setActivePersonId(null);
      setLogoPath(null);
      setEnglishPosterPath(null);
      setRuntimeLive(null);
      setCertificationLive(null);
      setStatusLive(null);
      /* eslint-enable react-hooks/set-state-in-effect */
      return;
    }
    let cancelled = false;
    setCreditsLoading(true);
    setCast([]); setDirectors([]); setWriters([]); setLogoPath(null); setEnglishPosterPath(null);
    setRuntimeLive(null); setCertificationLive(null); setStatusLive(null);

    apiCredits(id, "movie")
      .then((c) => {
        if (cancelled) return;
        setCast(c.cast);
        setDirectors(c.directors);
        setWriters(c.writers);
        setLogoPath(c.logo_path || null);
        setEnglishPosterPath(c.poster_path || null);
        if (c.runtime) setRuntimeLive(c.runtime);
        if (c.certification) setCertificationLive(c.certification);
        if (c.status) setStatusLive(c.status);
      })
      .catch(() => { })
      .finally(() => { if (!cancelled) setCreditsLoading(false); });
    return () => { cancelled = true; };
  }, [movie?.id, movie?.tmdb_id, isOpen]);

  // Fetch live IMDB rating on open.
  useEffect(() => {
    const id = movie?.tmdb_id ?? movie?.id;
    if (!isOpen || (!id && !movie?.imdb_id)) {
      setTimeout(() => { setImdbLive(null); }, 0);
      return;
    }
    let cancelled = false;
    setTimeout(() => { setImdbLive(null); }, 0);
    apiImdbTitle({ tmdbId: id, imdbId: movie?.imdb_id })
      .then((info) => {
        if (cancelled || !info) return;
        setImdbLive(info);
      })
      .catch(() => { });
    return () => { cancelled = true; };
  }, [movie?.id, movie?.tmdb_id, movie?.imdb_id, isOpen]);

  // Eagerly prefetch watch providers on modal open
  useEffect(() => {
    const id = movie?.tmdb_id ?? movie?.id;
    if (!isOpen || !id) return;
    fetchWatchProviders(id).catch(() => {});
  }, [movie?.id, movie?.tmdb_id, isOpen]);

  const handleActionClick = (action: "like" | "okay" | "dislike" | "watchlist" | "skip") => {
    if (!onAction) return;
    onAction(action);
    setSuccessAction(action);
    setTimeout(() => setSuccessAction(null), 2500);
  };

  const handleWatchTrailer = (langKey?: string) => {
    if (langKey) { setShowTrailerPlayer(true); setSelectedTrailerLang(langKey); return; }
    if (trailerFetched && !trailerLoading) {
      if (trailerKey) { setShowTrailerPlayer(true); }
      return;
    }
    const id = movie?.tmdb_id ?? movie?.id;
    if (!id) return;
    setTrailerLoading(true);
    fetch(`/api/tmdb-trailer?id=${id}`)
      .then((r) => r.json())
      .then((d: { key: string | null; languages: Array<{ lang: string; label: string; key: string }> }) => {
        const langs = d.languages ?? [];
        setTrailerLanguages(langs);
        setTrailerFetched(true);
        const origLang = movie?.original_language;
        const preferred = langs.find((l) => l.lang === origLang) ?? langs[0];
        const key = preferred?.key ?? d.key ?? null;
        setTrailerKey(key);
        setSelectedTrailerLang(key);
        if (key && langs.length <= 1) setShowTrailerPlayer(true);
      })
      .catch(() => setTrailerFetched(true))
      .finally(() => setTrailerLoading(false));
  };

  if (!movie) return null;

  // Hardcoded filter logic for "More Like This" based on the current movie
  let curatedSimilar = similar;
  const currentRating = movie.imdb_rating || movie.vote_average || 0;
  const isFamily = movie.genres?.some(g => g.toLowerCase() === 'family') || movie.primary_genre?.toLowerCase() === 'family';
  const MIN_SIMILAR = 4;

  if (currentRating >= 6.0) {
    const minR = 6.0;
    const rated = curatedSimilar.filter((m) => (m.imdb_rating || m.vote_average || 0) >= minR);
    if (rated.length >= MIN_SIMILAR) {
      curatedSimilar = rated;
    }
  }

  if (isFamily) {
    const familyPool = curatedSimilar.filter((m) => {
      const g = m.genres || (m.primary_genre ? [m.primary_genre] : []);
      return !g.some((x) => x.toLowerCase() === "horror");
    });
    if (familyPool.length >= MIN_SIMILAR) {
      curatedSimilar = familyPool;
    }
  }

  const filteredSimilar = curatedSimilar;

  const poster = posterUrl(englishPosterPath || movie.poster_path, "w780");
  const bgImage = movie.backdrop_path
    ? `https://image.tmdb.org/t/p/w1280${movie.backdrop_path}`
    : poster;

  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const year = movie.year || "";
  const genres = movie.genres?.join(", ") || movie.primary_genre || "";
  const imdb = imdbLive?.rating != null
    ? imdbLive.rating.toFixed(1)
    : movie.imdb_rating ? movie.imdb_rating.toFixed(1)
      : movie.vote_average ? movie.vote_average.toFixed(1) : null;
  const overview = movie.overview || "No overview available.";
  
  const effectiveMins =
    movie.runtime ?? runtimeLive ?? (imdbLive?.runtime ? parseInt(imdbLive.runtime, 10) : null);
  const runtime = effectiveMins && effectiveMins > 0 ? `${Math.floor(effectiveMins / 60)}h ${effectiveMins % 60}m` : null;
  const certification = certificationLive || movie.certification || null;
  const status = statusLive || movie.status || null;
  const isUpcoming =
    status === "Upcoming" ||
    status === "Post Production" ||
    status === "In Production" ||
    status === "Planned" ||
    (Boolean(movie.release_date) && new Date(movie.release_date!).getTime() > NOW_MS);

  const matchPct = movie.score !== undefined && movie.score >= 0.70 ? Math.round(movie.score * 100) : null;
  const matchColor = movie.score !== undefined && movie.score >= 0.85 ? "var(--color-success)" : "var(--color-yellow)";
  const guessedCountry = userRegion ? (REGION_TO_COUNTRY[userRegion] ?? "US") : "US";
  const hasTmdb = (movie.tmdb_id ?? movie.id) > 0;

  const whereToWatchIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect width="20" height="14" x="2" y="3" rx="3" />
      <path d="M8 21h8" />
      <path d="M12 17v4" />
    </svg>
  );

  /* ── Metadata & Title ── */

  const metaLine = (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center", color: "rgba(255, 255, 255, 0.7)", fontSize: "13.5px", fontWeight: 500 }}>
      {isUpcoming ? (
        <span style={{ padding: "2px 8px", borderRadius: "999px", background: "rgba(99, 102, 241, 0.25)", border: "1px solid rgba(99, 102, 241, 0.45)", color: "#a5b4fc", fontSize: "11px", fontWeight: 700 }}>
          Upcoming {year ? `(${year})` : ""}
        </span>
      ) : year ? (
        <span>{year}</span>
      ) : null}

      {certification && (
        <span style={{ padding: "1px 6px", borderRadius: "4px", border: "1px solid rgba(255, 255, 255, 0.25)", fontSize: "11px", fontWeight: 600, color: "rgba(255, 255, 255, 0.85)" }}>
          {certification}
        </span>
      )}

      {lang && <span style={{ opacity: 0.4 }}>•</span>}
      {lang && <span>{lang}</span>}

      {runtime && <span style={{ opacity: 0.4 }}>•</span>}
      {runtime && <span>{runtime}</span>}

      {imdb && <span style={{ opacity: 0.4 }}>•</span>}
      {imdb && (
        <span style={{ color: "#facc15", fontWeight: 700, display: "inline-flex", alignItems: "center", gap: "4px" }}>
          ★ {imdb}
        </span>
      )}
    </div>
  );

  const titleBlock = logoPath ? (
    <div style={{ marginBottom: "8px", marginTop: "2px" }}>
      <img
        src={posterUrl(logoPath, "w500")}
        alt={movie.title}
        style={{
          maxHeight: isMobile ? "52px" : "68px",
          maxWidth: isMobile ? "220px" : "320px",
          width: "auto",
          height: "auto",
          objectFit: "contain",
          filter: "drop-shadow(0 2px 8px rgba(0,0,0,0.6))",
        }}
        onError={() => setLogoPath(null)}
      />
    </div>
  ) : (
    <h2 style={{
      margin: "0 0 6px 0",
      fontSize: isMobile ? "clamp(22px, 5.5vw, 26px)" : "32px",
      fontWeight: 700,
      letterSpacing: "-0.02em",
      lineHeight: 1.15,
      color: "#ffffff",
    }}>
      {movie.title}
    </h2>
  );

  const genresRow = genres ? (
    <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
      {genres.split(",").map((g) => (
        <span
          key={g}
          style={{
            padding: "4px 12px",
            borderRadius: "999px",
            background: "rgba(255, 255, 255, 0.08)",
            border: "1px solid rgba(255, 255, 255, 0.12)",
            fontSize: "12px",
            fontWeight: 500,
            color: "rgba(255, 255, 255, 0.85)",
          }}
        >
          {g.trim()}
        </span>
      ))}
    </div>
  ) : null;

  /* ── Watch Trailer Button ── */
  const notFound = trailerFetched && !trailerLoading && !trailerKey;
  const isMultiLang = trailerLanguages.length > 1;
  const trailerNode = trailerLoading ? (
    <div style={{
      width: "100%", height: "44px", borderRadius: "999px",
      background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.15)",
      display: "flex", alignItems: "center", justifyContent: "center", gap: "10px",
      color: "#fff", fontSize: "14px", fontWeight: 600,
    }}>
      <div style={{
        width: "14px", height: "14px", borderRadius: "50%",
        border: "2px solid rgba(255,255,255,0.2)",
        borderTopColor: "#fff",
        animation: "spin 0.8s linear infinite",
      }} />
      <style>{"@keyframes spin{to{transform:rotate(360deg)}}"}</style>
      <span>Finding trailer…</span>
    </div>
  ) : notFound ? (
    <div style={{
      width: "100%", height: "44px", borderRadius: "999px",
      background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
      display: "flex", alignItems: "center", justifyContent: "center",
      color: "rgba(255,255,255,0.45)", fontSize: "13px",
    }}>
      No trailer available
    </div>
  ) : isMultiLang ? (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      <button
        type="button"
        onClick={() => handleWatchTrailer()}
        style={{
          width: "100%", height: "46px", borderRadius: "999px",
          background: "#ffffff", border: "none",
          color: "#000000", fontSize: "14.5px", fontWeight: 700,
          display: "flex", alignItems: "center", justifyContent: "center", gap: "8px",
          cursor: "pointer", boxShadow: "0 4px 14px rgba(0,0,0,0.3)",
          transition: "transform 0.12s ease, filter 0.12s ease",
        }}
        onPointerDown={(e) => { (e.currentTarget as HTMLElement).style.transform = "scale(0.98)"; }}
        onPointerUp={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3" />
        </svg>
        <span>Watch Trailer</span>
      </button>
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        <span style={{ fontSize: "12px", color: "rgba(255, 255, 255, 0.6)", fontWeight: 600, letterSpacing: "0.02em" }}>
          Trailer Languages / Dubs:
        </span>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
          {trailerLanguages.map((tl) => {
            const isActive = tl.key === selectedTrailerLang;
            return (
              <button
                key={tl.lang}
                onClick={() => handleWatchTrailer(tl.key)}
                style={{
                  padding: "7px 16px", borderRadius: "999px",
                  background: isActive ? "rgba(255, 255, 255, 0.26)" : "rgba(255, 255, 255, 0.08)",
                  border: isActive ? "1px solid rgba(255, 255, 255, 0.45)" : "1px solid rgba(255, 255, 255, 0.14)",
                  color: "#ffffff", fontSize: "13.5px", fontWeight: isActive ? 700 : 500,
                  cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "6px",
                  transition: "all 0.15s ease",
                }}
              >
                {tl.label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  ) : (
    <button
      type="button"
      onClick={() => handleWatchTrailer()}
      style={{
        width: "100%", height: "44px", borderRadius: "999px",
        background: "#ffffff", border: "none",
        color: "#000000", fontSize: "14px", fontWeight: 700,
        display: "flex", alignItems: "center", justifyContent: "center", gap: "8px",
        cursor: "pointer", boxShadow: "0 4px 14px rgba(0,0,0,0.3)",
        transition: "transform 0.12s ease, filter 0.12s ease",
      }}
      onPointerDown={(e) => { (e.currentTarget as HTMLElement).style.transform = "scale(0.98)"; }}
      onPointerUp={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
    >
      <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5 3 19 12 5 21 5 3" />
      </svg>
      <span>Watch Trailer</span>
    </button>
  );

  const overviewNode = (
    <p style={{ margin: 0, fontSize: "13px", lineHeight: 1.6, color: "rgba(255, 255, 255, 0.78)" }}>
      {overview}
    </p>
  );

  const reasonNode = movie.reason ? (
    <div style={{
      padding: "12px 14px", borderRadius: "12px",
      background: "rgba(255, 255, 255, 0.05)",
      border: "1px solid rgba(255, 255, 255, 0.08)",
    }}>
      <p style={{ margin: "0 0 4px", fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255, 255, 255, 0.45)" }}>
        Why recommended
      </p>
      <p style={{ margin: 0, fontSize: "12.5px", fontStyle: "italic", color: "rgba(255, 255, 255, 0.8)" }}>
        {movie.reason}
      </p>
    </div>
  ) : null;

  const creditsRow = (directors.length > 0 || writers.length > 0) ? (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "20px 32px", fontSize: "13px" }}>
      {directors.length > 0 && (
        <div>
          <div style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255, 255, 255, 0.45)", marginBottom: "3px" }}>
            {directors.length > 1 ? "DIRECTORS" : "DIRECTOR"}
          </div>
          <div style={{ color: "#ffffff", fontWeight: 600, fontSize: "13px" }}>
            {directors.map((d, i) => (
              <span key={`${d.id}-${i}`}>
                <button
                  onClick={() => setActivePersonId(d.id)}
                  style={{ padding: 0, background: "none", border: "none", cursor: "pointer", color: "inherit", textDecoration: "none", fontSize: "inherit", fontFamily: "inherit", fontWeight: "inherit" }}
                >
                  {d.name}
                </button>
                {i < directors.length - 1 ? ", " : ""}
              </span>
            ))}
          </div>
        </div>
      )}
      {writers.length > 0 && (
        <div>
          <div style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255, 255, 255, 0.45)", marginBottom: "3px" }}>
            {writers.length > 1 ? "WRITERS" : "WRITER"}
          </div>
          <div style={{ color: "#ffffff", fontWeight: 600, fontSize: "13px" }}>
            {(() => {
              const seen = new Set<number>();
              const uniq = writers.filter((w) => {
                if (seen.has(w.id)) return false;
                seen.add(w.id);
                return true;
              }).slice(0, 4);
              return uniq.map((w, i) => (
                <span key={`${w.id}-${i}`}>
                  <button
                    onClick={() => setActivePersonId(w.id)}
                    style={{ padding: 0, background: "none", border: "none", cursor: "pointer", color: "inherit", textDecoration: "none", fontSize: "inherit", fontFamily: "inherit", fontWeight: "inherit" }}
                  >
                    {w.name}
                  </button>
                  {i < uniq.length - 1 ? ", " : ""}
                </span>
              ));
            })()}
          </div>
        </div>
      )}
    </div>
  ) : null;

  /* ── Rate this movie buttons ── */
  const rateCardButton = (
    action: "like" | "okay" | "dislike",
    label: string,
    color: string,
    icon: React.ReactNode,
  ) => (
    <button
      onClick={() => handleActionClick(action)}
      aria-label={label}
      title={label}
      style={{
        flex: 1,
        height: "46px",
        borderRadius: "14px",
        background: "rgba(255, 255, 255, 0.08)",
        border: "1px solid rgba(255, 255, 255, 0.12)",
        color: color,
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transition: "background 0.15s, transform 0.1s, border-color 0.15s",
      }}
      onPointerDown={(e) => { (e.currentTarget as HTMLElement).style.transform = "scale(0.94)"; }}
      onPointerUp={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
      onPointerLeave={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
    >
      <AnimatePresence mode="wait">
        {successAction === action ? (
          <motion.svg key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></motion.svg>
        ) : (
          <motion.span key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex" }}>
            {icon}
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );

  const pillAction = (
    action: "watchlist" | "skip",
    label: string,
    doneLabel: string,
    icon: React.ReactNode,
    flex: number = 1,
  ) => (
    <button
      onClick={() => handleActionClick(action)}
      style={{
        flex,
        height: "42px",
        borderRadius: "24px",
        background: "rgba(255, 255, 255, 0.08)",
        border: "1px solid rgba(255, 255, 255, 0.12)",
        color: "#ffffff",
        fontSize: "13px",
        fontWeight: 600,
        cursor: "pointer",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        gap: "7px",
        transition: "background 0.15s, transform 0.1s, border-color 0.15s",
      }}
      onPointerDown={(e) => { (e.currentTarget as HTMLElement).style.transform = "scale(0.95)"; }}
      onPointerUp={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
      onPointerLeave={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
    >
      <AnimatePresence mode="wait">
        {successAction === action ? (
          <motion.span key="done" initial={{ scale: 0.7, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.7, opacity: 0 }} style={{ display: "inline-flex", alignItems: "center", gap: "6px", color: "var(--color-success)" }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
            {doneLabel}
          </motion.span>
        ) : (
          <motion.span key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "inline-flex", alignItems: "center", gap: "7px" }}>
            {icon}
            {label}
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  );

  const rateSection = onAction ? (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      <h4 style={{ margin: "0 0 4px 0", fontSize: "11px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255, 255, 255, 0.5)" }}>
        RATE THIS MOVIE
      </h4>
      <div style={{ display: "flex", gap: "8px" }}>
        {rateCardButton("dislike", "Not for me", "#ef4444",
          <span style={{ fontSize: "20px" }}>🙁</span>)}
        {rateCardButton("okay", "Like", "#3b82f6",
          <span style={{ fontSize: "20px" }}>😀</span>)}
        {rateCardButton("like", "Love", "#f59e0b",
          <span style={{ fontSize: "20px" }}>😍</span>)}
      </div>
      <div style={{ display: "flex", gap: "8px" }}>
        {pillAction("watchlist", "Add to Watchlist", "Added",
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" /></svg>, 1.35)}
        {pillAction("skip", "Skip", "Skipped",
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 4 15 12 5 20 5 4" fill="currentColor"></polygon><line x1="19" y1="5" x2="19" y2="19"></line></svg>, 1)}
      </div>
    </div>
  ) : null;

  /* ── Cast section ── */
  const castSection = (creditsLoading || cast.length > 0) ? (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <h4 style={{ margin: 0, fontSize: "11px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255, 255, 255, 0.5)" }}>
          CAST
        </h4>
        {!creditsLoading && cast.length > 0 && (
          <div style={{ display: "flex", gap: "6px" }}>
            <button
              onClick={() => {
                const el = castRowRef.current;
                if (!el) return;
                el.scrollBy({ left: -260, behavior: "smooth" });
              }}
              aria-label="Scroll cast left"
              style={{ width: "26px", height: "26px", borderRadius: "50%", border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.8)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6" /></svg>
            </button>
            <button
              onClick={() => {
                const el = castRowRef.current;
                if (!el) return;
                el.scrollBy({ left: 260, behavior: "smooth" });
              }}
              aria-label="Scroll cast right"
              style={{ width: "26px", height: "26px", borderRadius: "50%", border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.8)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
            </button>
          </div>
        )}
      </div>
      {creditsLoading && cast.length === 0 ? (
        <div style={{ display: "flex", gap: "16px", overflowX: "hidden" }}>
          {Array.from({ length: 9 }).map((_, i) => (
            <div key={i} style={{ width: "68px", flexShrink: 0, display: "flex", flexDirection: "column", alignItems: "center" }}>
              <div className="skeleton-shimmer" style={{ width: "64px", height: "64px", borderRadius: "50%", marginBottom: "8px" }} />
              <div className="skeleton-shimmer" style={{ height: "10px", width: "80%", borderRadius: "999px", marginBottom: "4px" }} />
            </div>
          ))}
        </div>
      ) : (
        <div ref={castRowRef} style={{ display: "flex", gap: "16px", overflowX: "auto", scrollbarWidth: "none", msOverflowStyle: "none", paddingBottom: "4px" }}>
          {cast.map((c) => (
            <button
              key={c.id}
              onClick={() => setActivePersonId(c.id)}
              style={{
                width: "68px", minWidth: "68px", maxWidth: "68px",
                flexShrink: 0,
                display: "flex", flexDirection: "column", alignItems: "center",
                textAlign: "center", textDecoration: "none", color: "inherit",
                background: "none", border: "none", padding: 0, cursor: "pointer", outline: "none", fontFamily: "inherit",
              }}
            >
              <div style={{
                width: "64px", height: "64px", minWidth: "64px", minHeight: "64px",
                borderRadius: "50%", overflow: "hidden",
                background: "rgba(255,255,255,0.08)",
                border: "1px solid rgba(255,255,255,0.12)",
                marginBottom: "8px",
                position: "relative",
              }}>
                {c.profile_path ? (
                  <img
                    src={posterUrl(c.profile_path, "w185")}
                    alt={c.name}
                    style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
                  />
                ) : (
                  <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "rgba(255,255,255,0.4)", fontSize: "18px" }}>
                    {c.name?.[0] || "?"}
                  </div>
                )}
              </div>
              <div style={{ fontSize: "11px", color: "rgba(255, 255, 255, 0.9)", fontWeight: 500, lineHeight: 1.25, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
                {c.name}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  ) : null;

  /* ── More like this section ── */
  const similarSection = (similarLoading || similar.length > 0) ? (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <h4 style={{ margin: 0, fontSize: "11px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255, 255, 255, 0.5)" }}>
          MORE LIKE THIS
        </h4>
        {!similarLoading && filteredSimilar.length > 0 && (
          <div style={{ display: "flex", gap: "6px" }}>
            <button
              onClick={() => {
                const el = similarRowRef.current;
                if (!el) return;
                el.scrollBy({ left: -320, behavior: "smooth" });
              }}
              aria-label="Scroll similar movies left"
              style={{ width: "26px", height: "26px", borderRadius: "50%", border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.8)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6" /></svg>
            </button>
            <button
              onClick={() => {
                const el = similarRowRef.current;
                if (!el) return;
                el.scrollBy({ left: 320, behavior: "smooth" });
              }}
              aria-label="Scroll similar movies right"
              style={{ width: "26px", height: "26px", borderRadius: "50%", border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.8)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
            </button>
          </div>
        )}
      </div>

      {similarLoading && (
        <div style={{ display: "flex", gap: "12px", overflowX: "hidden" }}>
          {Array.from({ length: 9 }).map((_, i) => (
            <div key={i} style={{ flexShrink: 0, width: "84px" }}>
              <div className="skeleton-shimmer" style={{ width: "84px", paddingBottom: "126px", borderRadius: "10px" }} />
            </div>
          ))}
        </div>
      )}

      {!similarLoading && similar.length > 0 && filteredSimilar.length === 0 && (
        <div style={{ fontSize: "13px", color: "rgba(255,255,255,0.5)", padding: "12px 0", fontStyle: "italic" }}>
          No similar movies found in your selected languages.
        </div>
      )}
      {!similarLoading && filteredSimilar.length > 0 && (
        <div
          ref={similarRowRef}
          style={{ display: "flex", gap: "12px", overflowX: "auto", paddingBottom: "4px", msOverflowStyle: "none", scrollbarWidth: "none" }}
        >
          {filteredSimilar.map((m) => (
            <SimilarCard
              key={m.tmdb_id ?? m.id}
              movie={m}
              onClick={() => {
                if (onMovieSelect) {
                  onMovieSelect({
                    id: m.tmdb_id ?? m.id,
                    tmdb_id: m.tmdb_id,
                    title: m.title,
                    poster_path: m.poster_path,
                    backdrop_path: m.backdrop_path,
                    year: m.year,
                    original_language: m.original_language,
                    imdb_rating: m.imdb_rating,
                    vote_average: m.vote_average,
                    genres: m.genres,
                    primary_genre: m.primary_genre,
                    overview: m.overview,
                    director: m.director,
                    runtime: m.runtime,
                    score: m.score,
                    reason: m.reason,
                  });
                }
              }}
            />
          ))}
        </div>
      )}
    </div>
  ) : null;

  if (typeof document === "undefined") return null;

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <div style={{ position: "fixed", inset: 0, zIndex: 100, display: "flex", alignItems: isMobile ? "flex-start" : "center", justifyContent: "center" }}>

          {/* Fullscreen trailer overlay */}
          <AnimatePresence>
            {showTrailerPlayer && selectedTrailerLang && (
              <TrailerOverlay
                videoKey={selectedTrailerLang}
                title={movie.title}
                languages={trailerLanguages}
                selectedLanguage={selectedTrailerLang}
                onSelectLanguage={(k) => setSelectedTrailerLang(k)}
                onClose={() => setShowTrailerPlayer(false)}
              />
            )}
          </AnimatePresence>

          {/* Backdrop overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
            style={{
              position: "absolute", inset: 0,
              background: "rgba(0, 0, 0, 0.75)",
              backdropFilter: "blur(14px)",
              WebkitBackdropFilter: "blur(14px)",
            }}
          />

          {/* Modal Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 15 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 15 }}
            transition={{ type: "spring", damping: 26, stiffness: 320 }}
            style={{
              position: "relative",
              width: isMobile ? "100%" : "92vw",
              maxWidth: isMobile ? "100%" : "1080px",
              maxHeight: isMobile ? "100dvh" : "90vh",
              height: isMobile ? "100dvh" : "auto",
              background: "#101217",
              border: isMobile ? "none" : "1px solid rgba(255, 255, 255, 0.12)",
              borderRadius: isMobile ? "0" : "28px",
              boxShadow: "0 32px 80px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.08)",
              overflow: "hidden",
              color: "#ffffff",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {/* Ambient blurred backdrop — fixed to modal container so it stays uniform across the entire scroll area */}
            <div
              aria-hidden
              style={{
                position: "absolute",
                inset: 0,
                overflow: "hidden",
                borderRadius: "inherit",
                pointerEvents: "none",
                zIndex: 0,
              }}
            >
              <img
                src={bgImage}
                alt=""
                style={{
                  position: "absolute",
                  inset: "-20%",
                  width: "140%",
                  height: "140%",
                  objectFit: "cover",
                  filter: "blur(60px) brightness(0.24) saturate(1.35)",
                  transform: "scale(1.15)",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "radial-gradient(ellipse at center, rgba(16,18,24,0.55) 0%, rgba(10,12,16,0.92) 100%)",
                }}
              />
            </div>

            {/* Close button — top-right */}
            <button
              onClick={() => showWatchProviders ? setShowWatchProviders(false) : onClose()}
              aria-label={showWatchProviders ? "Close watch providers" : "Close"}
              style={{
                position: "absolute",
                top: isMobile ? "calc(env(safe-area-inset-top) + 16px)" : "22px",
                right: "22px",
                zIndex: 30,
                width: "36px",
                height: "36px",
                borderRadius: "50%",
                background: "rgba(255, 255, 255, 0.12)",
                border: "1px solid rgba(255, 255, 255, 0.16)",
                backdropFilter: "blur(8px)",
                WebkitBackdropFilter: "blur(8px)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#ffffff",
                cursor: "pointer",
                transition: "background 150ms ease, transform 120ms ease",
              }}
              onPointerDown={(e) => { (e.currentTarget as HTMLElement).style.transform = "scale(0.92)"; }}
              onPointerUp={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
              onPointerLeave={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>

            {/* Scrollable Content Viewport */}
            <div
              style={{
                position: "relative",
                zIndex: 1,
                width: "100%",
                maxHeight: isMobile ? "100dvh" : "90vh",
                overflowY: "auto",
                overscrollBehavior: "none",
              }}
            >
              {isMobile ? (

                <div style={{ padding: "calc(env(safe-area-inset-top) + 20px) 18px 36px", display: "flex", flexDirection: "column", gap: "16px" }}>
                  {/* Top 2-Column Header: Poster on Left, Title/Meta/Genres on Right */}
                  <div style={{ display: "flex", gap: "14px", alignItems: "flex-start" }}>
                    {/* Left: Clean unclipped Poster */}
                    <div
                      style={{
                        width: "115px",
                        minWidth: "115px",
                        aspectRatio: "2 / 3",
                        borderRadius: "14px",
                        overflow: "hidden",
                        boxShadow: "0 12px 28px rgba(0, 0, 0, 0.7)",
                        border: "1px solid rgba(255, 255, 255, 0.15)",
                        position: "relative",
                        background: "rgba(255, 255, 255, 0.05)",
                        flexShrink: 0,
                      }}
                    >
                      <img
                        src={poster}
                        alt={movie.title}
                        style={{ width: "100%", height: "100%", objectFit: "cover" }}
                      />
                    </div>

                    {/* Right: Title, Match %, Meta & Genres */}
                    <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "6px", paddingRight: "28px" }}>
                      {matchPct && (
                        <div>
                          <span style={{
                            display: "inline-block",
                            background: `${matchColor}22`,
                            border: `1px solid ${matchColor}55`,
                            color: matchColor,
                            borderRadius: "var(--radius-pill)",
                            padding: "2px 8px",
                            fontSize: "11px",
                            fontWeight: 700,
                          }}>
                            {matchPct}% Match
                          </span>
                        </div>
                      )}
                      {titleBlock}
                      {metaLine}
                      {genresRow && <div style={{ marginTop: "4px" }}>{genresRow}</div>}
                    </div>
                  </div>

                  {/* Primary Action: Trailer Button */}
                  {trailerNode}

                  {/* Secondary Action: Where to Watch button (under trailer) */}
                  {hasTmdb && (
                    <button
                      type="button"
                      onClick={() => setShowWatchProviders((s) => !s)}
                      style={{
                        width: "100%",
                        height: "42px",
                        borderRadius: "999px",
                        background: showWatchProviders ? "rgba(255, 255, 255, 0.15)" : "rgba(255, 255, 255, 0.07)",
                        border: showWatchProviders ? "1px solid rgba(255, 255, 255, 0.35)" : "1px solid rgba(255, 255, 255, 0.14)",
                        color: "#ffffff",
                        fontSize: "13px",
                        fontWeight: 600,
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: "8px",
                        transition: "all 0.15s ease",
                      }}
                      onPointerDown={(e) => { (e.currentTarget as HTMLElement).style.transform = "scale(0.98)"; }}
                      onPointerUp={(e) => { (e.currentTarget as HTMLElement).style.transform = ""; }}
                    >
                      {whereToWatchIcon}
                      <span>{showWatchProviders ? "Hide Where to Watch" : "Where to Watch"}</span>
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        style={{
                          transform: showWatchProviders ? "rotate(180deg)" : "rotate(0deg)",
                          transition: "transform 0.2s ease",
                        }}
                      >
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                    </button>
                  )}

                  {/* Expandable Where-to-Watch Panel on Mobile */}
                  <AnimatePresence initial={false}>
                    {showWatchProviders && hasTmdb && (
                      <motion.div
                        key="mobile-watch-panel"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{
                          height: { duration: 0.28, ease: [0.16, 1, 0.3, 1] },
                          opacity: { duration: 0.22, ease: "easeOut" },
                        }}
                        style={{ overflow: "hidden", willChange: "height, opacity" }}
                      >
                        <WatchProvidersPanel
                          tmdbId={(movie.tmdb_id ?? movie.id) as number}
                          defaultCountry={guessedCountry}
                          movieTitle={movie.title}
                        />
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Synopsis / Overview */}
                  {overviewNode}

                  {/* Why Recommended */}
                  {reasonNode}

                  {/* Rate / Interaction Section */}
                  {rateSection}

                  {/* Directors & Writers */}
                  {creditsRow}

                  {/* Cast Carousel */}
                  {castSection}

                  {/* More Like This Carousel */}
                  {similarSection}
                </div>
              ) : (

                <div style={{ padding: "34px 38px 36px", display: "flex", flexDirection: "column", gap: "26px" }}>
                  {/* ── Top Half: 2 Columns (Poster + Details) ── */}
                  <div style={{ display: "flex", gap: "34px", alignItems: "flex-start" }}>
                    {/* Left Column: Full Unclipped Rounded Poster Card with Where-To-Watch */}
                    <div style={{ flex: "0 0 290px", width: "290px" }}>
                      <div
                        style={{
                          width: "100%",
                          aspectRatio: "2 / 3",
                          borderRadius: "20px",
                          overflow: "hidden",
                          boxShadow: "0 20px 48px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.14)",
                          position: "relative",
                          background: "rgba(255,255,255,0.04)",
                        }}
                      >
                        <img
                          src={poster}
                          alt={movie.title}
                          loading="eager"
                          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
                        />

                        {/* Bottom scrim for Where to Watch button */}
                        <div style={{
                          position: "absolute", bottom: 0, left: 0, right: 0, height: "45%",
                          background: "linear-gradient(to top, rgba(0,0,0,0.75) 0%, rgba(0,0,0,0.2) 60%, transparent 100%)",
                          pointerEvents: "none",
                        }} />
                        {hasTmdb && (
                          <motion.button
                            onClick={() => setShowWatchProviders((s) => !s)}
                            whileHover={{ scale: 1.02, y: -1 }}
                            whileTap={{ scale: 0.98 }}
                            aria-expanded={showWatchProviders}
                            style={{
                              position: "absolute",
                              bottom: "14px", left: "14px", right: "14px",
                              padding: "11px 16px",
                              borderRadius: "14px",
                              fontSize: "13px",
                              fontWeight: 700,
                              color: "#ffffff",
                              cursor: "pointer",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              gap: "8px",
                              background: "linear-gradient(130deg, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0.18) 100%)",
                              border: "1px solid rgba(255, 255, 255, 0.3)",
                              backdropFilter: "blur(16px) saturate(1.4)",
                              WebkitBackdropFilter: "blur(16px) saturate(1.4)",
                              boxShadow: "0 8px 24px rgba(0, 0, 0, 0.35)",
                              zIndex: 5,
                            }}
                          >
                            {whereToWatchIcon}
                            <span>{showWatchProviders ? "Hide Providers" : "Where to Watch"}</span>
                          </motion.button>
                        )}

                        {/* Watch Providers Overlay inside Poster Card */}
                        <AnimatePresence>
                          {showWatchProviders && hasTmdb && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              exit={{ opacity: 0 }}
                              transition={{ duration: 0.18 }}
                              style={{
                                position: "absolute",
                                inset: 0,
                                background: "rgba(12, 14, 20, 0.92)",
                                backdropFilter: "blur(16px) saturate(1.4)",
                                WebkitBackdropFilter: "blur(16px) saturate(1.4)",
                                display: "flex",
                                flexDirection: "column",
                                padding: "16px 14px",
                                zIndex: 10,
                                overflowY: "auto",
                              }}
                            >
                              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "10px" }}>
                                <h4 style={{ margin: 0, fontSize: "12px", fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "#fff" }}>
                                  Where to Watch
                                </h4>
                                <button
                                  onClick={() => setShowWatchProviders(false)}
                                  style={{
                                    background: "rgba(255,255,255,0.1)",
                                    border: "none",
                                    color: "rgba(255,255,255,0.8)",
                                    cursor: "pointer",
                                    fontSize: "12px",
                                    fontWeight: 600,
                                    borderRadius: "50%",
                                    width: "22px",
                                    height: "22px",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                  }}
                                >
                                  ✕
                                </button>
                              </div>
                              <WatchProvidersPanel
                                tmdbId={(movie.tmdb_id ?? movie.id) as number}
                                defaultCountry={guessedCountry}
                                movieTitle={movie.title}
                              />
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </div>

                    {/* Right Column: Title, Info, Trailer, Overview, Rate */}
                    <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: "20px" }}>
                      {/* Title, Meta, Genres */}
                      <div>
                        {matchPct && (
                          <div style={{ marginBottom: "8px" }}>
                            <span style={{
                              display: "inline-block",
                              background: `${matchColor}22`,
                              border: `1px solid ${matchColor}55`,
                              color: matchColor,
                              borderRadius: "var(--radius-pill)",
                              padding: "3px 12px",
                              fontSize: "12px",
                              fontWeight: 700,
                              letterSpacing: "0.02em",
                            }}>
                              {matchPct}% Match
                            </span>
                          </div>
                        )}
                        {titleBlock}
                        {metaLine}
                        {genresRow && <div style={{ marginTop: "12px" }}>{genresRow}</div>}
                      </div>

                      {/* Middle grid: Trailer/Overview/Credits on left + Rate box on right with subtle divider line */}
                      <div style={{ display: "grid", gridTemplateColumns: onAction ? "1.2fr 1fr" : "1fr", gap: "28px", alignItems: "stretch" }}>
                        {/* Left sub-column: Trailer, Overview, Reason, Credits */}
                        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                          {trailerNode}
                          {overviewNode}
                          {reasonNode}
                          {creditsRow}
                        </div>

                        {/* Right sub-column: Rate box with subtle vertical divider line */}
                        {rateSection && (
                          <div style={{
                            borderLeft: "1px solid rgba(255, 255, 255, 0.12)",
                            paddingLeft: "28px",
                            display: "flex",
                            flexDirection: "column",
                            justifyContent: "flex-start",
                          }}>
                            {rateSection}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* ── Bottom Half: FULL-WIDTH Cast & More Like This across the entire modal width ── */}
                  <div style={{ display: "flex", flexDirection: "column", gap: "24px", paddingTop: "8px", borderTop: "1px solid rgba(255,255,255,0.06)" }}>
                    {/* Cast Section (full width) */}
                    {castSection}

                    {/* More Like This Section (full width) */}
                    {similarSection}
                  </div>
                </div>
              )}
            </div>

            <AnimatePresence>
              {activePersonId && (
                <PersonDetailOverlay
                  personId={activePersonId}
                  onClose={() => setActivePersonId(null)}
                  onSelectMovie={(m) => {
                    setActivePersonId(null);
                    if (onMovieSelect) onMovieSelect(m);
                  }}
                />
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body
  );
}

/* ── Fullscreen Trailer Overlay ──────────────────────
 * Rendered in a separate portal so it sits above the movie modal.
 * Unmounting the iframe stops playback instantly.
 * ─────────────────────────────────────────────────── */
export function TrailerOverlay({
  videoKey,
  title,
  languages,
  selectedLanguage,
  onSelectLanguage,
  onClose,
}: {
  videoKey: string;
  title: string;
  languages?: Array<{ lang: string; label: string; key: string }>;
  selectedLanguage?: string | null;
  onSelectLanguage?: (key: string) => void;
  onClose: () => void;
}) {
  // ESC key closes
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    setMounted(true);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []);

  if (!mounted) return null;

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 200,
        background: "rgba(0,0,0,0.96)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "16px",
      }}
    >
      {/* Header row */}
      <div
        style={{
          width: "100%",
          maxWidth: "900px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "12px",
        }}
      >
        <p
          style={{
            margin: 0,
            fontSize: "15px",
            fontWeight: 600,
            color: "rgba(255,255,255,0.9)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            maxWidth: "calc(100% - 52px)",
          }}
        >
          {title}
        </p>
        <button
          onClick={onClose}
          style={{
            flexShrink: 0,
            width: "40px",
            height: "40px",
            borderRadius: "50%",
            background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.15)",
            color: "white",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* 16:9 iframe container */}
      <div
        style={{
          width: "100%",
          maxWidth: "900px",
          aspectRatio: "16 / 9",
          borderRadius: "14px",
          overflow: "hidden",
          background: "#000",
          boxShadow: "0 32px 80px rgba(0,0,0,0.8)",
        }}
      >
        {/^[A-Za-z0-9_-]{11}$/.test(videoKey) && (
          <iframe
            key={videoKey}
            width="100%"
            height="100%"
            src={`https://www.youtube-nocookie.com/embed/${videoKey}?autoplay=1&playsinline=1&rel=0&modestbranding=1&enablejsapi=1`}
            title={`${title} trailer`}
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowFullScreen
            referrerPolicy="strict-origin-when-cross-origin"
            style={{ border: "none", display: "block" }}
          />
        )}
      </div>

      {/* Language Switcher under player if multiple languages exist */}
      {languages && languages.length > 1 && (
        <div
          style={{
            width: "100%",
            maxWidth: "900px",
            display: "flex",
            flexWrap: "wrap",
            gap: "8px",
            alignItems: "center",
            marginTop: "14px",
            justifyContent: "center",
          }}
        >
          <span style={{ fontSize: "13px", color: "rgba(255, 255, 255, 0.6)", fontWeight: 500 }}>
            Trailer Language:
          </span>
          {languages.map((tl) => {
            const isActive = tl.key === (selectedLanguage ?? videoKey);
            return (
              <button
                key={tl.lang}
                onClick={() => onSelectLanguage?.(tl.key)}
                style={{
                  padding: "7px 16px",
                  borderRadius: "999px",
                  background: isActive ? "rgba(255, 255, 255, 0.28)" : "rgba(255, 255, 255, 0.08)",
                  border: isActive ? "1px solid rgba(255, 255, 255, 0.45)" : "1px solid rgba(255, 255, 255, 0.14)",
                  color: "#ffffff",
                  fontSize: "13.5px",
                  fontWeight: isActive ? 700 : 500,
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {tl.label}
              </button>
            );
          })}
        </div>
      )}

      {/* Tap-backdrop-to-close on mobile */}
      <div
        onClick={onClose}
        style={{ position: "absolute", inset: 0, zIndex: -1 }}
      />
    </motion.div>,
    document.body
  );
}

/* ── Similar Movie Mini-Card ─────────────────────── */
function SimilarCard({ movie, onClick }: { movie: Recommendation; onClick: () => void }) {
  const fallbackSrc = "/poster_placeholder.svg";
  const initialSrc = movie.poster_path ? posterUrl(movie.poster_path, "w185") : fallbackSrc;
  const [imgSrc, setImgSrc] = useState(initialSrc);

  useEffect(() => {
    /* eslint-disable react-hooks/set-state-in-effect */
    setImgSrc(initialSrc);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, [initialSrc]);

  return (
    <button
      onClick={onClick}
      style={{
        flex: "0 0 90px",
        width: "90px",
        minWidth: "90px",
        maxWidth: "90px",
        background: "none",
        border: "none",
        outline: "none",
        WebkitTapHighlightColor: "transparent",
        padding: 0,
        cursor: "pointer",
        textAlign: "left",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          width: "100%",
          aspectRatio: "2 / 3",
          borderRadius: "10px",
          overflow: "hidden",
          background: "var(--color-surface)",
          position: "relative",
        }}
      >
        <img
          src={imgSrc}
          alt={movie.title}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
          onError={() => {
            if (imgSrc !== fallbackSrc) setImgSrc(fallbackSrc);
          }}
        />
        {movie.imdb_rating && (
          <div
            style={{
              position: "absolute",
              bottom: "5px",
              left: "5px",
              background: "rgba(0,0,0,0.75)",
              backdropFilter: "blur(4px)",
              borderRadius: "5px",
              padding: "2px 5px",
              fontSize: "9px",
              fontWeight: 700,
              color: "var(--color-rating)",
            }}
          >
            {movie.imdb_rating.toFixed(1)}
          </div>
        )}
      </div>
      <p
        style={{
          margin: "6px 0 0",
          fontSize: "10px",
          fontWeight: 500,
          color: "var(--color-text-secondary)",
          lineHeight: 1.3,
          minHeight: "26px",
          display: "-webkit-box",
          WebkitLineClamp: 2,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
        }}
      >
        {movie.title}
      </p>
      {movie.year && (
        <p style={{ margin: "2px 0 0", fontSize: "9px", color: "var(--color-text-muted)" }}>
          {movie.year}
        </p>
      )}
    </button>
  );
}

