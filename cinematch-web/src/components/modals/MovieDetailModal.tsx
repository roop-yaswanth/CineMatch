import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState, useRef } from "react";
import Image from "next/image";
import { createPortal } from "react-dom";
import { posterUrl, languageLabel, apiSimilarMovies, type Recommendation } from "@/lib/api";
import WatchProvidersPanel, { REGION_TO_COUNTRY } from "@/components/WatchProvidersPanel";

export interface DetailMovie {
  id: number;
  tmdb_id?: number;
  title: string;
  poster_path?: string;
  backdrop_path?: string;
  year?: number | string;
  original_language?: string;
  imdb_rating?: number;
  imdb_votes?: number;
  vote_average?: number;
  vote_count?: number;
  genres?: string[];
  primary_genre?: string;
  overview?: string;
  director?: string;
  runtime?: number;
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
  const [trailerKey, setTrailerKey] = useState<string | null>(null);
  const [trailerLanguages, setTrailerLanguages] = useState<Array<{lang:string;label:string;key:string}>>([]);
  const [selectedTrailerLang, setSelectedTrailerLang] = useState<string | null>(null);
  const [trailerLoading, setTrailerLoading] = useState(false);
  const [trailerFetched, setTrailerFetched] = useState(false);
  const [showTrailerPlayer, setShowTrailerPlayer] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const similarRowRef = useRef<HTMLDivElement>(null);

  // Treat phones + small tablets as compact layout
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 900);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Prevent body scroll when open; reset trailer on close
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "auto";
      setTimeout(() => {
        setSuccessAction(null);
        setTrailerKey(null);
        setTrailerLanguages([]);
        setSelectedTrailerLang(null);
        setTrailerFetched(false);
        setShowTrailerPlayer(false);
        setShowWatchProviders(false);
      }, 0);
    }
    return () => {
      document.body.style.overflow = "auto";
    };
  }, [isOpen]);

  // Keep a stable ref to onClose so the history effect never re-runs just
  // because the parent re-created the callback (e.g. after onMovieSelect).
  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  // Browser-back gesture closes the modal (mobile back swipe + desktop back button).
  // We push a sentinel history entry on open; popping it triggers onClose.
  // The flag avoids treating our own history.back() in onClose as a "real" back-nav.
  useEffect(() => {
    if (!isOpen) return;
    let dismissedByBack = false;
    const sentinel = { __movieModal: true, ts: Date.now() };
    window.history.pushState(sentinel, "");

    const handlePop = () => {
      dismissedByBack = true;
      onCloseRef.current();
    };
    window.addEventListener("popstate", handlePop);

    return () => {
      window.removeEventListener("popstate", handlePop);
      // If the modal was closed by anything other than a back-nav (X, swipe,
      // backdrop click), pop the sentinel ourselves so the back button doesn't
      // re-open the modal afterward.
      if (!dismissedByBack && window.history.state && (window.history.state as { __movieModal?: boolean }).__movieModal) {
        window.history.back();
      }
    };
  }, [isOpen]); // ← intentionally NOT depending on onClose (captured via ref above)

  // Fetch similar movies whenever the movie changes
  useEffect(() => {
    const id = movie?.tmdb_id ?? movie?.id;
    if (!isOpen || !id) { 
      setTimeout(() => setSimilar([]), 0); // Async state update to avoid cascading re-renders
      return; 
    }
    
    // Defer the setSimilarLoading to escape the synchronous commit phase
    setTimeout(() => {
      setSimilarLoading(true);
      setSimilar([]);
    }, 0);
    
    apiSimilarMovies(id, sessionId ?? null, 20)
      .then(setSimilar)
      .catch(() => setSimilar([]))
      .finally(() => setSimilarLoading(false));
  }, [movie?.id, movie?.tmdb_id, isOpen, sessionId]);

  const handleActionClick = (action: "like" | "okay" | "dislike" | "watchlist" | "skip") => {
    if (!onAction) return;
    onAction(action);
    setSuccessAction(action);
    setTimeout(() => setSuccessAction(null), 2500);
  };

  const handleWatchTrailer = (langKey?: string) => {
    // Specific language button clicked — open that language's trailer directly
    if (langKey) { setShowTrailerPlayer(true); setSelectedTrailerLang(langKey); return; }
    // Already fetched and found — open player (first/primary language)
    if (trailerFetched && !trailerLoading) {
      if (trailerKey) { setShowTrailerPlayer(true); }
      return;
    }
    // First click — lazy fetch
    const id = movie?.tmdb_id ?? movie?.id;
    if (!id) return;
    setTrailerLoading(true);
    fetch(`/api/tmdb-trailer?id=${id}`)
      .then((r) => r.json())
      .then((d: { key: string | null; languages: Array<{lang:string;label:string;key:string}> }) => {
        const langs = d.languages ?? [];
        setTrailerLanguages(langs);
        setTrailerFetched(true);
        // Prefer original language of the movie, fall back to first available
        const origLang = movie?.original_language;
        const preferred = langs.find((l) => l.lang === origLang) ?? langs[0];
        const key = preferred?.key ?? d.key ?? null;
        setTrailerKey(key);
        setSelectedTrailerLang(key);
        // Auto-open player only for single-language movies.
        // For multilingual: just show the language pills — user picks first.
        if (key && langs.length <= 1) setShowTrailerPlayer(true);
      })
      .catch(() => setTrailerFetched(true))
      .finally(() => setTrailerLoading(false));
  };

  if (!movie) return null;

  const poster = posterUrl(movie.poster_path, "w500");
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const year = movie.year || "";
  const genres = movie.genres?.join(", ") || movie.primary_genre || "";
  const imdb = movie.imdb_rating ? movie.imdb_rating.toFixed(1) : movie.vote_average ? movie.vote_average.toFixed(1) : null;
  const overview = movie.overview || "No overview available.";
  const runtime = movie.runtime ? `${Math.floor(movie.runtime / 60)}h ${movie.runtime % 60}m` : null;
  const matchPct = movie.score !== undefined && movie.score >= 0.70 ? Math.round(movie.score * 100) : null;
  const matchColor = movie.score !== undefined && movie.score >= 0.85 ? "#22c55e" : "#eab308";
  const guessedCountry = userRegion ? (REGION_TO_COUNTRY[userRegion] ?? "US") : "US";
  const watchButtonStyle = {
    position: "absolute" as const,
    bottom: isMobile ? "14px" : "16px",
    left: isMobile ? "12px" : "16px",
    right: isMobile ? "12px" : "16px",
    padding: "12px 16px",
    borderRadius: "12px",
    fontSize: "13px",
    fontWeight: 700,
    color: "#fff",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "10px",
    background: "linear-gradient(130deg, rgba(255,255,255,0.24) 0%, rgba(255,255,255,0.10) 30%, rgba(255,255,255,0.18) 52%, rgba(255,255,255,0.07) 100%), rgba(255,255,255,0.03)",
    border: "1px solid rgba(255, 255, 255, 0.34)",
    backdropFilter: "blur(20px) saturate(1.55)",
    WebkitBackdropFilter: "blur(20px) saturate(1.55)",
    boxShadow: "0 8px 24px rgba(0, 0, 0, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.42), inset 0 -10px 16px rgba(255, 255, 255, 0.04)",
    zIndex: 10,
    textShadow: "0 2px 4px rgba(0, 0, 0, 0.3)",
    transition: "all 0.3s ease",
  };

  // Watch Trailer node — extracted so it can sit at a fixed spot above the
  // overview (consistent placement = muscle memory) instead of after it.
  const trailerNode = (() => {
    const notFound = trailerFetched && !trailerLoading && !trailerKey;
    const isMultiLang = trailerLanguages.length > 1;

    if (trailerLoading) {
      return (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          gap: "10px", padding: "12px 18px", borderRadius: "12px",
          background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)",
          color: "var(--color-text-muted)", fontSize: "13px",
        }}>
          <div style={{
            width: "14px", height: "14px", borderRadius: "50%",
            border: "2px solid rgba(255,255,255,0.15)",
            borderTopColor: "rgba(255,255,255,0.6)",
            animation: "spin 0.8s linear infinite", flexShrink: 0,
          }} />
          <style>{"@keyframes spin{to{transform:rotate(360deg)}}"}</style>
          <span>Finding trailer...</span>
        </div>
      );
    }

    if (notFound) {
      return (
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          gap: "8px", padding: "12px 18px", borderRadius: "12px",
          background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
          color: "rgba(255,255,255,0.35)", fontSize: "12px",
        }}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>No trailer available</span>
        </div>
      );
    }

    if (isMultiLang) {
      return (
        <div>
          <p style={{ margin: "0 0 8px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--color-text-muted)", opacity: 0.6 }}>
            Watch Trailer in
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {trailerLanguages.map((tl) => {
              const isActive = tl.key === selectedTrailerLang;
              return (
                <motion.button
                  key={tl.lang}
                  whileHover={{ scale: 1.04 }}
                  whileTap={{ scale: 0.96 }}
                  onClick={() => handleWatchTrailer(tl.key)}
                  style={{
                    display: "flex", alignItems: "center", gap: "6px",
                    padding: "9px 16px", borderRadius: "100px",
                    background: isActive ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.06)",
                    border: `1px solid ${isActive ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.1)"}`,
                    color: isActive ? "var(--color-text-primary)" : "var(--color-text-muted)",
                    fontSize: "13px", fontWeight: isActive ? 600 : 500,
                    cursor: "pointer", transition: "all 0.18s",
                  }}
                >
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" style={{ opacity: isActive ? 1 : 0.7 }}>
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  {tl.label}
                </motion.button>
              );
            })}
          </div>
        </div>
      );
    }

    return (
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.97 }}
        onClick={() => handleWatchTrailer()}
        style={{
          display: "flex", alignItems: "center", justifyContent: "center",
          gap: "10px", width: "100%", padding: "12px 18px", borderRadius: "12px",
          background: "rgba(255,255,255,0.08)", border: "1px solid rgba(255,255,255,0.14)",
          color: "var(--color-text-primary)", cursor: "pointer",
          fontSize: "14px", fontWeight: 600, letterSpacing: "0.01em",
        }}
      >
        <div style={{
          width: "28px", height: "28px", borderRadius: "50%",
          background: "rgba(255,255,255,0.12)",
          display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
        }}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
        </div>
        <span>Watch Trailer</span>
      </motion.button>
    );
  })();

  if (typeof document === 'undefined') return null;

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <div style={{ position: "fixed", inset: 0, zIndex: 100, display: "flex", alignItems: isMobile ? "flex-start" : "center", justifyContent: "center" }}>

          {/* Fullscreen trailer overlay — sits above the modal */}
          <AnimatePresence>
            {showTrailerPlayer && selectedTrailerLang && (
              <TrailerOverlay
                videoKey={selectedTrailerLang}
                title={movie.title}
                onClose={() => setShowTrailerPlayer(false)}
              />
            )}
          </AnimatePresence>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            // Only close when the backdrop itself is clicked, not when events
            // bubble up from inside the modal (e.g. SimilarCard buttons).
            onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
            style={{
              position: "absolute",
              inset: 0,
              background: "rgba(0, 0, 0, 0.7)",
              backdropFilter: "blur(12px)",
              WebkitBackdropFilter: "blur(12px)",
            }}
          />
          
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="glass-modal"
            style={{
              position: "relative",
              width: isMobile ? "100%" : "94vw",
              maxWidth: isMobile ? "100%" : "980px",
              maxHeight: isMobile ? "100dvh" : "92vh",
              height: isMobile ? "100dvh" : "auto",
              minHeight: isMobile ? undefined : "72vh",
              borderRadius: isMobile ? "0" : undefined,
              boxShadow: isMobile ? "none" : "0 25px 80px -12px rgba(0, 0, 0, 0.8)",
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {/* No sticky header — close X lives on the poster itself (see below) */}

            {/* Content wrapper - no top padding to remove excess space */}
            <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", position: "relative", zIndex: 1 }}>
              {/* Left Column: Poster + Where-to-Watch */}
              <div
                style={{
                  flex: isMobile ? "1 1 100%" : "1 1 240px",
                  maxWidth: isMobile ? "100%" : "340px",
                  margin: "0 auto",
                  width: "100%",
                  padding: 0,
                  display: "flex",
                  flexDirection: "column",
                  gap: 0,
                }}
              >
                <div
                  style={{
                    position: "relative",
                    // On compact screens: tall hero area with full poster visibility
                    // On desktop: maintain 2:3 aspect ratio, capped at 50vh.
                    ...(isMobile
                      ? { height: "62vh", maxHeight: "620px", minHeight: "340px" }
                      : { aspectRatio: "2/3", maxHeight: "62vh" }),
                    width: "100%",
                    background: "var(--color-surface)",
                    borderRadius: isMobile ? "0" : "12px",
                    overflow: "hidden",
                    boxShadow: isMobile ? "none" : "0 12px 32px -8px rgba(0,0,0,0.6)",
                  }}
                >
                  {/* Unified poster area: blurred bg + sharp centred poster + buttons.
                      Same layout on both mobile and desktop — no branching needed. */}

                  {/* Blurred background fill (mobile only visually useful, harmless on desktop) */}
                  <Image
                    src={poster}
                    alt=""
                    aria-hidden
                    fill
                    style={{
                      objectFit: "cover",
                      objectPosition: "center center",
                      filter: "blur(20px) saturate(1.1) brightness(0.45)",
                      transform: "scale(1.08)",
                      zIndex: 0,
                    }}
                    unoptimized
                  />
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      background: "linear-gradient(180deg, rgba(0,0,0,0.12) 0%, rgba(0,0,0,0.28) 100%)",
                      zIndex: 1,
                    }}
                  />

                  {/* Sharp centred poster */}
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      zIndex: 2,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <div
                      style={{
                        position: "relative",
                        height: "100%",
                        maxWidth: "100%",
                        aspectRatio: "2 / 3",
                      }}
                    >
                      <Image
                        src={poster}
                        alt={movie.title}
                        fill
                        style={{ objectFit: "cover", objectPosition: "center center" }}
                        unoptimized
                      />
                    </div>
                  </div>

                  {/* Smart X button — floats top-right of the poster.
                      Closes WatchProviders overlay if open, otherwise closes the modal. */}
                  <button
                    onClick={() => showWatchProviders ? setShowWatchProviders(false) : onClose()}
                    aria-label={showWatchProviders ? "Close streaming info" : "Close"}
                    style={{
                      position: "absolute",
                      top: isMobile ? "calc(env(safe-area-inset-top) + 10px)" : "10px",
                      right: "10px",
                      zIndex: 25,
                      width: "32px",
                      height: "32px",
                      borderRadius: "50%",
                      background: "rgba(0,0,0,0.55)",
                      border: "1px solid rgba(255,255,255,0.18)",
                      backdropFilter: "blur(8px)",
                      WebkitBackdropFilter: "blur(8px)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: "#fff",
                      cursor: "pointer",
                      transition: "background 0.2s",
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>

                  {/* Where to Watch button — same design on mobile + desktop */}
                  {(movie.tmdb_id ?? movie.id) > 0 && (
                    <motion.button
                      onClick={() => setShowWatchProviders((s) => !s)}
                      whileHover={{ scale: 1.05, y: -1 }}
                      whileTap={{ scale: 0.95 }}
                      style={watchButtonStyle}
                      aria-expanded={showWatchProviders}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="2" y="5" width="20" height="14" rx="3" ry="3" />
                        <line x1="8" y1="3" x2="6" y2="5" />
                        <line x1="13" y1="3" x2="11" y2="5" />
                        <line x1="18" y1="3" x2="16" y2="5" />
                        <polygon points="10,10 15,12 10,14" fill="currentColor" stroke="none" />
                      </svg>
                      <span>{showWatchProviders ? "Hide" : "Where to Watch"}</span>
                    </motion.button>
                  )}

                  {/* Watch Providers Overlay on Poster */}
                  {showWatchProviders && (movie.tmdb_id ?? movie.id) > 0 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      style={{
                        position: "absolute",
                        inset: 0,
                        background: "linear-gradient(165deg, rgba(16, 18, 24, 0.40) 0%, rgba(12, 14, 20, 0.26) 100%)",
                        backdropFilter: "blur(14px) saturate(1.35) brightness(1.06)",
                        WebkitBackdropFilter: "blur(14px) saturate(1.35) brightness(1.06)",
                        display: "flex",
                        flexDirection: "column",
                        padding: isMobile ? "calc(env(safe-area-inset-top) + 10px) 16px 16px" : "16px",
                        zIndex: 20,
                        borderRadius: isMobile ? "0" : "12px",
                        border: "1px solid rgba(255, 255, 255, 0.14)",
                        overflowY: "auto",
                      }}
                    >
                      <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ type: "spring", damping: 20 }}
                        style={{ width: "100%" }}
                      >
                        {/* Header row: title + close button always visible */}
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                          <h3 style={{ margin: 0, fontSize: "14px", fontWeight: 600, color: "#fff" }}>Where to Watch</h3>
                        </div>
                        <WatchProvidersPanel
                          tmdbId={(movie.tmdb_id ?? movie.id) as number}
                          defaultCountry={guessedCountry}
                          movieTitle={movie.title}
                        />
                      </motion.div>
                    </motion.div>
                  )}
                </div>
              </div>

              {/* Right Column: Details and Actions */}
              <div
                style={{
                  flex: isMobile ? "1 1 100%" : "2 1 340px",
                  padding: isMobile ? "14px 14px 20px" : "24px 20px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "14px",
                  background: movie.backdrop_path ? "transparent" : "var(--color-bg)",
                }}
              >
                <div>
                  {/* Match score badge */}
                  {matchPct && (
                    <div style={{ marginBottom: "12px" }}>
                      <span style={{
                        display: "inline-block",
                        background: `${matchColor}22`,
                        border: `1px solid ${matchColor}55`,
                        color: matchColor,
                        borderRadius: "20px",
                        padding: "3px 12px",
                        fontSize: "12px",
                        fontWeight: 700,
                        letterSpacing: "0.02em",
                      }}>
                        {matchPct}% Match
                      </span>
                    </div>
                  )}

                  <h2 style={{ margin: "0 0 6px 0", fontSize: "26px", fontWeight: 700, color: "var(--color-text-primary)", lineHeight: 1.1 }}>
                    {movie.title}
                  </h2>

                  <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center", color: "var(--color-text-muted)", fontSize: "13px" }}>
                    {year && <span>{year}</span>}
                    {year && lang && <span style={{ opacity: 0.4 }}>·</span>}
                    {lang && <span>{lang}</span>}
                    {(year || lang) && runtime && <span style={{ opacity: 0.4 }}>·</span>}
                    {runtime && <span>{runtime}</span>}
                    {(year || lang || runtime) && imdb && <span style={{ opacity: 0.4 }}>·</span>}
                    {imdb && <span style={{ color: "var(--color-accent-warm)", fontWeight: 600 }}>IMDb {imdb}</span>}
                  </div>

                  {movie.director && (
                    <p style={{ marginTop: "8px", fontSize: "13px", color: "var(--color-text-muted)" }}>
                      <span style={{ opacity: 0.6 }}>Directed by</span>{" "}
                      <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{movie.director}</span>
                    </p>
                  )}

                  {genres && (
                    <div style={{ marginTop: "12px", display: "flex", gap: "6px", flexWrap: "wrap" }}>
                      {genres.split(",").map(g => (
                        <span key={g} style={{
                          background: "rgba(255,255,255,0.05)",
                          padding: "4px 10px",
                          borderRadius: "100px",
                          fontSize: "12px",
                          border: "1px solid rgba(255,255,255,0.1)"
                        }}>
                          {g.trim()}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Watch Trailer — fixed position above overview for muscle memory */}
                {trailerNode}

                <div style={{ fontSize: "13px", lineHeight: 1.6, color: "var(--color-text-secondary)" }}>
                  {overview}
                </div>

                {/* Why recommended */}
                {movie.reason && (
                  <div style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: "12px",
                    padding: "14px 16px",
                  }}>
                    <p style={{ margin: "0 0 4px", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--color-text-muted)", opacity: 0.7 }}>
                      Why recommended
                    </p>
                    <p style={{ margin: 0, fontSize: "13px", lineHeight: 1.6, color: "var(--color-text-secondary)", fontStyle: "italic" }}>
                      {movie.reason}
                    </p>
                  </div>
                )}

                {onAction && (
                    <div style={{ marginTop: "auto", paddingTop: "16px", borderTop: "1px solid var(--color-border-subtle)" }}>
                        <h4 style={{ margin: "0 0 16px 0", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--color-text-muted)" }}>
                            Rate this movie
                        </h4>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
                            {/* Like */}
                            <button className="rating-btn rating-btn--like" onClick={() => handleActionClick("like")} style={{ flex: 1, padding: "12px", fontWeight: 600, height: "48px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "like" ? (
                                        <motion.div key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-like)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "var(--color-like)", fontWeight: 700 }}>Liked</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--color-like)" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" /></svg>
                                            Like
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                            {/* Okay */}
                            <button className="rating-btn rating-btn--okay" onClick={() => handleActionClick("okay")} style={{ flex: 1, padding: "12px", fontWeight: 600, height: "48px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "okay" ? (
                                        <motion.div key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-okay)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "var(--color-okay)", fontWeight: 700 }}>Okay</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--color-okay)" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" /><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" /></svg>
                                            Okay
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                            {/* Dislike */}
                            <button className="rating-btn rating-btn--dislike" onClick={() => handleActionClick("dislike")} style={{ flex: 1, padding: "12px", fontWeight: 600, height: "48px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "dislike" ? (
                                        <motion.div key="done" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-dislike)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "var(--color-dislike)", fontWeight: 700 }}>Noted</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--color-dislike)" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" /><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" /></svg>
                                            Dislike
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                        </div>
                        <div style={{ display: "flex", gap: "12px", marginTop: "12px" }}>
                            <button className="glass-button" onClick={() => handleActionClick("watchlist")} style={{ flex: 1, padding: "12px", color: "var(--color-text-primary)", fontSize: "14px", display: "flex", justifyContent: "center", alignItems: "center", height: "46px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "watchlist" ? (
                                        <motion.div key="check" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ color: "#22c55e", fontWeight: 700 }}>Added</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M5 12h14"/></svg>
                                            Watchlist
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                            <button className="glass-button" onClick={() => handleActionClick("skip")} style={{ flex: 1, padding: "12px", color: "var(--color-text-primary)", fontSize: "14px", display: "flex", justifyContent: "center", alignItems: "center", height: "46px" }}>
                                <AnimatePresence mode="wait">
                                    {successAction === "skip" ? (
                                        <motion.div key="check" initial={{ scale: 0.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.5, opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
                                            <span style={{ fontWeight: 700 }}>Skipped</span>
                                        </motion.div>
                                    ) : (
                                        <motion.div key="icon" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="5 4 15 12 5 20 5 4"></polygon><line x1="19" y1="5" x2="19" y2="19"></line></svg>
                                            Skip
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </button>
                        </div>
                    </div>
                )}
              </div>
            </div>

            {/* ── More Like This ───────────────────────── */}
            {(similarLoading || similar.length > 0) && (
              <div style={{
                borderTop: "1px solid var(--color-border-subtle)",
                padding: "16px 20px 20px",
                position: "relative",
                zIndex: 1,
              }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
                  <h4 style={{
                    margin: 0,
                    fontSize: "12px",
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    color: "var(--color-text-muted)",
                    fontWeight: 600,
                  }}>
                    More like this
                  </h4>
                  {!similarLoading && similar.length > 0 && (
                    <div style={{ display: "flex", gap: "6px" }}>
                      <button
                        onClick={() => similarRowRef.current?.scrollBy({ left: -600, behavior: "smooth" })}
                        style={{
                          width: "28px", height: "28px", borderRadius: "50%",
                          border: "1px solid var(--color-border-subtle)",
                          background: "rgba(255,255,255,0.06)",
                          color: "var(--color-text-secondary)",
                          cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                        }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="15 18 9 12 15 6" />
                        </svg>
                      </button>
                      <button
                        onClick={() => similarRowRef.current?.scrollBy({ left: 600, behavior: "smooth" })}
                        style={{
                          width: "28px", height: "28px", borderRadius: "50%",
                          border: "1px solid var(--color-border-subtle)",
                          background: "rgba(255,255,255,0.06)",
                          color: "var(--color-text-secondary)",
                          cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
                        }}
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="9 18 15 12 9 6" />
                        </svg>
                      </button>
                    </div>
                  )}
                </div>

                {/* Skeleton row while loading */}
                {similarLoading && (
                  <div style={{ display: "flex", gap: "12px", overflowX: "hidden" }}>
                    {Array.from({ length: 6 }).map((_, i) => (
                      <div key={i} style={{ flexShrink: 0, width: "90px" }}>
                        <div className="skeleton-shimmer" style={{ width: "90px", paddingBottom: "135px", borderRadius: "10px" }} />
                        <div className="skeleton-shimmer" style={{ height: "10px", width: "75%", borderRadius: "999px", marginTop: "8px" }} />
                      </div>
                    ))}
                  </div>
                )}

                {/* Similar movies row */}
                {!similarLoading && similar.length > 0 && (
                  <div
                    ref={similarRowRef}
                    style={{
                      display: "flex",
                      gap: "12px",
                      overflowX: "auto",
                      paddingBottom: "4px",
                      scrollSnapType: "x mandatory",
                      msOverflowStyle: "none",
                      scrollbarWidth: "none",
                    }}
                  >
                    {similar.map((m) => (
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
            )}

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
  onClose,
}: {
  videoKey: string;
  title: string;
  onClose: () => void;
}) {
  // ESC key closes
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  if (typeof document === "undefined") return null;

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      style={{
        position: "fixed", inset: 0, zIndex: 200,
        background: "rgba(0,0,0,0.96)",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "16px",
      }}
    >
      {/* Header row */}
      <div style={{
        width: "100%", maxWidth: "900px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: "12px",
      }}>
        <p style={{
          margin: 0, fontSize: "14px", fontWeight: 600,
          color: "rgba(255,255,255,0.8)",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          maxWidth: "calc(100% - 52px)",
        }}>
          {title}
        </p>
        <button
          onClick={onClose}
          style={{
            flexShrink: 0,
            width: "40px", height: "40px", borderRadius: "50%",
            background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.15)",
            color: "white", cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* 16:9 iframe container */}
      <div style={{
        width: "100%", maxWidth: "900px",
        aspectRatio: "16 / 9",
        borderRadius: "12px",
        overflow: "hidden",
        background: "#000",
        boxShadow: "0 32px 80px rgba(0,0,0,0.8)",
      }}>
        <iframe
          src={`https://www.youtube.com/embed/${videoKey}?autoplay=1&playsinline=1&rel=0&modestbranding=1`}
          title={`${title} trailer`}
          allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
          allowFullScreen
          style={{ width: "100%", height: "100%", border: "none", display: "block" }}
        />
      </div>

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
    setImgSrc(initialSrc);
  }, [initialSrc]);

  return (
    <button
      onClick={onClick}
      style={{
        flex: "0 0 90px",
        width: "90px",
        minWidth: "90px",
        maxWidth: "90px",
        scrollSnapAlign: "start",
        background: "none",
        border: "none",
        padding: 0,
        cursor: "pointer",
        textAlign: "left",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div style={{
        width: "100%",
        aspectRatio: "2 / 3",
        borderRadius: "10px",
        overflow: "hidden",
        background: "var(--color-surface)",
        position: "relative",
      }}>
        <Image
          src={imgSrc}
          alt={movie.title}
          fill
          sizes="90px"
          style={{ objectFit: "cover" }}
          unoptimized
          onError={() => {
            if (imgSrc !== fallbackSrc) setImgSrc(fallbackSrc);
          }}
        />
        {movie.imdb_rating && (
          <div style={{
            position: "absolute",
            bottom: "5px",
            left: "5px",
            background: "rgba(0,0,0,0.75)",
            backdropFilter: "blur(4px)",
            borderRadius: "5px",
            padding: "2px 5px",
            fontSize: "9px",
            fontWeight: 700,
            color: "#fbbf24",
          }}>
            {movie.imdb_rating.toFixed(1)}
          </div>
        )}
      </div>
      <p style={{
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
      }}>
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
