import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState, useRef } from "react";
import Image from "next/image";
import Link from "next/link";
import { createPortal } from "react-dom";
import { posterUrl, languageLabel, apiSimilarMovies, apiCredits, type Recommendation, type CastMember, type CrewMember } from "@/lib/api";
import { PersonDetailOverlay } from "./PersonDetailOverlay";
import WatchProvidersPanel, { REGION_TO_COUNTRY } from "@/components/WatchProvidersPanel";
import { pushBackHandler } from "@/lib/backStack";

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
  const [cast, setCast] = useState<CastMember[]>([]);
  const [directors, setDirectors] = useState<CrewMember[]>([]);
  const [writers, setWriters] = useState<CrewMember[]>([]);
  const [creditsLoading, setCreditsLoading] = useState(false);
  const [trailerKey, setTrailerKey] = useState<string | null>(null);
  const [trailerLanguages, setTrailerLanguages] = useState<Array<{ lang: string; label: string; key: string }>>([]);
  const [selectedTrailerLang, setSelectedTrailerLang] = useState<string | null>(null);
  const [trailerLoading, setTrailerLoading] = useState(false);
  const [trailerFetched, setTrailerFetched] = useState(false);
  const [showTrailerPlayer, setShowTrailerPlayer] = useState(false);
  const [activePersonId, setActivePersonId] = useState<number | null>(null);
  const [isMobile, setIsMobile] = useState(false);
  const similarRowRef = useRef<HTMLDivElement>(null);
  const castRowRef = useRef<HTMLDivElement>(null);

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

  // PWA back-gesture: push a fake history entry when the modal opens so that
  // a back-swipe / back-button closes this modal instead of navigating away.
  // Uses the centralized backStack so nested modals don't conflict.
  useEffect(() => {
    if (!isOpen) return;
    const cleanup = pushBackHandler(() => onCloseRef.current());
    return cleanup; // called when isOpen flips false (via UI close, not back gesture)
  }, [isOpen]);

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

  // Fetch cast & crew whenever the movie changes
  useEffect(() => {
    const id = movie?.tmdb_id ?? movie?.id;
    if (!isOpen || !id) {
      setCast([]); setDirectors([]); setWriters([]);
      setActivePersonId(null);
      return;
    }
    let cancelled = false;
    setCreditsLoading(true);
    setCast([]); setDirectors([]); setWriters([]);
    
    apiCredits(id, "movie")
      .then((c) => {
        if (cancelled) return;
        setCast(c.cast);
        setDirectors(c.directors);
        setWriters(c.writers);
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setCreditsLoading(false); });
    return () => { cancelled = true; };
  }, [movie?.id, movie?.tmdb_id, isOpen]);

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
      .then((d: { key: string | null; languages: Array<{ lang: string; label: string; key: string }> }) => {
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
  // On mobile we use the wider backdrop(saves vertical space).
  // Fall back to the portrait poster if backdrop_path is missing.
  const mobileHero = movie.backdrop_path
    ? `https://image.tmdb.org/t/p/w1280${movie.backdrop_path}`
    : poster;
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
            className={isMobile ? "" : "glass-modal"}
            style={{
              position: "relative",
              width: isMobile ? "100%" : "94vw",
              maxWidth: isMobile ? "100%" : "980px",
              maxHeight: isMobile ? "100dvh" : "92vh",
              height: isMobile ? "100dvh" : "auto",
              minHeight: isMobile ? undefined : "72vh",
              background: isMobile ? "var(--color-bg)" : undefined,
              borderRadius: isMobile ? "0" : undefined,
              boxShadow: isMobile ? "none" : "0 25px 80px -12px rgba(0, 0, 0, 0.8)",
              overflowY: "auto",
              overscrollBehavior: "none",  // prevent iOS bounce revealing top pill
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

                    ...(isMobile
                      ? { aspectRatio: "16/9", width: "100%" }
                      : { aspectRatio: "2/3", maxHeight: "62vh" }),
                    background: "var(--color-surface)",
                    borderRadius: isMobile ? "0" : "12px",
                    overflow: "hidden",
                    boxShadow: isMobile ? "none" : "0 12px 32px -8px rgba(0,0,0,0.6)",
                  }}
                >
                  {/* ── Hero image ───────────────────────────────────────────
                      Mobile : full-width 16:9 backdrop.
                               Falls back to poster with objectPosition top.
                      Desktop: blurred bg fill + centred sharp portrait.     */}

                  {isMobile ? (
                    /* Landscape backdrop — single image, edge-to-edge */
                    <Image
                      src={mobileHero}
                      alt={movie.title}
                      fill
                      sizes="100vw"
                      style={{
                        objectFit: "cover",
                        objectPosition: movie.backdrop_path ? "center center" : "center top",
                        zIndex: 0,
                      }}
                      priority
                    />
                  ) : (
                    /* Desktop: blurred bg fill */
                    <Image
                      src={poster}
                      alt=""
                      aria-hidden
                      fill
                      sizes="340px"
                      style={{
                        objectFit: "cover",
                        objectPosition: "center center",
                        filter: "blur(20px) saturate(1.1) brightness(0.45)",
                        transform: "scale(1.08)",
                        zIndex: 0,
                      }}
                    />
                  )}
                  <div
                    style={{
                      position: "absolute",
                      inset: 0,
                      background: "linear-gradient(180deg, rgba(0,0,0,0.12) 0%, rgba(0,0,0,0.28) 100%)",
                      zIndex: 1,
                    }}
                  />

                  {/* Top gradient for safe-area and X button contrast (mobile only) */}
                  {isMobile && (
                    <div
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        right: 0,
                        height: "90px",
                        background: "linear-gradient(to bottom, rgba(0,0,0,0.75) 0%, transparent 100%)",
                        zIndex: 3,
                        pointerEvents: "none",
                      }}
                    />
                  )}

                  {/* Bottom gradient → bleeds into dark modal content below (mobile only) */}
                  {isMobile && (
                    <div
                      style={{
                        position: "absolute",
                        bottom: 0,
                        left: 0,
                        right: 0,
                        height: "55%",
                        background: "linear-gradient(to top, rgba(12,13,18,1) 0%, rgba(12,13,18,0.65) 45%, transparent 100%)",
                        zIndex: 3,
                        pointerEvents: "none",
                      }}
                    />
                  )}

                  {/* Desktop only: centred sharp portrait on top of blurred bg */}
                  {!isMobile && (
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
                          sizes="340px"
                          style={{ objectFit: "cover", objectPosition: "center center" }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Mobile: title + rating overlaid at the poster bottom.
                      Key info is visible without scrolling down. */}
                  {isMobile && !showWatchProviders && (
                    <div
                      style={{
                        position: "absolute",
                        bottom: 0,
                        left: 0,
                        right: 0,
                        padding: "0 14px 16px", // Reduced since Where-to-Watch is now below the hero
                        zIndex: 4,
                        pointerEvents: "none",
                      }}
                    >
                      <h2
                        style={{
                          margin: "0 0 4px",
                          fontSize: "clamp(18px, 5vw, 24px)",
                          fontWeight: 700,
                          color: "#fff",
                          lineHeight: 1.15,
                          textShadow: "0 2px 8px rgba(0,0,0,0.7)",
                          letterSpacing: "-0.02em",
                        }}
                      >
                        {movie.title}
                      </h2>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
                        {movie.year && (
                          <span style={{ fontSize: "12px", color: "rgba(255,255,255,0.65)", textShadow: "0 1px 4px rgba(0,0,0,0.6)" }}>
                            {movie.year}
                          </span>
                        )}
                        {lang && (
                          <span style={{ fontSize: "12px", color: "rgba(255,255,255,0.65)", textShadow: "0 1px 4px rgba(0,0,0,0.6)" }}>
                            {lang}
                          </span>
                        )}
                        {imdb && (
                          <span style={{
                            display: "inline-flex", alignItems: "center", gap: "4px",
                            fontSize: "12px", fontWeight: 700,
                            color: "#fbbf24",
                            textShadow: "0 1px 4px rgba(0,0,0,0.7)",
                          }}>
                            ⭐ {imdb}
                          </span>
                        )}
                        {runtime && (
                          <span style={{ fontSize: "12px", color: "rgba(255,255,255,0.55)", textShadow: "0 1px 4px rgba(0,0,0,0.6)" }}>
                            {runtime}
                          </span>
                        )}
                      </div>
                      {genres && (
                        <div style={{ marginTop: "6px", fontSize: "12px", color: "rgba(255,255,255,0.8)", fontWeight: 500, textShadow: "0 1px 4px rgba(0,0,0,0.8)", letterSpacing: "0.01em" }}>
                          {genres.split(",").map(g => g.trim()).join(" • ")}
                        </div>
                      )}
                    </div>
                  )}

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

                  {/* Where to Watch button — desktop only (absolute on portrait poster).
                      Mobile version lives below the hero as a standalone row. */}
                  {!isMobile && (movie.tmdb_id ?? movie.id) > 0 && (
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

                  {/* Watch Providers — desktop: absolute overlay on the portrait poster */}
                  {!isMobile && showWatchProviders && (movie.tmdb_id ?? movie.id) > 0 && (
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
                        padding: "16px",
                        zIndex: 20,
                        borderRadius: "12px",
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

                {/* Mobile-only: Where to Watch button sits below the hero, above the panel */}
                {isMobile && (movie.tmdb_id ?? movie.id) > 0 && (
                  <button
                    onClick={() => setShowWatchProviders((s) => !s)}
                    aria-expanded={showWatchProviders}
                    style={{
                      width: "100%",
                      padding: "13px 16px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: "10px",
                      background: showWatchProviders
                        ? "rgba(255,255,255,0.08)"
                        : "linear-gradient(180deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%)",
                      backdropFilter: "blur(12px)",
                      WebkitBackdropFilter: "blur(12px)",
                      border: "none",
                      borderTop: "1px solid rgba(255,255,255,0.05)",
                      borderBottom: "1px solid rgba(255,255,255,0.08)",
                      color: "#fff",
                      fontSize: "14px",
                      fontWeight: 600,
                      cursor: "pointer",
                      transition: "background 0.2s",
                      letterSpacing: "-0.01em",
                    }}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="2" y="5" width="20" height="14" rx="3" ry="3" />
                      <line x1="8" y1="3" x2="6" y2="5" />
                      <line x1="13" y1="3" x2="11" y2="5" />
                      <line x1="18" y1="3" x2="16" y2="5" />
                      <polygon points="10,10 15,12 10,14" fill="currentColor" stroke="none" />
                    </svg>
                    <span>{showWatchProviders ? "Hide" : "Where to Watch"}</span>
                    <svg
                      width="13" height="13" viewBox="0 0 24 24" fill="none"
                      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                      style={{ marginLeft: "auto", opacity: 0.5, transform: showWatchProviders ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}
                    >
                      <polyline points="6 9 12 15 18 9" />
                    </svg>
                  </button>
                )}

                {/* Watch Providers — mobile: in-flow panel below the button.
                    Uses opacity+y only (no height animation) — fast, GPU-composited. */}
                <AnimatePresence initial={false}>
                  {isMobile && showWatchProviders && (movie.tmdb_id ?? movie.id) > 0 && (
                    <motion.div
                      key="mobile-w2w"
                      initial={{ opacity: 0, y: -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -6 }}
                      transition={{ duration: 0.18, ease: "easeOut" }}
                    >
                      <div
                        style={{
                          padding: "16px 16px 4px",
                          background: "rgba(12,13,18,1)",
                          borderBottom: "1px solid rgba(255,255,255,0.07)",
                        }}
                      >
                        <WatchProvidersPanel
                          tmdbId={(movie.tmdb_id ?? movie.id) as number}
                          defaultCountry={guessedCountry}
                          movieTitle={movie.title}
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
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

                  {/* On mobile, title and meta are already overlaid on the hero poster */}
                  {!isMobile && (
                    <>
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
                    </>
                  )}

                  {movie.director && (
                    <p style={{ marginTop: "8px", fontSize: "13px", color: "var(--color-text-muted)" }}>
                      <span style={{ opacity: 0.6 }}>Directed by</span>{" "}
                      <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>{movie.director}</span>
                    </p>
                  )}

                  {!isMobile && genres && (
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
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 5v14M5 12h14" /></svg>
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

            {/* ── Cast & Crew ──────────────────────────── */}
            {(creditsLoading || cast.length > 0 || directors.length > 0 || writers.length > 0) && (
              <div style={{ padding: "8px 20px 0", position: "relative", zIndex: 1 }}>
                {(directors.length > 0 || writers.length > 0) && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "16px 24px", marginBottom: "16px", fontSize: "13px" }}>
                    {directors.length > 0 && (
                      <div>
                        <div style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-muted)", marginBottom: "3px" }}>
                          {directors.length > 1 ? "Directors" : "Director"}
                        </div>
                        <div style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>
                          {directors.map((d, i) => (
                            <span key={`${d.id}-${i}`}>
                              <button
                                onClick={() => setActivePersonId(d.id)}
                                style={{ padding: 0, background: "none", borderBottom: "1px dotted rgba(255,255,255,0.25)", cursor: "pointer", color: "inherit", textDecoration: "none", fontSize: "inherit", fontFamily: "inherit" }}
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
                        <div style={{ fontSize: "10px", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-muted)", marginBottom: "3px" }}>
                          {writers.length > 1 ? "Writers" : "Writer"}
                        </div>
                        <div style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>
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
                                  style={{ padding: 0, background: "none", borderBottom: "1px dotted rgba(255,255,255,0.25)", cursor: "pointer", color: "inherit", textDecoration: "none", fontSize: "inherit", fontFamily: "inherit" }}
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
                )}

                {(creditsLoading || cast.length > 0) && (
                  <div>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
                      <h4 style={{ margin: 0, fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-muted)", fontWeight: 600 }}>
                        Cast
                      </h4>
                      {!creditsLoading && cast.length > 0 && (
                        <div style={{ display: "flex", gap: "6px" }}>
                          <button
                            onClick={() => castRowRef.current?.scrollBy({ left: -(window.innerWidth * 0.6), behavior: "smooth" })}
                            aria-label="Scroll cast left"
                            style={{ width: "28px", height: "28px", borderRadius: "50%", border: "1px solid var(--color-border-subtle)", background: "rgba(255,255,255,0.06)", color: "var(--color-text-secondary)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6" /></svg>
                          </button>
                          <button
                            onClick={() => castRowRef.current?.scrollBy({ left: (window.innerWidth * 0.6), behavior: "smooth" })}
                            aria-label="Scroll cast right"
                            style={{ width: "28px", height: "28px", borderRadius: "50%", border: "1px solid var(--color-border-subtle)", background: "rgba(255,255,255,0.06)", color: "var(--color-text-secondary)", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
                          </button>
                        </div>
                      )}
                    </div>
                    {creditsLoading && cast.length === 0 ? (
                      <div style={{ color: "var(--color-text-muted)", fontSize: "13px", padding: "8px 0" }}>Loading…</div>
                    ) : (
                      <div ref={castRowRef} style={{ display: "flex", gap: "12px", overflowX: "auto", paddingBottom: "0", scrollbarWidth: "none", msOverflowStyle: "none" }}>
                        {cast.map((c) => (
                          <button
                            key={c.id}
                            onClick={() => setActivePersonId(c.id)}
                            style={{
                              width: "92px", flexShrink: 0, textAlign: "center", textDecoration: "none", color: "inherit",
                              background: "none", border: "none", padding: 0, cursor: "pointer", outline: "none", fontFamily: "inherit"
                            }}
                          >
                            <div style={{
                              width: "92px",
                              height: "92px",
                              borderRadius: "50%",
                              overflow: "hidden",
                              background: "var(--color-surface)",
                              marginBottom: "8px",
                              position: "relative",
                            }}>
                              {c.profile_path ? (
                                <Image
                                  src={posterUrl(c.profile_path, "w185")}
                                  alt={c.name}
                                  fill
                                  sizes="92px"
                                  style={{ objectFit: "cover" }}

                                />
                              ) : (
                                <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: "24px" }}>
                                  {c.name?.[0] || "?"}
                                </div>
                              )}
                            </div>
                            <div style={{ fontSize: "12px", color: "var(--color-text-primary)", fontWeight: 500, lineHeight: 1.25, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
                              {c.name}
                            </div>
                            {c.character && (
                              <div style={{ fontSize: "11px", color: "var(--color-text-muted)", marginTop: "2px", lineHeight: 1.25, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
                                {c.character}
                              </div>
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* ── More Like This ───────────────────────── */}
            {(similarLoading || similar.length > 0) && (
              <div style={{
                padding: "4px 20px 20px",
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
                        onClick={() => similarRowRef.current?.scrollBy({ left: -(window.innerWidth * 0.8), behavior: "smooth" })}
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
                        onClick={() => similarRowRef.current?.scrollBy({ left: (window.innerWidth * 0.8), behavior: "smooth" })}
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
