"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { motion, AnimatePresence, useReducedMotion } from "framer-motion";

import { posterUrl, languageLabel, recommendationId, type Recommendation } from "@/lib/api";
import { useBackdrop, usePoster } from "@/lib/usePoster";

const ROTATE_MS = 8000;
const MAX_ITEMS = 5;

function formatRuntime(minutes?: number | string | null): string {
  if (!minutes) return "";
  const num = typeof minutes === "number" ? minutes : parseFloat(String(minutes).replace(/[^\d.]/g, ""));
  if (!Number.isFinite(num) || num <= 0) return "";
  const h = Math.floor(num / 60);
  const m = Math.round(num % 60);
  if (h > 0) {
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
  }
  return `${m}m`;
}

interface Props {
  movies: Recommendation[];
  onOpenDetail: (movie: Recommendation) => void;
  onWatchlist?: (movie: Recommendation) => void;
  onLike?: (movie: Recommendation) => void;
  onAction?: (movie: Recommendation, action: "dislike" | "like" | "love" | "watchlist") => void;
}

export default function HeroCarousel({ movies, onOpenDetail, onWatchlist, onLike, onAction }: Props) {
  const items = useMemo(() => movies.slice(0, MAX_ITEMS), [movies]);
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const wasInteractedRef = useRef(false);
  const reduceMotion = useReducedMotion();

  const [liveLogos, setLiveLogos] = useState<Record<number, string | null>>({});

  // Clamp index when the movie list itself changes (e.g. rated movie removed).
  useEffect(() => {
    Promise.resolve().then(() => setIndex((prev) => Math.min(prev, Math.max(0, items.length - 1))));
    wasInteractedRef.current = false;

    // Fetch live title logos from TMDB for the hero items
    items.forEach((m) => {
      const tmdbId = recommendationId(m);
      if (!tmdbId) return;
      fetch(`/api/tmdb?id=${tmdbId}`)
        .then((res) => res.json())
        .then((data) => {
          if (data.logo_path) {
            setLiveLogos((prev) => ({ ...prev, [m.id]: data.logo_path }));
          }
        })
        .catch(() => { });
    });
  }, [items]);

  // Auto-rotate. Pauses while `paused` is true (user touch/hover).
  useEffect(() => {
    if (items.length < 2 || paused) return;
    const t = setInterval(() => {
      setIndex((i) => (i + 1) % items.length);
    }, ROTATE_MS);
    return () => clearInterval(t);
  }, [items.length, paused]);

  const goTo = useCallback(
    (targetIndex: number) => {
      wasInteractedRef.current = true;
      setIndex(((targetIndex % items.length) + items.length) % items.length);
    },
    [items.length]
  );

  // Parallax: translate the backdrop at a fraction of scroll. rAF-throttled,
  // transform-only, skipped once the hero is off-screen.
  const parallaxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (reduceMotion) return;
    let raf = 0;
    const onScroll = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const el = parallaxRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        if (rect.bottom < 0) return;
        el.style.transform = `translate3d(0, ${Math.min(window.scrollY * 0.18, 120)}px, 0)`;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [reduceMotion]);

  const touchStartXRef = useRef<number | null>(null);
  const touchStartYRef = useRef<number | null>(null);

  const handleTouchStart = (e: React.TouchEvent) => {
    setPaused(true);
    touchStartXRef.current = e.touches[0].clientX;
    touchStartYRef.current = e.touches[0].clientY;
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (touchStartXRef.current === null || touchStartYRef.current === null) return;
    const deltaX = e.changedTouches[0].clientX - touchStartXRef.current;
    const deltaY = e.changedTouches[0].clientY - touchStartYRef.current;

    // Minimum swipe threshold (40px); horizontal swipe must beat vertical.
    if (Math.abs(deltaX) > 40 && Math.abs(deltaX) > Math.abs(deltaY)) {
      if (deltaX < 0) {
        goTo(index + 1); // Swiped left -> next title
      } else {
        goTo(index - 1); // Swiped right -> previous title
      }
    }
    touchStartXRef.current = null;
    touchStartYRef.current = null;
  };

  if (items.length === 0) return null;
  const movie = items[index];

  return (
    <section
      aria-label="Featured"
      style={{
        position: "relative",
        width: "100%",
        height: "clamp(460px, 58vh, 640px)",
        minHeight: 420,
        overflow: "hidden",
        touchAction: "pan-y",
        userSelect: "none",
        WebkitUserSelect: "none",
        WebkitMaskImage:
          "linear-gradient(180deg, #000 0%, #000 82%, transparent 100%)",
        maskImage:
          "linear-gradient(180deg, #000 0%, #000 82%, transparent 100%)",
      }}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      <div ref={parallaxRef} style={{ position: "absolute", inset: 0, willChange: "transform" }}>
        <AnimatePresence mode="popLayout">
          {movie && (
            <HeroSlide
              key={movie.id}
              movie={movie}
              liveLogo={liveLogos[movie.id] ?? null}
              onOpenDetail={onOpenDetail}
              onWatchlist={onWatchlist}
              onLike={onLike}
              onAction={onAction}
            />
          )}
        </AnimatePresence>
      </div>

      {/* Pagination dots */}
      {items.length > 1 && (
        <div
          style={{
            position: "absolute",
            bottom: "clamp(16px, 3vh, 24px)",
            left: "var(--rail-x)",
            zIndex: 10,
            display: "flex",
            gap: 2,
          }}
          aria-hidden
        >
          {items.map((_, i) => (
            <button
              key={i}
              type="button"
              onClick={() => goTo(i)}
              aria-label={`Show featured item ${i + 1}`}
              style={{
                padding: "8px 4px",
                background: "transparent",
                border: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
              }}
            >
              <span
                style={{
                  position: "relative",
                  display: "block",
                  width: i === index ? 26 : 8,
                  height: 4,
                  borderRadius: 999,
                  overflow: "hidden",
                  background: i === index ? "rgba(255,255,255,0.25)" : "rgba(255,255,255,0.30)",
                  transition: "width 240ms ease, background 240ms ease",
                }}
              >
                {i === index && !paused && !reduceMotion && (
                  <motion.span
                    key={`prog-${index}`}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: ROTATE_MS / 1000, ease: "linear" }}
                    style={{
                      position: "absolute",
                      inset: 0,
                      transformOrigin: "left center",
                      borderRadius: 999,
                      background: "rgba(255,255,255,0.95)",
                    }}
                  />
                )}
                {i === index && (paused || reduceMotion) && (
                  <span style={{ position: "absolute", inset: 0, borderRadius: 999, background: "rgba(255,255,255,0.95)" }} />
                )}
              </span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function HeroSlide({
  movie,
  liveLogo,
  onOpenDetail,
  onWatchlist,
  onLike,
  onAction,
}: {
  movie: Recommendation;
  liveLogo: string | null;
  onOpenDetail: (m: Recommendation) => void;
  onWatchlist?: (m: Recommendation) => void;
  onLike?: (m: Recommendation) => void;
  onAction?: (m: Recommendation, action: "dislike" | "like" | "love" | "watchlist") => void;
}) {
  const tmdbId = recommendationId(movie);
  const backdrop = useBackdrop(movie.backdrop_path, tmdbId, "original");
  const poster = usePoster(movie.poster_path, tmdbId, "w780", movie.original_language);

  const backdropUrl = backdrop.src || poster;
  const hasRealBackdrop = !!backdrop.src;

  const year = movie.year ? String(movie.year) : "";
  const lang = movie.original_language ? languageLabel(movie.original_language) : "";
  const runtime = formatRuntime(movie.runtime);
  const score = movie.imdb_rating ? movie.imdb_rating.toFixed(1) : (movie.vote_average ? movie.vote_average.toFixed(1) : null);
  const chips = [year, lang, runtime, score ? `★ ${score}` : ""].filter(Boolean);

  return (
    <motion.div
      key={movie.id}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
      style={{ position: "absolute", inset: 0 }}
    >
      {/* Background layer with Ken Burns animation */}
      <motion.div
        initial={{ scale: 1.06 }}
        animate={{ scale: 1 }}
        transition={{ duration: 7.5, ease: "easeOut" }}
        style={{ position: "absolute", inset: "-4% 0" }}
      >
        <img
          src={backdropUrl}
          alt=""
          className={`hero-backdrop-img ${!hasRealBackdrop ? "hero-backdrop-blur" : ""}`}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            objectPosition: "center 22%",
            ...(!hasRealBackdrop && { filter: "blur(24px) saturate(1.4)", transform: "scale(1.1)" }),
          }}
        />
        <div
          aria-hidden
          style={{
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(180deg, rgba(5,5,7,0.45) 0%, rgba(5,5,7,0.02) 20%, rgba(5,5,7,0.0) 42%, rgba(5,5,7,0.35) 62%, rgba(5,5,7,0.85) 84%, var(--color-bg) 100%), radial-gradient(ellipse 100% 70% at 50% 30%, transparent 45%, rgba(5,5,7,0.45) 100%)",
          }}
        />
      </motion.div>

      {/* Foreground copy layer */}
      <div
        style={{
          position: "relative",
          zIndex: 2,
          height: "100%",
          padding: "0 var(--rail-x) clamp(44px, 6.5vh, 60px)",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
          maxWidth: 760,
        }}
      >
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.35, ease: "easeOut" }}
        >

          <div
            style={{
              minHeight: "54px",
              display: "flex",
              alignItems: "flex-end",
              margin: "4px 0 8px",
            }}
          >
            {liveLogo ? (
              <img
                src={posterUrl(liveLogo, "w500")}
                alt={movie.title}
                style={{
                  maxHeight: "clamp(50px, 8.5vw, 68px)",
                  maxWidth: "min(320px, 78vw)",
                  width: "auto",
                  height: "auto",
                  objectFit: "contain",
                  filter: "drop-shadow(0 4px 18px rgba(0,0,0,0.95)) drop-shadow(0 1px 4px rgba(0,0,0,0.8))",
                }}
              />
            ) : (
              <h1
                style={{
                  margin: 0,
                  fontSize: "clamp(24px, 5vw, 40px)",
                  fontWeight: 800,
                  letterSpacing: "-0.03em",
                  lineHeight: 1.1,
                  color: "#fff",
                  textShadow: "0 2px 14px rgba(0,0,0,0.85)",
                  maxWidth: 620,
                }}
              >
                {movie.title}
              </h1>
            )}
          </div>

          {chips.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 10 }} aria-hidden>
              {chips.map((c) => (
                <span key={c} className="hero-chip">
                  {c}
                </span>
              ))}
            </div>
          )}

          {movie.overview && (
            <p
              className="desktop-only"
              style={{
                margin: "12px 0 0",
                fontSize: 14,
                lineHeight: 1.55,
                color: "rgba(255,255,255,0.85)",
                maxWidth: 540,
                WebkitLineClamp: 3,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                textShadow: "0 1px 8px rgba(0,0,0,0.7)",
              }}
            >
              {movie.overview}
            </p>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: "clamp(6px, 2vw, 10px)", marginTop: 18, flexWrap: "nowrap" }}>
            <button
              type="button"
              className="btn btn-primary"
              style={{ minHeight: 44, padding: "0 clamp(12px, 3vw, 20px)", whiteSpace: "nowrap" }}
              onClick={() => {
                onOpenDetail(movie);
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2" />
                <line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                <circle cx="12" cy="16" r="1" fill="currentColor" />
              </svg>
              More info
            </button>
            {onWatchlist && (
              <button
                type="button"
                className="btn btn-secondary"
                style={{ minHeight: 44, padding: "0 clamp(12px, 2.5vw, 18px)", whiteSpace: "nowrap" }}
                onClick={() => {
                  onWatchlist(movie);
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
                </svg>
                Watchlist
              </button>
            )}
            {(onAction || onLike) && (
              <HeroReactionButton
                movie={movie}
                onAction={onAction}
                onLike={onLike}
              />
            )}
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

function HeroReactionButton({
  movie,
  onAction,
  onLike,
}: {
  movie: Recommendation;
  onAction?: (movie: Recommendation, action: "dislike" | "like" | "love" | "watchlist") => void;
  onLike?: (movie: Recommendation) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsOpen(true);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => {
      setIsOpen(false);
    }, 200);
  };

  // Close tray when clicking/tapping outside on touch/mobile screens
  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e: MouseEvent | TouchEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("touchstart", handleClickOutside, { passive: true });
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("touchstart", handleClickOutside);
    };
  }, [isOpen]);

  const handleSelect = (action: "dislike" | "like" | "love") => {
    setIsOpen(false);
    if (onAction) onAction(movie, action);
    else if (onLike) onLike(movie);
  };

  const handleTriggerClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsOpen((prev) => !prev);
  };

  return (
    <div
      ref={containerRef}
      className="shelf-reaction-group"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{ position: "relative", zIndex: 20 }}
    >
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="shelf-reaction-tray"
            initial={{ opacity: 0, y: 8, scale: 0.9, x: "-50%" }}
            animate={{ opacity: 1, y: 0, scale: 1, x: "-50%" }}
            exit={{ opacity: 0, y: 6, scale: 0.9, x: "-50%" }}
            transition={{ duration: 0.16, ease: "easeOut" }}
            style={{
              position: "absolute",
              bottom: "calc(100% + 10px)",
              left: "50%",
              pointerEvents: "auto",
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "6px 10px",
              borderRadius: 999,
              background: "rgba(12, 14, 20, 0.96)",
              backdropFilter: "blur(16px)",
              WebkitBackdropFilter: "blur(16px)",
              border: "1px solid rgba(255, 255, 255, 0.2)",
              boxShadow: "0 12px 28px rgba(0, 0, 0, 0.65), 0 2px 8px rgba(0, 0, 0, 0.4)",
              zIndex: 50,
            }}
            role="group"
            aria-label="Reaction options"
          >
            <button
              type="button"
              className="shelf-reaction-item"
              aria-label={`Dislike ${movie.title}`}
              onClick={(e) => {
                e.stopPropagation();
                handleSelect("dislike");
              }}
              style={{ width: 36, height: 36, fontSize: 20 }}
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
                handleSelect("like");
              }}
              style={{ width: 36, height: 36, fontSize: 20 }}
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
                handleSelect("love");
              }}
              style={{ width: 36, height: 36, fontSize: 20 }}
            >
              <span aria-hidden>😍</span>
              <span className="shelf-tooltip">Love this!</span>
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        type="button"
        className="btn btn-secondary hero-rate-btn"
        style={{
          minHeight: 44,
          minWidth: 44,
          padding: "0 clamp(10px, 2vw, 16px)",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 6,
          borderRadius: 999,
          whiteSpace: "nowrap",
        }}
        aria-label={`Rate ${movie.title}`}
        aria-expanded={isOpen}
        onClick={handleTriggerClick}
      >
        <span aria-hidden style={{ fontSize: 18, lineHeight: 1 }}>😀</span>
        <span className="desktop-only" style={{ fontSize: 13, fontWeight: 600 }}>Rate</span>
      </button>
    </div>
  );
}
