"use client";

/**
 * Dashboard shelving system.
 */

import { memo, useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { createPortal } from "react-dom";
import { useMounted } from "@/lib/useMounted";

import { languageLabel, recommendationId, type Recommendation } from "@/lib/api";
import { usePoster, useBackdrop, prefetchBackdrops } from "@/lib/usePoster";
import type { Shelf } from "./shelves";

export type QuickAction = "dislike" | "like" | "love" | "watchlist";

const NOW_MS = new Date().getTime();

/** "142" → "2h 22m"; empty string hides the segment when runtime unknown. */
function formatRuntime(minutes?: number | string | null): string {
  if (!minutes) return "";
  const num = typeof minutes === "number" ? minutes : parseFloat(String(minutes).replace(/[^\d.]/g, ""));
  if (!Number.isFinite(num) || num <= 0) return "";
  const h = Math.floor(num / 60);
  const m = Math.round(num % 60);
  return h > 0 ? `${h}h${m > 0 ? ` ${m}m` : ""}` : `${m}m`;
}

interface ShelfRowProps {
  shelf: Shelf;
  index: number;
  /** First visible shelf renders images eagerly (LCP candidate). */
  priority?: boolean;
  onOpenMovie: (movie: Recommendation) => void;
  onOpenShelf: (shelf: Shelf) => void;
  onQuickAction: (movie: Recommendation, action: QuickAction) => void;
}

/* ─── Section header ──────────────────────────────────────────────── */

function SectionHeader({ shelf, onSeeAll }: { shelf: Shelf; onSeeAll: () => void }) {
  const canSeeAll = !shelf.hideSeeAll;

  return (
    <div className="shelf-header">
      <div style={{ minWidth: 0 }}>
        {shelf.eyebrow && (
          <span className="shelf-eyebrow">{shelf.eyebrow}</span>
        )}
        <h3
          className="heading-section shelf-title"
          onClick={canSeeAll ? onSeeAll : undefined}
          style={{
            cursor: canSeeAll ? "pointer" : "default",
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
          }}
          role={canSeeAll ? "button" : undefined}
          tabIndex={canSeeAll ? 0 : undefined}
          onKeyDown={canSeeAll ? (e) => { if (e.key === "Enter" || e.key === " ") onSeeAll(); } : undefined}
        >
          <span>{shelf.title}</span>
          {canSeeAll && (
            <span
              className="shelf-title-chevron"
              style={{ fontSize: "14px", display: "inline-flex", alignItems: "center", opacity: 0.6, transition: "transform 200ms ease, opacity 200ms ease" }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </span>
          )}
        </h3>
      </div>
      {canSeeAll && (
        <button
          type="button"
          className="glass-pill shelf-see-all"
          onClick={onSeeAll}
          aria-label={`See all ${shelf.title}`}
        >
          <span>See all</span>
          <svg
            width="11"
            height="11"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
            style={{ display: "inline-block", flexShrink: 0 }}
          >
            <polyline points="9 6 15 12 9 18" />
          </svg>
        </button>
      )}
    </div>
  );
}

/* ─── Poster card ─────────────────────────────────────────────────── */

function PosterCard({
  movie,
  rank,
  priority,
  onOpen,
  onQuickAction,
}: {
  movie: Recommendation;
  rank?: number;
  priority?: boolean;
  onOpen: () => void;
  onQuickAction?: (movie: Recommendation, action: QuickAction) => void;
}) {
  const tmdbId = recommendationId(movie);
  const poster = usePoster(movie.poster_path, tmdbId, "w342", movie.original_language);
  const backdrop = useBackdrop(movie.backdrop_path, tmdbId, "w780");
  const previewImage = backdrop.src || poster;
  const imdb = movie.imdb_rating ? movie.imdb_rating.toFixed(1) : null;
  const [isHovered, setIsHovered] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number; width: number } | null>(null);
  const hoverTimerRef = useRef<NodeJS.Timeout | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const mounted = useMounted();

  const handleMouseEnter = () => {
    if (typeof window !== "undefined" && window.innerWidth >= 1200 && window.matchMedia("(hover: hover)").matches) {
      hoverTimerRef.current = setTimeout(() => {
        if (cardRef.current) {
          const rect = cardRef.current.getBoundingClientRect();
          // Suppress if the card is under or near the navigation arrow zones
          if (rect.left < 80 || rect.right > window.innerWidth - 80) {
            return;
          }
          const targetW = 340;
          const targetH = 345;
          const computedLeft = Math.max(
            16,
            Math.min(window.innerWidth - targetW - 16, rect.left + rect.width / 2 - targetW / 2)
          );
          // Center vertically on hovered card, then safely clamp to viewport top/bottom
          let computedTop = rect.top + rect.height / 2 - targetH / 2;
          computedTop = Math.max(16, Math.min(window.innerHeight - targetH - 16, computedTop));
          setCoords({ top: computedTop, left: computedLeft, width: targetW });
          setIsHovered(true);
        }
      }, 700);
    }
  };

  const handleMouseLeave = () => {
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    setIsHovered(false);
    setCoords(null);
  };

  // When user enters the action buttons area (Rate / Watchlist), immediately disable/cancel preview popup
  const handleActionsMouseEnter = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (hoverTimerRef.current) {
      clearTimeout(hoverTimerRef.current);
      hoverTimerRef.current = null;
    }
    setIsHovered(false);
  };

  // Close preview immediately on scroll
  useEffect(() => {
    if (!isHovered) return;
    const handleScroll = () => {
      handleMouseLeave();
    };
    window.addEventListener("scroll", handleScroll, { passive: true, capture: true });
    return () => window.removeEventListener("scroll", handleScroll, { capture: true });
  }, [isHovered]);

  const handleOpen = () => {
    onOpen();
  };

  const handleQuick = (action: QuickAction) => {
    handleMouseLeave();
    onQuickAction?.(movie, action);
  };

  const handlePortalQuick = (action: QuickAction) => {
    handleMouseLeave();
    onQuickAction?.(movie, action);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleOpen();
    }
  };

  const genres = movie.genres?.length ? movie.genres.slice(0, 3) : movie.primary_genre ? [movie.primary_genre] : [];

  return (
    <div
      ref={cardRef}
      className="shelf-card-wrap"
      style={{ position: "relative" }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div
        role="button"
        tabIndex={0}
        aria-label={`${movie.title}${yearLabel(movie) ? ` (${yearLabel(movie)})` : ""} — open details`}
        onClick={handleOpen}
        onKeyDown={onKeyDown}
        className="shelf-card"
      >
        <div className="shelf-poster-box">
          {rank != null && (
            <span className="shelf-rank-num" aria-hidden>{rank}</span>
          )}

          <img
            src={poster}
            alt=""
            loading={priority ? "eager" : "lazy"}
            decoding="async"
            draggable={false}
          />
          {(imdb || yearLabel(movie)) && (
            <span className={`shelf-card-badge${rank != null ? " ranked-badge" : ""}`}>
              {imdb ? `★ ${imdb}` : yearLabel(movie)}
            </span>
          )}

          {/* Standard Quick Actions on Poster */}
          {onQuickAction && (
            <div className="shelf-card-actions" onMouseEnter={handleActionsMouseEnter}>
              <div className="shelf-reaction-group">
                <div className="shelf-reaction-tray" role="group" aria-label="Reaction options">
                  <button
                    type="button"
                    className="shelf-reaction-item"
                    aria-label={`Dislike ${movie.title}`}
                    onClick={(e) => { e.stopPropagation(); e.currentTarget.blur(); handleQuick("dislike"); }}
                  >
                    <span aria-hidden>🙁</span>
                    <span className="shelf-tooltip">Not for me</span>
                  </button>
                  <button
                    type="button"
                    className="shelf-reaction-item"
                    aria-label={`Like ${movie.title}`}
                    onClick={(e) => { e.stopPropagation(); e.currentTarget.blur(); handleQuick("like"); }}
                  >
                    <span aria-hidden>😀</span>
                    <span className="shelf-tooltip">I like this</span>
                  </button>
                  <button
                    type="button"
                    className="shelf-reaction-item"
                    aria-label={`Love ${movie.title}`}
                    onClick={(e) => { e.stopPropagation(); e.currentTarget.blur(); handleQuick("love"); }}
                  >
                    <span aria-hidden>😍</span>
                    <span className="shelf-tooltip">Love this!</span>
                  </button>
                </div>

                <button
                  type="button"
                  className="shelf-action-btn shelf-action-btn--reaction"
                  aria-label={`Rate ${movie.title}`}
                  onClick={(e) => { e.stopPropagation(); e.currentTarget.blur(); handleQuick("like"); }}
                >
                  <span aria-hidden style={{ fontSize: 16 }}>😀</span>
                  <span className="shelf-tooltip">Rate</span>
                </button>
              </div>

              <button
                type="button"
                className="shelf-action-btn shelf-action-btn--watchlist"
                aria-label={`Add ${movie.title} to watchlist`}
                onClick={(e) => { e.stopPropagation(); e.currentTarget.blur(); handleQuick("watchlist"); }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                  <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
                </svg>
                <span className="shelf-tooltip">Add to Watchlist</span>
              </button>
            </div>
          )}
        </div>

        <div className="shelf-card-info">
          <p className="poster-info-title">{movie.title}</p>
          <p className="poster-info-meta-text">
            {[yearLabel(movie), movie.original_language ? languageLabel(movie.original_language) : "", formatRuntime(movie.runtime)]
              .filter(Boolean)
              .join(" · ")}
          </p>
        </div>
      </div>

      {/* ── EXPANDED PREVIEW PORTAL (Hover delay popup, viewport-clamped & smooth) ── */}
      {mounted && isHovered && coords && createPortal(
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0, scale: 0.92, y: 16, filter: "blur(8px)" }}
            animate={{ opacity: 1, scale: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0, scale: 0.94, y: 8, transition: { duration: 0.22 } }}
            transition={{ duration: 0.58, ease: [0.16, 1, 0.3, 1] }}
            onMouseEnter={() => {
              if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
              setIsHovered(true);
            }}
            onMouseLeave={handleMouseLeave}
            style={{
              position: "fixed",
              top: coords.top,
              left: coords.left,
              width: coords.width,
              zIndex: 99999,
              borderRadius: "16px",
              background: "rgba(18, 19, 26, 0.98)",
              border: "1px solid rgba(255, 255, 255, 0.14)",
              boxShadow: "0 24px 60px rgba(0, 0, 0, 0.9), 0 4px 20px rgba(0, 0, 0, 0.6)",
              overflow: "hidden",
              pointerEvents: "auto",
              transformOrigin: "center center",
              willChange: "transform, opacity",
            }}
          >
            {/* Landscape Header (16:9) */}
            <div
              style={{
                position: "relative",
                width: "100%",
                height: "170px",
                overflow: "hidden",
                cursor: "pointer",
                background: "#0c0d12",
              }}
              onClick={() => {
                handleMouseLeave();
                onOpen();
              }}
            >
              <img
                src={previewImage}
                alt={movie.title}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  objectPosition: "center 25%",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background:
                    "linear-gradient(to top, rgba(18, 19, 26, 1) 0%, rgba(18, 19, 26, 0.45) 45%, transparent 100%)",
                }}
              />

              {/* Floating IMDb badge */}
              {imdb && (
                <span
                  style={{
                    position: "absolute",
                    top: "10px",
                    right: "10px",
                    background: "rgba(10, 11, 16, 0.75)",
                    backdropFilter: "blur(8px)",
                    WebkitBackdropFilter: "blur(8px)",
                    border: "1px solid rgba(255, 255, 255, 0.15)",
                    borderRadius: "6px",
                    padding: "3px 8px",
                    fontSize: "11.5px",
                    fontWeight: 700,
                    color: "var(--color-rating, #f5c518)",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "3px",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.5)",
                  }}
                >
                  <span style={{ fontSize: "11px" }}>★</span> {imdb}
                </span>
              )}

              {/* Optional rank badge */}
              {rank != null && (
                <span
                  style={{
                    position: "absolute",
                    top: "10px",
                    left: "10px",
                    background: "linear-gradient(135deg, #a855f7, #6366f1)",
                    borderRadius: "6px",
                    padding: "3px 8px",
                    fontSize: "11px",
                    fontWeight: 800,
                    color: "#ffffff",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.5)",
                  }}
                >
                  #{rank}
                </span>
              )}
            </div>

            {/* Content Body */}
            <div
              style={{
                padding: "0 18px 16px",
                marginTop: "-12px",
                position: "relative",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
              }}
            >
              <h4
                style={{
                  margin: 0,
                  fontSize: "16px",
                  fontWeight: 700,
                  color: "#ffffff",
                  lineHeight: 1.25,
                  letterSpacing: "-0.01em",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
                onClick={() => {
                  handleMouseLeave();
                  onOpen();
                }}
              >
                {movie.title}
              </h4>

              {/* Chips row */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "5px", alignItems: "center" }}>
                {(movie.status === "Upcoming" || (movie.release_date && new Date(movie.release_date).getTime() > NOW_MS)) ? (
                  <span
                    style={{
                      fontSize: "11px",
                      fontWeight: 700,
                      padding: "2px 7px",
                      borderRadius: "6px",
                      background: "rgba(99, 102, 241, 0.25)",
                      border: "1px solid rgba(99, 102, 241, 0.45)",
                      color: "#a5b4fc",
                    }}
                  >
                    Upcoming
                  </span>
                ) : yearLabel(movie) ? (
                  <span
                    style={{
                      fontSize: "11.5px",
                      fontWeight: 600,
                      padding: "2px 7px",
                      borderRadius: "6px",
                      background: "rgba(255, 255, 255, 0.08)",
                      color: "rgba(255, 255, 255, 0.88)",
                    }}
                  >
                    {yearLabel(movie)}
                  </span>
                ) : null}

                {movie.certification && (
                  <span
                    style={{
                      fontSize: "11px",
                      fontWeight: 700,
                      padding: "1px 6px",
                      borderRadius: "4px",
                      border: "1px solid rgba(255, 255, 255, 0.25)",
                      color: "rgba(255, 255, 255, 0.85)",
                    }}
                  >
                    {movie.certification}
                  </span>
                )}

                {movie.original_language && (
                  <span
                    style={{
                      fontSize: "11.5px",
                      fontWeight: 600,
                      padding: "2px 7px",
                      borderRadius: "6px",
                      background: "rgba(255, 255, 255, 0.08)",
                      color: "rgba(255, 255, 255, 0.88)",
                    }}
                  >
                    {languageLabel(movie.original_language)}
                  </span>
                )}
                {formatRuntime(movie.runtime) && (
                  <span
                    style={{
                      fontSize: "11.5px",
                      fontWeight: 500,
                      padding: "2px 7px",
                      borderRadius: "6px",
                      background: "rgba(255, 255, 255, 0.08)",
                      color: "rgba(255, 255, 255, 0.7)",
                    }}
                  >
                    {formatRuntime(movie.runtime)}
                  </span>
                )}
                {genres.slice(0, 2).map((g) => (
                  <span
                    key={g}
                    style={{
                      fontSize: "11.5px",
                      fontWeight: 500,
                      padding: "2px 7px",
                      borderRadius: "6px",
                      background: "rgba(255, 255, 255, 0.08)",
                      color: "rgba(255, 255, 255, 0.7)",
                    }}
                  >
                    {g}
                  </span>
                ))}
              </div>

              {/* Synopsis excerpt */}
              {movie.overview && (
                <p
                  style={{
                    margin: 0,
                    fontSize: "12px",
                    lineHeight: 1.45,
                    color: "rgba(255, 255, 255, 0.68)",
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}
                >
                  {movie.overview}
                </p>
              )}

              {/* Action Buttons: View, Rate emoji, Watchlist */}
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "4px" }}>
                <button
                  type="button"
                  onClick={() => {
                    handleMouseLeave();
                    onOpen();
                  }}
                  style={{
                    flex: 1,
                    height: "38px",
                    padding: "0 16px",
                    fontSize: "13px",
                    fontWeight: 700,
                    borderRadius: "999px",
                    background: "#ffffff",
                    color: "#000000",
                    border: "none",
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "6px",
                    cursor: "pointer",
                    transition: "transform 140ms ease, background-color 140ms ease",
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = "#f0f0f5"; e.currentTarget.style.transform = "scale(1.02)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = "#ffffff"; e.currentTarget.style.transform = "none"; }}
                >
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="#000000">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  <span>View Details</span>
                </button>

                {onQuickAction && (
                  <>
                    <div className="shelf-reaction-group" style={{ position: "relative" }}>
                      <div className="shelf-reaction-tray" role="group" aria-label="Reaction options" style={{ bottom: "44px" }}>
                        <button
                          type="button"
                          className="shelf-reaction-item"
                          aria-label={`Dislike ${movie.title}`}
                          onClick={() => handlePortalQuick("dislike")}
                        >
                          <span aria-hidden>🙁</span>
                          <span className="shelf-tooltip">Not for me</span>
                        </button>
                        <button
                          type="button"
                          className="shelf-reaction-item"
                          aria-label={`Like ${movie.title}`}
                          onClick={() => handlePortalQuick("like")}
                        >
                          <span aria-hidden>😀</span>
                          <span className="shelf-tooltip">I like this</span>
                        </button>
                        <button
                          type="button"
                          className="shelf-reaction-item"
                          aria-label={`Love ${movie.title}`}
                          onClick={() => handlePortalQuick("love")}
                        >
                          <span aria-hidden>😍</span>
                          <span className="shelf-tooltip">Love this!</span>
                        </button>
                      </div>

                      <button
                        type="button"
                        style={{
                          width: "38px",
                          height: "38px",
                          borderRadius: "50%",
                          background: "rgba(255, 255, 255, 0.10)",
                          border: "1px solid rgba(255, 255, 255, 0.12)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          cursor: "pointer",
                          transition: "background-color 140ms ease, transform 140ms ease",
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255, 255, 255, 0.20)"; e.currentTarget.style.transform = "scale(1.05)"; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(255, 255, 255, 0.10)"; e.currentTarget.style.transform = "none"; }}
                        aria-label={`Rate ${movie.title}`}
                        onClick={() => handlePortalQuick("like")}
                      >
                        <span aria-hidden style={{ fontSize: 16 }}>😀</span>
                      </button>
                    </div>

                    <button
                      type="button"
                      style={{
                        width: "38px",
                        height: "38px",
                        borderRadius: "50%",
                        background: "rgba(255, 255, 255, 0.10)",
                        border: "1px solid rgba(255, 255, 255, 0.12)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "#ffffff",
                        cursor: "pointer",
                        transition: "background-color 140ms ease, transform 140ms ease",
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255, 255, 255, 0.20)"; e.currentTarget.style.transform = "scale(1.05)"; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(255, 255, 255, 0.10)"; e.currentTarget.style.transform = "none"; }}
                      aria-label={`Add ${movie.title} to watchlist`}
                      onClick={() => handlePortalQuick("watchlist")}
                    >
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                        <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
                      </svg>
                    </button>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        </AnimatePresence>,
        document.body
      )}
    </div>
  );
}

function yearLabel(m: Recommendation): string {
  const y = typeof m.year === "number" ? m.year : parseInt(String(m.year ?? ""), 10);
  return Number.isFinite(y) && y > 0 ? String(y) : "";
}

/* ─── Spotlight card ──────────────────────────────────────────────── */

function SpotlightCard({
  movie,
  priority,
  onOpen,
  onQuickAction,
}: {
  movie: Recommendation;
  priority?: boolean;
  onOpen: () => void;
  onQuickAction: (movie: Recommendation, action: QuickAction) => void;
}) {
  const backdrop = useBackdrop(movie.backdrop_path, recommendationId(movie));
  const poster = usePoster(movie.poster_path, recommendationId(movie), "w342", movie.original_language);
  const hasBackdrop = !!backdrop.src;
  const imdb = movie.imdb_rating ? `★ ${movie.imdb_rating.toFixed(1)}` : (movie.vote_average ? `★ ${movie.vote_average.toFixed(1)}` : null);
  const runtime = formatRuntime(movie.runtime);
  const blurb = movie.reason?.trim() || movie.overview || "";

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={`${movie.title} — open details`}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(); }
      }}
      className="spotlight-card"
    >
      <div className="spotlight-art">
        {hasBackdrop ? (
          <img
            src={backdrop.src!}
            alt=""
            loading={priority ? "eager" : "lazy"}
            decoding="async"
            draggable={false}
          />
        ) : (
          <div className="spotlight-fallback">
            <img src={poster} alt="" className="spotlight-fallback-blur" aria-hidden draggable={false} />
            <img src={poster} alt={movie.title} className="spotlight-fallback-poster" draggable={false} />
          </div>
        )}
        <div className="spotlight-scrim" aria-hidden />
      </div>

      <div className="spotlight-body">
        <h4 className="spotlight-title">{movie.title}</h4>
        <div className="spotlight-meta">
          {yearLabel(movie) && <span>{yearLabel(movie)}</span>}
          {movie.original_language && <span>{languageLabel(movie.original_language)}</span>}
          {runtime && <span>{runtime}</span>}
          {imdb && <span className="spotlight-imdb">{imdb}</span>}
        </div>
        {blurb && <p className="spotlight-blurb">{blurb}</p>}
      </div>

      <div className="spotlight-actions" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
        <div className="shelf-reaction-group">
          <div className="shelf-reaction-tray" role="group" aria-label="Reaction options">
            <button
              type="button"
              className="shelf-reaction-item"
              aria-label={`Dislike ${movie.title}`}
              onClick={(e) => { e.stopPropagation(); onQuickAction(movie, "dislike"); }}
            >
              <span aria-hidden>🙁</span>
              <span className="shelf-tooltip">Not for me</span>
            </button>
            <button
              type="button"
              className="shelf-reaction-item"
              aria-label={`Like ${movie.title}`}
              onClick={(e) => { e.stopPropagation(); onQuickAction(movie, "like"); }}
            >
              <span aria-hidden>😀</span>
              <span className="shelf-tooltip">I like this</span>
            </button>
            <button
              type="button"
              className="shelf-reaction-item"
              aria-label={`Love ${movie.title}`}
              onClick={(e) => { e.stopPropagation(); onQuickAction(movie, "love"); }}
            >
              <span aria-hidden>😍</span>
              <span className="shelf-tooltip">Love this!</span>
            </button>
          </div>

          <button
            type="button"
            className="glass-pill"
            style={{ fontSize: 12, fontWeight: 600, padding: "7px 14px", cursor: "pointer", color: "#fff", background: "rgba(255,255,255,0.14)", display: "inline-flex", alignItems: "center", gap: "6px", whiteSpace: "nowrap" }}
            aria-label={`Rate ${movie.title}`}
            onClick={(e) => { e.stopPropagation(); onQuickAction(movie, "like"); }}
          >
            <span>😀</span> Rate
          </button>
        </div>

        <button
          type="button"
          className="glass-pill"
          style={{ fontSize: 12, fontWeight: 600, padding: "7px 14px", cursor: "pointer", color: "#fff", background: "rgba(255,255,255,0.14)", display: "inline-flex", alignItems: "center", gap: "6px", whiteSpace: "nowrap" }}
          onClick={(e) => { e.stopPropagation(); onQuickAction(movie, "watchlist"); }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
          </svg>
          Watchlist
        </button>
      </div>
    </div>
  );
}

/* ─── The row ─────────────────────────────────────────────────────── */

function ShelfRowInner({ shelf, index, priority, onOpenMovie, onOpenShelf, onQuickAction }: ShelfRowProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ start: false, end: false });

  const updateEdges = useCallback(() => {
    const t = trackRef.current;
    if (!t) return;
    const canLeft = t.scrollLeft > 6;
    const canRight = t.scrollLeft < t.scrollWidth - t.clientWidth - 6;
    setEdges((prev) => {
      if (prev.start === canLeft && prev.end === canRight) return prev;
      return { start: canLeft, end: canRight };
    });
  }, []);

  useEffect(() => {
    const t = trackRef.current;
    if (!t) return;
    t.addEventListener("scroll", updateEdges, { passive: true });
    const ro = new ResizeObserver(updateEdges);
    ro.observe(t);
    updateEdges();
    return () => {
      t.removeEventListener("scroll", updateEdges);
      ro.disconnect();
    };
  }, [updateEdges, shelf.movies.length]);

  useEffect(() => {
    if (shelf.variant === "spotlight" && shelf.movies.length > 0) {
      prefetchBackdrops(shelf.movies);
    }
  }, [shelf.variant, shelf.movies]);

  // A hidden "See all" anchor lives here so SectionHeader's pill can trigger
  // the overlay without threading another callback through the DOM order.
  const openShelf = useCallback(() => onOpenShelf(shelf), [onOpenShelf, shelf]);

  const scrollByPage = (dir: 1 | -1) => {
    const t = trackRef.current;
    if (!t) return;

    const distance = Math.max(240, Math.floor(t.clientWidth * 0.85));
    t.scrollBy({ left: dir * distance, behavior: "smooth" });
  };

  return (
    <motion.section
      className="shelf-section"
      id={`shelf-${shelf.id}`}
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "0px 0px -40px 0px" }}
      transition={{ duration: 0.35, delay: Math.min(index, 2) * 0.05, ease: [0.22, 1, 0.36, 1] }}
    >
      <SectionHeader shelf={shelf} onSeeAll={openShelf} />

      <div className="shelf-rail">
        <div ref={trackRef} className="shelf-track hide-scrollbar" role="list" aria-label={shelf.title}>
          {shelf.variant === "spotlight"
            ? shelf.movies.map((m, i) => (
              <div role="listitem" key={recommendationId(m)} className="shelf-cell spotlight-cell">
                <SpotlightCard
                  movie={m}
                  priority={priority && i < 2}
                  onOpen={() => onOpenMovie(m)}
                  onQuickAction={onQuickAction}
                />
              </div>
            ))
            : shelf.movies.map((m, i) => (
              <div role="listitem" key={recommendationId(m)} className="shelf-cell">
                <PosterCard
                  movie={m}
                  rank={shelf.variant === "ranked" ? i + 1 : undefined}
                  priority={priority && i < 4}
                  onOpen={() => onOpenMovie(m)}
                  onQuickAction={onQuickAction}
                />
              </div>
            ))}
        </div>

        {edges.start && <div className="carousel-scrim left" />}
        {edges.end && <div className="carousel-scrim right" />}

        {edges.start && (
          <button
            type="button"
            className="carousel-btn carousel-btn-left"
            aria-label={`Scroll ${shelf.title} back`}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              scrollByPage(-1);
            }}
          >
            <svg style={{ pointerEvents: "none" }} width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6" /></svg>
          </button>
        )}
        {edges.end && (
          <button
            type="button"
            className="carousel-btn carousel-btn-right"
            aria-label={`Scroll ${shelf.title} forward`}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              scrollByPage(1);
            }}
          >
            <svg style={{ pointerEvents: "none" }} width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
          </button>
        )}
      </div>
    </motion.section>
  );
}

export const ShelfRow = memo(ShelfRowInner);

/* ─── Grid item for the See-all overlay ───────────────────────────── */

export function ShelfCardGridItem({
  movie,
  onClick,
  onQuickAction,
}: {
  movie: Recommendation;
  onClick: () => void;
  onQuickAction?: (movie: Recommendation, action: QuickAction) => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ duration: 0.18, ease: "easeOut" }}
    >
      <PosterCard movie={movie} onOpen={onClick} onQuickAction={onQuickAction} />
    </motion.div>
  );
}
