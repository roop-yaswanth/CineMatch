"use client";

/**
 * Top-of-dashboard hero — anchors the page emotionally so it doesn't launch
 * straight into rows of cards. Picks the top N picks from a single Stack and
 * auto-rotates them every ROTATE_MS, pausing on user interaction.
 *
 * Visual:
 *   - 56vh on mobile, 64vh capped at 580px on desktop.
 *   - Full-bleed backdrop with a title gradient.
 *   - "More info" opens the detail modal; "Watchlist" fires a quick action.
 *   - Pagination dots at the bottom.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";

import { posterUrl, type Recommendation } from "@/lib/api";

const ROTATE_MS = 8000;
const MAX_ITEMS = 5;

interface Props {
  movies: Recommendation[];
  onOpenDetail: (movie: Recommendation) => void;
  onWatchlist?: (movie: Recommendation) => void;
}

export default function HeroFeature({ movies, onOpenDetail, onWatchlist }: Props) {
  const items = movies.slice(0, MAX_ITEMS);
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const wasInteractedRef = useRef(false);

  const [liveBackdrops, setLiveBackdrops] = useState<Record<number, string | null>>({});

  // Reset when the movie list itself changes (e.g. new recommendations).
  useEffect(() => {
    setIndex(0);
    wasInteractedRef.current = false;
    
    // Fetch live high-quality backdrops from TMDB for the hero items
    items.forEach((m) => {
      const tmdbId = m.tmdb_id || m.id;
      if (!tmdbId) return;
      fetch(`/api/tmdb?id=${tmdbId}`)
        .then((res) => res.json())
        .then((data) => {
          if (data.backdrop_path) {
            setLiveBackdrops((prev) => ({ ...prev, [m.id]: data.backdrop_path }));
          }
        })
        .catch(() => {});
    });
  }, [items.length, items[0]?.id]);

  // Auto-rotate. Pauses while `paused` is true (user touch/hover).
  useEffect(() => {
    if (items.length < 2 || paused) return;
    const t = setInterval(() => {
      setIndex((i) => (i + 1) % items.length);
    }, ROTATE_MS);
    return () => clearInterval(t);
  }, [items.length, paused]);

  const goTo = useCallback((i: number) => {
    wasInteractedRef.current = true;
    setIndex(((i % items.length) + items.length) % items.length);
  }, [items.length]);

  if (items.length === 0) return null;
  const movie = items[index];
  const backdropPath = liveBackdrops[movie.id] || movie.backdrop_path;
  const backdropUrl = backdropPath ? posterUrl(backdropPath, "original") : "/poster_placeholder.svg";

  return (
    <section
      aria-label="Featured"
      style={{
        position: "relative",
        width: "100%",
        height: "min(64vh, 580px)",
        minHeight: 360,
        overflow: "hidden",
        marginBottom: 24,
        // Soft mask into the page background — avoids a hard horizontal seam
        // where the hero ends and the rails begin.
        WebkitMaskImage:
          "linear-gradient(180deg, #000 0%, #000 78%, transparent 100%)",
        maskImage:
          "linear-gradient(180deg, #000 0%, #000 78%, transparent 100%)",
      }}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onTouchStart={() => setPaused(true)}
    >
      <AnimatePresence mode="popLayout">
        <motion.div
          key={movie.id}
          initial={{ opacity: 0, scale: 1.04 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          style={{ position: "absolute", inset: 0 }}
        >
          <Image
            src={backdropUrl}
            alt=""
            fill
            priority={index === 0}
            sizes="100vw"
            style={{ objectFit: "cover", objectPosition: "center 15%" }}
          />
          {/* Dual gradient — bottom to anchor text, left for desktop side-text legibility. */}
          <div
            aria-hidden
            style={{
              position: "absolute",
              inset: 0,
              background:
                "linear-gradient(180deg, rgba(0,0,0,0) 35%, rgba(0,0,0,0.55) 75%, rgba(0,0,0,0.92) 100%), linear-gradient(90deg, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.10) 45%, rgba(0,0,0,0) 70%)",
            }}
          />
        </motion.div>
      </AnimatePresence>

      {/* Foreground content */}
      <div
        style={{
          position: "relative",
          zIndex: 2,
          height: "100%",
          padding: "0 20px 28px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "flex-end",
          maxWidth: 720,
        }}
      >
        <AnimatePresence mode="wait">
          <motion.div
            key={`copy-${movie.id}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          >
            <div
              style={{
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: "0.12em",
                color: "rgba(255,255,255,0.7)",
                fontWeight: 600,
                marginBottom: 8,
              }}
            >
              Featured for you
            </div>
            <h1
              style={{
                margin: 0,
                fontSize: "clamp(26px, 6vw, 44px)",
                fontWeight: 800,
                letterSpacing: "-0.03em",
                lineHeight: 1.05,
                color: "#fff",
                textShadow: "0 2px 10px rgba(0,0,0,0.5)",
                maxWidth: 600,
              }}
            >
              {movie.title}
            </h1>
            {movie.overview && (
              <p
                style={{
                  margin: "12px 0 0",
                  fontSize: 14,
                  lineHeight: 1.5,
                  color: "rgba(255,255,255,0.85)",
                  maxWidth: 480,
                  display: "-webkit-box",
                  WebkitLineClamp: 3,
                  WebkitBoxOrient: "vertical",
                  overflow: "hidden",
                  textShadow: "0 1px 6px rgba(0,0,0,0.6)",
                }}
              >
                {movie.overview}
              </p>
            )}

            <div style={{ display: "flex", gap: 10, marginTop: 18 }}>
              <button type="button" className="btn btn-primary" onClick={() => onOpenDetail(movie)}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                  <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2" />
                  <line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  <circle cx="12" cy="16" r="1" fill="currentColor" />
                </svg>
                More info
              </button>
              {onWatchlist && (
                <button type="button" className="btn btn-secondary" onClick={() => onWatchlist(movie)}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
                  </svg>
                  Watchlist
                </button>
              )}
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Pagination dots */}
        {items.length > 1 && (
          <div style={{ marginTop: 20, display: "flex", gap: 6 }} aria-hidden>
            {items.map((_, i) => (
              <button
                key={i}
                type="button"
                onClick={() => goTo(i)}
                aria-label={`Show featured item ${i + 1}`}
                style={{
                  width: i === index ? 22 : 8,
                  height: 4,
                  padding: 0,
                  borderRadius: 999,
                  border: "none",
                  background: i === index ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.30)",
                  cursor: "pointer",
                  transition: "width 240ms ease, background 240ms ease",
                }}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
