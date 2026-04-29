"use client";

/**
 * Netflix-style horizontal rail of compact posters. Used for the secondary
 * dashboard buckets (Hollywood / Global) so they don't compete with the
 * primary swipeable Matched stack — small posters mean ~3 visible per phone
 * viewport instead of 2, and the user can passively scroll without committing
 * to a like/dislike action.
 *
 * Click on a poster opens the detail modal (same flow as cast cards on
 * /person and grid cards on /explore). The "View all" header click reuses the
 * existing stack-detail overlay via `onOpenDetail`.
 */

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";

import MovieCard from "@/components/MovieCard";
import type { Recommendation } from "@/lib/api";

interface Props {
  label: string;
  subtitle?: string;
  movies: Recommendation[];
  onMovieClick: (movie: Recommendation) => void;
  /** Optional handler for the "View all →" header link (opens stack overlay). */
  onOpenDetail?: () => void;
}

export default function CompactRail({ label, subtitle, movies, onMovieClick, onOpenDetail }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [scrollState, setScrollState] = useState({ canLeft: false, canRight: false });

  useEffect(() => {
    const t = trackRef.current;
    if (!t) return;
    const update = () => {
      const { scrollLeft, scrollWidth, clientWidth } = t;
      setScrollState({
        canLeft: scrollLeft > 2,
        canRight: scrollLeft < scrollWidth - clientWidth - 2,
      });
    };
    t.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(t);
    update();
    return () => {
      t.removeEventListener("scroll", update);
      ro.disconnect();
    };
  }, [movies.length]);

  const scroll = (dir: "left" | "right") => {
    const t = trackRef.current;
    if (!t) return;
    t.scrollBy({ left: (dir === "right" ? 1 : -1) * t.clientWidth * 0.85, behavior: "smooth" });
  };

  if (movies.length === 0) return null;

  return (
    <section style={{ width: "100%", overflow: "hidden", position: "relative" }}>
      {/* Header */}
      <div
        style={{
          padding: "0 20px",
          marginBottom: 12,
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <div style={{ minWidth: 0, flex: 1 }}>
          {onOpenDetail ? (
            <button
              type="button"
              onClick={onOpenDetail}
              style={{
                background: "none",
                border: "none",
                padding: 0,
                cursor: "pointer",
                color: "var(--color-text-primary)",
                textAlign: "left",
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
              }}
            >
              <h3
                style={{
                  margin: 0,
                  fontSize: "clamp(1.05rem, 2.2vw, 1.3rem)",
                  fontWeight: 600,
                  letterSpacing: "-0.02em",
                  lineHeight: 1.1,
                }}
              >
                {label}
              </h3>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
                <path d="M5 3L9 7L5 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          ) : (
            <h3 style={{ margin: 0, fontSize: "clamp(1.05rem, 2.2vw, 1.3rem)", fontWeight: 600, letterSpacing: "-0.02em" }}>
              {label}
            </h3>
          )}
          {subtitle && (
            <p style={{ margin: "2px 0 0", fontSize: 12, color: "var(--color-text-muted)" }}>{subtitle}</p>
          )}
        </div>

        {/* Desktop chevrons */}
        <div className="compact-rail-chevrons" style={{ display: "none", gap: 6 }}>
          <button
            type="button"
            onClick={() => scroll("left")}
            disabled={!scrollState.canLeft}
            aria-label="Scroll left"
            style={chevronStyle(scrollState.canLeft)}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => scroll("right")}
            disabled={!scrollState.canRight}
            aria-label="Scroll right"
            style={chevronStyle(scrollState.canRight)}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>
        </div>
      </div>

      {/* Track */}
      <div
        ref={trackRef}
        className="hide-scrollbar"
        style={{
          display: "flex",
          gap: 12,
          overflowX: "auto",
          scrollSnapType: "x proximity",
          padding: "4px 20px 4px",
          scrollPaddingLeft: 20,
        }}
      >
        {movies.map((m) => (
          <motion.button
            key={m.tmdb_id ?? m.id}
            onClick={() => onMovieClick(m)}
            whileTap={{ scale: 0.97 }}
            style={{
              width: "min(32vw, 130px)",
              minWidth: 110,
              flexShrink: 0,
              scrollSnapAlign: "start",
              background: "none",
              border: "none",
              padding: 0,
              cursor: "pointer",
              textAlign: "left",
            }}
          >
            <MovieCard movie={m} compact noLayout />
          </motion.button>
        ))}
        {/* "See all" tail card on mobile when there are many items. */}
        {onOpenDetail && movies.length >= 8 && (
          <Link
            href="#"
            onClick={(e) => { e.preventDefault(); onOpenDetail(); }}
            style={{
              width: "min(32vw, 130px)",
              minWidth: 110,
              flexShrink: 0,
              scrollSnapAlign: "start",
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.10)",
              background: "rgba(28,30,36,0.66)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
              padding: "20px 12px",
              color: "var(--color-text-primary)",
              textDecoration: "none",
              fontSize: 13,
              fontWeight: 600,
              aspectRatio: "2 / 3",
            }}
          >
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6" />
            </svg>
            See all
          </Link>
        )}
      </div>

      <style>{`
        @media (min-width: 900px) {
          .compact-rail-chevrons { display: flex !important; }
        }
      `}</style>
    </section>
  );
}

function chevronStyle(active: boolean): React.CSSProperties {
  return {
    width: 32,
    height: 32,
    borderRadius: "50%",
    border: "1px solid rgba(255,255,255,0.10)",
    background: "rgba(28,30,36,0.66)",
    color: active ? "var(--color-text-primary)" : "var(--color-text-muted)",
    cursor: active ? "pointer" : "default",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    opacity: active ? 1 : 0.4,
    transition: "opacity 160ms ease, color 160ms ease",
  };
}
