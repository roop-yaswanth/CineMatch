"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";

import BackButton from "@/components/ui/BackButton";
import { recommendationId, type Recommendation } from "@/lib/api";
import { pushBackHandler } from "@/lib/backStack";
import { useMounted } from "@/lib/useMounted";
import { ShelfCardGridItem } from "./ShelfRow";

export interface Collection {
  id: string;
  title: string;
  subtitle?: string;
  movies: Recommendation[];
}

interface Props {
  collection: Collection;
  onBack: () => void;
  onMovieClick: (movie: Recommendation) => void;
  onQuickAction?: (movie: Recommendation, action: "dislike" | "like" | "love" | "watchlist") => void;
  /** Fetch the next batch when the user reaches the end. Return new items,
   *  or null when exhausted — the overlay then shows "You've seen it all". */
  onLoadMore?: () => Promise<Recommendation[] | null>;
}

export default function CollectionOverlay({ collection, onBack, onMovieClick, onQuickAction, onLoadMore }: Props) {
  const mounted = useMounted();

  // Snapshot the collection: live shelf updates (refills, actions elsewhere)
  // must never shrink or close an open overlay while the user is browsing.
  const [items, setItems] = useState<Recommendation[]>(() => [...collection.movies]);
  const [loadingMore, setLoadingMore] = useState(false);
  const [exhausted, setExhausted] = useState(!onLoadMore);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const loadMore = useCallback(async () => {
    if (!onLoadMore || loadingMore || exhausted) return;
    setLoadingMore(true);
    try {
      const more = await onLoadMore();
      if (more && more.length > 0) {
        setItems((prev) => {
          const seen = new Set(prev.map(recommendationId));
          const fresh = more.filter((m) => !seen.has(recommendationId(m)));
          return fresh.length ? [...prev, ...fresh] : prev;
        });
        if (more.length < 10) setExhausted(true);
      } else {
        setExhausted(true);
      }
    } catch {
      setExhausted(true);
    } finally {
      setLoadingMore(false);
    }
  }, [onLoadMore, loadingMore, exhausted]);

  // Infinite scroll sentinel
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || exhausted) return;
    const io = new IntersectionObserver(
      (entries) => { if (entries.some((e) => e.isIntersecting)) void loadMore(); },
      { rootMargin: "600px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [loadMore, exhausted, items.length]);

  useEffect(() => {
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, []);

  // Centralized back-handler: back swipe/button closes this overlay without
  // conflicting with MovieDetailModal's own handler (they share one listener).
  useEffect(() => {
    const cleanup = pushBackHandler(onBack);
    return cleanup;
  }, [onBack]);

  const content = (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.25 }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 50,
        overflowY: "auto",
        overflowX: "hidden",
        overscrollBehavior: "none",
      }}
      className="dash-root"
    >
      <div
        className="glass"
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          padding: "calc(18px + env(safe-area-inset-top, 0px)) clamp(20px, 4vw, 40px) 14px",
          display: "flex",
          alignItems: "center",
          gap: "14px",
        }}
      >
        <BackButton onClick={onBack} />
        <div>
          <h2 className="heading-section" style={{ fontSize: "20px", margin: 0, color: "var(--color-text-primary)" }}>
            {collection.title}
          </h2>
        </div>
      </div>

      <div
        className="app-container collection-grid"
        style={{
          padding: "24px clamp(20px, 4vw, 40px) calc(60px + env(safe-area-inset-bottom))",
        }}
      >
        <AnimatePresence initial={false}>
          {items.map((movie) => (
            <ShelfCardGridItem
              key={recommendationId(movie)}
              movie={movie}
              onClick={() => onMovieClick(movie)}
              onQuickAction={onQuickAction}
            />
          ))}
        </AnimatePresence>
        {/* Infinite-scroll sentinel + status row */}
        <div ref={sentinelRef} style={{ height: 8 }} />
        {loadingMore && (
          <div style={{ display: "flex", justifyContent: "center", padding: "18px 0 26px" }}>
            <div className="skeleton-shimmer" style={{ width: 120, height: 26, borderRadius: 999 }} />
          </div>
        )}
        {exhausted && items.length > 0 && (
          <div style={{ textAlign: "center", padding: "16px 0 30px", color: "rgba(255,255,255,0.45)", fontSize: 13 }}>
            You&apos;ve reached the end — {items.length} titles
          </div>
        )}
        {items.length === 0 && (
          <p style={{ fontSize: 13, color: "var(--color-text-muted)", gridColumn: "1 / -1" }}>
            Nothing here yet.
          </p>
        )}
      </div>
    </motion.div>
  );

  if (!mounted) return null;
  return createPortal(content, document.body);
}
