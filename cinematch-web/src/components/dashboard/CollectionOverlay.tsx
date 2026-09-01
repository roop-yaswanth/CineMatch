"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";

import BackButton from "@/components/ui/BackButton";
import { recommendationId, type Recommendation } from "@/lib/api";
import { pushBackHandler } from "@/lib/backStack";
import { useMounted } from "@/lib/useMounted";
import { ShelfCardGridItem } from "./ShelfRow";
import { SkeletonCard } from "@/components/ui/Skeleton";

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
  /** Set of acted-upon / seen movie IDs to keep the overlay in sync */
  seenIds?: Set<number>;
}

export default function CollectionOverlay({ collection, onBack, onMovieClick, onQuickAction, onLoadMore, seenIds }: Props) {
  const mounted = useMounted();

  const [extraItems, setExtraItems] = useState<Recommendation[]>([]);
  const [dismissedIds, setDismissedIds] = useState<Set<number>>(() => new Set());
  const [loadingMore, setLoadingMore] = useState(false);
  const [exhausted, setExhausted] = useState(!onLoadMore);
  const sentinelRef = useRef<HTMLDivElement>(null);

  const items = useMemo(() => {
    const all = [...collection.movies, ...extraItems];
    const seen = new Set<number>();
    const out: Recommendation[] = [];
    for (const m of all) {
      const id = recommendationId(m);
      if (!seen.has(id) && (!seenIds || !seenIds.has(id)) && !dismissedIds.has(id)) {
        seen.add(id);
        out.push(m);
      }
    }
    return out;
  }, [collection.movies, extraItems, seenIds, dismissedIds]);

  const handleCardQuickAction = useCallback((movie: Recommendation, action: "dislike" | "like" | "love" | "watchlist") => {
    setDismissedIds((prev) => {
      const next = new Set(prev);
      next.add(recommendationId(movie));
      return next;
    });
    onQuickAction?.(movie, action);
  }, [onQuickAction]);

  const loadMore = useCallback(async () => {
    if (!onLoadMore || loadingMore || exhausted) return;
    setLoadingMore(true);
    try {
      const more = await onLoadMore();
      if (more && more.length > 0) {
        setExtraItems((prev) => {
          const currentSeen = new Set([
            ...collection.movies.map(recommendationId),
            ...prev.map(recommendationId),
          ]);
          const fresh = more.filter((m) => {
            const id = recommendationId(m);
            return !currentSeen.has(id) && (!seenIds || !seenIds.has(id)) && !dismissedIds.has(id);
          });
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
  }, [onLoadMore, loadingMore, exhausted, collection.movies, seenIds, dismissedIds]);

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
              onQuickAction={handleCardQuickAction}
            />
          ))}
        </AnimatePresence>

        {loadingMore &&
          Array.from({ length: items.length === 0 ? 12 : 6 }).map((_, i) => (
            <SkeletonCard key={`collection-skel-${i}`} compact />
          ))}

        {/* Infinite-scroll sentinel */}
        <div ref={sentinelRef} style={{ gridColumn: "1 / -1", height: 1, margin: 0 }} />

        {exhausted && items.length > 0 && (
          <div
            style={{
              gridColumn: "1 / -1",
              textAlign: "center",
              padding: "24px 0 36px",
              color: "rgba(255, 255, 255, 0.45)",
              fontSize: 13,
              fontWeight: 500,
            }}
          >
            You&apos;ve reached the end — {items.length} titles
          </div>
        )}

        {items.length === 0 && !loadingMore && exhausted && (
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
