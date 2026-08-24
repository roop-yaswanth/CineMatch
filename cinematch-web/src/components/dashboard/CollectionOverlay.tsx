"use client";

import { useEffect } from "react";
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
}

export default function CollectionOverlay({ collection, onBack, onMovieClick, onQuickAction }: Props) {
  const mounted = useMounted();

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
          {collection.movies.map((movie) => (
            <ShelfCardGridItem
              key={recommendationId(movie)}
              movie={movie}
              onClick={() => onMovieClick(movie)}
              onQuickAction={onQuickAction}
            />
          ))}
        </AnimatePresence>
        {collection.movies.length === 0 && (
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
