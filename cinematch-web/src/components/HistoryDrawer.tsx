"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { apiGetHistory, posterUrl, type HistoryItem } from "@/lib/api";

interface Props {
  sessionId: string;
  onClose: () => void;
}

export default function HistoryDrawer({ sessionId, onClose }: Props) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGetHistory(sessionId)
      .then(setItems)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [sessionId]);

  const ratingLabel = (r: string) => {
    const labels: Record<string, string> = {
      like: "Liked",
      okay: "Okay",
      dislike: "Skipped",
      not_watched: "Not watched",
      remove: "Removed",
    };
    return labels[r] || r;
  };

  const ratingColor = (r: string) => {
    if (r === "like") return "text-[var(--color-success)]";
    if (r === "dislike" || r === "remove") return "text-[var(--color-danger)]";
    return "text-[var(--color-text-muted)]";
  };

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm"
      />

      {/* Drawer */}
      <motion.div
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", damping: 30, stiffness: 300 }}
        className="fixed top-0 right-0 bottom-0 z-50 w-full max-w-md bg-[var(--color-bg)] border-l border-[var(--color-border)] overflow-y-auto"
      >
        {/* Header */}
        <div className="sticky top-0 z-10 glass px-6 py-4 flex items-center justify-between">
          <h2 className="text-sm font-medium tracking-[-0.01em]">
            Rating history
          </h2>
          <button
            onClick={onClose}
            className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
          >
            Close
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4">
          {loading && (
            <div className="space-y-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="flex gap-3 animate-pulse">
                  <div className="w-10 h-14 rounded bg-[var(--color-surface)]" />
                  <div className="flex-1">
                    <div className="h-3 w-3/4 rounded bg-[var(--color-surface)]" />
                    <div className="mt-2 h-2 w-1/3 rounded bg-[var(--color-surface)]" />
                  </div>
                </div>
              ))}
            </div>
          )}

          {!loading && items.length === 0 && (
            <p className="text-sm text-[var(--color-text-muted)] font-light text-center py-12">
              No ratings yet.
            </p>
          )}

          <AnimatePresence>
            {items.map((item, idx) => (
              <motion.div
                key={item.tmdb_id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.03 }}
                className="flex items-center gap-3 py-3 border-b border-[var(--color-border-subtle)]"
              >
                <div className="relative w-10 h-14 rounded overflow-hidden bg-[var(--color-surface)] shrink-0">
                  <Image
                    src={posterUrl(item.poster_path, "w92")}
                    alt={item.title}
                    fill
                    sizes="40px"
                    className="object-cover"
                    unoptimized
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-[var(--color-text-primary)] truncate">
                    {item.title}
                  </p>
                  <div className="mt-0.5 flex items-center gap-2">
                    <span className={`text-[10px] font-medium ${ratingColor(item.rating)}`}>
                      {ratingLabel(item.rating)}
                    </span>
                    {item.year && (
                      <span className="text-[10px] text-[var(--color-text-muted)]">
                        {item.year}
                      </span>
                    )}
                    <span className="text-[10px] text-[var(--color-text-muted)] capitalize">
                      {item.context}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </motion.div>
    </>
  );
}
