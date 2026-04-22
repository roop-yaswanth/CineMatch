"use client";

import { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { apiGetHistory, type HistoryItem } from "@/lib/api";
import { usePoster } from "@/lib/usePoster";

interface Props {
  sessionId: string;
  onClose: () => void;
}

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

function readHistoryCache(sessionId: string): { data: HistoryItem[]; isFresh: boolean } | null {
  try {
    const cached = localStorage.getItem(`history_cache_${sessionId}`);
    if (!cached) return null;
    const parsed = JSON.parse(cached) as { data?: HistoryItem[]; ts?: number };
    if (!Array.isArray(parsed.data) || typeof parsed.ts !== "number") return null;
    return { data: parsed.data, isFresh: Date.now() - parsed.ts < CACHE_TTL_MS };
  } catch {
    return null;
  }
}

export default function HistoryDrawer({ sessionId, onClose }: Props) {
  const initialCache = readHistoryCache(sessionId);
  const [items, setItems] = useState<HistoryItem[]>(() => initialCache?.data ?? []);
  const [loading, setLoading] = useState(!initialCache);

  useEffect(() => {
    if (initialCache?.isFresh) return;

    apiGetHistory(sessionId)
      .then((data) => {
        setItems(data);
        try {
          localStorage.setItem(`history_cache_${sessionId}`, JSON.stringify({ data, ts: Date.now() }));
        } catch { /* storage full */ }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [sessionId, initialCache]);

  const handleDelete = (tmdbId: number) => {
    setItems((prev) => prev.filter((item) => item.tmdb_id !== tmdbId));
  };

  // Group by context
  const grouped = useMemo(() => {
    const onboarding = items.filter((i) => i.context === "onboarding");
    const recommendation = items.filter((i) => i.context === "recommendation");
    return { onboarding, recommendation };
  }, [items]);

  const ratingConfig: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
    like: {
      label: "Liked",
      color: "var(--color-like)",
      icon: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
        </svg>
      ),
    },
    okay: {
      label: "Okay",
      color: "var(--color-okay)",
      icon: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
          <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
        </svg>
      ),
    },
    dislike: {
      label: "Disliked",
      color: "var(--color-dislike)",
      icon: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
          <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
        </svg>
      ),
    },
    not_watched: {
      label: "Skipped",
      color: "var(--color-text-muted)",
      icon: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="13 17 18 12 13 7" />
          <polyline points="6 17 11 12 6 7" />
        </svg>
      ),
    },
    remove: {
      label: "Removed",
      color: "var(--color-danger)",
      icon: (
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      ),
    },
  };

  const renderItemEl = (item: HistoryItem, idx: number) => (
    <HistoryItemCard key={`${item.tmdb_id}-${item.context}`} item={item} idx={idx} ratingConfig={ratingConfig} onDelete={handleDelete} />
  );




  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        style={{
          position: "fixed", inset: 0, zIndex: 50,
          backgroundColor: "rgba(0,0,0,0.4)",
          backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
        }}
      />

      {/* Drawer */}
      <motion.div
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", damping: 30, stiffness: 300 }}
        className="glass-modal"
        style={{
          position: "fixed", top: 0, right: 0, bottom: 0, zIndex: 50,
          width: "100%", maxWidth: "420px",
          borderRadius: "24px 0 0 24px",
          overflowY: "auto",
          padding: 0,
        }}
      >
        {/* Header */}
        <div className="glass" style={{
          position: "sticky", top: 0, zIndex: 10,
          padding: "16px 20px 16px 16px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          borderBottom: "none",
          borderRadius: "24px 0 0 0",
        }}>
          <h2 style={{ fontSize: "16px", fontWeight: 600, letterSpacing: "-0.02em", margin: 0 }}>
            Rating History
          </h2>
          <button
            onClick={onClose}
            className="glass-pill"
            style={{ fontSize: "12px", color: "var(--color-text-muted)", cursor: "pointer", padding: "6px 14px" }}
          >
            Close
          </button>
        </div>

        {/* Content */}
        <div style={{ padding: "16px 16px 24px", paddingLeft: "max(16px, env(safe-area-inset-left))" }}>
          {loading && (
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="skeleton-shimmer" style={{ height: "74px", borderRadius: "14px" }} />
              ))}
            </div>
          )}

          {!loading && items.length === 0 && (
            <div style={{ textAlign: "center", padding: "60px 0" }}>
              <p style={{ fontSize: "14px", color: "var(--color-text-muted)", fontWeight: 300 }}>
                No ratings yet.
              </p>
              <p style={{ fontSize: "12px", color: "var(--color-text-muted)", marginTop: "8px" }}>
                Rate some movies to see your history here.
              </p>
            </div>
          )}

          {/* Recommendation section */}
          {grouped.recommendation.length > 0 && (
            <div style={{ marginBottom: "28px" }}>
              <div style={{
                fontSize: "11px", color: "var(--color-text-secondary)",
                fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase",
                marginBottom: "12px", paddingLeft: "4px",
              }}>
                Recommendations ({grouped.recommendation.length})
              </div>
              <AnimatePresence>
                {grouped.recommendation.map((item, idx) => renderItemEl(item, idx))}
              </AnimatePresence>
            </div>
          )}

          {/* Onboarding section */}
          {grouped.onboarding.length > 0 && (
            <div>
              <div style={{
                fontSize: "11px", color: "var(--color-text-secondary)",
                fontWeight: 500, letterSpacing: "0.05em", textTransform: "uppercase",
                marginBottom: "12px", paddingLeft: "4px",
              }}>
                Onboarding ({grouped.onboarding.length})
              </div>
              <AnimatePresence>
                {grouped.onboarding.map((item, idx) => renderItemEl(item, idx))}
              </AnimatePresence>
            </div>
          )}
        </div>
      </motion.div>
    </>
  );
}

/* ─── History Item Card (uses hooks for poster fallback) ─── */

function HistoryItemCard({
  item,
  idx,
  ratingConfig,
  onDelete,
}: {
  item: HistoryItem;
  idx: number;
  ratingConfig: Record<string, { label: string; color: string; icon: React.ReactNode }>;
  onDelete: (tmdbId: number) => void;
}) {
  const poster = usePoster(item.poster_path, item.tmdb_id, "w92");
  const config = ratingConfig[item.rating] || {
    label: item.rating,
    color: "var(--color-text-muted)",
    icon: "",
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -12, height: 0, marginBottom: 0, padding: 0 }}
      transition={{ delay: idx * 0.02, duration: 0.25 }}
      className="glass-card"
      style={{
        display: "flex", alignItems: "center", gap: "12px",
        padding: "12px", marginBottom: "8px",
        borderRadius: "14px",
      }}
    >
      {/* Poster */}
      <div style={{
        position: "relative", width: "44px", height: "62px",
        borderRadius: "10px", overflow: "hidden",
        background: "var(--color-surface)", flexShrink: 0,
      }}>
        <Image
          src={poster}
          alt={item.title}
          fill
          sizes="44px"
          className="object-cover"
          unoptimized
        />
      </div>

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{
          fontSize: "13px", fontWeight: 500,
          color: "var(--color-text-primary)",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          margin: 0,
        }}>
          {item.title}
        </p>
        <div style={{ marginTop: "4px", display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{
            fontSize: "11px", fontWeight: 500, color: config.color,
            display: "flex", alignItems: "center", gap: "3px",
          }}>
            <span style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>{config.icon}</span>
            {config.label}
          </span>
          {item.year && (
            <span style={{ fontSize: "10px", color: "var(--color-text-muted)" }}>
              {item.year}
            </span>
          )}
        </div>
      </div>

      {/* Delete button */}
      <motion.button
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        onClick={() => onDelete(item.tmdb_id)}
        style={{
          width: "28px", height: "28px", borderRadius: "8px",
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.06)",
          color: "var(--color-text-muted)",
          cursor: "pointer", fontSize: "12px", flexShrink: 0,
          transition: "all 0.2s",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.background = "rgba(248,113,113,0.12)";
          e.currentTarget.style.borderColor = "rgba(248,113,113,0.2)";
          e.currentTarget.style.color = "var(--color-danger)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background = "rgba(255,255,255,0.04)";
          e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)";
          e.currentTarget.style.color = "var(--color-text-muted)";
        }}
        title="Remove from history"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </motion.button>
    </motion.div>
  );
}
