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

export default function HistoryDrawer({ sessionId, onClose }: Props) {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGetHistory(sessionId)
      .then(setItems)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [sessionId]);

  const handleDelete = (tmdbId: number) => {
    setItems((prev) => prev.filter((item) => item.tmdb_id !== tmdbId));
  };

  // Group by context
  const grouped = useMemo(() => {
    const onboarding = items.filter((i) => i.context === "onboarding");
    const recommendation = items.filter((i) => i.context === "recommendation");
    return { onboarding, recommendation };
  }, [items]);

  const ratingConfig: Record<string, { label: string; color: string; icon: string }> = {
    like: { label: "Liked", color: "var(--color-like)", icon: "♥" },
    okay: { label: "Okay", color: "var(--color-okay)", icon: "○" },
    dislike: { label: "Disliked", color: "var(--color-dislike)", icon: "✕" },
    not_watched: { label: "Skipped", color: "var(--color-text-muted)", icon: "→" },
    remove: { label: "Removed", color: "var(--color-danger)", icon: "✕" },
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
          padding: "20px 24px",
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
        <div style={{ padding: "16px 20px 24px" }}>
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
  ratingConfig: Record<string, { label: string; color: string; icon: string }>;
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
            <span style={{ fontSize: "9px" }}>{config.icon}</span>
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
        ✕
      </motion.button>
    </motion.div>
  );
}

