"use client";

import { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";
import { apiGetHistory, languageLabel, type HistoryItem } from "@/lib/api";
import { usePoster } from "@/lib/usePoster";

interface Props {
  sessionId: string;
  onClose: () => void;
}

type InteractionFilter = "all" | "like" | "okay" | "dislike" | "not_watched";
type HistoryListItem = HistoryItem & { genres?: string[] };

const RATING_CONFIG: Record<string, { label: string; color: string; icon: string }> = {
  like: { label: "Liked", color: "var(--color-like)", icon: "♥" },
  okay: { label: "Okay", color: "var(--color-okay)", icon: "○" },
  dislike: { label: "Disliked", color: "var(--color-dislike)", icon: "✕" },
  not_watched: { label: "Skipped", color: "var(--color-text-muted)", icon: "→" },
};

const INTERACTION_FILTERS: Array<{ value: InteractionFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "like", label: "Liked" },
  { value: "okay", label: "Okay" },
  { value: "dislike", label: "Disliked" },
  { value: "not_watched", label: "Skipped" },
];

const IconHeart = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
  </svg>
);

export default function YourLikesView({ sessionId, onClose }: Props) {
  const [items, setItems] = useState<HistoryListItem[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Filters
  const [interactionFilter, setInteractionFilter] = useState<InteractionFilter>("all");
  const [genreFilter, setGenreFilter] = useState<string>("all");
  const [languageFilter, setLanguageFilter] = useState<string>("all");

  useEffect(() => {
    apiGetHistory(sessionId)
      .then(setItems)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [sessionId]);

  // Extract unique genres and languages from items
  const { genres, languages } = useMemo(() => {
    const genreSet = new Set<string>();
    const langSet = new Set<string>();
    
    items.forEach((item) => {
      if (item.genres && Array.isArray(item.genres)) {
        item.genres.forEach((g) => genreSet.add(g));
      }
      if (item.primary_genre) {
        genreSet.add(item.primary_genre);
      }
      if (item.original_language) {
        langSet.add(item.original_language);
      }
    });

    return {
      genres: Array.from(genreSet).sort(),
      languages: Array.from(langSet).sort(),
    };
  }, [items]);

  // Apply filters
  const filteredItems = useMemo(() => {
    let filtered = items;

    // Interaction filter
    if (interactionFilter !== "all") {
      filtered = filtered.filter((item) => item.rating === interactionFilter);
    }

    // Genre filter
    if (genreFilter !== "all") {
      filtered = filtered.filter((item) => {
        if (item.genres && Array.isArray(item.genres)) {
          return item.genres.includes(genreFilter);
        }
        if (item.primary_genre) {
          return item.primary_genre === genreFilter;
        }
        return false;
      });
    }

    // Language filter
    if (languageFilter !== "all") {
      filtered = filtered.filter((item) => item.original_language === languageFilter);
    }

    return filtered;
  }, [items, interactionFilter, genreFilter, languageFilter]);

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 50,
          backgroundColor: "rgba(0,0,0,0.6)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
        }}
      />

      {/* Full Screen View */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.97 }}
        transition={{ type: "spring", damping: 30, stiffness: 300 }}
        className="glass-modal likes-modal-container"
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 51,
          margin: "20px",
          borderRadius: "24px",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Header */}
        <div
          className="glass likes-header"
          style={{
            position: "sticky",
            top: 0,
            zIndex: 10,
            padding: "20px 24px",
            borderBottom: "1px solid var(--color-border-subtle)",
            borderRadius: "24px 24px 0 0",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ color: "var(--color-like)", display: "flex" }}>
              <IconHeart />
            </span>
            <h2
              style={{
                fontSize: "18px",
                fontWeight: 600,
                letterSpacing: "-0.02em",
                margin: 0,
              }}
            >
              Your Likes
            </h2>
          </div>
          <button
            onClick={onClose}
            className="glass-pill"
            style={{
              fontSize: "13px",
              color: "var(--color-text-muted)",
              cursor: "pointer",
              padding: "8px 18px",
            }}
          >
            Close
          </button>
        </div>

        {/* Filters */}
        <div className="likes-filters"
          style={{
            padding: "16px 24px",
            borderBottom: "1px solid var(--color-border-subtle)",
            display: "flex",
            flexWrap: "wrap",
            gap: "12px",
            alignItems: "center",
          }}
        >
          {/* Interaction Filter */}
          <div style={{ display: "flex", gap: "6px", flexWrap: "nowrap" }}>
            {INTERACTION_FILTERS.map((filter) => (
              <button
                key={filter.value}
                onClick={() => setInteractionFilter(filter.value)}
                className="filter-pill"
                data-active={interactionFilter === filter.value}
              >
                {filter.label}
              </button>
            ))}
          </div>

          <div className="likes-filter-sep" style={{ width: "1px", height: "24px", background: "var(--color-border-subtle)" }} />

          {/* Genre Filter */}
          {genres.length > 0 && (
            <select
              value={genreFilter}
              onChange={(e) => setGenreFilter(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Genres</option>
              {genres.map((genre) => (
                <option key={genre} value={genre}>
                  {genre}
                </option>
              ))}
            </select>
          )}

          {/* Language Filter */}
          {languages.length > 0 && (
            <select
              value={languageFilter}
              onChange={(e) => setLanguageFilter(e.target.value)}
              className="filter-select"
            >
              <option value="all">All Languages</option>
              {languages.map((lang) => (
                <option key={lang} value={lang}>
                  {languageLabel(lang)}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Content */}
        <div className="likes-content" style={{ flex: 1, padding: "24px", overflowY: "auto" }}>
          {loading && (
            <div className="likes-grid"
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
                gap: "16px",
              }}
            >
              {Array.from({ length: 12 }).map((_, i) => (
                <div
                  key={i}
                  className="skeleton-shimmer"
                  style={{ height: "280px", borderRadius: "16px" }}
                />
              ))}
            </div>
          )}

          {!loading && filteredItems.length === 0 && (
            <div style={{ textAlign: "center", padding: "80px 0" }}>
              <p
                style={{
                  fontSize: "16px",
                  color: "var(--color-text-muted)",
                  fontWeight: 400,
                }}
              >
                No movies found.
              </p>
              <p
                style={{
                  fontSize: "13px",
                  color: "var(--color-text-muted)",
                  marginTop: "8px",
                }}
              >
                {interactionFilter !== "all" || genreFilter !== "all" || languageFilter !== "all"
                  ? "Try adjusting your filters"
                  : "Rate some movies to see them here"}
              </p>
            </div>
          )}

          {!loading && filteredItems.length > 0 && (
            <AnimatePresence>
              <div className="likes-grid"
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
                  gap: "16px",
                }}
              >
                {filteredItems.map((item, idx) => (
                  <MovieCard key={`${item.tmdb_id}-${idx}`} item={item} idx={idx} />
                ))}
              </div>
            </AnimatePresence>
          )}
        </div>
      </motion.div>

      <style>{`
        .filter-pill {
          padding: 7px 16px;
          border-radius: 20px;
          border: 1px solid var(--color-border-subtle);
          background: rgba(255, 255, 255, 0.03);
          color: var(--color-text-secondary);
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          white-space: nowrap;
        }
        .filter-pill:hover {
          background: rgba(255, 255, 255, 0.06);
          border-color: var(--color-border);
        }
        .filter-pill[data-active="true"] {
          background: var(--color-like);
          color: #fff;
          border-color: var(--color-like);
        }
        .filter-select {
          padding: 7px 14px;
          border-radius: 12px;
          border: 1px solid var(--color-border-subtle);
          background: rgba(255, 255, 255, 0.03);
          color: var(--color-text-primary);
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
        }
        .filter-select:hover {
          background: rgba(255, 255, 255, 0.06);
          border-color: var(--color-border);
        }
        .filter-select option {
          background: var(--color-surface);
          color: var(--color-text-primary);
        }

        @media (max-width: 640px) {
          .likes-modal-container {
            margin: 0 !important;
            border-radius: 0 !important;
          }
          .likes-header {
            padding: 14px 16px !important;
            border-radius: 0 !important;
          }
          .likes-filters {
            padding: 10px 16px !important;
            gap: 8px !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
          }
          .likes-filters::-webkit-scrollbar {
            display: none;
          }
          .likes-filter-sep {
            display: none;
          }
          .likes-content {
            padding: 12px !important;
          }
          .likes-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 10px !important;
          }
          .filter-pill {
            padding: 6px 12px;
            font-size: 12px;
          }
          .filter-select {
            padding: 6px 10px;
            font-size: 12px;
          }
        }
      `}</style>
    </>
  );
}

/* ─── Movie Card Component ─── */

function MovieCard({ item, idx }: { item: HistoryItem; idx: number }) {
  const poster = usePoster(item.poster_path, item.tmdb_id, "w342");
  const config = RATING_CONFIG[item.rating] || {
    label: item.rating,
    color: "var(--color-text-muted)",
    icon: "",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ delay: idx * 0.02, duration: 0.3 }}
      className="glass-card"
      style={{
        borderRadius: "16px",
        overflow: "hidden",
        cursor: "default",
      }}
    >
      {/* Poster */}
      <div
        style={{
          position: "relative",
          width: "100%",
          paddingBottom: "150%",
          background: "var(--color-surface)",
          overflow: "hidden",
        }}
      >
        <Image
          src={poster}
          alt={item.title}
          fill
          sizes="(max-width: 768px) 50vw, (max-width: 1200px) 33vw, 20vw"
          className="object-cover"
          unoptimized
        />
        
        {/* Rating Badge */}
        <div
          style={{
            position: "absolute",
            top: "8px",
            right: "8px",
            padding: "6px 10px",
            borderRadius: "8px",
            background: "rgba(0, 0, 0, 0.75)",
            backdropFilter: "blur(8px)",
            display: "flex",
            alignItems: "center",
            gap: "4px",
            fontSize: "11px",
            fontWeight: 600,
            color: config.color,
          }}
        >
          <span style={{ fontSize: "10px" }}>{config.icon}</span>
          {config.label}
        </div>
      </div>

      {/* Info */}
      <div style={{ padding: "12px" }}>
        <p
          style={{
            fontSize: "14px",
            fontWeight: 600,
            color: "var(--color-text-primary)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            lineHeight: 1.3,
            margin: 0,
            minHeight: "36px",
          }}
        >
          {item.title}
        </p>
        {item.year && (
          <p
            style={{
              fontSize: "12px",
              color: "var(--color-text-muted)",
              marginTop: "6px",
              margin: 0,
            }}
          >
            {item.year}
          </p>
        )}
      </div>
    </motion.div>
  );
}
