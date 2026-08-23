"use client";

import { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";

import dynamic from "next/dynamic";
import {
  apiGetHistory,
  languageLabel,
  apiRecommendationAction,
  readHistoryCache,
  writeHistoryCache,
  type HistoryItem,
} from "@/lib/api";
import { usePoster } from "@/lib/usePoster";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import { useRouter } from "next/navigation";
import MobileMenu from "@/components/MobileMenu";
import { useSession } from "@/context/SessionContext";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";

const MovieDetailModal = dynamic(() => import("@/components/modals/MovieDetailModal"), { ssr: false });

interface Props {
  sessionId: string;
  onClose: () => void;
  initialFilter?: InteractionFilter;
}

type InteractionFilter = "all" | "like" | "okay" | "dislike" | "not_watched" | "watchlist";
type HistoryListItem = HistoryItem & { genres?: string[] };

const RATING_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
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
  watchlist: {
    label: "Watchlist",
    color: "var(--color-accent)",
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="5" x2="12" y2="19" />
        <line x1="5" y1="12" x2="19" y2="12" />
      </svg>
    ),
  },
};

const INTERACTION_FILTERS: Array<{ value: InteractionFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "like", label: "Liked" },
  { value: "okay", label: "Okay" },
  { value: "dislike", label: "Disliked" },
  { value: "not_watched", label: "Skipped" },
  { value: "watchlist", label: "Watchlist" },
];

function toDetailMovie(item: HistoryListItem): DetailMovie {
  return {
    id: item.tmdb_id,
    tmdb_id: item.tmdb_id,
    title: item.title,
    poster_path: item.poster_path, // maybe undefined
    year: item.year,
    original_language: item.original_language,
    primary_genre: item.primary_genre,
    genres: item.genres,
  };
}

export default function YourLikesView({ sessionId, onClose, initialFilter = "all" }: Props) {
  const { logout } = useSession();
  const router = useRouter();
  // Stale-while-revalidate: if we have ANY cached items at all, paint them
  // immediately and skip the skeleton state. The background refetch below
  // ALWAYS runs and reconciles silently. Only show skeletons when the user
  // is genuinely cold (no cache) — that's the "slow first time" case we
  // can't avoid without a server-side prefetch.
  const [cachedItems] = useState<HistoryListItem[]>(() => readHistoryCache<HistoryListItem>(sessionId) ?? []);
  const [items, setItems] = useState<HistoryListItem[]>(cachedItems);
  const [loading, setLoading] = useState(cachedItems.length === 0);
  const [activeMovie, setActiveMovie] = useState<DetailMovie | null>(null);

  // Filters
  const [interactionFilter, setInteractionFilter] = useState<InteractionFilter>(initialFilter);
  const [genreFilter, setGenreFilter] = useState<string>("all");
  const [languageFilter, setLanguageFilter] = useState<string>("all");

  useEffect(() => {
    /* eslint-disable-next-line react-hooks/set-state-in-effect */
    setInteractionFilter(initialFilter);
  }, [initialFilter]);
  // We don't watch searchParams here since it's passed from parent as initialFilter

  useEffect(() => {
    // Always reconcile against the server on mount. Skipping this when the
    // cache looked "fresh" left movies added to the watchlist elsewhere
    // (dashboard hero/card "+", search, detail modal) invisible here for up
    // to 5 minutes — those surfaces mutate server-side history without
    // touching this localStorage cache. With cached items painted above,
    // the refetch is invisible (no skeleton flash).
    let cancelled = false;
    apiGetHistory(sessionId)
      .then((data) => {
        if (cancelled) return;
        setItems(data);
        writeHistoryCache(sessionId, data);
      })
      .catch(console.error)
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
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
    // The backend returns history in chronological insertion order
    // (onboarding rates, then recommendation rates). Reverse so the most
    // recently rated movie shows first — matches user expectation that
    // "what I just rated" lands at the top of the grid.
    let filtered = items.slice().reverse();

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
      {/* Full Page View */}
      <div
        className="likes-modal-container"
        style={{
          minHeight: "100dvh",
          display: "flex",
          flexDirection: "column",
          background: "var(--color-bg)",
        }}
      >
        {/* Header — shared <PageHeader> component with centered title */}
        <PageHeader
          onBack={onClose}
          hideBackButton={!onClose}
          backAriaLabel="Go back"
          title={
            <span style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
              Your Collection
            </span>
          }
          rightSlot={
            <MobileMenu onLogout={() => { logout(); router.replace("/login"); }} />
          }
        />

        {/* Streamlined Filter Bar — Sleek horizontal scroll strip */}
        <div
          className="likes-filters"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            overflowX: "auto",
            scrollbarWidth: "none",
            padding: "8px var(--s-header-x) 12px",
            borderBottom: "1px solid rgba(255, 255, 255, 0.06)",
            background: "var(--color-bg)",
          }}
        >
          {/* Interaction filter pills */}
          <div
            className="interaction-pills"
            role="tablist"
            aria-label="Filter by interaction"
            style={{
              display: "flex",
              gap: "8px",
              alignItems: "center",
              flexShrink: 0,
            }}
          >
            {INTERACTION_FILTERS.map((filter) => {
              const active = interactionFilter === filter.value;
              return (
                <button
                  key={filter.value}
                  role="tab"
                  aria-selected={active}
                  onClick={() => setInteractionFilter(filter.value)}
                  style={{
                    padding: "6px 14px",
                    borderRadius: "999px",
                    border: active ? "1px solid rgba(255,255,255,0.28)" : "1px solid rgba(255,255,255,0.10)",
                    background: active ? "rgba(255,255,255,0.14)" : "rgba(28,30,36,0.58)",
                    color: active ? "#ffffff" : "rgba(255,255,255,0.65)",
                    fontSize: "12.5px",
                    fontWeight: active ? 600 : 500,
                    whiteSpace: "nowrap",
                    cursor: "pointer",
                    flexShrink: 0,
                    transition: "all 160ms ease",
                  }}
                >
                  {filter.label}
                </button>
              );
            })}
          </div>

          {/* Genre Filter */}
          {genres.length > 0 && (
            <select
              value={genreFilter}
              onChange={(e) => setGenreFilter(e.target.value)}
              style={{
                appearance: "none",
                WebkitAppearance: "none",
                background: genreFilter !== "all" ? "rgba(255, 255, 255, 0.16)" : "rgba(28, 30, 36, 0.58)",
                border: genreFilter !== "all" ? "1px solid rgba(255, 255, 255, 0.35)" : "1px solid rgba(255, 255, 255, 0.10)",
                borderRadius: "999px",
                color: genreFilter !== "all" ? "#ffffff" : "rgba(255, 255, 255, 0.65)",
                padding: "6px 28px 6px 14px",
                fontSize: "12.5px",
                fontWeight: genreFilter !== "all" ? 600 : 500,
                cursor: "pointer",
                outline: "none",
                flexShrink: 0,
                backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='rgba(255,255,255,0.5)' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E")`,
                backgroundRepeat: "no-repeat",
                backgroundPosition: "right 10px center",
                backgroundSize: "10px",
              }}
            >
              <option value="all" style={{ background: "#161820", color: "#fff" }}>All Genres</option>
              {genres.map((genre) => (
                <option key={genre} value={genre} style={{ background: "#161820", color: "#fff" }}>
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
              style={{
                appearance: "none",
                WebkitAppearance: "none",
                background: languageFilter !== "all" ? "rgba(255, 255, 255, 0.16)" : "rgba(28, 30, 36, 0.58)",
                border: languageFilter !== "all" ? "1px solid rgba(255, 255, 255, 0.35)" : "1px solid rgba(255, 255, 255, 0.10)",
                borderRadius: "999px",
                color: languageFilter !== "all" ? "#ffffff" : "rgba(255, 255, 255, 0.65)",
                padding: "6px 28px 6px 14px",
                fontSize: "12.5px",
                fontWeight: languageFilter !== "all" ? 600 : 500,
                cursor: "pointer",
                outline: "none",
                flexShrink: 0,
                backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='rgba(255,255,255,0.5)' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E")`,
                backgroundRepeat: "no-repeat",
                backgroundPosition: "right 10px center",
                backgroundSize: "10px",
              }}
            >
              <option value="all" style={{ background: "#161820", color: "#fff" }}>All Languages</option>
              {languages.map((lang) => (
                <option key={lang} value={lang} style={{ background: "#161820", color: "#fff" }}>
                  {languageLabel(lang)}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Content */}
        <div
          className="likes-content"
          style={{
            flex: 1,
            // Reserve room at the bottom so the last grid row clears the
            // floating bottom nav (pill ≈ 72 px + 28 px static lift +
            // safe-area). Without this, the bottom-most cards' titles are
            // hidden behind the nav.
            padding: "24px 24px calc(120px + env(safe-area-inset-bottom))",
            overflowY: "auto",
          }}
        >
          {loading && (
            <div className="likes-grid"
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
                gap: "16px",
              }}
            >
              {Array.from({ length: 12 }).map((_, i) => (
                <div key={i} style={{ borderRadius: "16px", overflow: "hidden" }}>
                  <div
                    className="skeleton-shimmer"
                    style={{ width: "100%", paddingBottom: "150%", borderRadius: "12px" }}
                  />
                  <div style={{ padding: "10px 4px 4px", display: "flex", flexDirection: "column", gap: "6px" }}>
                    <div className="skeleton-shimmer" style={{ height: "13px", width: "85%", borderRadius: "999px" }} />
                    <div className="skeleton-shimmer" style={{ height: "11px", width: "50%", borderRadius: "999px" }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {!loading && filteredItems.length === 0 && (
            (() => {
              const hasFilters =
                interactionFilter !== "all" ||
                genreFilter !== "all" ||
                languageFilter !== "all";
              if (hasFilters) {
                return (
                  <EmptyState
                    title="No matches"
                    description="Try resetting one of the filters above to see more."
                    cta={{
                      kind: "button",
                      label: "Reset filters",
                      onClick: () => {
                        setInteractionFilter("all");
                        setGenreFilter("all");
                        setLanguageFilter("all");
                      },
                    }}
                  />
                );
              }
              return (
                <EmptyState
                  title="Your collection is empty"
                  description="Rate movies you've seen and add titles to your watchlist — they'll show up here."
                  cta={{ kind: "link", href: "/explore", label: "Browse Trending" }}
                />
              );
            })()
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
                  <MovieCard
                    key={`${item.tmdb_id}-${idx}`}
                    item={item}
                    idx={idx}
                    onClick={() => setActiveMovie(toDetailMovie(item))}
                  />
                ))}
              </div>
            </AnimatePresence>
          )}
        </div>
      </div>

      {activeMovie && (
        <MovieDetailModal
          isOpen={activeMovie !== null}
          onClose={() => setActiveMovie(null)}
          movie={activeMovie}
          sessionId={sessionId}
          onAction={async (action) => {
            const targetId = activeMovie.id || activeMovie.tmdb_id!;
            // Optimistic local update so the grid reflects the new rating
            // *before* the server round-trip and refetch return. Without
            // this, on slow connections (or whenever the SW serves the
            // /api/history response from its 5-min cache) the rated movie
            // visibly stays in the previous filter bucket until the next
            // full page refresh — which is exactly the "I have to refresh
            // to see it disappear" symptom.
            setItems((prev) => {
              const next = prev.map((it) =>
                it.tmdb_id === targetId ? { ...it, rating: action } : it
              );
              writeHistoryCache(sessionId, next);
              return next;
            });
            if (action !== "watchlist") setActiveMovie(null);

            // Persist + reconcile against the server in the background.
            try {
              await apiRecommendationAction(sessionId, targetId, action);
              const data = await apiGetHistory(sessionId);
              setItems(data);
              writeHistoryCache(sessionId, data);
            } catch (e) {
              console.error(e);
            }
          }}
          onMovieSelect={(mov) => setActiveMovie(mov)}
        />
      )}

      <style>{`
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
          .filter-select {
            padding: 6px 10px;
            font-size: 12px;
          }
          .interaction-pills {
            display: none !important;
          }
          .interaction-select-mobile {
            display: block !important;
          }
        }
      `}</style>
    </>
  );
}

/* ─── Movie Card Component ─── */

function MovieCard({ item, idx, onClick }: { item: HistoryItem; idx: number; onClick: () => void }) {
  const poster = usePoster(item.poster_path, item.tmdb_id, "w342");
  const config = RATING_CONFIG[item.rating] || {
    label: item.rating,
    color: "var(--color-text-muted)",
    icon: null,
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.94 }}
      // Tightened: was `delay: idx * 0.02, duration: 0.3`. With 24+ cards
      // the last card was waiting almost 0.8s before settling, which
      // looks "slow" on a fast device. Cap the stagger and shorten the
      // duration so the whole grid lands in well under 250 ms.
      transition={{ delay: Math.min(idx, 12) * 0.012, duration: 0.18, ease: "easeOut" }}
      className="glass-card likes-card"
      onClick={onClick}
      style={{
        borderRadius: "16px",
        overflow: "hidden",
        cursor: "pointer",
        // Off-screen cards skip layout/paint work until they scroll near
        // the viewport — big win on long collections.
        contentVisibility: "auto",
        containIntrinsicSize: "240px",
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
        <img
          src={poster}
          alt={item.title}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
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
          <span style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>{config.icon}</span>
          {config.label}
        </div>
      </div>

      {/* Info — same global title/meta treatment as every other surface */}
      <div style={{ padding: "12px" }}>
        <p className="poster-info-title">
          {item.title}
        </p>
        {item.year && (
          <p className="poster-info-meta-text" style={{ marginTop: "5px" }}>
            {item.year}
          </p>
        )}
      </div>
    </motion.div>
  );
}
