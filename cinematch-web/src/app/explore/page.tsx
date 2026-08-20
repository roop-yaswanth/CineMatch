"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion } from "framer-motion";

import dynamic from "next/dynamic";
import MovieCard from "@/components/MovieCard";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";
import MobileMenu from "@/components/MobileMenu";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import { SkeletonGrid, SkeletonRail } from "@/components/ui/Skeleton";

const MovieDetailModal = dynamic(() => import("@/components/modals/MovieDetailModal"), { ssr: false });
import { PosterCard } from "@/components/RecommendationsView";
import { toast } from "@/components/ui/Toast";
import { useSession } from "@/context/SessionContext";
import {
  apiDiscover,
  apiExplore,
  apiGenres,
  apiRecommendationAction,
  LANGUAGE_LABELS,
  languageLabel,
  type DiscoverFilters,
  type DiscoverSort,
  type ExploreCategory,
  type ExploreMovie,
  type TmdbGenre,
} from "@/lib/api";

interface CategoryDef {
  id: ExploreCategory;
  label: string;
  subtitle: string;
}

const CATEGORIES: CategoryDef[] = [
  { id: "trending_day", label: "Trending Today", subtitle: "What everyone's watching right now." },
  { id: "popular", label: "Popular", subtitle: "Audience favorites this week." },
  { id: "top_rated", label: "Top Rated", subtitle: "All-time highest rated on TMDB." },
  { id: "now_playing", label: "Now Playing", subtitle: "In theatres now." },
  { id: "upcoming", label: "Upcoming", subtitle: "Coming soon." },
];

type TabId = ExploreCategory | "discover";

const TAB_OPTIONS: Array<{ id: TabId; label: string }> = [
  { id: "trending_day", label: "Trending" },
  { id: "popular", label: "Popular" },
  { id: "top_rated", label: "Top Rated" },
  { id: "now_playing", label: "Now Playing" },
  { id: "upcoming", label: "Upcoming" },
  { id: "discover", label: "Discover" },
];

const SORT_OPTIONS: Array<{ value: DiscoverSort; label: string }> = [
  { value: "popularity.desc", label: "Most Popular" },
  { value: "vote_average.desc", label: "Highest Rated" },
  { value: "primary_release_date.desc", label: "Newest" },
  { value: "primary_release_date.asc", label: "Oldest" },
  { value: "revenue.desc", label: "Highest Revenue" },
  { value: "title.asc", label: "Title (A–Z)" },
];

const LANGUAGE_OPTIONS = ["", "en", "te", "hi", "ta", "ml", "kn", "ja", "ko", "zh", "es", "fr", "de", "it", "pt", "ru"];

function toDetailMovie(m: ExploreMovie): DetailMovie {
  return { ...m, id: m.tmdb_id };
}

export default function ExplorePage() {
  return (
    <Suspense fallback={null}>
      <ExplorePageInner />
    </Suspense>
  );
}

/* ─── Filter pill select ─── */
function PillSelect({
  value,
  onChange,
  options,
  active,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
  active?: boolean;
}) {
  const isActive = active ?? (value !== "" && value !== options[0]?.value);
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        appearance: "none",
        WebkitAppearance: "none",
        background: isActive
          ? "rgba(var(--rgb-accent), 0.14)"
          : "var(--glass-chrome)",
        border: isActive
          ? "1px solid rgba(var(--rgb-accent), 0.45)"
          : "1px solid var(--hairline)",
        borderRadius: "var(--radius-pill)",
        color: isActive ? "var(--color-accent)" : "var(--color-text-secondary)",
        padding: "7px 28px 7px 14px",
        fontSize: "12.5px",
        fontWeight: isActive ? 600 : 500,
        cursor: "pointer",
        outline: "none",
        transition: "all var(--dur-base) var(--ease-out)",
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 24 24' fill='none' stroke='rgba(255,255,255,0.5)' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E")`,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right 10px center",
        backgroundSize: "10px",
        minWidth: 0,
        whiteSpace: "nowrap" as const,
      }}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} style={{ background: "#161820", color: "#fff" }}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

function ExplorePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { session, isLoading, logout } = useSession();

  useEffect(() => {
    if (!isLoading && !session) {
      if (typeof window !== "undefined") {
        window.location.replace("/login");
      } else {
        router.replace("/login");
      }
    }
  }, [session, isLoading, router]);

  const initialTab: TabId = (() => {
    const t = searchParams?.get("tab") || "";
    const known: TabId[] = ["trending_day", "popular", "top_rated", "now_playing", "upcoming", "discover"];
    return (known as string[]).includes(t) ? (t as TabId) : "trending_day";
  })();
  const [tab, setTabState] = useState<TabId>(initialTab);

  const setTab = (next: TabId) => {
    setTabState(next);
    const url = next === "trending_day" ? "/explore" : `/explore?tab=${encodeURIComponent(next)}`;
    window.history.replaceState(null, "", url);
  };

  // Global Explore Filters
  const [selectedLanguage, setSelectedLanguage] = useState<string>("");
  const [selectedGenre, setSelectedGenre] = useState<string>("");
  const [sortByFilter, setSortByFilter] = useState<DiscoverSort>("popularity.desc");
  const [genresList, setGenresList] = useState<TmdbGenre[]>([]);

  useEffect(() => {
    apiGenres().then(setGenresList).catch(() => { });
  }, []);

  const [grid, setGrid] = useState<ExploreMovie[]>([]);
  const [gridPage, setGridPage] = useState(1);
  const [gridTotalPages, setGridTotalPages] = useState(1);
  const [gridLoading, setGridLoading] = useState(false);

  const [active, setActive] = useState<DetailMovie | null>(null);
  const seenIds = useRef<Set<number>>(new Set());

  const region = session?.profile?.region || undefined;

  // Load paginated grid for a specific category
  useEffect(() => {
    if (tab === "discover") return;
    let cancelled = false;
    /* eslint-disable react-hooks/set-state-in-effect */
    setGrid([]);
    setGridPage(1);
    setGridTotalPages(1);
    /* eslint-enable react-hooks/set-state-in-effect */
    seenIds.current = new Set();
    setGridLoading(true);

    const lang = selectedLanguage || undefined;
    const genre = selectedGenre || undefined;
    const sort = sortByFilter !== "popularity.desc" ? sortByFilter : undefined;

    apiExplore(tab, 1, region, lang, genre, sort)
      .then((r) => {
        if (cancelled) return;
        const fresh = r.results.filter((m) => {
          if (seenIds.current.has(m.tmdb_id)) return false;
          seenIds.current.add(m.tmdb_id);
          return true;
        });
        setGrid(fresh);
        setGridPage(r.page);
        setGridTotalPages(r.total_pages);
      })
      .catch(() => { })
      .finally(() => { if (!cancelled) setGridLoading(false); });
    return () => { cancelled = true; };
  }, [tab, region, selectedLanguage, selectedGenre, sortByFilter]);

  const loadMore = useCallback(async () => {
    if (tab === "discover" || gridLoading || gridPage >= gridTotalPages) return;
    setGridLoading(true);
    const lang = selectedLanguage || undefined;
    const genre = selectedGenre || undefined;
    const sort = sortByFilter !== "popularity.desc" ? sortByFilter : undefined;

    try {
      const next = await apiExplore(tab, gridPage + 1, region, lang, genre, sort);
      const fresh = next.results.filter((m) => {
        if (seenIds.current.has(m.tmdb_id)) return false;
        seenIds.current.add(m.tmdb_id);
        return true;
      });
      setGrid((prev) => [...prev, ...fresh]);
      setGridPage(next.page);
      setGridTotalPages(next.total_pages);
    } catch { /* ignore */ }
    finally { setGridLoading(false); }
  }, [tab, gridPage, gridTotalPages, gridLoading, region, selectedLanguage, selectedGenre, sortByFilter]);

  const handleAction = useCallback(
    async (m: { id?: number; tmdb_id?: number; title: string }, action: "like" | "okay" | "dislike" | "watchlist" | "skip") => {
      if (!session) return;
      try {
        await apiRecommendationAction(session.session_id, (m.id || m.tmdb_id)!, action);
        if (action === "watchlist") {
          toast({ message: `Added "${m.title}" to your watchlist`, tone: "success" });
        } else if (action === "like") {
          toast({ message: `Liked "${m.title}"`, tone: "success" });
        } else if (action === "dislike") {
          toast({ message: `Disliked "${m.title}"`, tone: "neutral" });
        }
      } catch (err) {
        console.error("Action failed:", err);
      }
    },
    [session]
  );

  if (isLoading || !session) {
    return <div style={{ minHeight: "100dvh", background: "var(--color-bg)" }} />;
  }

  const hasActiveFilters = Boolean(selectedLanguage || selectedGenre || sortByFilter !== "popularity.desc");
  const isDiscover = tab === "discover";

  return (
    <div style={{ minHeight: "100dvh", background: "var(--color-bg)", display: "flex", flexDirection: "column" }}>
      <PageHeader
        title="Explore"
        hideBackButton
        rightSlot={
          session ? (
            <MobileMenu onLogout={() => { logout(); router.replace("/login"); }} />
          ) : null
        }
      />

      {/* Category Tabs & Filter Bar at top of page body */}
      <div style={{ width: "100%", background: "var(--color-bg)", borderBottom: "1px solid rgba(255, 255, 255, 0.06)", paddingTop: "6px" }}>
        {/* Category Tabs — Sleek horizontal scroll strip */}
        <div
          style={{
            display: "flex",
            gap: "8px",
            overflowX: "auto",
            scrollbarWidth: "none",
            padding: "4px var(--s-header-x) 10px",
            alignItems: "center",
          }}
        >
          {TAB_OPTIONS.map((t) => {
            const isActive = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`tab-pill ${isActive ? "glass-pill-active active" : "glass-pill"}`}
                style={{
                  padding: "7px 16px",
                  borderRadius: "999px",
                  fontSize: "13px",
                  fontWeight: isActive ? 600 : 500,
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                }}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        {/* Filter Bar — Sleek compact single horizontal scroll strip */}
        {!isDiscover && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              overflowX: "auto",
              scrollbarWidth: "none",
              padding: "8px var(--s-header-x) 12px",
              borderTop: "1px solid rgba(255, 255, 255, 0.06)",
            }}
          >
            <PillSelect
              value={selectedLanguage}
              onChange={(v) => setSelectedLanguage(v)}
              options={[
                { value: "", label: "All Languages" },
                ...LANGUAGE_OPTIONS.filter((c) => c !== "").map((c) => ({
                  value: c,
                  label: LANGUAGE_LABELS[c] || languageLabel(c),
                })),
              ]}
            />

            <PillSelect
              value={selectedGenre}
              onChange={(v) => setSelectedGenre(v)}
              options={[
                { value: "", label: "All Genres" },
                ...genresList.map((g) => ({ value: String(g.id), label: g.name })),
              ]}
            />

            <PillSelect
              value={sortByFilter}
              onChange={(v) => setSortByFilter(v as DiscoverSort)}
              active={sortByFilter !== "popularity.desc"}
              options={SORT_OPTIONS.map((s) => ({ value: s.value, label: s.label }))}
            />

            {hasActiveFilters && (
              <button
                onClick={() => {
                  setSelectedLanguage("");
                  setSelectedGenre("");
                  setSortByFilter("popularity.desc");
                }}
                style={{
                  background: "rgba(239, 68, 68, 0.12)",
                  border: "1px solid rgba(239, 68, 68, 0.3)",
                  color: "#ef4444",
                  borderRadius: "999px",
                  padding: "6px 14px",
                  fontSize: "12px",
                  fontWeight: 500,
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "4px",
                }}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
                Reset
              </button>
            )}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="app-container" style={{ flex: 1, width: "100%", padding: "32px 24px var(--s-bottom-clearance)" }}>
        {isDiscover ? (
          <Discover
            region={region}
            onSelect={(m) => setActive(toDetailMovie(m))}
          />
        ) : (
          <Grid
            movies={grid}
            loading={gridLoading}
            canLoadMore={gridPage < gridTotalPages}
            onLoadMore={loadMore}
            onSelect={(m) => setActive(toDetailMovie(m))}
            categoryId={tab}
          />
        )}
      </div>

      <MovieDetailModal
        isOpen={!!active}
        onClose={() => setActive(null)}
        movie={active}
        onMovieSelect={(m) => setActive(m)}
        sessionId={session?.session_id ?? null}
        userRegion={session?.profile?.region ?? null}
      />
    </div>
  );
}

/* ─── Grid View ─── */
function Grid({
  movies,
  loading,
  canLoadMore,
  onLoadMore,
  onSelect,
  categoryId,
}: {
  movies: ExploreMovie[];
  loading: boolean;
  canLoadMore: boolean;
  onLoadMore: () => void;
  onSelect: (m: ExploreMovie) => void;
  categoryId: ExploreCategory;
}) {
  const cat = CATEGORIES.find((c) => c.id === categoryId);

  return (
    <div>
      <div style={{ marginBottom: "24px" }}>
        <h2 className="h-section" style={{ fontSize: "20px", fontWeight: 700 }}>
          {cat?.label || "Explore"}
        </h2>
        {cat?.subtitle && (
          <p className="t-meta" style={{ margin: "4px 0 0" }}>
            {cat.subtitle}
          </p>
        )}
      </div>

      {movies.length === 0 && !loading ? (
        <EmptyState title="No movies found" description="Try clearing some filters or changing your selection." />
      ) : (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(165px, 1fr))",
              gap: "24px 18px",
            }}
          >
            {movies.map((m) => (
              <div key={m.tmdb_id} style={{ cursor: "pointer" }} onClick={() => onSelect(m)}>
                <MovieCard movie={m} />
              </div>
            ))}
          </div>

          {loading && (
            <div style={{ marginTop: "20px" }}>
              <SkeletonGrid count={8} />
            </div>
          )}

          {canLoadMore && !loading && (
            <div style={{ textAlign: "center", marginTop: "32px" }}>
              <button
                onClick={onLoadMore}
                className="btn btn-secondary"
              >
                Load More
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ─── Discover ─── */
const CURRENT_YEAR = new Date().getFullYear();

function Discover({
  region,
  onSelect,
}: {
  region?: string;
  onSelect: (m: ExploreMovie) => void;
}) {
  const [genres, setGenres] = useState<TmdbGenre[]>([]);
  const [filters, setFilters] = useState<DiscoverFilters>({
    sort_by: "popularity.desc",
    with_genres: [],
    region,
  });

  const [results, setResults] = useState<ExploreMovie[]>([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(false);
  const seen = useRef<Set<number>>(new Set());

  useEffect(() => {
    apiGenres().then(setGenres).catch(() => { });
  }, []);

  useEffect(() => {
    let cancelled = false;
    /* eslint-disable react-hooks/set-state-in-effect */
    setResults([]);
    setPage(1);
    setTotalPages(1);
    /* eslint-enable react-hooks/set-state-in-effect */
    seen.current = new Set();
    setLoading(true);
    apiDiscover({ ...filters, page: 1 })
      .then((r) => {
        if (cancelled) return;
        const fresh = r.results.filter((m) => {
          if (seen.current.has(m.tmdb_id)) return false;
          seen.current.add(m.tmdb_id);
          return true;
        });
        setResults(fresh);
        setPage(r.page);
        setTotalPages(r.total_pages);
      })
      .catch(() => { })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [filters]);

  const loadMore = useCallback(async () => {
    if (loading || page >= totalPages) return;
    setLoading(true);
    try {
      const next = await apiDiscover({ ...filters, page: page + 1 });
      const fresh = next.results.filter((m) => {
        if (seen.current.has(m.tmdb_id)) return false;
        seen.current.add(m.tmdb_id);
        return true;
      });
      setResults((p) => [...p, ...fresh]);
      setPage(next.page);
      setTotalPages(next.total_pages);
    } catch { /* ignore */ }
    finally { setLoading(false); }
  }, [filters, page, totalPages, loading]);

  const toggleGenre = (id: number) => {
    setFilters((f) => {
      const cur = f.with_genres || [];
      const next = cur.includes(id) ? cur.filter((g) => g !== id) : [...cur, id];
      return { ...f, with_genres: next };
    });
  };

  const reset = () => setFilters({ sort_by: "popularity.desc", with_genres: [], region });

  const hasFilters = Boolean(
    filters.with_original_language ||
    (filters.with_genres && filters.with_genres.length > 0) ||
    filters.sort_by !== "popularity.desc" ||
    filters.year_from ||
    filters.year_to
  );

  return (
    <div>
      {/* ── Discover Filter Panel ── */}
      <div
        className="glass-card"
        style={{
          padding: "20px",
          marginBottom: "24px",
        }}
      >
        {/* Top row: dropdowns */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
            gap: "12px",
            marginBottom: "16px",
          }}
        >
          <FilterField label="Sort by">
            <PillSelect
              value={filters.sort_by || "popularity.desc"}
              onChange={(v) => setFilters((f) => ({ ...f, sort_by: v as DiscoverSort }))}
              active={filters.sort_by !== "popularity.desc"}
              options={SORT_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
            />
          </FilterField>

          <FilterField label="Language">
            <PillSelect
              value={filters.with_original_language || ""}
              onChange={(v) => setFilters((f) => ({ ...f, with_original_language: v || undefined }))}
              options={[
                { value: "", label: "Any Language" },
                ...LANGUAGE_OPTIONS.filter((c) => c !== "").map((c) => ({
                  value: c,
                  label: LANGUAGE_LABELS[c] || languageLabel(c),
                })),
              ]}
            />
          </FilterField>

          <FilterField label="Year from">
            <NumberInput
              value={filters.year_from}
              min={1900}
              max={CURRENT_YEAR + 5}
              placeholder="Any"
              onChange={(v) => setFilters((f) => ({ ...f, year_from: v }))}
            />
          </FilterField>

          <FilterField label="Year to">
            <NumberInput
              value={filters.year_to}
              min={1900}
              max={CURRENT_YEAR + 5}
              placeholder="Any"
              onChange={(v) => setFilters((f) => ({ ...f, year_to: v }))}
            />
          </FilterField>
        </div>

        {/* Genre chips */}
        <div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
            <span className="h-eyebrow">
              Genres
            </span>
            {Boolean(filters.with_genres?.length) && (
              <span style={{ fontSize: "11px", color: "var(--color-accent-strong)", fontWeight: 500 }}>
                {filters.with_genres!.length} selected
              </span>
            )}
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {genres.map((g) => {
              const isActive = (filters.with_genres || []).includes(g.id);
              return (
                <button
                  key={g.id}
                  onClick={() => toggleGenre(g.id)}
                  style={{
                    padding: "6px 13px",
                    borderRadius: "var(--radius-pill)",
                    border: isActive
                      ? "1px solid rgba(var(--rgb-accent), 0.45)"
                      : "1px solid var(--hairline)",
                    background: isActive
                      ? "rgba(var(--rgb-accent), 0.14)"
                      : "var(--glass-chrome)",
                    color: isActive ? "var(--color-accent)" : "var(--color-text-muted)",
                    fontSize: "12px",
                    fontWeight: isActive ? 600 : 500,
                    cursor: "pointer",
                    transition: "all var(--dur-base) var(--ease-out)",
                  }}
                >
                  {g.name}
                </button>
              );
            })}
          </div>
        </div>

        {/* Reset */}
        {hasFilters && (
          <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "12px" }}>
            <button
              onClick={reset}
              style={{
                background: "rgba(var(--rgb-dislike), 0.10)",
                border: "1px solid rgba(var(--rgb-dislike), 0.28)",
                color: "var(--color-danger)",
                borderRadius: "20px",
                padding: "5px 14px",
                fontSize: "12px",
                fontWeight: 500,
                cursor: "pointer",
                transition: "all 0.15s ease",
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
              }}
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
              Reset filters
            </button>
          </div>
        )}
      </div>

      {/* ── Results Grid ── */}
      {results.length === 0 && !loading ? (
        <EmptyState title="No matching movies" description="Try broadening your Discover filters." />
      ) : (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(165px, 1fr))",
              gap: "24px 18px",
            }}
          >
            {results.map((m) => (
              <div key={m.tmdb_id} style={{ cursor: "pointer" }} onClick={() => onSelect(m)}>
                <MovieCard movie={m} />
              </div>
            ))}
          </div>

          {loading && (
            <div style={{ marginTop: "20px" }}>
              <SkeletonGrid count={8} />
            </div>
          )}

          {page < totalPages && !loading && (
            <div style={{ textAlign: "center", marginTop: "32px" }}>
              <button
                onClick={loadMore}
                className="btn btn-secondary"
              >
                Load More
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ─── Shared sub-components ─── */

function FilterField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
      <label className="h-eyebrow">
        {label}
      </label>
      {children}
    </div>
  );
}

function NumberInput({
  value,
  min,
  max,
  placeholder,
  onChange,
}: {
  value?: number;
  min: number;
  max: number;
  placeholder: string;
  onChange: (v?: number) => void;
}) {
  return (
    <input
      type="number"
      min={min}
      max={max}
      value={value ?? ""}
      placeholder={placeholder}
      onChange={(e) => {
        const v = parseInt(e.target.value, 10);
        onChange(Number.isNaN(v) ? undefined : v);
      }}
      style={{
        appearance: "none",
        WebkitAppearance: "none",
        background: "var(--glass-chrome)",
        border: "1px solid var(--hairline)",
        borderRadius: "var(--radius-pill)",
        color: "var(--color-text-secondary)",
        padding: "7px 14px",
        fontSize: "12.5px",
        fontWeight: 500,
        outline: "none",
        width: "100%",
        boxSizing: "border-box",
        transition: "all var(--dur-base) var(--ease-out)",
      }}
    />
  );
}
