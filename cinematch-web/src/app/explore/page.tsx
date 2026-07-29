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
          ? "linear-gradient(135deg, rgba(139,92,246,0.25), rgba(59,130,246,0.18))"
          : "rgba(255,255,255,0.05)",
        border: isActive
          ? "1px solid rgba(139,92,246,0.45)"
          : "1px solid rgba(255,255,255,0.10)",
        borderRadius: "20px",
        color: isActive ? "#e0d4fc" : "rgba(255,255,255,0.65)",
        padding: "6px 28px 6px 12px",
        fontSize: "12.5px",
        fontWeight: 500,
        cursor: "pointer",
        outline: "none",
        transition: "all 0.18s ease",
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
    setGrid([]);
    setGridPage(1);
    setGridTotalPages(1);
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

  if (isLoading || !session) {
    return <div style={{ minHeight: "100vh", background: "var(--color-bg)" }} />;
  }

  const handleAction = useCallback(
    async (m: any, action: "like" | "okay" | "dislike" | "watchlist" | "skip") => {
      if (!session) return;
      try {
        await apiRecommendationAction(session.session_id, m.id || m.tmdb_id, action);
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

  const hasActiveFilters = Boolean(selectedLanguage || selectedGenre || sortByFilter !== "popularity.desc");
  const isDiscover = tab === "discover";

  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg)", display: "flex", flexDirection: "column" }}>
      <PageHeader
        title="Explore"
        rightSlot={
          session ? (
            <MobileMenu onLogout={() => { logout(); router.replace("/login"); }} />
          ) : null
        }
      >
        {/* Category Tabs */}
        <div style={{ display: "flex", gap: "6px", overflowX: "auto", scrollbarWidth: "none", paddingBottom: "8px" }}>
          {TAB_OPTIONS.map((t) => {
            const isActive = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                style={{
                  padding: "7px 16px",
                  borderRadius: "999px",
                  border: isActive ? "1px solid rgba(255,255,255,0.32)" : "1px solid rgba(255,255,255,0.08)",
                  background: isActive ? "rgba(255,255,255,0.14)" : "transparent",
                  color: isActive ? "#fff" : "rgba(255,255,255,0.55)",
                  fontSize: "13px",
                  fontWeight: isActive ? 600 : 450,
                  whiteSpace: "nowrap",
                  cursor: "pointer",
                  transition: "all 0.18s ease",
                  letterSpacing: "-0.01em",
                }}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        {/* Filter Bar — hidden on Discover (Discover has its own rich filter panel) */}
        {!isDiscover && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: "8px",
              paddingTop: "8px",
              borderTop: "1px solid rgba(255,255,255,0.05)",
            }}
          >
            <PillSelect
              value={selectedLanguage}
              onChange={(v) => setSelectedLanguage(v)}
              options={[
                { value: "", label: "🌐  All Languages" },
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
                { value: "", label: "🎭  All Genres" },
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
                  background: "rgba(239, 68, 68, 0.10)",
                  border: "1px solid rgba(239, 68, 68, 0.25)",
                  color: "#f87171",
                  borderRadius: "20px",
                  padding: "6px 14px",
                  fontSize: "12px",
                  fontWeight: 500,
                  cursor: "pointer",
                  transition: "all 0.15s ease",
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
      </PageHeader>

      {/* Content */}
      <div className="app-container" style={{ flex: 1, width: "100%", padding: "var(--s-5) 24px var(--s-bottom-clearance)" }}>
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
      <div style={{ marginBottom: "20px" }}>
        <h2 style={{ fontSize: "22px", fontWeight: 700, color: "var(--color-text-primary)", margin: 0 }}>
          {cat?.label || "Explore"}
        </h2>
        {cat?.subtitle && (
          <p style={{ fontSize: "13px", color: "var(--color-text-muted)", margin: "4px 0 0" }}>
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
              gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
              gap: "16px",
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
                style={{
                  padding: "10px 24px",
                  borderRadius: "999px",
                  background: "rgba(255,255,255,0.08)",
                  border: "1px solid rgba(255,255,255,0.16)",
                  color: "var(--color-text-primary)",
                  fontSize: "13px",
                  fontWeight: 600,
                  cursor: "pointer",
                }}
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
    setResults([]);
    setPage(1);
    setTotalPages(1);
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
        style={{
          background: "linear-gradient(135deg, rgba(20,22,30,0.75), rgba(30,28,40,0.55))",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: "16px",
          padding: "20px",
          marginBottom: "24px",
          backdropFilter: "blur(12px)",
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
            <span style={{ fontSize: "11px", fontWeight: 600, color: "rgba(255,255,255,0.4)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              Genres
            </span>
            {Boolean(filters.with_genres?.length) && (
              <span style={{ fontSize: "11px", color: "rgba(139,92,246,0.8)", fontWeight: 500 }}>
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
                    padding: "5px 12px",
                    borderRadius: "999px",
                    border: isActive
                      ? "1px solid rgba(139,92,246,0.45)"
                      : "1px solid rgba(255,255,255,0.08)",
                    background: isActive
                      ? "linear-gradient(135deg, rgba(139,92,246,0.22), rgba(59,130,246,0.14))"
                      : "rgba(255,255,255,0.03)",
                    color: isActive ? "#e0d4fc" : "rgba(255,255,255,0.5)",
                    fontSize: "12px",
                    fontWeight: isActive ? 600 : 400,
                    cursor: "pointer",
                    transition: "all 0.18s ease",
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
                background: "rgba(239, 68, 68, 0.10)",
                border: "1px solid rgba(239, 68, 68, 0.25)",
                color: "#f87171",
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
              gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
              gap: "16px",
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
                style={{
                  padding: "10px 24px",
                  borderRadius: "999px",
                  background: "rgba(255,255,255,0.08)",
                  border: "1px solid rgba(255,255,255,0.16)",
                  color: "var(--color-text-primary)",
                  fontSize: "13px",
                  fontWeight: 600,
                  cursor: "pointer",
                }}
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
      <label style={{ fontSize: "11px", fontWeight: 600, color: "rgba(255,255,255,0.4)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
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
        background: "rgba(255,255,255,0.05)",
        border: "1px solid rgba(255,255,255,0.10)",
        borderRadius: "20px",
        color: "rgba(255,255,255,0.65)",
        padding: "6px 12px",
        fontSize: "12.5px",
        fontWeight: 500,
        outline: "none",
        width: "100%",
        boxSizing: "border-box",
        transition: "all 0.18s ease",
      }}
    />
  );
}
