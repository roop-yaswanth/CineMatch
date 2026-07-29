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

type TabId = "all" | ExploreCategory | "discover";

const TAB_OPTIONS: Array<{ id: TabId; label: string }> = [
  { id: "all", label: "All" },
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
    const known: TabId[] = ["all", "trending_day", "popular", "top_rated", "now_playing", "upcoming", "discover"];
    return (known as string[]).includes(t) ? (t as TabId) : "all";
  })();
  const [tab, setTabState] = useState<TabId>(initialTab);

  const setTab = (next: TabId) => {
    setTabState(next);
    const url = next === "all" ? "/explore" : `/explore?tab=${encodeURIComponent(next)}`;
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

  const [rails, setRails] = useState<Record<string, ExploreMovie[]>>({});
  const [railLoading, setRailLoading] = useState(false);

  const [grid, setGrid] = useState<ExploreMovie[]>([]);
  const [gridPage, setGridPage] = useState(1);
  const [gridTotalPages, setGridTotalPages] = useState(1);
  const [gridLoading, setGridLoading] = useState(false);

  const [active, setActive] = useState<DetailMovie | null>(null);
  const seenIds = useRef<Set<number>>(new Set());

  const region = session?.profile?.region || undefined;

  // Load all rails in parallel for "All" view
  useEffect(() => {
    if (tab !== "all") return;
    let cancelled = false;
    setRailLoading(true);
    const lang = selectedLanguage || undefined;
    const genre = selectedGenre || undefined;
    const sort = sortByFilter !== "popularity.desc" ? sortByFilter : undefined;

    Promise.all(
      CATEGORIES.map((c) =>
        apiExplore(c.id, 1, region, lang, genre, sort)
          .then((r) => [c.id, r.results] as [string, ExploreMovie[]])
          .catch(() => [c.id, [] as ExploreMovie[]] as [string, ExploreMovie[]])
      )
    ).then((entries) => {
      if (cancelled) return;
      const next: Record<string, ExploreMovie[]> = {};
      for (const [k, v] of entries) next[k] = v;
      setRails(next);
      setRailLoading(false);
    });
    return () => { cancelled = true; };
  }, [tab, region, selectedLanguage, selectedGenre, sortByFilter]);

  // Load paginated grid for a specific category
  useEffect(() => {
    if (tab === "all" || tab === "discover") return;
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
    if (tab === "all" || tab === "discover" || gridLoading || gridPage >= gridTotalPages) return;
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
        {/* Row 1: Category Tabs */}
        <div style={{ display: "flex", gap: "var(--s-2)", overflowX: "auto", scrollbarWidth: "none", paddingBottom: "10px" }}>
          {TAB_OPTIONS.map((t) => {
            const active = tab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                style={{
                  padding: "7px 14px",
                  borderRadius: "999px",
                  border: active ? "1px solid rgba(255,255,255,0.32)" : "1px solid rgba(255,255,255,0.10)",
                  background: active ? "rgba(255,255,255,0.14)" : "rgba(28,30,36,0.66)",
                  color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                  fontSize: "13px",
                  fontWeight: 500,
                  whiteSpace: "nowrap",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        {/* Row 2: Explore Filter Bar */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "10px",
            paddingTop: "6px",
            borderTop: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          {/* Language Selector */}
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "12px", color: "rgba(255,255,255,0.5)", fontWeight: 500 }}>Language:</span>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              style={{
                background: selectedLanguage ? "rgba(255, 255, 255, 0.16)" : "rgba(255, 255, 255, 0.08)",
                border: selectedLanguage ? "1px solid rgba(255, 255, 255, 0.3)" : "1px solid rgba(255, 255, 255, 0.14)",
                borderRadius: "10px",
                color: "#ffffff",
                padding: "5px 10px",
                fontSize: "12px",
                fontWeight: 500,
                cursor: "pointer",
                outline: "none",
              }}
            >
              <option value="" style={{ background: "#161820", color: "#fff" }}>All Languages</option>
              {LANGUAGE_OPTIONS.filter((c) => c !== "").map((c) => (
                <option key={c} value={c} style={{ background: "#161820", color: "#fff" }}>
                  {LANGUAGE_LABELS[c] || languageLabel(c)}
                </option>
              ))}
            </select>
          </div>

          {/* Genre Selector */}
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "12px", color: "rgba(255,255,255,0.5)", fontWeight: 500 }}>Genre:</span>
            <select
              value={selectedGenre}
              onChange={(e) => setSelectedGenre(e.target.value)}
              style={{
                background: selectedGenre ? "rgba(255, 255, 255, 0.16)" : "rgba(255, 255, 255, 0.08)",
                border: selectedGenre ? "1px solid rgba(255, 255, 255, 0.3)" : "1px solid rgba(255, 255, 255, 0.14)",
                borderRadius: "10px",
                color: "#ffffff",
                padding: "5px 10px",
                fontSize: "12px",
                fontWeight: 500,
                cursor: "pointer",
                outline: "none",
              }}
            >
              <option value="" style={{ background: "#161820", color: "#fff" }}>All Genres</option>
              {genresList.map((g) => (
                <option key={g.id} value={String(g.id)} style={{ background: "#161820", color: "#fff" }}>
                  {g.name}
                </option>
              ))}
            </select>
          </div>

          {/* Sort Selector */}
          {tab !== "all" && tab !== "discover" && (
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ fontSize: "12px", color: "rgba(255,255,255,0.5)", fontWeight: 500 }}>Sort:</span>
              <select
                value={sortByFilter}
                onChange={(e) => setSortByFilter(e.target.value as DiscoverSort)}
                style={{
                  background: sortByFilter !== "popularity.desc" ? "rgba(255, 255, 255, 0.16)" : "rgba(255, 255, 255, 0.08)",
                  border: sortByFilter !== "popularity.desc" ? "1px solid rgba(255, 255, 255, 0.3)" : "1px solid rgba(255, 255, 255, 0.14)",
                  borderRadius: "10px",
                  color: "#ffffff",
                  padding: "5px 10px",
                  fontSize: "12px",
                  fontWeight: 500,
                  cursor: "pointer",
                  outline: "none",
                }}
              >
                {SORT_OPTIONS.map((s) => (
                  <option key={s.value} value={s.value} style={{ background: "#161820", color: "#fff" }}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Reset Filters button */}
          {hasActiveFilters && (
            <button
              onClick={() => {
                setSelectedLanguage("");
                setSelectedGenre("");
                setSortByFilter("popularity.desc");
              }}
              style={{
                background: "rgba(239, 68, 68, 0.14)",
                border: "1px solid rgba(239, 68, 68, 0.3)",
                color: "#f87171",
                borderRadius: "10px",
                padding: "5px 12px",
                fontSize: "12px",
                fontWeight: 500,
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              Reset Filters
            </button>
          )}
        </div>
      </PageHeader>

      {/* Content */}
      <div className="app-container" style={{ flex: 1, width: "100%", padding: "var(--s-5) 24px var(--s-bottom-clearance)" }}>
        {tab === "all" ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
            {CATEGORIES.map((cat) => (
              <Rail
                key={cat.id}
                category={cat}
                movies={rails[cat.id] || []}
                loading={railLoading && !rails[cat.id]}
                onSeeAll={() => setTab(cat.id)}
                onSelect={(m) => setActive(toDetailMovie(m))}
                onAction={handleAction}
              />
            ))}
          </div>
        ) : tab === "discover" ? (
          <Discover
            region={region}
            onSelect={(m) => setActive(toDetailMovie(m))}
            initialLanguage={selectedLanguage}
            initialGenre={selectedGenre}
            initialSort={sortByFilter}
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

/* ─── Rail ─── */
function Rail({
  category,
  movies,
  loading,
  onSeeAll,
  onSelect,
  onAction,
}: {
  category: CategoryDef;
  movies: ExploreMovie[];
  loading: boolean;
  onSeeAll: () => void;
  onSelect: (m: ExploreMovie) => void;
  onAction: (movie: any, action: any) => void;
}) {
  return (
    <section style={{ overflow: "visible" }}>
      <div style={{ padding: "0 20px 10px", display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <div>
          <h2 style={{ fontSize: "18px", fontWeight: 600, color: "var(--color-text-primary)", margin: 0, letterSpacing: "-0.02em" }}>
            {category.label}
          </h2>
          <p style={{ fontSize: "12px", color: "var(--color-text-muted)", margin: "2px 0 0", fontWeight: 400 }}>
            {category.subtitle}
          </p>
        </div>
        <button
          onClick={onSeeAll}
          style={{
            background: "rgba(255,255,255,0.06)",
            border: "1px solid rgba(255,255,255,0.12)",
            color: "var(--color-text-primary)",
            fontSize: "12px",
            fontWeight: 600,
            cursor: "pointer",
            padding: "6px 14px",
            borderRadius: "999px",
            display: "inline-flex",
            alignItems: "center",
            gap: "5px",
            transition: "all 0.18s ease",
          }}
        >
          <span>See all</span>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </button>
      </div>

      {loading ? (
        <SkeletonRail count={6} />
      ) : movies.length === 0 ? (
        <EmptyState title="Nothing here yet" description="Check back soon — TMDB updates this list often." />
      ) : (
        <div
          className="hide-scrollbar"
          style={{
            display: "flex",
            gap: "14px",
            overflowX: "auto",
            padding: "4px 20px 16px",
            scrollSnapType: "x mandatory",
            scrollbarWidth: "none",
          }}
        >
          {movies.slice(0, 18).map((m) => (
            <PosterCard
              key={m.tmdb_id}
              movie={m as any}
              disabled={false}
              onAction={onAction}
              onClick={() => onSelect(m)}
            />
          ))}
        </div>
      )}
    </section>
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
  initialLanguage = "",
  initialGenre = "",
  initialSort = "popularity.desc",
}: {
  region?: string;
  onSelect: (m: ExploreMovie) => void;
  initialLanguage?: string;
  initialGenre?: string;
  initialSort?: DiscoverSort;
}) {
  const [genres, setGenres] = useState<TmdbGenre[]>([]);
  const [filters, setFilters] = useState<DiscoverFilters>({
    sort_by: initialSort,
    with_genres: initialGenre ? [Number(initialGenre)] : [],
    with_original_language: initialLanguage || undefined,
    region,
  });

  useEffect(() => {
    setFilters((f) => ({
      ...f,
      with_original_language: initialLanguage || undefined,
      with_genres: initialGenre ? [Number(initialGenre)] : f.with_genres,
      sort_by: initialSort || f.sort_by,
    }));
  }, [initialLanguage, initialGenre, initialSort]);

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

  return (
    <div style={{ padding: "0 clamp(20px, 4vw, 40px)" }}>
      <div
        style={{
          background: "rgba(20, 22, 28, 0.62)",
          border: "1px solid rgba(255,255,255,0.10)",
          borderRadius: "16px",
          padding: "16px",
          marginBottom: "20px",
          display: "flex",
          flexDirection: "column",
          gap: "14px",
        }}
      >
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "10px" }}>
          <FilterField label="Sort by">
            <SelectInput
              value={filters.sort_by || "popularity.desc"}
              onChange={(v) => setFilters((f) => ({ ...f, sort_by: v as DiscoverSort }))}
              options={SORT_OPTIONS.map((o) => ({ value: o.value, label: o.label }))}
            />
          </FilterField>

          <FilterField label="Language">
            <SelectInput
              value={filters.with_original_language || ""}
              onChange={(v) => setFilters((f) => ({ ...f, with_original_language: v || undefined }))}
              options={LANGUAGE_OPTIONS.map((c) => ({
                value: c,
                label: c === "" ? "Any" : (LANGUAGE_LABELS[c] || languageLabel(c)),
              }))}
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

        <div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "11px", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Genres
            </span>
            {Boolean(filters.with_genres?.length) && (
              <span style={{ fontSize: "11px", color: "var(--color-text-secondary)" }}>
                {filters.with_genres!.length} selected
              </span>
            )}
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
            {genres.map((g) => {
              const active = (filters.with_genres || []).includes(g.id);
              return (
                <button
                  key={g.id}
                  onClick={() => toggleGenre(g.id)}
                  style={{
                    padding: "4px 10px",
                    borderRadius: "999px",
                    border: active ? "1px solid rgba(255,255,255,0.3)" : "1px solid rgba(255,255,255,0.08)",
                    background: active ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.03)",
                    color: active ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                    fontSize: "12px",
                    fontWeight: active ? 600 : 400,
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                  }}
                >
                  {g.name}
                </button>
              );
            })}
          </div>
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={reset}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--color-text-muted)",
              fontSize: "12px",
              cursor: "pointer",
              textDecoration: "underline",
            }}
          >
            Reset discover filters
          </button>
        </div>
      </div>

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

function FilterField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <label style={{ fontSize: "11px", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </label>
      {children}
    </div>
  );
}

function SelectInput({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "8px",
        color: "var(--color-text-primary)",
        padding: "6px 10px",
        fontSize: "12px",
        outline: "none",
        cursor: "pointer",
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
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "8px",
        color: "var(--color-text-primary)",
        padding: "6px 10px",
        fontSize: "12px",
        outline: "none",
        width: "100%",
        boxSizing: "border-box",
      }}
    />
  );
}
