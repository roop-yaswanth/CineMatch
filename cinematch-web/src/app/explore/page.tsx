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
import { useSession } from "@/context/SessionContext";
import {
  apiDiscover,
  apiExplore,
  apiGenres,
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

const LANGUAGE_OPTIONS = ["", "en", "hi", "te", "ta", "ml", "kn", "ja", "ko", "zh", "es", "fr", "de", "it", "pt", "ru"];

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

  // Initial tab read from URL (?tab=…) so deep links and back-nav restore the
  // user's last view. We validate against the known TabId set and fall back
  // to "all" for anything else.
  const initialTab: TabId = (() => {
    const t = searchParams?.get("tab") || "";
    const known: TabId[] = ["all", "trending_day", "popular", "top_rated", "now_playing", "upcoming", "discover"];
    return (known as string[]).includes(t) ? (t as TabId) : "all";
  })();
  const [tab, setTabState] = useState<TabId>(initialTab);

  // Wrapped setter that mirrors the tab to the URL without a full reroute.
  const setTab = (next: TabId) => {
    setTabState(next);
    const url = next === "all" ? "/explore" : `/explore?tab=${encodeURIComponent(next)}`;
    window.history.replaceState(null, "", url);
  };

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
    Promise.all(
      CATEGORIES.map((c) =>
        apiExplore(c.id, 1, region).then((r) => [c.id, r.results] as [string, ExploreMovie[]]).catch(() => [c.id, [] as ExploreMovie[]] as [string, ExploreMovie[]])
      )
    ).then((entries) => {
      if (cancelled) return;
      const next: Record<string, ExploreMovie[]> = {};
      for (const [k, v] of entries) next[k] = v;
      setRails(next);
      setRailLoading(false);
    });
    return () => { cancelled = true; };
  }, [tab, region]);

  // Load paginated grid for a specific category
  useEffect(() => {
    if (tab === "all" || tab === "discover") return;
    let cancelled = false;
    setGrid([]);
    setGridPage(1);
    setGridTotalPages(1);
    seenIds.current = new Set();
    setGridLoading(true);
    apiExplore(tab, 1, region)
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
      .catch(() => {})
      .finally(() => { if (!cancelled) setGridLoading(false); });
    return () => { cancelled = true; };
  }, [tab, region]);

  const loadMore = useCallback(async () => {
    if (tab === "all" || tab === "discover" || gridLoading || gridPage >= gridTotalPages) return;
    setGridLoading(true);
    try {
      const next = await apiExplore(tab, gridPage + 1, region);
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
  }, [tab, gridPage, gridTotalPages, gridLoading, region]);

  if (isLoading) return null;

  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg)", display: "flex", flexDirection: "column" }}>
      {/* Header — uses the shared <PageHeader> so back-button placement,
          title centering and right-slot spacing match every other page. */}
      <PageHeader
        title="Explore"
        rightSlot={
          session ? (
            <MobileMenu onLogout={() => { logout(); router.replace("/login"); }} />
          ) : null
        }
      >
        {/* Category tabs scroll horizontally below the header row. */}
        <div style={{ padding: "0 var(--s-header-x) var(--s-3)", display: "flex", gap: "var(--s-2)", overflowX: "auto", scrollbarWidth: "none" }}>
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
      </PageHeader>

      {/* Content. Bottom padding (`--s-bottom-clearance`) reserves space
          for the floating bottom-nav so the last row isn't hidden. */}
      <div className="app-container" style={{ flex: 1, width: "100%", padding: "var(--s-5) 0 var(--s-bottom-clearance)" }}>
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
              />
            ))}
          </div>
        ) : tab === "discover" ? (
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

/* ─── Rail ─── */
function Rail({
  category,
  movies,
  loading,
  onSeeAll,
  onSelect,
}: {
  category: CategoryDef;
  movies: ExploreMovie[];
  loading: boolean;
  onSeeAll: () => void;
  onSelect: (m: ExploreMovie) => void;
}) {
  return (
    <section>
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
          style={{ background: "none", border: "none", color: "var(--color-text-secondary)", fontSize: "13px", fontWeight: 500, cursor: "pointer", padding: "4px 8px" }}
        >
          See all →
        </button>
      </div>

      {loading ? (
        <SkeletonRail count={6} />
      ) : movies.length === 0 ? (
        <EmptyState title="Nothing here yet" description="Check back soon — TMDB updates this list often." />
      ) : (
        <div
          style={{
            display: "flex",
            gap: "12px",
            overflowX: "auto",
            padding: "0 20px 4px",
            scrollSnapType: "x mandatory",
            scrollbarWidth: "none",
          }}
        >
          {movies.slice(0, 18).map((m) => (
            <motion.div
              key={m.tmdb_id}
              onClick={() => onSelect(m)}
              style={{ width: "140px", flexShrink: 0, scrollSnapAlign: "start", cursor: "pointer" }}
              whileTap={{ scale: 0.97 }}
            >
              <MovieCard movie={m} compact noLayout showFullDate={category.id === "upcoming"} />
            </motion.div>
          ))}
        </div>
      )}
    </section>
  );
}

/* ─── Grid ─── */
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
  categoryId?: string;
}) {
  if (movies.length === 0 && loading) {
    return <div style={{ padding: "0 20px" }}><SkeletonGrid count={12} /></div>;
  }
  if (movies.length === 0) {
    return <EmptyState title="No results" description="Try a different category or filter." />;
  }
  return (
    <div style={{ padding: "0 20px" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
          gap: "20px 14px",
        }}
      >
        {movies.map((m) => (
          <motion.div
            key={m.tmdb_id}
            onClick={() => onSelect(m)}
            style={{ cursor: "pointer" }}
            whileTap={{ scale: 0.97 }}
          >
            <MovieCard movie={m} compact noLayout showFullDate={categoryId === "upcoming"} />
          </motion.div>
        ))}
      </div>

      {canLoadMore && (
        <div style={{ display: "flex", justifyContent: "center", marginTop: "32px" }}>
          <button
            onClick={onLoadMore}
            disabled={loading}
            style={{
              padding: "10px 22px",
              borderRadius: "12px",
              border: "1px solid rgba(255,255,255,0.16)",
              background: "rgba(28,30,36,0.78)",
              color: "var(--color-text-primary)",
              fontSize: "13px",
              fontWeight: 500,
              cursor: loading ? "default" : "pointer",
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "Loading…" : "Load more"}
          </button>
        </div>
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
    apiGenres().then(setGenres).catch(() => {});
  }, []);

  // Re-query whenever filters change (page resets)
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
      .catch(() => {})
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
    <div style={{ padding: "0 20px" }}>
      {/* Filter panel */}
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
        {/* Sort + Language + Year + Min rating row */}
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

          <FilterField label="Min rating">
            <NumberInput
              value={filters.vote_average_gte}
              min={0}
              max={10}
              step={0.5}
              placeholder="Any"
              onChange={(v) => setFilters((f) => ({ ...f, vote_average_gte: v }))}
            />
          </FilterField>
        </div>

        {/* Genre chips */}
        {genres.length > 0 && (
          <div>
            <div style={{ fontSize: "11px", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "8px" }}>
              Genres
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {genres.map((g) => {
                const selected = (filters.with_genres || []).includes(g.id);
                return (
                  <button
                    key={g.id}
                    onClick={() => toggleGenre(g.id)}
                    style={{
                      padding: "5px 11px",
                      borderRadius: "999px",
                      border: selected ? "1px solid rgba(255,255,255,0.32)" : "1px solid rgba(255,255,255,0.10)",
                      background: selected ? "rgba(255,255,255,0.16)" : "rgba(28,30,36,0.58)",
                      color: selected ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                      fontSize: "12px",
                      cursor: "pointer",
                      transition: "all 0.15s",
                    }}
                  >
                    {g.name}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={reset}
            style={{
              padding: "6px 12px",
              borderRadius: "8px",
              border: "1px solid rgba(255,255,255,0.10)",
              background: "transparent",
              color: "var(--color-text-secondary)",
              fontSize: "12px",
              cursor: "pointer",
            }}
          >
            Reset filters
          </button>
        </div>
      </div>

      {/* Results */}
      {results.length === 0 && loading ? (
        <SkeletonGrid count={12} />
      ) : results.length === 0 ? (
        <EmptyState title="No results" description="Try widening your filters or clearing the genre selection." cta={{ kind: "button", label: "Reset filters", onClick: reset }} />
      ) : (
        <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
              gap: "20px 14px",
            }}
          >
            {results.map((m) => (
              <motion.div key={m.tmdb_id} onClick={() => onSelect(m)} style={{ cursor: "pointer" }} whileTap={{ scale: 0.97 }}>
                <MovieCard movie={m} compact noLayout />
              </motion.div>
            ))}
          </div>
          {page < totalPages && (
            <div style={{ display: "flex", justifyContent: "center", marginTop: "32px" }}>
              <button
                onClick={loadMore}
                disabled={loading}
                style={{
                  padding: "10px 22px",
                  borderRadius: "12px",
                  border: "1px solid rgba(255,255,255,0.16)",
                  background: "rgba(28,30,36,0.78)",
                  color: "var(--color-text-primary)",
                  fontSize: "13px",
                  fontWeight: 500,
                  cursor: loading ? "default" : "pointer",
                  opacity: loading ? 0.6 : 1,
                }}
              >
                {loading ? "Loading…" : "Load more"}
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
    <label style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      <span style={{ fontSize: "11px", color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {label}
      </span>
      {children}
    </label>
  );
}

const inputStyle: React.CSSProperties = {
  background: "rgba(28,30,36,0.78)",
  border: "1px solid rgba(255,255,255,0.10)",
  borderRadius: "10px",
  padding: "8px 10px",
  color: "var(--color-text-primary)",
  fontSize: "13px",
  outline: "none",
  width: "100%",
  boxSizing: "border-box",
  height: "36px",
};

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
    <select value={value} onChange={(e) => onChange(e.target.value)} style={inputStyle}>
      {options.map((o) => (
        <option key={o.value} value={o.value} style={{ background: "#1a1c22" }}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  placeholder,
}: {
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
}) {
  return (
    <input
      type="number"
      value={value ?? ""}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      onChange={(e) => {
        const v = e.target.value;
        onChange(v === "" ? undefined : Number(v));
      }}
      style={inputStyle}
    />
  );
}
