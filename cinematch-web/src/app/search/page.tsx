"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";

import dynamic from "next/dynamic";
import MobileMenu from "@/components/MobileMenu";
import BackButton from "@/components/ui/BackButton";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";
import { useSession } from "@/context/SessionContext";

const MovieDetailModal = dynamic(() => import("@/components/modals/MovieDetailModal"), { ssr: false });
import {
  apiSearchMulti,
  languageLabel,
  peekMultiSearchCache,
  posterUrl,
  type MultiSearchMovie,
  type MultiSearchPerson,
  type MultiSearchResponse,
  type MultiSearchTv,
} from "@/lib/api";
import { getRecentSearches, rememberRecentSearch, clearRecentSearches } from "@/lib/recent-searches";
import HighlightedText from "@/components/ui/HighlightedText";
import EmptyState from "@/components/ui/EmptyState";
import { SkeletonGrid } from "@/components/ui/Skeleton";

type Tab = "movies" | "tv" | "people";

const TABS: Array<{ id: Tab; label: string }> = [
  // Movies first because (a) it's the default surface, and (b) it can
  // answer instantly from the local CSV-backed library before we ever hit
  // TMDB. TV / People always go through TMDB and are typically slower.
  { id: "movies", label: "Movies" },
  { id: "tv", label: "TV" },
  { id: "people", label: "People" },
];

export default function SearchPageWrapper() {
  return (
    <Suspense fallback={null}>
      <SearchPage />
    </Suspense>
  );
}

function SearchPage() {
  const router = useRouter();
  const params = useSearchParams();
  const { session, isLoading, logout } = useSession();
  const initialQ = params.get("q") || "";

  const [query, setQuery] = useState(initialQ);
  const [debounced, setDebounced] = useState(initialQ);
  const [tab, setTab] = useState<Tab>("movies");
  // Movies search starts with library (DB) results only; the user opts in to
  // see TMDB matches with a "Show TMDB results" button. Keeps the page snappy
  // and avoids paying TMDB latency unless the user asks for it.
  const [showTmdbMovies, setShowTmdbMovies] = useState(false);
  // Collapse on every fresh debounced query so each search starts library-first.
  useEffect(() => { setShowTmdbMovies(false); }, [debounced]);
  // Holds the most recent successful response. Crucially we keep showing the
  // last results while the user types — only when *new* results land for the
  // current query do we swap them in. No flash of empty state on every keystroke.
  const [resultRecord, setResultRecord] = useState<{ forQuery: string; data: MultiSearchResponse } | null>(
    () => {
      // Seed from cache so a deep-link to ?q=… renders instantly with no fetch wait.
      if (initialQ) {
        const cached = peekMultiSearchCache(initialQ);
        if (cached) return { forQuery: initialQ.trim(), data: cached };
      }
      return null;
    }
  );
  const [active, setActive] = useState<DetailMovie | null>(null);
  const [recents, setRecents] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  // Debounce. Shorter (200ms) — the cache-peek covers cheap re-types; the
  // debounce only matters for genuinely new fetches.
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 200);
    return () => clearTimeout(t);
  }, [query]);

  // Reflect query in URL.
  useEffect(() => {
    const url = debounced ? `/search?q=${encodeURIComponent(debounced)}` : "/search";
    window.history.replaceState(null, "", url);
  }, [debounced]);

  // Synchronous cache-peek when the user's query changes — applies before the
  // debounced fetch, so re-typing a recent query feels truly instant.
  useEffect(() => {
    if (!debounced) return;
    const cached = peekMultiSearchCache(debounced);
    if (cached) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setResultRecord({ forQuery: debounced, data: cached });
    }
  }, [debounced]);

  // Fetch (de-duped + cached by apiSearchMulti).
  useEffect(() => {
    if (!debounced) return;
    let cancelled = false;
    apiSearchMulti(debounced)
      .then((r) => {
        if (cancelled) return;
        setResultRecord({ forQuery: debounced, data: r });
        rememberRecentSearch(debounced);
        setRecents(getRecentSearches());
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [debounced]);

  // Hydrate recents on mount.
  useEffect(() => { setRecents(getRecentSearches()); }, []);

  // While the user is typing, show stale results from the previous successful
  // fetch. Loading indicator shows only when there's a true mismatch and we
  // don't have anything to display yet.
  const showingStale =
    !!debounced && (!!resultRecord && resultRecord.forQuery !== debounced);
  const results: MultiSearchResponse =
    !debounced
      ? { movies: [], tv: [], people: [] }
      : resultRecord
      ? resultRecord.data
      : { movies: [], tv: [], people: [] };
  const loading =
    !!debounced && (!resultRecord || resultRecord.forQuery !== debounced);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const openMovie = useCallback((m: MultiSearchMovie) => {
    setActive({
      id: m.tmdb_id,
      tmdb_id: m.tmdb_id,
      title: m.title,
      poster_path: m.poster_path,
      backdrop_path: m.backdrop_path,
      year: m.year,
      original_language: m.original_language,
      imdb_rating: m.imdb_rating,
      vote_average: m.vote_average,
      genres: m.genres,
      primary_genre: m.primary_genre,
      overview: m.overview,
    });
  }, []);

  if (isLoading) return null;

  const totalCount = results.movies.length + results.tv.length + results.people.length;

  return (
    <div style={{ minHeight: "100vh", background: "var(--color-bg)", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <header className="glass" style={{ position: "sticky", top: 0, zIndex: 40 }}>
        <div style={{ width: "100%", padding: "12px 20px 10px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <BackButton />

          <h1
            className="h-page h-page--brand"
            style={{ flex: 1, textAlign: "center" }}
          >
            Search
          </h1>

          <div style={{ width: "40px", flexShrink: 0, display: "flex", justifyContent: "flex-end" }}>
            {session && <MobileMenu onLogout={() => { logout(); router.replace("/login"); }} />}
          </div>
        </div>

        {/* Search input */}
        <div style={{ padding: "0 20px 12px", position: "relative" }}>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search movies, TV shows, people…"
            className="app-search-input"
          />
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ position: "absolute", left: "34px", top: "13px" }}>
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          {/* Inline loading spinner — sits in place of the clear button while
              a fetch is in flight, so users always have a visible signal that
              search is working. Backend latency on cold paths can hit ~1s. */}
          {loading && (
            <div
              aria-hidden
              style={{
                position: "absolute",
                right: "34px",
                top: "16px",
                width: 14,
                height: 14,
                border: "2px solid rgba(255,255,255,0.18)",
                borderTopColor: "rgba(255,255,255,0.85)",
                borderRadius: "50%",
                animation: "spin 0.7s linear infinite",
              }}
            />
          )}
          {query && !loading && (
            <button
              onClick={() => setQuery("")}
              aria-label="Clear"
              style={{ position: "absolute", right: "32px", top: "12px", background: "none", border: "none", cursor: "pointer", color: "var(--color-text-muted)", padding: "4px" }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          )}
        </div>

        {/* Tabs */}
        <div style={{ padding: "0 20px 12px", display: "flex", gap: "8px", overflowX: "auto", scrollbarWidth: "none" }}>
          {TABS.map((t) => {
            const isActive = tab === t.id;
            // Movies count: only the library hits are shown by default, so
            // the badge reflects that. TMDB matches are opt-in via the
            // "Search TMDB" button below the library results.
            const count =
              t.id === "movies"
                ? results.movies.filter((m) => m.source === "db").length
                : t.id === "tv"
                ? results.tv.length
                : results.people.length;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                style={{
                  padding: "7px 14px",
                  borderRadius: "999px",
                  border: isActive ? "1px solid rgba(255,255,255,0.32)" : "1px solid rgba(255,255,255,0.10)",
                  background: isActive ? "rgba(255,255,255,0.14)" : "rgba(28,30,36,0.66)",
                  color: isActive ? "var(--color-text-primary)" : "var(--color-text-secondary)",
                  fontSize: "13px",
                  fontWeight: 500,
                  whiteSpace: "nowrap",
                  cursor: "pointer",
                }}
              >
                {t.label}{debounced && ` (${count})`}
              </button>
            );
          })}
        </div>
      </header>

      {/* Body */}
      <div className="app-container" style={{ flex: 1, width: "100%", padding: "20px 20px 120px" }}>
        {!debounced ? (
          <RecentSearchesPanel
            recents={recents}
            onPick={(q) => { setQuery(q); inputRef.current?.focus(); }}
            onClear={() => { clearRecentSearches(); setRecents([]); }}
          />
        ) : loading && totalCount === 0 ? (
          <SkeletonGrid count={12} />
        ) : totalCount === 0 ? (
          <EmptyState
            title={`No results for "${debounced}"`}
            description="Try a different spelling or shorten the query."
          />
        ) : (
          <div
            style={{
              opacity: showingStale ? 0.6 : 1,
              transition: "opacity 160ms ease",
            }}
          >
            {tab === "movies" && (() => {
              // Split results by source so the local-library hits feel
              // immediate and TMDB-only matches are opt-in.
              const dbMovies = results.movies.filter((m) => m.source === "db");
              const tmdbMovies = results.movies.filter((m) => m.source === "tmdb");
              const hasDb = dbMovies.length > 0;
              const hasTmdb = tmdbMovies.length > 0;
              return (
                <>
                  {hasDb && (
                    <Section title={null}>
                      <MovieGrid movies={dbMovies} onSelect={openMovie} query={debounced} />
                    </Section>
                  )}

                  {hasTmdb && !showTmdbMovies && (
                    <div style={{ display: "flex", justifyContent: "center", padding: "12px 0 4px" }}>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => setShowTmdbMovies(true)}
                      >
                        Search TMDB ({tmdbMovies.length})
                      </button>
                    </div>
                  )}

                  {hasTmdb && showTmdbMovies && (
                    <Section title="From TMDB">
                      <MovieGrid movies={tmdbMovies} onSelect={openMovie} query={debounced} />
                    </Section>
                  )}

                  {!hasDb && !hasTmdb && (
                    <EmptyState
                      title={`No movies for "${debounced}"`}
                      tone="search"
                      description="Try a different spelling or shorten the query."
                    />
                  )}
                </>
              );
            })()}

            {tab === "tv" && (
              results.tv.length > 0 ? (
                <Section title={null}>
                  <TvGrid items={results.tv} query={debounced} />
                </Section>
              ) : (
                <EmptyState title={`No TV results for "${debounced}"`} tone="search" />
              )
            )}

            {tab === "people" && (
              results.people.length > 0 ? (
                <Section title={null}>
                  <PeopleGrid people={results.people} query={debounced} />
                </Section>
              ) : (
                <EmptyState title={`No people for "${debounced}"`} tone="search" />
              )
            )}
          </div>
        )}

        {/* Subtle inline loading bar visible while a fetch races for stale results. */}
        {loading && totalCount > 0 && (
          <div
            aria-hidden
            style={{
              marginTop: 16,
              height: 2,
              borderRadius: 1,
              background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)",
              backgroundSize: "200% 100%",
              animation: "shimmer 1.2s linear infinite",
            }}
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

/* ─── Section helpers ─── */
function Section({ title, children }: { title: string | null; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: "32px" }}>
      {title && (
        <h2 style={{ fontSize: "14px", textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-muted)", fontWeight: 600, margin: "0 0 14px" }}>
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}

function ShowMore({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        marginTop: "14px",
        background: "none",
        border: "none",
        color: "var(--color-text-secondary)",
        fontSize: "13px",
        fontWeight: 500,
        cursor: "pointer",
      }}
    >
      {label} →
    </button>
  );
}

/* ─── Recent searches panel (empty input state) ─── */
function RecentSearchesPanel({
  recents,
  onPick,
  onClear,
}: {
  recents: string[];
  onPick: (q: string) => void;
  onClear: () => void;
}) {
  if (recents.length === 0) {
    return (
      <EmptyState
        title="Find a movie, show, or person"
        description="Search the CineMatch library plus everything on TMDB."
      />
    );
  }
  return (
    <section style={{ padding: "20px 0" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <h2 style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-muted)", fontWeight: 600, margin: 0 }}>
          Recent
        </h2>
        <button
          onClick={onClear}
          style={{ background: "none", border: "none", color: "var(--color-text-muted)", fontSize: 12, cursor: "pointer", padding: 4 }}
        >
          Clear
        </button>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {recents.map((r) => (
          <button
            key={r}
            onClick={() => onPick(r)}
            style={{
              padding: "8px 14px",
              borderRadius: 999,
              border: "1px solid rgba(255,255,255,0.10)",
              background: "rgba(28,30,36,0.66)",
              color: "var(--color-text-primary)",
              fontSize: 13,
              cursor: "pointer",
              transition: "background 160ms ease",
            }}
          >
            {r}
          </button>
        ))}
      </div>
    </section>
  );
}

/* ─── Movie grid ─── */
function MovieGrid({ movies, onSelect, query }: { movies: MultiSearchMovie[]; onSelect: (m: MultiSearchMovie) => void; query: string }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: "20px 14px" }}>
      {movies.map((m) => (
        <motion.button
          key={`${m.tmdb_id}-${m.source}`}
          onClick={() => onSelect(m)}
          whileTap={{ scale: 0.97 }}
          style={{ background: "none", border: "none", padding: 0, cursor: "pointer", textAlign: "left", display: "flex", flexDirection: "column", outline: "none" }}
        >
          <div style={{ position: "relative", width: "100%", aspectRatio: "2 / 3", borderRadius: "14px", overflow: "hidden", background: "var(--color-surface)" }}>
            {m.poster_path ? (
              <Image src={posterUrl(m.poster_path, "w342")} alt={m.title} fill sizes="140px" style={{ objectFit: "cover" }} />
            ) : null}
          </div>
          <div style={{ padding: "8px 2px 0" }}>
            <div style={{ fontSize: "13px", fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1.25, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
              <HighlightedText text={m.title} query={query} />
            </div>
            <div style={{ marginTop: "3px", fontSize: "11px", color: "var(--color-text-muted)" }}>
              {m.year || ""}{m.year && m.original_language ? " · " : ""}{m.original_language ? languageLabel(m.original_language) : ""}
            </div>
          </div>
        </motion.button>
      ))}
    </div>
  );
}

/* ─── TV grid (TMDB only — open TMDB page in new tab) ─── */
function TvGrid({ items, query }: { items: MultiSearchTv[]; query: string }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: "20px 14px" }}>
      {items.map((t) => (
        <a
          key={t.tmdb_id}
          href={`https://www.themoviedb.org/tv/${t.tmdb_id}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ textDecoration: "none", color: "inherit", display: "flex", flexDirection: "column" }}
        >
          <div style={{ position: "relative", width: "100%", aspectRatio: "2 / 3", borderRadius: "14px", overflow: "hidden", background: "var(--color-surface)" }}>
            {t.poster_path ? (
              <Image src={posterUrl(t.poster_path, "w342")} alt={t.name} fill sizes="140px" style={{ objectFit: "cover" }} />
            ) : (
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: "12px" }}>No poster</div>
            )}
            <div style={{ position: "absolute", top: "6px", right: "6px", background: "rgba(0,0,0,0.7)", color: "#fff", fontSize: "9px", padding: "2px 6px", borderRadius: "4px", letterSpacing: "0.04em" }}>TV</div>
          </div>
          <div style={{ padding: "8px 2px 0" }}>
            <div style={{ fontSize: "13px", fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1.25, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
              <HighlightedText text={t.name} query={query} />
            </div>
            <div style={{ marginTop: "3px", fontSize: "11px", color: "var(--color-text-muted)" }}>
              {t.year || ""}{t.year && t.original_language ? " · " : ""}{t.original_language ? languageLabel(t.original_language) : ""}
            </div>
          </div>
        </a>
      ))}
    </div>
  );
}

/* ─── People grid ─── */
function PeopleGrid({ people, query }: { people: MultiSearchPerson[]; query: string }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: "24px 14px" }}>
      {people.map((p) => (
        <Link
          key={p.tmdb_id}
          href={`/person/${p.tmdb_id}`}
          style={{ textDecoration: "none", color: "inherit", display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}
        >
          <div style={{ position: "relative", width: "120px", height: "120px", borderRadius: "50%", overflow: "hidden", background: "var(--color-surface)" }}>
            {p.profile_path ? (
              <Image src={posterUrl(p.profile_path, "w185")} alt={p.name} fill sizes="120px" style={{ objectFit: "cover" }} />
            ) : (
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: "32px" }}>
                {p.name?.[0] || "?"}
              </div>
            )}
          </div>
          <div style={{ padding: "10px 4px 0" }}>
            <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--color-text-primary)", lineHeight: 1.25 }}>
              <HighlightedText text={p.name} query={query} />
            </div>
            {p.known_for_department && (
              <div style={{ marginTop: "3px", fontSize: "11px", color: "var(--color-text-muted)" }}>
                {p.known_for_department}
              </div>
            )}
            {p.known_for.length > 0 && (
              <div style={{ marginTop: "4px", fontSize: "11px", color: "var(--color-text-muted)", lineHeight: 1.3, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
                {p.known_for.map((k) => k.title).filter(Boolean).slice(0, 3).join(", ")}
              </div>
            )}
          </div>
        </Link>
      ))}
    </div>
  );
}
