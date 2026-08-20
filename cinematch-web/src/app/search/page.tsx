"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import Link from "next/link";
import { motion } from "framer-motion";

import dynamic from "next/dynamic";
import MobileMenu from "@/components/MobileMenu";
import PageHeader from "@/components/ui/PageHeader";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";
import { useSession } from "@/context/SessionContext";

const MovieDetailModal = dynamic(() => import("@/components/modals/MovieDetailModal"), { ssr: false });
import { toast } from "@/components/ui/Toast";
import {
  apiSearchMulti,
  languageLabel,
  peekMultiSearchCache,
  posterUrl,
  apiRecommendationAction,
  type MultiSearchMovie,
  type MultiSearchPerson,
  type MultiSearchResponse,
  type MultiSearchTopItem,
  type MultiSearchTv,
} from "@/lib/api";
import { getRecentSearches, rememberRecentSearch, clearRecentSearches } from "@/lib/recent-searches";
import HighlightedText from "@/components/ui/HighlightedText";
import EmptyState from "@/components/ui/EmptyState";
import { SkeletonGrid } from "@/components/ui/Skeleton";

// Removed TABS definition

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

  // Auth gate: this is an app page, so require a logged-in session (it must not
  // be reachable by typing the URL while signed out).
  useEffect(() => {
    if (!isLoading && !session) {
      if (typeof window !== "undefined") {
        window.location.replace("/login");
      } else {
        router.replace("/login");
      }
    }
  }, [session, isLoading, router]);

  const initialQ = params.get("q") || "";

  const [query, setQuery] = useState(initialQ);
  const [debounced, setDebounced] = useState(initialQ);
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
      .catch(() => { });
    return () => { cancelled = true; };
  }, [debounced]);

  // Hydrate recents on mount.
  useEffect(() => { Promise.resolve().then(() => setRecents(getRecentSearches())); }, []);

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

  const handleAction = useCallback(
    async (action: "like" | "okay" | "dislike" | "watchlist" | "skip") => {
      if (!session || !active) return;
      try {
        await apiRecommendationAction(session.session_id, active.id, action);
        if (action === "watchlist") {
          toast({
            message: `Added "${active.title}" to your watchlist`,
            tone: "success",
          });
        } else if (action === "like") {
          toast({
            message: `Liked "${active.title}"`,
            tone: "success",
          });
        } else if (action === "okay") {
          toast({
            message: `Rated "${active.title}" as Okay`,
            tone: "success",
          });
        } else if (action === "dislike") {
          toast({
            message: `Disliked "${active.title}"`,
            tone: "neutral",
          });
        }
      } catch (err) {
        console.error("Action failed:", err);
      }
    },
    [session, active]
  );

  if (isLoading || !session) {
    return <div style={{ minHeight: "100dvh", background: "var(--color-bg)" }} />;
  }

  const totalCount = results.movies.length + results.tv.length + results.people.length;

  return (
    <div style={{ minHeight: "100dvh", background: "var(--color-bg)", display: "flex", flexDirection: "column" }}>
      {/* Header — shared <PageHeader> for layout consistency. */}
      <PageHeader
        title="Search"
        hideBackButton
        rightSlot={
          session ? <MobileMenu onLogout={() => { logout(); router.replace("/login"); }} /> : null
        }
      >
        {/* Search input */}
        <div style={{ padding: "0 var(--s-header-x) var(--s-3)" }}>
          <div style={{ position: "relative" }}>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search movies, TV shows, people…"
              className="app-search-input"
            />
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)" }}>
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            {/* Inline loading spinner — sits in place of the clear button while
              a fetch is in flight, so users always have a visible signal that
              search is working. Backend latency on cold paths can hit ~1s.
              Matches the search icon's 18×18 size and is vertically centered the
              same way (top:50% + translateY) so the two never look mismatched. */}
            {loading && (
              <div
                aria-hidden
                style={{
                  position: "absolute",
                  right: "14px",
                  top: 0,
                  bottom: 0,
                  margin: "auto 0",
                  width: 18,
                  height: 18,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  pointerEvents: "none",
                }}
              >
                <div
                  style={{
                    width: 18,
                    height: 18,
                    border: "2px solid rgba(255,255,255,0.18)",
                    borderTopColor: "rgba(255,255,255,0.85)",
                    borderRadius: "50%",
                    animation: "spin 0.7s linear infinite",
                  }}
                />
              </div>
            )}
            {query && !loading && (
              <button
                onClick={() => setQuery("")}
                aria-label="Clear"
                // 40 px touch target — was `padding: 4px` which gave a
                // ~16×16 hit area, well below the iOS 44 px guideline. The
                // visual icon stays 16×16; the surrounding box just becomes
                // easier to tap.
                style={{
                  position: "absolute",
                  right: "4px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  width: "40px",
                  height: "40px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "none",
                  border: "none",
                  borderRadius: "999px",
                  cursor: "pointer",
                  color: "var(--color-text-muted)",
                  padding: 0,
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            )}
          </div>
        </div>

      </PageHeader>

      {/* Body */}
      <div className="app-container" style={{ flex: 1, width: "100%", padding: "var(--s-5) var(--s-header-x) var(--s-bottom-clearance)" }}>
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
            {(() => {
              // The route ranks every source on one relevance scale — render
              // movies as a single ranked grid instead of Library/TMDB silos,
              // and lead with the mixed best-match row so a TV-only title
              // (absent from the movie library) still tops the page.
              const topItems = results.top ?? [];
              const hasTv = results.tv.length > 0;
              const hasPeople = results.people.length > 0;

              return (
                <>
                  {topItems.length > 0 && (
                    <Section title="Top Results">
                      <TopResultsRow items={topItems} onSelectMovie={openMovie} query={debounced} />
                    </Section>
                  )}

                  {results.movies.length > 0 && (
                    <Section title="Movies">
                      <MovieGrid movies={results.movies} onSelect={openMovie} query={debounced} />
                    </Section>
                  )}

                  {hasTv && (
                    <Section title="TV Shows">
                      <TvGrid items={results.tv} query={debounced} />
                    </Section>
                  )}

                  {hasPeople && (
                    <Section title="People">
                      <PeopleGrid people={results.people} query={debounced} />
                    </Section>
                  )}
                </>
              );
            })()}
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
        onAction={handleAction}
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

/* ─── Top results — mixed movies/TV/people on one relevance scale ─── */
function TopResultsRow({
  items,
  onSelectMovie,
  query,
}: {
  items: MultiSearchTopItem[];
  onSelectMovie: (m: MultiSearchMovie) => void;
  query: string;
}) {
  const cardBase: React.CSSProperties = {
    flex: "0 0 auto",
    width: 132,
    display: "flex",
    flexDirection: "column",
    textAlign: "left",
    textDecoration: "none",
    color: "inherit",
    background: "none",
    border: "none",
    padding: 0,
    cursor: "pointer",
    outline: "none",
  };
  const posterBase: React.CSSProperties = {
    position: "relative",
    width: "100%",
    aspectRatio: "2 / 3",
    borderRadius: "var(--radius-poster)",
    overflow: "hidden",
    background: "var(--color-surface)",
  };
  const chip = (label: string) => (
    <span
      style={{
        position: "absolute",
        top: 6,
        left: 6,
        padding: "2px 7px",
        borderRadius: 5,
        fontSize: 9,
        fontWeight: 700,
        letterSpacing: "0.05em",
        textTransform: "uppercase",
        background: "rgba(5,5,7,0.72)",
        color: "var(--color-accent)",
        border: "1px solid rgba(var(--rgb-accent), 0.35)",
        backdropFilter: "blur(6px)",
        WebkitBackdropFilter: "blur(6px)",
      }}
    >
      {label}
    </span>
  );
  const meta = (title: string, sub: string) => (
    <div style={{ padding: "8px 2px 0" }}>
      <div style={{ fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1.25, overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
        <HighlightedText text={title} query={query} />
      </div>
      {sub && <div style={{ marginTop: 3, fontSize: 11, color: "var(--color-text-muted)" }}>{sub}</div>}
    </div>
  );

  return (
    <div className="hide-scrollbar" style={{ display: "flex", gap: 14, overflowX: "auto", paddingBottom: 4, WebkitOverflowScrolling: "touch" }}>
      {items.map((it) => {
        if (it.media_type === "movie") {
          return (
            <motion.button key={`top-m-${it.tmdb_id}`} whileTap={{ scale: 0.97 }} onClick={() => onSelectMovie(it)} style={cardBase}>
              <div style={posterBase}>
                {it.poster_path && <img src={posterUrl(it.poster_path, "w342")} alt={it.title} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />}
                {chip("Movie")}
              </div>
              {meta(it.title, [it.year, it.original_language ? languageLabel(it.original_language) : ""].filter(Boolean).join(" · "))}
            </motion.button>
          );
        }
        if (it.media_type === "tv") {
          return (
            <a key={`top-t-${it.tmdb_id}`} href={`https://www.themoviedb.org/tv/${it.tmdb_id}`} target="_blank" rel="noopener noreferrer" style={cardBase}>
              <div style={posterBase}>
                {it.poster_path && <img src={posterUrl(it.poster_path, "w342")} alt={it.name} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />}
                {chip("TV")}
              </div>
              {meta(it.name, [it.year, it.original_language ? languageLabel(it.original_language) : ""].filter(Boolean).join(" · "))}
            </a>
          );
        }
        return (
          <Link key={`top-p-${it.tmdb_id}`} href={`/person/${it.tmdb_id}`} style={cardBase}>
            <div style={{ ...posterBase, aspectRatio: "1 / 1", borderRadius: "50%", width: 108, margin: "12px auto" }}>
              {it.profile_path ? (
                <img src={posterUrl(it.profile_path, "w185")} alt={it.name} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
              ) : (
                <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: 28 }}>
                  {it.name?.[0] || "?"}
                </div>
              )}
            </div>
            <div style={{ textAlign: "center" }}>{meta(it.name, it.known_for_department || "")}</div>
          </Link>
        );
      })}
    </div>
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
            className="glass-pill"
            style={{
              padding: "8px 14px",
              fontSize: 13,
              cursor: "pointer",
              color: "var(--color-text-primary)",
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
      {movies.map((m) => {
        const isImdb = m.source === "imdb";
        const inner = (
          <>
            <div style={{ position: "relative", width: "100%", aspectRatio: "2 / 3", borderRadius: "14px", overflow: "hidden", background: "var(--color-surface)" }}>
              {m.poster_path ? (
                <img src={posterUrl(m.poster_path, "w342")} alt={m.title} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
              ) : null}
              {isImdb && (
                <div style={{ position: "absolute", top: "6px", right: "6px", background: "rgba(var(--rgb-rating),0.92)", color: "#000", fontSize: "9px", fontWeight: 700, padding: "2px 6px", borderRadius: "4px", letterSpacing: "0.04em" }}>IMDb</div>
              )}
            </div>
            <div style={{ padding: "8px 2px 0" }}>
              <div className="poster-info-title">
                <HighlightedText text={m.title} query={query} />
              </div>
              <div className="poster-info-meta-text" style={{ marginTop: "3px" }}>
                {[
                  m.year,
                  m.original_language ? languageLabel(m.original_language) : "",
                ].filter(Boolean).join(" · ")}
                {(m.imdb_rating || m.vote_average) ? (
                  <span style={{ color: "var(--color-rating)", fontWeight: 700 }}>
                    {" "}· ★ {(m.imdb_rating ?? m.vote_average)!.toFixed(1)}
                  </span>
                ) : null}
              </div>
            </div>
          </>
        );

        if (isImdb && m.imdb_url) {
          return (
            <a key={`${m.imdb_url}-imdb`} href={m.imdb_url} target="_blank" rel="noopener noreferrer" style={{ textDecoration: "none", color: "inherit", display: "flex", flexDirection: "column" }}>
              {inner}
            </a>
          );
        }

        return (
          <motion.button
            key={`${m.tmdb_id}-${m.source}`}
            onClick={() => onSelect(m)}
            whileTap={{ scale: 0.97 }}
            style={{ background: "none", border: "none", padding: 0, cursor: "pointer", textAlign: "left", display: "flex", flexDirection: "column", outline: "none" }}
          >
            {inner}
          </motion.button>
        );
      })}
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
              <img src={posterUrl(t.poster_path, "w342")} alt={t.name} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
            ) : (
              <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-muted)", fontSize: "12px" }}>No poster</div>
            )}
            <div style={{ position: "absolute", top: "6px", right: "6px", background: "rgba(0,0,0,0.7)", color: "#fff", fontSize: "9px", padding: "2px 6px", borderRadius: "4px", letterSpacing: "0.04em" }}>TV</div>
          </div>
          <div style={{ padding: "8px 2px 0" }}>
            <div className="poster-info-title">
              <HighlightedText text={t.name} query={query} />
            </div>
            <div className="poster-info-meta-text" style={{ marginTop: "3px" }}>
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
              <img src={posterUrl(p.profile_path, "w185")} alt={p.name} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
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
