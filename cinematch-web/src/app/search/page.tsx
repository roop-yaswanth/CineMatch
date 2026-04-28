"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";

import dynamic from "next/dynamic";
import MobileMenu from "@/components/MobileMenu";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";
import { useSession } from "@/context/SessionContext";

const MovieDetailModal = dynamic(() => import("@/components/modals/MovieDetailModal"), { ssr: false });
import {
  apiSearchMulti,
  languageLabel,
  posterUrl,
  type MultiSearchMovie,
  type MultiSearchPerson,
  type MultiSearchResponse,
  type MultiSearchTv,
} from "@/lib/api";

type Tab = "all" | "movies" | "tv" | "people";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "all", label: "All" },
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
  const [tab, setTab] = useState<Tab>("all");
  // Tag results with the query they were fetched for; `loading` and the
  // displayed `results` are derived without effect-driven setState.
  const [resultRecord, setResultRecord] = useState<{ forQuery: string; data: MultiSearchResponse } | null>(null);
  const [active, setActive] = useState<DetailMovie | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Debounce
  useEffect(() => {
    const t = setTimeout(() => setDebounced(query.trim()), 300);
    return () => clearTimeout(t);
  }, [query]);

  // Reflect query in URL
  useEffect(() => {
    const url = debounced ? `/search?q=${encodeURIComponent(debounced)}` : "/search";
    window.history.replaceState(null, "", url);
  }, [debounced]);

  // Fetch
  useEffect(() => {
    if (!debounced) return;
    let cancelled = false;
    apiSearchMulti(debounced)
      .then((r) => { if (!cancelled) setResultRecord({ forQuery: debounced, data: r }); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [debounced]);

  const results: MultiSearchResponse = !debounced
    ? { movies: [], tv: [], people: [] }
    : resultRecord && resultRecord.forQuery === debounced
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
          <button
            onClick={() => router.back()}
            className="glass-button"
            aria-label="Back"
            style={{ width: "40px", height: "40px", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--color-text-primary)", padding: 0, cursor: "pointer" }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>

          <h1
            className="heading-display"
            style={{
              flex: 1,
              fontSize: "21px",
              fontWeight: 700,
              letterSpacing: "-0.035em",
              background: "linear-gradient(180deg, #ffffff 0%, #a0a0a0 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              margin: 0,
              textAlign: "center",
            }}
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
            style={{
              width: "100%",
              padding: "12px 40px 12px 42px",
              borderRadius: "14px",
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(28, 30, 36, 0.82)",
              color: "var(--color-text-primary)",
              fontSize: "15px",
              outline: "none",
            }}
          />
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--color-text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ position: "absolute", left: "34px", top: "11px" }}>
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          {query && (
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
            const count = t.id === "all" ? totalCount
              : t.id === "movies" ? results.movies.length
              : t.id === "tv" ? results.tv.length
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
      <div className="app-container" style={{ flex: 1, width: "100%", padding: "20px 20px 80px" }}>
        {!debounced ? (
          <div style={{ padding: "60px 0", textAlign: "center", color: "var(--color-text-muted)" }}>
            Search for movies, TV shows, or people.
          </div>
        ) : loading && totalCount === 0 ? (
          <div style={{ padding: "40px 0", textAlign: "center", color: "var(--color-text-muted)" }}>Searching…</div>
        ) : totalCount === 0 ? (
          <div style={{ padding: "40px 0", textAlign: "center", color: "var(--color-text-muted)" }}>No results for &ldquo;{debounced}&rdquo;.</div>
        ) : (
          <>
            {(tab === "all" || tab === "movies") && results.movies.length > 0 && (
              <Section title={tab === "all" ? "Movies" : null}>
                <MovieGrid movies={tab === "all" ? results.movies.slice(0, 12) : results.movies} onSelect={openMovie} />
                {tab === "all" && results.movies.length > 12 && (
                  <ShowMore label={`Show all ${results.movies.length} movies`} onClick={() => setTab("movies")} />
                )}
              </Section>
            )}
            {(tab === "all" || tab === "tv") && results.tv.length > 0 && (
              <Section title={tab === "all" ? "TV Shows" : null}>
                <TvGrid items={tab === "all" ? results.tv.slice(0, 8) : results.tv} />
                {tab === "all" && results.tv.length > 8 && (
                  <ShowMore label={`Show all ${results.tv.length} TV results`} onClick={() => setTab("tv")} />
                )}
              </Section>
            )}
            {(tab === "all" || tab === "people") && results.people.length > 0 && (
              <Section title={tab === "all" ? "People" : null}>
                <PeopleGrid people={tab === "all" ? results.people.slice(0, 8) : results.people} />
                {tab === "all" && results.people.length > 8 && (
                  <ShowMore label={`Show all ${results.people.length} people`} onClick={() => setTab("people")} />
                )}
              </Section>
            )}
          </>
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

/* ─── Movie grid ─── */
function MovieGrid({ movies, onSelect }: { movies: MultiSearchMovie[]; onSelect: (m: MultiSearchMovie) => void }) {
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
              {m.title}
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
function TvGrid({ items }: { items: MultiSearchTv[] }) {
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
              {t.name}
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
function PeopleGrid({ people }: { people: MultiSearchPerson[] }) {
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
              {p.name}
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
