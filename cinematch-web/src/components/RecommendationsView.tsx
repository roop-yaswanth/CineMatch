"use client";

import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";


import dynamic from "next/dynamic";
import BackButton from "@/components/ui/BackButton";
import HeroFeature from "@/components/dashboard/HeroFeature";
import CompactRail from "@/components/dashboard/CompactRail";
import { toast } from "@/components/ui/Toast";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";

const MovieDetailModal = dynamic(() => import("@/components/modals/MovieDetailModal"), { ssr: false });
import MobileMenu from "@/components/MobileMenu";
import {
  apiMultiRecommendations,
  apiRecommendationAction,
  apiSearchMovies,
  languageLabel,
  posterUrl,
  prefetchPosters,
  preferencesFromProfile,
  recommendationId,
  type MultiBucketResponse,
  type Recommendation,
  type RecommendationPreferences,
  type SearchResult,
  type UserSession,
} from "@/lib/api";
import { usePoster } from "@/lib/usePoster";
import { pushBackHandler } from "@/lib/backStack";

interface Props {
  session: UserSession;
  onSessionUpdate: (s: UserSession) => void;
  onBackToOnboarding: () => void;
  onLogout: () => void;
}

type RecommendationAction = "like" | "okay" | "dislike" | "remove" | "watchlist" | "skip";
type StackId = "hollywood" | "matched" | "other";

interface Stack {
  id: StackId;
  label: string;
  subtitle: string;
  movies: Recommendation[];
}

// ── Language keyword extractor ──────────────────────────────────────────────
// Allows queries like "rebel telugu", "parasite korean", "intouchables french"
// to be interpreted as: search "rebel" and boost/filter by language "te".
const LANG_KEYWORDS: Record<string, string> = {
  telugu: "te", hindi: "hi", tamil: "ta", malayalam: "ml", kannada: "kn",
  marathi: "mr", bengali: "bn", gujarati: "gu", punjabi: "pa", urdu: "ur",
  korean: "ko", japanese: "ja", chinese: "zh", mandarin: "zh", cantonese: "cn",
  french: "fr", spanish: "es", german: "de", italian: "it", portuguese: "pt",
  russian: "ru", arabic: "ar", turkish: "tr", thai: "th", indonesian: "id",
  persian: "fa", farsi: "fa", swedish: "sv", danish: "da", dutch: "nl",
  polish: "pl", ukrainian: "uk", greek: "el", hebrew: "he", english: "en",
};

function extractLangKeyword(query: string): { cleanQuery: string; langCode: string | null; year: number | null } {
  let working = query;
  let langCode: string | null = null;
  let year: number | null = null;

  // Detect language keyword
  const tokens = working.toLowerCase().split(/\s+/);
  for (const token of tokens) {
    const code = LANG_KEYWORDS[token];
    if (code) {
      langCode = code;
      working = working.replace(new RegExp(`\\b${token}\\b`, "gi"), "").replace(/\s{2,}/g, " ").trim();
      break;
    }
  }

  // Detect 4-digit year (1888–2099)
  const yearMatch = working.match(/\b(18[89]\d|19\d{2}|20\d{2})\b/);
  if (yearMatch) {
    year = parseInt(yearMatch[1], 10);
    working = working.replace(yearMatch[0], "").replace(/\s{2,}/g, " ").trim();
  }

  return { cleanQuery: working || query, langCode, year };
}
// ────────────────────────────────────────────────────────────────────────────

function toDetailMovie(movie: Recommendation): DetailMovie {
  return { ...movie };
}

function partitionFromBuckets(
  resp: MultiBucketResponse,
  preferences: RecommendationPreferences
): { stacks: Stack[]; allMovies: Recommendation[] } {
  const { english, regional, global: globalMovies } = resp.buckets;

  const regionalEntries = Object.entries(regional).filter(
    ([key]) => key !== "_merged"
  );
  const hasRegional = regionalEntries.some(([, arr]) => arr.length > 0);

  const regionalMerged: Recommendation[] = [];
  if (hasRegional) {
    const buckets = regionalEntries.map(([, arr]) => [...arr]);
    const cursors = buckets.map(() => 0);
    let added = true;
    while (added) {
      added = false;
      for (let i = 0; i < buckets.length; i++) {
        if (cursors[i] < buckets[i].length) {
          regionalMerged.push(buckets[i][cursors[i]]);
          cursors[i]++;
          added = true;
        }
      }
    }
  }

  const selectedNonEnglish = (preferences.languages || [])
    .filter((l) => l && l.toLowerCase() !== "en");

  const matchedLabel = selectedNonEnglish.length > 0
    ? selectedNonEnglish.map((lang) => languageLabel(lang)).join(", ")
    : regionalEntries.map(([lang]) => languageLabel(lang)).join(", ");

  const result: Stack[] = [];

  if (hasRegional || (regional._merged && regional._merged.length > 0)) {
    const movies = hasRegional ? regionalMerged : (regional._merged || []);
    result.push({
      id: "matched",
      label: matchedLabel ? `${matchedLabel} Cinema` : "Regional Favorites",
      subtitle: "Best from your preferred languages.",
      movies,
    });
  }

  if ((english || []).length > 0) {
    result.push({
      id: "hollywood",
      label: "Hollywood",
      subtitle: "Handpicked from your taste profile.",
      movies: english || [],
    });
  }

  if ((globalMovies || []).length > 0) {
    result.push({
      id: "other",
      label: "Global Cinema",
      subtitle: "Hidden gems across cultures — curated by plot similarity.",
      movies: globalMovies || [],
    });
  }

  const allMovies = [
    ...(hasRegional ? regionalMerged : (regional._merged || [])),
    ...(english || []),
    ...(globalMovies || []),
  ];

  return { stacks: result, allMovies };
}

const BUCKET_DISPLAY = 50;
const BUCKET_FETCH = 100;
const CACHE_REFETCH_THRESHOLD = 20;

type StackCache = Record<StackId, Recommendation[]>;
const EMPTY_CACHE = (): StackCache => ({ hollywood: [], matched: [], other: [] });

const RECS_CACHE_KEY = "cinematch_recs_cache";
const RECS_CACHE_TTL_MS = 10 * 60 * 1000;

interface RecsCache {
  stacks: Stack[];
  movies: Recommendation[];
  bucketCache: StackCache;
  seenIds: number[];
  displayedIds: number[];
  ts: number;
}

function readRecsCache(sessionId: string): RecsCache | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(`${RECS_CACHE_KEY}_${sessionId}`);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as RecsCache;
    if (!parsed || !Array.isArray(parsed.stacks) || Date.now() - parsed.ts > RECS_CACHE_TTL_MS) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function writeRecsCache(sessionId: string, data: Omit<RecsCache, "ts">) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(
      `${RECS_CACHE_KEY}_${sessionId}`,
      JSON.stringify({ ...data, ts: Date.now() })
    );
  } catch { /* localStorage full — non-critical */ }
}

export default function RecommendationsView({
  session,
  onSessionUpdate,
  onBackToOnboarding,
  onLogout,
}: Props) {
  const router = useRouter();

  // ── Try to restore from navigation cache on mount ─────────────────────────
  const cachedRef = useRef(readRecsCache(session.session_id));
  const hadCache = cachedRef.current !== null;

  const [stacks, setStacks] = useState<Stack[]>(() => cachedRef.current?.stacks ?? []);
  const [movies, setMovies] = useState<Recommendation[]>(() => cachedRef.current?.movies ?? []);
  const [loading, setLoading] = useState(false);
  const [initialLoad, setInitialLoad] = useState(!hadCache);

  const [showUpdateToast] = useState(false);
  const [activeStack, setActiveStack] = useState<StackId | null>(null);
  const [preferences, setPreferences] = useState<RecommendationPreferences>(
    () => preferencesFromProfile(session.profile)
  );

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [showSearchOverlay, setShowSearchOverlay] = useState(false);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  // Detail Modal state
  const [activeMovie, setActiveMovie] = useState<DetailMovie | null>(null);

  // Route-based navigation for sub-pages
  const openYourLikes = () => router.push("/your-likes");
  const openWatchlist = () => router.push("/your-likes?filter=watchlist");
  const openPrefs = () => router.push("/preferences");


  const [isUpdating, setIsUpdating] = useState(false);


  // Action counter for auto-rerun
  const actionCountRef = useRef({ positive: 0, negative: 0, total: 0 });

  // Every movie ID the user has acted on — prevents re-showing after refresh
  const seenIdsRef = useRef<Set<number>>(
    new Set(cachedRef.current?.seenIds ?? [])
  );

  // All movie IDs ever displayed — sent to backend on auto-rerun so it generates truly new movies
  const displayedIdsRef = useRef<Set<number>>(
    new Set(cachedRef.current?.displayedIds ?? [])
  );


  const bucketCacheRef = useRef<StackCache>(
    cachedRef.current?.bucketCache ?? EMPTY_CACHE()
  );

  // Search handler with debounce
  const handleSearch = useCallback(async (query: string) => {
    setSearchQuery(query);

    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (!query.trim() || query.trim().length < 2) {
      setSearchResults([]);
      setShowSearch(false);
      return;
    }

    searchTimeoutRef.current = setTimeout(async () => {
      setSearchLoading(true);
      setShowSearch(true);
      try {
        // Detect language keywords ("telugu", "hindi", "korean") and 4-digit years
        // from the query. E.g. "rebel telugu 2012" → search "rebel", lang="te", year=2012.
        const { cleanQuery, langCode, year } = extractLangKeyword(query.trim());
        const resp = await apiSearchMovies(cleanQuery, 15);

        // Re-sort: apply language boost then year proximity boost
        let results = resp.results;
        if (langCode) {
          const match = results.filter((r) => r.original_language === langCode);
          const rest  = results.filter((r) => r.original_language !== langCode);
          results = [...match, ...rest];
        }
        if (year) {
          // Float exact/near-year matches (±1) to the very top within each language group
          const exact = results.filter((r) => r.year && Math.abs(r.year - year) <= 1);
          const other = results.filter((r) => !r.year || Math.abs(r.year - year) > 1);
          results = [...exact, ...other];
        }
        setSearchResults(results);
      } catch (err) {
        console.error("Search error:", err);
        setSearchResults([]);
      } finally {
        setSearchLoading(false);
      }
    }, 250); // 250ms debounce — fast search
  }, []);

  const applyFilters = useCallback((arr: Recommendation[], prefs: RecommendationPreferences): Recommendation[] => {
    if (!prefs.include_classics) {
      const modern = arr.filter((m) => {
        const y = typeof m.year === "number" ? m.year : parseInt(String(m.year ?? ""), 10);
        return isNaN(y) || y >= 2000;
      });
      if (modern.length > 0) return modern;
    }
    return arr;
  }, []);

  const applyBucketResponse = useCallback(
    (resp: MultiBucketResponse, prefs: RecommendationPreferences) => {
      const filterBucket = (arr: Recommendation[]) =>
        applyFilters(
          arr.filter((m) => !seenIdsRef.current.has(recommendationId(m))),
          prefs
        );

      const fEn = filterBucket(resp.buckets.english || []);
      const fGlob = filterBucket(resp.buckets.global || []);

      // Interleave regional languages, then filter
      const regionalEntries = Object.entries(resp.buckets.regional || {});
      const regionalMergedRaw: Recommendation[] = [];
      if (regionalEntries.length > 0) {
        const buckets = regionalEntries.map(([, arr]) => [...(arr || [])]);
        const cursors = buckets.map(() => 0);
        let added = true;
        while (added) {
          added = false;
          for (let i = 0; i < buckets.length; i++) {
            if (cursors[i] < buckets[i].length) {
              regionalMergedRaw.push(buckets[i][cursors[i]++]);
              added = true;
            }
          }
        }
      }
      const fReg = filterBucket(regionalMergedRaw);

      // Split each into display slice and cache reserve
      const displayEn = fEn.slice(0, BUCKET_DISPLAY);
      const displayReg = fReg.slice(0, BUCKET_DISPLAY);
      const displayGlob = fGlob.slice(0, BUCKET_DISPLAY);

      bucketCacheRef.current = {
        hollywood: fEn.slice(BUCKET_DISPLAY),
        matched: fReg.slice(BUCKET_DISPLAY),
        other: fGlob.slice(BUCKET_DISPLAY),
      };

      const displayResp: MultiBucketResponse = {
        ...resp,
        buckets: {
          english: displayEn,
          regional: regionalEntries.length > 0 ? { _merged: displayReg } : {},
          global: displayGlob,
        },
      };

      const { stacks: newStacks, allMovies } = partitionFromBuckets(displayResp, prefs);
      for (const m of allMovies) displayedIdsRef.current.add(recommendationId(m));
      setStacks(newStacks);
      setMovies(allMovies);
    },
    [applyFilters]
  );

  const silentRefreshInFlight = useRef(false);
  const silentRefresh = useCallback(async (prefs: RecommendationPreferences) => {
    if (silentRefreshInFlight.current) return;
    silentRefreshInFlight.current = true;
    try {
      const resp = await apiMultiRecommendations(session.session_id, {
        languages: prefs.languages,
        genres: prefs.genres,
        age_group: prefs.age_group,
        region: prefs.region,
        include_classics: prefs.include_classics,
        semantic_index: prefs.semantic_index,
        per_bucket_k: BUCKET_FETCH,
      });

      const totalMovies = [
        ...(resp.buckets.english || []),
        ...Object.values(resp.buckets.regional || {}).flat(),
        ...(resp.buckets.global || []),
      ];
      const hasUnseen = totalMovies.some(
        (m) => !seenIdsRef.current.has(recommendationId(m))
      );
      if (!hasUnseen && totalMovies.length > 0) {
        seenIdsRef.current = new Set();
        actionCountRef.current = { positive: 0, negative: 0, total: 0 };
      }

      await prefetchPosters(totalMovies);
      applyBucketResponse(resp, prefs);
      if (resp.session) onSessionUpdate(resp.session);
    } catch (err) {
      console.error("[silentRefresh] Failed:", err);
    } finally {
      silentRefreshInFlight.current = false;
    }
  }, [applyBucketResponse, onSessionUpdate, session.session_id]);

  const generate = useCallback(
    async (
      nextPreferences: RecommendationPreferences = preferences,
      { autoRerun = false }: { autoRerun?: boolean } = {}
    ) => {
      if (autoRerun) {
        setIsUpdating(true);
      } else {
        setLoading(true);
        setActiveStack(null);
        setStacks([]);
        setMovies([]);
        bucketCacheRef.current = EMPTY_CACHE();
        seenIdsRef.current = new Set();
        displayedIdsRef.current = new Set();
        actionCountRef.current = { positive: 0, negative: 0, total: 0 };
      }
      try {
        const excludeIds = autoRerun
          ? Array.from(displayedIdsRef.current)
          : undefined;
        const resp = await apiMultiRecommendations(session.session_id, {
          languages: nextPreferences.languages,
          genres: nextPreferences.genres,
          age_group: nextPreferences.age_group,
          region: nextPreferences.region,
          include_classics: nextPreferences.include_classics,
          semantic_index: nextPreferences.semantic_index,
          per_bucket_k: BUCKET_FETCH,
          exclude_ids: excludeIds,
        });

        const allMovies = [
          ...(resp.buckets.english || []),
          ...Object.values(resp.buckets.regional || {}).flat(),
          ...(resp.buckets.global || []),
        ];
        await prefetchPosters(allMovies);
        // On auto-rerun: clear stacks now that new data is ready, then apply
        if (autoRerun) {
          bucketCacheRef.current = EMPTY_CACHE();
          seenIdsRef.current = new Set();
          displayedIdsRef.current = new Set();
          actionCountRef.current = { positive: 0, negative: 0, total: 0 };
        }
        applyBucketResponse(resp, nextPreferences);
        if (resp.session) onSessionUpdate(resp.session);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
        setIsUpdating(false);
      }
    },
    [applyBucketResponse, onSessionUpdate, preferences, session.session_id]
  );



  useEffect(() => {
    if (!initialLoad) return;
    void generate(preferences);
    setInitialLoad(false);
  }, [generate, initialLoad, preferences]);

  // Keep navigation cache in sync with optimistic stack updates (actions, etc.)
  useEffect(() => {
    if (stacks.length === 0) return; // Don't cache empty state
    writeRecsCache(session.session_id, {
      stacks,
      movies,
      bucketCache: bucketCacheRef.current,
      seenIds: Array.from(seenIdsRef.current),
      displayedIds: Array.from(displayedIdsRef.current),
    });
  }, [stacks, movies, session.session_id]);

  const handleAction = useCallback(
    async (movie: Recommendation | DetailMovie, action: RecommendationAction) => {
      const tmdbId = "tmdb_id" in movie && movie.tmdb_id ? movie.tmdb_id : movie.id;

      seenIdsRef.current.add(tmdbId);

      setMovies((prev) => prev.filter((m) => recommendationId(m) !== tmdbId));
      let targetStackId: StackId | null = null;

      setStacks((prev) =>
        prev.map((s) => {
          const inThis = s.movies.some((m) => recommendationId(m) === tmdbId);
          if (!inThis) return s;
          targetStackId = s.id as StackId;
          const remaining = s.movies.filter((m) => recommendationId(m) !== tmdbId);
          const cache = bucketCacheRef.current[s.id as StackId];
          const needed = BUCKET_DISPLAY - remaining.length;
          const toAdd = needed > 0 && cache && cache.length > 0 ? cache.splice(0, needed) : [];
          return { ...s, movies: [...remaining, ...toAdd] };
        })
      );

      actionCountRef.current.total++;
      if (action === "like" || action === "okay") actionCountRef.current.positive++;
      if (action === "dislike") actionCountRef.current.negative++;
      if (action === "remove" || action === "skip") actionCountRef.current.negative += 0.5;

      const { positive, negative, total } = actionCountRef.current;
      const shouldAutoRerun = negative >= 10 || total >= 10 || positive >= 10;

      if (shouldAutoRerun) {
        actionCountRef.current = { positive: 0, negative: 0, total: 0 };
        setIsUpdating(true);
        try {
          await apiRecommendationAction(session.session_id, tmdbId, action);
          await generate(preferences, { autoRerun: true });
        } catch (err) {
          console.error("Taste profile update failed:", err);
          try { await generate(preferences, { autoRerun: true }); } catch { setIsUpdating(false); }
        }
        return;
      }

      apiRecommendationAction(session.session_id, tmdbId, action)
        .then((result) => {
          onSessionUpdate(result.session);
          if (targetStackId) {
            const cacheRemaining = bucketCacheRef.current[targetStackId]?.length || 0;
            setStacks(currentStacks => {
              const s = currentStacks.find(st => st.id === targetStackId);
              if (s && s.movies.length < CACHE_REFETCH_THRESHOLD && cacheRemaining === 0) {
                void silentRefresh(preferences);
              }
              return currentStacks;
            });
          }
        })
        .catch((err) => console.error("Recommendation action failed:", err));
    },
    [activeMovie, generate, onSessionUpdate, preferences, session.session_id, silentRefresh]
  );

  const handlePreferenceUpdate = useCallback(
    (nextPreferences: RecommendationPreferences) => {
      setPreferences(nextPreferences);
      startTransition(() => {
        void generate(nextPreferences);
      });
    },
    [generate]
  );

  // Pick up preference updates from /preferences page via sessionStorage
  useEffect(() => {
    const handleVisibility = () => {
      if (document.visibilityState !== "visible") return;
      try {
        const raw = sessionStorage.getItem("cinematch_prefs_update");
        if (raw) {
          sessionStorage.removeItem("cinematch_prefs_update");
          const nextPrefs = JSON.parse(raw) as RecommendationPreferences;
          handlePreferenceUpdate(nextPrefs);
        }
      } catch { /* ignore */ }
    };

    // Also check immediately on mount (covers back-navigation)
    try {
      const raw = sessionStorage.getItem("cinematch_prefs_update");
      if (raw) {
        sessionStorage.removeItem("cinematch_prefs_update");
        const nextPrefs = JSON.parse(raw) as RecommendationPreferences;
        handlePreferenceUpdate(nextPrefs);
      }
    } catch { /* ignore */ }

    document.addEventListener("visibilitychange", handleVisibility);
    return () => document.removeEventListener("visibilitychange", handleVisibility);
  }, [handlePreferenceUpdate]);

  return (
    <div
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-sans)",
        width: "100%",
        // Note: don't use 100vw here — vw units don't scale with html { zoom },
        // so on desktop the page would only fill 85% of physical viewport.
        // width: 100% inherits correctly from the zoomed html element.
        maxWidth: "100%",
        overflowX: "hidden",
        position: "relative",
      }}
    >
      <div style={{ position: "relative", zIndex: 1, display: "flex", flexDirection: "column", minHeight: "100dvh" }}>
        {/* Main Page Content Wrapper — Fades out when Detail View opens */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            flex: 1,
            opacity: activeStack ? 0 : 1,
            transition: "opacity 0.25s ease",
            pointerEvents: activeStack ? "none" : "auto",
          }}
        >

          {/* ─── Full-Page Search Overlay ─── */}
          {showSearchOverlay && (
            <div
              style={{
                position: "fixed",
                inset: 0,
                zIndex: 9000,
                background: "rgba(8, 8, 12, 0.96)",
                backdropFilter: "blur(32px) saturate(1.2)",
                WebkitBackdropFilter: "blur(32px) saturate(1.2)",
                display: "flex",
                flexDirection: "column",
                padding: "env(safe-area-inset-top, 0px) 0 0 0",
              }}
            >
              {/* Overlay header with search input */}
              <div style={{
                padding: "16px 20px 12px",
                display: "flex",
                alignItems: "center",
                gap: "12px",
                borderBottom: "1px solid rgba(255,255,255,0.08)",
              }}>
                {/* Back / close */}
                <button
                  onClick={() => {
                    setShowSearchOverlay(false);
                    setSearchQuery("");
                    setSearchResults([]);
                    setShowSearch(false);
                  }}
                  style={{
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    color: "var(--color-text-muted)",
                    display: "flex",
                    alignItems: "center",
                    padding: "6px",
                    flexShrink: 0,
                  }}
                  aria-label="Close search"
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="15 18 9 12 15 6" />
                  </svg>
                </button>

                {/* Search input */}
                <div style={{ flex: 1, position: "relative" }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }}>
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                  <input
                    ref={searchInputRef}
                    type="text"
                    placeholder="Search movies, shows, people…"
                    value={searchQuery}
                    autoFocus
                    onChange={(e) => handleSearch(e.target.value)}
                    style={{
                      width: "100%",
                      padding: "13px 44px 13px 44px",
                      borderRadius: "14px",
                      border: "1px solid rgba(255,255,255,0.15)",
                      background: "rgba(255,255,255,0.07)",
                      color: "var(--color-text-primary)",
                      fontSize: "16px",
                      fontWeight: 400,
                      letterSpacing: "-0.005em",
                      outline: "none",
                    }}
                  />
                  {searchQuery && (
                    <button
                      onClick={() => { setSearchQuery(""); setSearchResults([]); setShowSearch(false); }}
                      style={{
                        position: "absolute", right: "12px", top: "50%", transform: "translateY(-50%)",
                        background: "rgba(255,255,255,0.15)", border: "none", cursor: "pointer",
                        padding: "3px", borderRadius: "50%", color: "rgba(255,255,255,0.7)",
                        width: "22px", height: "22px", display: "flex", alignItems: "center", justifyContent: "center",
                      }}
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>

              {/* Results area */}
              <div style={{ flex: 1, overflowY: "auto", padding: "8px 12px" }}>
                {/* Query hint */}
                {searchQuery && !searchLoading && (
                  <div style={{ padding: "10px 8px 4px", fontSize: "12px", color: "var(--color-text-muted)", fontWeight: 500 }}>
                    {searchQuery}
                  </div>
                )}

                {searchLoading && (
                  <div style={{ padding: "40px", textAlign: "center", color: "var(--color-text-muted)" }}>Searching…</div>
                )}

                {!searchLoading && searchQuery && searchResults.length === 0 && (
                  <div style={{ padding: "40px", textAlign: "center", color: "var(--color-text-muted)" }}>No results for &ldquo;{searchQuery}&rdquo;</div>
                )}

                {!searchLoading && searchResults.map((movie) => (
                  <div
                    key={movie.tmdb_id}
                    style={{
                      display: "flex",
                      gap: "14px",
                      padding: "12px 12px",
                      borderRadius: "14px",
                      background: "transparent",
                      cursor: "pointer",
                      borderBottom: "1px solid rgba(255,255,255,0.05)",
                      transition: "background 0.15s ease",
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.06)"; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                    onClick={() => {
                      setActiveMovie({ ...movie, id: movie.tmdb_id });
                      setShowSearchOverlay(false);
                      setSearchQuery("");
                      setSearchResults([]);
                    }}
                  >
                    <div style={{ width: "52px", height: "78px", borderRadius: "8px", overflow: "hidden", flexShrink: 0, background: "rgba(255,255,255,0.06)" }}>
                      {movie.poster_path && (
                        <Image src={posterUrl(movie.poster_path, "w92")} alt={movie.title} width={52} height={78} style={{ objectFit: "cover", width: "100%", height: "100%" }} />
                      )}
                    </div>
                    <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", justifyContent: "center" }}>
                      <div style={{ fontWeight: 600, color: "var(--color-text-primary)", fontSize: "15px", marginBottom: "4px" }}>{movie.title}</div>
                      <div style={{ fontSize: "12px", color: "var(--color-text-muted)" }}>
                        {[movie.original_language ? languageLabel(movie.original_language) : null, movie.year].filter(Boolean).join(" · ")}
                      </div>
                      {movie.imdb_rating && (
                        <div style={{ fontSize: "11px", color: "#fbbf24", marginTop: "4px" }}>IMDb {movie.imdb_rating.toFixed(1)}</div>
                      )}
                    </div>
                    {/* Play icon */}
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: "rgba(255,255,255,0.35)" }}>
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="10" />
                        <polygon points="10 8 16 12 10 16 10 8" fill="currentColor" stroke="none" />
                      </svg>
                    </div>
                  </div>
                ))}

                {/* Full search link */}
                {searchQuery.trim().length > 0 && !searchLoading && (
                  <button
                    onClick={() => {
                      router.push(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
                      setShowSearchOverlay(false);
                      setSearchQuery("");
                    }}
                    style={{
                      marginTop: "12px",
                      width: "100%",
                      textAlign: "center",
                      padding: "12px",
                      borderRadius: "12px",
                      background: "rgba(255,255,255,0.06)",
                      border: "1px solid rgba(255,255,255,0.10)",
                      color: "var(--color-text-primary)",
                      fontSize: "13px",
                      fontWeight: 500,
                      cursor: "pointer",
                    }}
                  >
                    Search movies, TV &amp; people for &ldquo;{searchQuery.trim()}&rdquo; →
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Header */}
          <header
            className="glass"
            style={{
              position: "sticky",
              top: 0,
              zIndex: 40,
            }}
          >
            <div
              style={{
                width: "100%",
                padding: "12px 16px 12px",
                display: "flex",
                alignItems: "center",
                position: "relative",   // needed for absolute title centering
              }}
            >
              {/* Left: empty flex spacer */}
              <div style={{ flex: 1 }} />

              {/* Center: title — absolutely positioned so it's always truly centered */}
              <h1
                className="heading-display"
                style={{
                  position: "absolute",
                  left: "50%",
                  transform: "translateX(-50%)",
                  fontSize: "21px",
                  fontWeight: 700,
                  letterSpacing: "-0.035em",
                  background: "linear-gradient(180deg, #ffffff 0%, #a0a0a0 100%)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  backgroundClip: "text",
                  margin: 0,
                  whiteSpace: "nowrap",
                  pointerEvents: "none",
                }}
              >
                CineMatch
              </h1>

              {/* Right: search + hamburger */}
              <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "4px" }}>
                {/* Desktop mini search box — hidden on mobile via CSS */}
                <button
                  className="header-search-box"
                  onClick={() => setShowSearchOverlay(true)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "8px 14px 8px 12px",
                    borderRadius: "10px",
                    border: "1px solid rgba(255,255,255,0.12)",
                    background: "rgba(255,255,255,0.06)",
                    color: "rgba(255,255,255,0.45)",
                    fontSize: "13px",
                    cursor: "pointer",
                    minWidth: "220px",
                    textAlign: "left",
                  }}
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                  Search…
                </button>

                {/* Mobile search icon only */}
                <button
                  className="header-search-icon"
                  onClick={() => setShowSearchOverlay(true)}
                  aria-label="Search"
                  style={{
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                    color: "var(--color-text-primary)",
                    padding: "6px",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8" />
                    <line x1="21" y1="21" x2="16.65" y2="16.65" />
                  </svg>
                </button>

                {/* Hamburger menu — right of search */}
                <MobileMenu
                  onLogout={onLogout}
                  onReset={onBackToOnboarding}
                  onPreferences={openPrefs}
                  onYourLikes={openYourLikes}
                  onWatchlist={openWatchlist}
                />
              </div>
            </div>
          </header>

          {/* Content */}
          <div className="app-container" style={{ flex: 1, width: "100%", padding: "24px 0 48px" }}>
            {/* Title bar */}
            <div
              style={{
                padding: "0 20px",
                marginBottom: "28px",
                display: "flex",
                alignItems: "flex-start",
                justifyContent: "space-between",
                gap: "14px",
                flexWrap: "wrap",
              }}
            >
              <div>
                <h2
                  style={{
                    fontSize: "clamp(1.8rem, 4vw, 2.8rem)",
                    fontWeight: 700,
                    letterSpacing: "-0.04em",
                    margin: 0,
                    color: "var(--color-text-primary)",
                  }}
                >
                  {/* For You */}
                </h2>
              </div>

            </div>

            {/* Loading skeleton */}
            {loading && movies.length === 0 && (
              <div style={{ display: "grid", gap: "48px", padding: "0 20px" }}>
                {[0, 1, 2].map((i) => (
                  <div key={i}>
                    <div
                      className="skeleton-shimmer"
                      style={{
                        height: "18px",
                        width: i === 0 ? "240px" : "200px",
                        borderRadius: "999px",
                        marginBottom: "14px",
                      }}
                    />
                    <div style={{ display: "flex", gap: "16px", overflow: "hidden", paddingBottom: "8px" }}>
                      {Array.from({ length: 6 }).map((_, j) => (
                        <div key={j} style={{ width: "min(40vw, 150px)", flexShrink: 0 }}>
                          <div
                            className="skeleton-shimmer skeleton-grain"
                            style={{
                              aspectRatio: "2 / 3",
                              borderRadius: "16px",
                              marginBottom: "12px"
                            }}
                          />
                          <div
                            className="skeleton-shimmer"
                            style={{
                              height: "10px",
                              width: "85%",
                              borderRadius: "4px",
                              marginBottom: "6px"
                            }}
                          />
                          <div
                            className="skeleton-shimmer"
                            style={{
                              height: "8px",
                              width: "50%",
                              borderRadius: "4px",
                              opacity: 0.6
                            }}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Featured hero — drawn from the top of the user's most-targeted
                bucket (Matched if present, else Hollywood, else Global). */}
            {!loading && stacks.length > 0 && (() => {
              const heroStack =
                stacks.find((s) => s.id === "matched") ??
                stacks.find((s) => s.id === "hollywood") ??
                stacks[0];
              if (!heroStack || heroStack.movies.length === 0) return null;
              return (
                <div style={{ marginBottom: 8 }}>
                  <HeroFeature
                    movies={heroStack.movies}
                    onOpenDetail={(m) => setActiveMovie(toDetailMovie(m))}
                    onWatchlist={(m) => {
                      handleAction(m, "watchlist");
                      toast({
                        message: `Added "${m.title}" to your watchlist`,
                        tone: "success",
                        action: {
                          label: "Undo",
                          onClick: () => handleAction(m, "remove"),
                        },
                      });
                    }}
                  />
                </div>
              );
            })()}

            {/* Stacks
                - "matched" stays as the big-card swipeable carousel (StackRow).
                - "hollywood" / "other" become compact rails so users can
                  passively browse without committing to ratings on every card. */}
            {!loading && (
              <div style={{ display: "grid", gap: "40px" }}>
                {stacks.map((stack) =>
                  stack.id === "matched" ? (
                    <StackRow
                      key={stack.id}
                      stack={stack}
                      disabled={loading}
                      onAction={handleAction}
                      onOpenDetail={() => setActiveStack(stack.id)}
                      onMovieClick={(m) => setActiveMovie(toDetailMovie(m))}
                    />
                  ) : (
                    <CompactRail
                      key={stack.id}
                      label={stack.label}
                      subtitle={stack.subtitle}
                      movies={stack.movies}
                      onMovieClick={(m) => setActiveMovie(toDetailMovie(m))}
                      onOpenDetail={() => setActiveStack(stack.id)}
                    />
                  )
                )}
              </div>
            )}


          </div>
        </div>

        {/* Stack detail overlay */}
        <AnimatePresence>
          {activeStack && stacks.find((s) => s.id === activeStack) && (
            <StackDetailView
              key={"detail-view-" + activeStack}
              stack={stacks.find((s) => s.id === activeStack)!}
              onBack={() => setActiveStack(null)}
              onAction={handleAction}
              onMovieClick={(m) => setActiveMovie(toDetailMovie(m))}
              disabled={loading}
            />
          )}
        </AnimatePresence>


        {typeof document !== 'undefined' && createPortal(
          <AnimatePresence>
            {(showUpdateToast || isUpdating) && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                style={{
                  position: "fixed",
                  inset: 0,
                  zIndex: 10000000,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "rgba(0, 0, 0, 0.6)",
                  backdropFilter: "blur(16px)",
                  WebkitBackdropFilter: "blur(16px)",
                }}
              >
                <motion.div
                  animate={{ y: [0, -18, 0] }}
                  transition={{ repeat: Infinity, duration: 0.7, ease: "easeInOut" }}
                  style={{ fontSize: "72px" }}
                >
                  🍿
                </motion.div>
                <motion.p
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3, type: "spring", stiffness: 200, damping: 20 }}
                  style={{
                    marginTop: "20px",
                    color: "white",
                    fontSize: "18px",
                    fontWeight: 600,
                    letterSpacing: "-0.02em",
                    textShadow: "0 2px 12px rgba(0,0,0,0.5)",
                  }}
                >
                  Updating Taste Profile...
                </motion.p>
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.6 }}
                  transition={{ delay: 0.6 }}
                  style={{
                    marginTop: "8px",
                    color: "white",
                    fontSize: "13px",
                    fontWeight: 400,
                  }}
                >
                  Rebuilding recommendations from your feedback
                </motion.p>
              </motion.div>
            )}
          </AnimatePresence>,
          document.body
        )}



        {/* Movie Details Modal */}
        <MovieDetailModal
          isOpen={!!activeMovie}
          onClose={() => setActiveMovie(null)}
          movie={activeMovie}
          sessionId={session.session_id}
          userRegion={session.profile?.region ?? null}
          onAction={(action) => {
            if (activeMovie) {
              handleAction(activeMovie, action);
            }
          }}
          onMovieSelect={(m) => setActiveMovie(m)}
        />
        {/* <BottomNav 
        onYourLikes={openYourLikes}
        onPreferences={openPrefs}
        onRefresh={() => void generate(preferences)}
      /> */}
      </div>
    </div>
  );
}

/* ─── Stack Detail (Full-Screen Grid) ──────────── */

function StackDetailView({
  stack,
  onBack,
  onAction,
  onMovieClick,
  disabled,
}: {
  stack: Stack;
  onBack: () => void;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  onMovieClick: (movie: Recommendation) => void;
  disabled: boolean;
}) {
  useEffect(() => {
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, []);

  // Centralized back-handler: back swipe/button closes this overlay without
  // conflicting with MovieDetailModal's own handler (they share one listener).
  useEffect(() => {
    const cleanup = pushBackHandler(onBack);
    return cleanup;
  }, [onBack]);

  const content = (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.25 }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 50,
        background: "transparent",
        overflowY: "auto",
        overflowX: "hidden",
        overscrollBehavior: "none",   // prevent iOS rubber-band bounce to top pill
      }}
    >
      {/* Detail header */}
      <div
        className="glass"
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          padding: "calc(18px + env(safe-area-inset-top, 0px)) clamp(20px, 4vw, 40px) 14px",
          display: "flex",
          alignItems: "center",
          gap: "14px",
        }}
      >
        <BackButton onClick={onBack} />
        <div style={{ minWidth: 0 }}>
          <h2
            className="heading-section"
            style={{
              fontSize: "20px",
              fontWeight: 700,
              letterSpacing: "-0.03em",
              margin: 0,
              color: "var(--color-text-primary)",
            }}
          >
            {stack.label}
          </h2>
          <p style={{ fontSize: "12px", color: "var(--color-text-muted)", marginTop: "3px", fontWeight: 500, letterSpacing: "-0.005em" }}>
            {stack.movies.length > 50
              ? `Top 50 of ${stack.movies.length} movies`
              : `${stack.movies.length} movie${stack.movies.length !== 1 ? "s" : ""}`}
          </p>
        </div>
      </div>

      {/* Grid */}
      <div
        className="stack-detail-grid app-container"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
          gap: "24px",
          padding: "24px clamp(20px, 4vw, 40px) 80px",
        }}
      >
        <AnimatePresence initial={false}>
          {stack.movies.slice(0, 50).map((movie, index) => (
            <PosterCard
              key={recommendationId(movie)}
              movie={movie}
              disabled={disabled}
              onAction={onAction}
              onClick={() => onMovieClick(movie)}
              priority={index < 4}
              showFullInfo
            />
          ))}
        </AnimatePresence>
        {stack.movies.length === 0 && (
          <p style={{ fontSize: "13px", color: "var(--color-text-muted)", gridColumn: "1 / -1" }}>
            No movies in this category yet.
          </p>
        )}
      </div>
    </motion.div>
  );

  if (typeof document === "undefined") return null;
  return createPortal(content, document.body);
}

/* ─── Stack Row — Paged Carousel ───────────────── */

function StackRow({
  stack,
  disabled,
  onAction,
  onOpenDetail,
  onMovieClick,
}: {
  stack: Stack;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  onOpenDetail: () => void;
  onMovieClick: (movie: Recommendation) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [scrollInfo, setScrollInfo] = useState({ canLeft: false, canRight: false });

  const updateScroll = useCallback(() => {
    const t = trackRef.current;
    if (!t) return;
    const { scrollLeft, scrollWidth, clientWidth } = t;
    setScrollInfo({
      canLeft: scrollLeft > 2,
      canRight: scrollLeft < scrollWidth - clientWidth - 2,
    });
  }, []);

  useEffect(() => {
    const t = trackRef.current;
    if (!t) return;
    t.addEventListener("scroll", updateScroll, { passive: true });
    const ro = new ResizeObserver(updateScroll);
    ro.observe(t);
    updateScroll();
    return () => {
      t.removeEventListener("scroll", updateScroll);
      ro.disconnect();
    };
  }, [updateScroll, stack.movies.length]);

  const scrollBy = (dir: "left" | "right") => {
    const t = trackRef.current;
    if (!t) return;
    const amount = t.clientWidth * 0.82;
    t.scrollBy({ left: dir === "right" ? amount : -amount, behavior: "smooth" });
  };

  const { canLeft, canRight } = scrollInfo;

  return (
    <section className="stack-section" style={{ width: "100%", overflow: "hidden", position: "relative" }}>
      {/* Stack header */}
      <div
        style={{
          padding: "0 20px",
          marginBottom: "14px",
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: "12px",
          position: "relative",
          zIndex: 3,
          pointerEvents: "auto",
        }}
      >
        <div
          className="stack-name-btn"
          onClick={(e) => { e.stopPropagation(); onOpenDetail(); }}
          role="button"
          tabIndex={0}
          style={{
            cursor: "pointer",
            textAlign: "left",
            padding: "4px 8px 4px 0",
            minWidth: 0,
            flex: 1,
            margin: "-4px 0 -4px -4px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            position: "relative",
            zIndex: 3,
          }}
        >
          <h3
            className="heading-section"
            style={{
              fontSize: "clamp(1.1rem, 2.4vw, 1.35rem)",
              fontWeight: 600,
              letterSpacing: "-0.025em",
              color: "var(--color-text-primary)",
              margin: 0,
              display: "flex",
              alignItems: "center",
              gap: "8px",
              textTransform: "none",
            }}
          >
            {stack.label}
            <svg
              className="chevron-icon"
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              aria-hidden="true"
            >
              <path d="M5 3L9 7L5 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </h3>
          {stack.subtitle && (
            <p style={{
              margin: "2px 0 0",
              fontSize: "12px",
              fontWeight: 500,
              color: "var(--color-text-muted)",
              letterSpacing: "-0.005em",
            }}>
              {stack.subtitle}
            </p>
          )}
        </div>

        <button
          type="button"
          className="glass-pill"
          onClick={(e) => { e.stopPropagation(); onOpenDetail(); }}
          style={{
            cursor: "pointer",
            fontSize: "12px",
            fontWeight: 600,
            padding: "6px 12px",
            letterSpacing: "-0.005em",
            flexShrink: 0,
            display: "inline-flex",
            alignItems: "center",
            gap: "4px",
            position: "relative",
            zIndex: 3,
          }}
        >
          View All
          <svg width="11" height="11" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M5 3L9 7L5 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      {/* Empty state */}
      {stack.movies.length === 0 && (
        <div style={{ padding: "32px 20px", textAlign: "center", color: "var(--color-text-muted)", fontSize: "13px" }}>
          No movies in this category yet.
        </div>
      )}

      {/* Carousel */}
      {stack.movies.length > 0 && (
        <div style={{ position: "relative" }}>
          {/* Edge scrims */}
          {canLeft && <div className="carousel-scrim left" />}
          {canRight && <div className="carousel-scrim right" />}

          {/* Desktop-only left chevron — hidden on touch via CSS */}
          <button
            className="carousel-btn carousel-btn-left"
            onClick={() => scrollBy("left")}
            aria-label="Previous"
            style={{ opacity: canLeft ? 1 : 0, pointerEvents: canLeft ? "auto" : "none" }}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>

          {/* Desktop-only right chevron */}
          <button
            className="carousel-btn carousel-btn-right"
            onClick={() => scrollBy("right")}
            aria-label="Next"
            style={{ opacity: canRight ? 1 : 0, pointerEvents: canRight ? "auto" : "none" }}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>

          {/* Native-scroll track */}
          <div ref={trackRef} className="hide-scrollbar carousel-track">
            <AnimatePresence initial={false}>
              {stack.movies.map((movie, index) => (
                <PosterCard
                  key={recommendationId(movie)}
                  movie={movie}
                  disabled={disabled}
                  onAction={onAction}
                  onClick={() => onMovieClick(movie)}
                  priority={index === 0}
                />
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}
    </section>
  );
}







function PosterCard({
  movie,
  disabled,
  onAction,
  priority = false,
  showFullInfo = false,
  onClick,
}: {
  movie: Recommendation;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  priority?: boolean;
  showFullInfo?: boolean;
  onClick?: () => void;
}) {
  const poster = usePoster(movie.poster_path, recommendationId(movie), "w342");
  const backdrop = usePoster(movie.backdrop_path || movie.poster_path, recommendationId(movie), "w500");
  const hasBackdrop = !!movie.backdrop_path;

  // Expanded Hover State
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandPos, setExpandPos] = useState<DOMRect | null>(null);

  const cardRef = useRef<HTMLDivElement>(null); // For outer article
  const posterRef = useRef<HTMLDivElement>(null); // For the poster itself

  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const hoverCloseTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const isDesktopHoverEnabled = useCallback(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(min-width: 1024px) and (hover: hover) and (pointer: fine)").matches;
  }, []);

  const measureAndExpand = useCallback(() => {
    if (!posterRef.current) return;
    // Anchor purely to the poster image, not the title below it
    const rect = posterRef.current.getBoundingClientRect();
    setExpandPos(rect);
    setIsExpanded(true);
  }, []);

  const scheduleHoverOpen = useCallback(() => {
    if (disabled || showFullInfo || !isDesktopHoverEnabled()) return;
    if (hoverCloseTimeoutRef.current) {
      clearTimeout(hoverCloseTimeoutRef.current);
      hoverCloseTimeoutRef.current = null;
    }
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    hoverTimeoutRef.current = setTimeout(() => {
      measureAndExpand();
    }, 280);
  }, [disabled, isDesktopHoverEnabled, measureAndExpand, showFullInfo]);

  const scheduleHoverClose = useCallback((delay = 120) => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    if (hoverCloseTimeoutRef.current) clearTimeout(hoverCloseTimeoutRef.current);
    hoverCloseTimeoutRef.current = setTimeout(() => {
      setIsExpanded(false);
      setTimeout(() => setExpandPos(null), 300);
    }, delay);
  }, []);

  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
      if (hoverCloseTimeoutRef.current) clearTimeout(hoverCloseTimeoutRef.current);
    };
  }, []);

  useEffect(() => {
    if (!isExpanded) return;
    const hide = () => { setIsExpanded(false); setExpandPos(null); };
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    return () => {
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
    };
  }, [isExpanded]);

  const lang = languageLabel(movie.original_language || "");
  const imdb = movie.imdb_rating ? movie.imdb_rating.toFixed(1) : movie.vote_average ? movie.vote_average.toFixed(1) : null;
  const genreList = (movie.genres && movie.genres.length > 0) ? movie.genres.slice(0, 3) : (movie.primary_genre ? [movie.primary_genre] : []);

  const isActuallyExpanded = isExpanded && expandPos && isDesktopHoverEnabled();

  // Portal render logic
  let portalElement = null;
  if (isActuallyExpanded && expandPos && typeof document !== "undefined") {
    const zoom = parseFloat(getComputedStyle(document.documentElement).zoom || "1");
    const s = {
      top: expandPos.top / zoom,
      left: expandPos.left / zoom,
      width: expandPos.width / zoom,
      height: expandPos.height / zoom
    };

    // Compute bounds checking limits with inverted zoom too
    const vw = window.innerWidth / zoom;
    const vh = window.innerHeight / zoom;

    // Scale wider for aesthetic 
    const scaleFactor = hasBackdrop ? 2.0 : 1.65;

    const expandedWidth = Math.max(280, s.width * scaleFactor);
    const expandedImgHeight = hasBackdrop ? (expandedWidth * 9 / 16) : (expandedWidth * 1.5);
    const detailsHeight = 170; // Adjusted for overlap
    const targetHeight = expandedImgHeight + detailsHeight;

    // tLeft centers the width
    let tLeft = s.left - (expandedWidth - s.width) / 2;
    // tTop centers the IMAGE part only!
    let tTop = s.top - (expandedImgHeight - s.height) / 2;

    // Bounds check to keep entirely within screen
    if (tLeft < 25) tLeft = 25;
    if (tLeft + expandedWidth > vw - 25) tLeft = vw - expandedWidth - 25;
    if (tTop < 25) tTop = 25;
    if (tTop + targetHeight > vh - 25) tTop = vh - targetHeight - 25;

    portalElement = createPortal(
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{
              opacity: 0,
              top: s.top,
              left: s.left,
              width: s.width,
              height: s.height,
              borderRadius: "8px"
            }}
            animate={{
              opacity: 1,
              top: tTop,
              left: tLeft,
              width: expandedWidth,
              height: targetHeight,
              borderRadius: "16px"
            }}
            exit={{
              opacity: 0,
              top: s.top,
              left: s.left,
              width: s.width,
              height: s.height,
              borderRadius: "8px",
              transition: { duration: 0.2, ease: "easeIn" }
            }}
            transition={{ type: "spring", stiffness: 380, damping: 30, mass: 0.8 }}
            style={{
              position: "fixed",
              zIndex: 999999,
              background: "var(--color-surface, #18191c)",
              boxShadow: "0 30px 60px rgba(0,0,0,0.85), 0 0 0 1px rgba(255,255,255,0.08) inset",
              overflow: "hidden",
              display: "flex",
              flexDirection: "column",
              cursor: "pointer",
              pointerEvents: "auto",
            }}
            onMouseEnter={() => {
              if (hoverCloseTimeoutRef.current) clearTimeout(hoverCloseTimeoutRef.current);
            }}
            onMouseLeave={() => scheduleHoverClose(80)}
            onClick={(e) => { e.stopPropagation(); setIsExpanded(false); if (onClick) onClick(); }}
          >
            {/* The Image Header (Poster or Backdrop) */}
            <div style={{ position: "relative", width: "100%", height: expandedImgHeight, flexShrink: 0 }}>
              <Image src={hasBackdrop ? backdrop : poster} alt={movie.title} fill sizes="400px" style={{ objectFit: "cover", objectPosition: "center 20%" }} />
              <div style={{ position: "absolute", bottom: -2, left: 0, right: 0, height: "65%", background: "linear-gradient(to top, #18191c 0%, rgba(24,25,28,0.95) 20%, rgba(24,25,28,0.6) 50%, rgba(24,25,28,0) 100%)", pointerEvents: "none" }} />
              {imdb && (
                <div style={{ position: "absolute", top: "12px", right: "12px", padding: "4px 8px", borderRadius: "8px", background: "rgba(0,0,0,0.7)", fontSize: "11px", fontWeight: 700, color: "#e8c84a", display: "flex", alignItems: "center", gap: "4px", boxShadow: "0 4px 12px rgba(0,0,0,0.4)" }}>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" /></svg>
                  {imdb}
                </div>
              )}
            </div>

            {/* The Extra Info Panel under the Image */}
            <div style={{ marginTop: "-40px", padding: "0 18px 20px 18px", flex: 1, display: "flex", flexDirection: "column", gap: "8px", zIndex: 2, background: "transparent" }}>
              <h3 style={{ fontSize: "18px", fontWeight: 700, color: "#fff", margin: 0, lineHeight: 1.2 }}>{movie.title}</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  <span style={{ color: "rgba(255,255,255,0.9)", border: "1px solid rgba(255,255,255,0.2)", borderRadius: "6px", padding: "2px 6px", fontSize: "10px", fontWeight: 600 }}>{movie.year || "--"}</span>
                  <span style={{ color: "rgba(255,255,255,0.9)", border: "1px solid rgba(255,255,255,0.2)", borderRadius: "6px", padding: "2px 6px", fontSize: "10px", fontWeight: 600 }}>{lang || "Global"}</span>
                </div>
                {genreList.length > 0 && (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                    {genreList.map((g) => (
                      <span key={g} style={{ color: "rgba(255,255,255,0.75)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: "6px", padding: "2px 6px", fontSize: "10px", fontWeight: 600 }}>{g}</span>
                    ))}
                  </div>
                )}
              </div>
              {movie.overview && (
                <p style={{ margin: "2px 0 0", fontSize: "12px", color: "rgba(255,255,255,0.75)", lineHeight: 1.4, display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{movie.overview}</p>
              )}
              <div style={{ marginTop: "auto", marginBottom: "4px", display: "flex", gap: "10px", alignItems: "center" }}>
                <button onClick={(e) => { e.stopPropagation(); if (onClick) onClick(); setIsExpanded(false); }} style={{ background: "#fff", color: "#000", border: "none", borderRadius: "100px", padding: "8px 16px", fontSize: "13px", fontWeight: 700, display: "flex", alignItems: "center", gap: "6px", cursor: "pointer", flex: 1, justifyContent: "center" }}>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg> View
                </button>
                <button onClick={(e) => { e.stopPropagation(); onAction(movie, "watchlist"); setIsExpanded(false); }} title="Add to watchlist" aria-label="Add to watchlist" style={{ background: "rgba(255,255,255,0.1)", color: "#fff", border: "1px solid rgba(255,255,255,0.2)", borderRadius: "50%", width: "34px", height: "34px", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", flexShrink: 0 }}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 5v14M5 12h14" /></svg>
                </button>
                <button onClick={(e) => { e.stopPropagation(); onAction(movie, "dislike"); setIsExpanded(false); }} title="Show me less of this" aria-label="Show me less of this" style={{ background: "rgba(255,255,255,0.1)", color: "#fff", border: "1px solid rgba(255,255,255,0.2)", borderRadius: "50%", width: "34px", height: "34px", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", flexShrink: 0 }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round"><path d="M18 6L6 18M6 6l12 12" /></svg>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>,
      document.body
    );
  }

  return (
    <motion.article
      ref={cardRef}
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.88, transition: { duration: 0.2, ease: "easeIn" } }}
      className="poster-card"
      style={{
        width: "min(42vw, 165px)",
        minWidth: "130px",
        flexShrink: 0,
        scrollSnapAlign: "start",
        paddingBottom: "8px",
        position: "relative",
      }}
      onMouseEnter={scheduleHoverOpen}
      onMouseLeave={() => scheduleHoverClose(100)}
    >
      <div
        style={{
          opacity: isActuallyExpanded ? 0 : 1,
          transition: "opacity 0.15s ease"
        }}
      >
        <div
          ref={posterRef}
          onClick={(e) => { e.stopPropagation(); if (onClick) onClick(); }}
          className="poster-container"
          style={{ position: "relative", aspectRatio: "2 / 3", borderRadius: "12px", overflow: "hidden", background: "transparent", cursor: "pointer", border: "1px solid transparent", transition: "border-color 0.22s ease" }}
        >
          <Image src={poster} alt={movie.title} fill sizes="(max-width: 640px) 48vw, 180px" style={{ objectFit: "cover" }} priority={priority} />
          {imdb && (
            <div style={{ position: "absolute", top: "8px", right: "8px", padding: "4px 8px", borderRadius: "8px", background: "rgba(0,0,0,0.82)", fontSize: "10px", fontWeight: 700, color: "#e8c84a", display: "flex", alignItems: "center", gap: "3px" }}>
              <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" /></svg>
              {imdb}
            </div>
          )}
        </div>
        <div onClick={(e) => { e.stopPropagation(); if (onClick) onClick(); }} style={{ padding: "10px 8px 12px", cursor: "pointer" }}>
          <p style={{ fontSize: "11px", fontWeight: 600, color: "var(--color-text-primary)", margin: 0, lineHeight: 1.3, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{movie.title}</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "3px", marginTop: "5px" }}>
            <p style={{ fontSize: "10px", color: "var(--color-text-muted)", margin: 0 }}>{[movie.year, lang].filter(Boolean).join(" · ")}</p>
          </div>
        </div>
      </div>
      {portalElement}
    </motion.article>
  );
}
