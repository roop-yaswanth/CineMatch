"use client";

import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import YourLikesView from "@/components/YourLikesView";
import PreferencesModal from "@/components/PreferencesModal";
import MobileMenu from "@/components/MobileMenu";
import {
  apiMultiRecommendations,
  apiRecommendationAction,
  apiSearchMovies,
  LANGUAGE_LABELS,
  languageLabel,
  posterUrl,
  prefetchPosters,
  preferencesFromProfile,
  recommendationId,
  regionLanguages,
  type MultiBucketResponse,
  type Recommendation,
  type RecommendationPreferences,
  type SearchResult,
  type UserSession,
} from "@/lib/api";
import { usePoster } from "@/lib/usePoster";

interface Props {
  session: UserSession;
  onSessionUpdate: (s: UserSession) => void;
  onBackToOnboarding: () => void;
  onLogout: () => void;
}

type RecommendationAction = "like" | "okay" | "dislike" | "remove";
type StackId = "hollywood" | "matched" | "other";

interface Stack {
  id: StackId;
  label: string;
  subtitle: string;
  movies: Recommendation[];
}

const CARD_ACTIONS: Array<{
  action: RecommendationAction;
  label: string;
  icon: ReactNode;
  color: string;
}> = [
  {
    action: "like",
    label: "Like",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
      </svg>
    ),
    color: "var(--color-like)",
  },
  {
    action: "okay",
    label: "Okay",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
        <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
      </svg>
    ),
    color: "var(--color-okay)",
  },
  {
    action: "dislike",
    label: "Dislike",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
        <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
      </svg>
    ),
    color: "var(--color-dislike)",
  },
  {
    action: "remove",
    label: "Skip",
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="5 4 15 12 5 20 5 4" fill="currentColor" stroke="none" />
        <line x1="19" y1="5" x2="19" y2="19" />
      </svg>
    ),
    color: "var(--color-skip)",
  },
];

/**
 * Converts the pre-partitioned /api/recommendations/multi response into Stacks.
 * The backend already separated movies by language bucket, so we just map them.
 */
function partitionFromBuckets(
  resp: MultiBucketResponse,
  preferences: RecommendationPreferences
): { stacks: Stack[]; allMovies: Recommendation[] } {
  const { english, regional, global: globalMovies } = resp.buckets;

  // Collect all regional movies (interleaved across languages for fairness)
  const regionalEntries = Object.entries(regional).filter(
    ([key]) => key !== "_merged"  // Skip internal merge key
  );
  const hasRegional = regionalEntries.some(([, arr]) => arr.length > 0);

  // Interleave regional languages so they mix evenly in the stack
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

  // Build label from actual language codes, not internal keys
  // Use selected languages from preferences for accurate labeling
  const selectedNonEnglish = (preferences.languages || [])
    .filter((l) => l && l.toLowerCase() !== "en");
  
  const matchedLabel = selectedNonEnglish.length > 0
    ? selectedNonEnglish.map((lang) => languageLabel(lang)).join(", ")
    : regionalEntries.map(([lang]) => languageLabel(lang)).join(", ");

  const result: Stack[] = [];

  // Regional language stack FIRST (if user selected non-English languages)
  if (hasRegional || (regional._merged && regional._merged.length > 0)) {
    const movies = hasRegional ? regionalMerged : (regional._merged || []);
    result.push({
      id: "matched",
      label: matchedLabel ? `${matchedLabel} Cinema` : "Regional Favorites",
      subtitle: "Best from your preferred languages.",
      movies,
    });
  }

  // Hollywood SECOND
  result.push({
    id: "hollywood",
    label: "Hollywood",
    subtitle: "Handpicked from your taste profile.",
    movies: english || [],
  });

  // Global Cinema THIRD (IMDb ≥ 7.0, non-selected languages)
  result.push({
    id: "other",
    label: "Global Cinema",
    subtitle: "Hidden gems across cultures — curated by plot similarity.",
    movies: globalMovies || [],
  });

  const allMovies = [
    ...(hasRegional ? regionalMerged : (regional._merged || [])),
    ...(english || []),
    ...(globalMovies || []),
  ];

  return { stacks: result, allMovies };
}
// ── Cache + display sizing constants ──────────────────────────────────────
const BUCKET_DISPLAY  = 50;   // max shown per stack at any time
const BUCKET_FETCH    = 100;  // movies fetched per bucket from backend
const CACHE_REFETCH_THRESHOLD = 20; // trigger a silent API refetch when a stack's display falls below this AND cache is empty

type StackCache = Record<StackId, Recommendation[]>;
const EMPTY_CACHE = (): StackCache => ({ hollywood: [], matched: [], other: [] });

export default function RecommendationsView({
  session,
  onSessionUpdate,
  onBackToOnboarding,
  onLogout,
}: Props) {
  const [stacks, setStacks] = useState<Stack[]>([]);
  const [movies, setMovies] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [initialLoad, setInitialLoad] = useState(true);
  const [showYourLikes, setShowYourLikes] = useState(false);
  const [showPrefs, setShowPrefs] = useState(false);
  const [showUpdateToast, setShowUpdateToast] = useState(false);
  const [activeStack, setActiveStack] = useState<StackId | null>(null);
  const [preferences, setPreferences] = useState<RecommendationPreferences>(
    () => preferencesFromProfile(session.profile)
  );

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Action counter for auto-rerun (mirrors Gradio trigger logic)
  const actionCountRef = useRef({ positive: 0, negative: 0, total: 0 });

  // Every movie ID the user has acted on — prevents re-showing after refresh
  const seenIdsRef = useRef<Set<number>>(new Set());

  // ── Bucket cache ───────────────────────────────────────────────────────────
  // Stores the "reserve" movies (indices 50-100) that are not yet displayed.
  // When a displayed movie is acted on the stack is immediately topped back up
  // from here — no API call needed until the cache itself runs dry.
  const bucketCacheRef = useRef<StackCache>(EMPTY_CACHE());

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
        const resp = await apiSearchMovies(query.trim(), 15);
        setSearchResults(resp.results);
      } catch (err) {
        console.error("Search error:", err);
        setSearchResults([]);
      } finally {
        setSearchLoading(false);
      }
    }, 300); // 300ms debounce
  }, []);

  // Apply frontend classics filter (genres are handled by backend, not here)
  const applyFilters = useCallback((arr: Recommendation[], prefs: RecommendationPreferences): Recommendation[] => {
    if (!prefs.include_classics) {
      const modern = arr.filter((m: any) => {
        const y = typeof m.year === "number" ? m.year : parseInt(m.year, 10);
        return isNaN(y) || y >= 2000;
      });
      if (modern.length > 0) return modern;
    }
    return arr;
  }, []);

  // ── applyBucketResponse ────────────────────────────────────────────────────
  // Converts the raw API response (100 per bucket) into:
  //   • display stacks  (first BUCKET_DISPLAY movies per stack)
  //   • bucketCacheRef  (remainder → used to silently top up stacks)
  const applyBucketResponse = useCallback(
    (resp: MultiBucketResponse, prefs: RecommendationPreferences) => {
      const filterBucket = (arr: Recommendation[]) =>
        applyFilters(
          arr.filter((m) => !seenIdsRef.current.has(recommendationId(m))),
          prefs
        );

      const fEn  = filterBucket(resp.buckets.english || []);
      const fGlob = filterBucket(resp.buckets.global   || []);

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
      const displayEn  = fEn.slice(0,  BUCKET_DISPLAY);
      const displayReg = fReg.slice(0, BUCKET_DISPLAY);
      const displayGlob = fGlob.slice(0,BUCKET_DISPLAY);

      bucketCacheRef.current = {
        hollywood: fEn.slice(BUCKET_DISPLAY),
        matched:   fReg.slice(BUCKET_DISPLAY),
        other:     fGlob.slice(BUCKET_DISPLAY),
      };

      // Build display resp (only first 50 per bucket for partitionFromBuckets)
      const displayResp: MultiBucketResponse = {
        ...resp,
        buckets: {
          english:  displayEn,
          regional: regionalEntries.length > 0 ? { _merged: displayReg } : {},
          global:   displayGlob,
        },
      };

      const { stacks: newStacks, allMovies } = partitionFromBuckets(displayResp, prefs);
      setStacks(newStacks);
      setMovies(allMovies);
    },
    [applyFilters]
  );

  // ── replenishFromCache ─────────────────────────────────────────────────────
  // After a movie is removed from a stack, pull from the cache reserve to keep
  // the stack at BUCKET_DISPLAY. Returns true if any movies were added.
  const replenishFromCache = useCallback((stackId: StackId): boolean => {
    const cache = bucketCacheRef.current[stackId];
    if (!cache || cache.length === 0) return false;

    let addedAny = false;
    setStacks((prev) =>
      prev.map((s) => {
        if (s.id !== stackId) return s;
        const needed = BUCKET_DISPLAY - s.movies.length;
        if (needed <= 0) return s;
        // Take from cache (mutating the ref)
        const toAdd = cache.splice(0, needed);
        bucketCacheRef.current[stackId] = cache;
        addedAny = true;
        return { ...s, movies: [...s.movies, ...toAdd] };
      })
    );
    return addedAny;
  }, []);

  // ── silentRefresh ──────────────────────────────────────────────────────────
  // Fires ONLY when: cache for a stack is exhausted AND display < CACHE_REFETCH_THRESHOLD.
  const silentRefreshInFlight = useRef(false);
  const silentRefresh = useCallback(async (prefs: RecommendationPreferences) => {
    if (silentRefreshInFlight.current) return;
    silentRefreshInFlight.current = true;
    try {
      const resp = await apiMultiRecommendations(session.session_id, {
        languages:        prefs.languages,
        genres:           prefs.genres,
        age_group:        prefs.age_group,
        region:           prefs.region,
        include_classics: prefs.include_classics,
        semantic_index:   prefs.semantic_index,
        per_bucket_k:     BUCKET_FETCH,
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

  // ── generate (full/manual refresh) ────────────────────────────────────────
  const generate = useCallback(
    async (nextPreferences: RecommendationPreferences = preferences) => {
      setLoading(true);
      setStacks([]);
      setMovies([]);
      bucketCacheRef.current = EMPTY_CACHE();
      seenIdsRef.current = new Set();
      actionCountRef.current = { positive: 0, negative: 0, total: 0 };
      try {
        const resp = await apiMultiRecommendations(session.session_id, {
          languages:        nextPreferences.languages,
          genres:           nextPreferences.genres,
          age_group:        nextPreferences.age_group,
          region:           nextPreferences.region,
          include_classics: nextPreferences.include_classics,
          semantic_index:   nextPreferences.semantic_index,
          per_bucket_k:     BUCKET_FETCH,
        });

        const allMovies = [
          ...(resp.buckets.english || []),
          ...Object.values(resp.buckets.regional || {}).flat(),
          ...(resp.buckets.global || []),
        ];
        await prefetchPosters(allMovies);
        applyBucketResponse(resp, nextPreferences);
        if (resp.session) onSessionUpdate(resp.session);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    },
    [applyBucketResponse, onSessionUpdate, preferences, session.session_id]
  );

  useEffect(() => {
    if (!initialLoad) return;
    void generate(preferences);
    setInitialLoad(false);
  }, [generate, initialLoad, preferences]);

  const handleAction = useCallback(
    async (movie: Recommendation, action: RecommendationAction) => {
      const tmdbId = recommendationId(movie);

      // 1. Mark as seen
      seenIdsRef.current.add(tmdbId);

      // 2. Optimistic removal from stacks + cache replenish
      setMovies((prev) => prev.filter((m) => recommendationId(m) !== tmdbId));
      let targetStackId: StackId | null = null;

      setStacks((prev) =>
        prev.map((s) => {
          const inThis = s.movies.some((m) => recommendationId(m) === tmdbId);
          if (!inThis) return s;
          targetStackId = s.id as StackId;
          const remaining = s.movies.filter((m) => recommendationId(m) !== tmdbId);
          // Pull from cache immediately
          const cache = bucketCacheRef.current[s.id as StackId];
          const needed = BUCKET_DISPLAY - remaining.length;
          const toAdd  = needed > 0 && cache && cache.length > 0 ? cache.splice(0, needed) : [];
          return { ...s, movies: [...remaining, ...toAdd] };
        })
      );

      // 3. Update action counters
      actionCountRef.current.total++;
      if (action === "like" || action === "okay") actionCountRef.current.positive++;
      if (action === "dislike") actionCountRef.current.negative++;

      const { positive, negative, total } = actionCountRef.current;
      const shouldAutoRerun = negative >= 10 || total >= 10 || positive >= 10;

      // 4. Taste Profile Update
      if (shouldAutoRerun) {
        actionCountRef.current = { positive: 0, negative: 0, total: 0 };
        setShowUpdateToast(true);
        try {
          await apiRecommendationAction(session.session_id, tmdbId, action);
          await generate(preferences);
        } catch (err) {
          console.error("Taste profile update failed:", err);
          await generate(preferences);
        }
        setTimeout(() => setShowUpdateToast(false), 1500);
        return;
      }

      // 5. Fire-and-forget backend update
      apiRecommendationAction(session.session_id, tmdbId, action)
        .then((result) => {
          onSessionUpdate(result.session);
          
          // 6. If target stack cache is empty AND displayed count is thin → silent re-fetch
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
    [generate, onSessionUpdate, preferences, session.session_id, silentRefresh]
  );

  const handlePreferenceUpdate = useCallback(
    (nextPreferences: RecommendationPreferences) => {
      setPreferences(nextPreferences);
      setShowPrefs(false);
      startTransition(() => {
        void generate(nextPreferences);
      });
    },
    [generate]
  );

  return (
    <div
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-sans)",
        width: "100%",
        maxWidth: "100vw",
        overflowX: "hidden",
      }}
    >
      {/* Header */}
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 40,
          background: "var(--color-bg)",
          borderBottom: "1px solid var(--color-border-subtle)",
        }}
      >
        <div
          style={{
            width: "100%",
            padding: "10px 20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ width: "40px" }} /> {/* Spacer to balance menu */}

          <h1
            style={{
              fontSize: "20px",
              fontWeight: 700,
              letterSpacing: "-0.03em",
              background: "linear-gradient(135deg, #ffffff 30%, #a78bfa 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              margin: 0,
              textAlign: "center",
            }}
          >
            CineMatch
          </h1>

          <MobileMenu
            onLogout={onLogout}
            onPreferences={() => setShowPrefs(true)}
            onYourLikes={() => setShowYourLikes(true)}
            onRefresh={() => void generate(preferences)}
            onReset={onBackToOnboarding}
          />
        </div>
        
        {/* Search Bar */}
        <div style={{ padding: "0 20px 12px", position: "relative" }}>
          <div style={{ position: "relative" }}>
            <input
              type="text"
              placeholder="Search movies..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              style={{
                width: "100%",
                padding: "10px 16px 10px 40px",
                borderRadius: "12px",
                border: "1px solid var(--color-border-subtle)",
                background: "var(--color-surface)",
                color: "var(--color-text-primary)",
                fontSize: "16px",  // 16px prevents iOS zoom on focus
                outline: "none",
              }}
            />
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--color-text-muted)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)" }}
            >
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery("");
                  setSearchResults([]);
                  setShowSearch(false);
                }}
                style={{
                  position: "absolute",
                  right: "12px",
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  padding: "4px",
                  color: "var(--color-text-muted)",
                }}
              >
                ✕
              </button>
            )}
          </div>
          
          {/* Search Results Dropdown */}
          {showSearch && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                left: "20px",
                right: "20px",
                background: "var(--color-surface)",
                border: "1px solid var(--color-border-subtle)",
                borderRadius: "12px",
                maxHeight: "400px",
                overflowY: "auto",
                zIndex: 50,
                boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
              }}
            >
              {searchLoading ? (
                <div style={{ padding: "20px", textAlign: "center", color: "var(--color-text-muted)" }}>
                  Searching...
                </div>
              ) : searchResults.length === 0 ? (
                <div style={{ padding: "20px", textAlign: "center", color: "var(--color-text-muted)" }}>
                  No movies found
                </div>
              ) : (
                searchResults.map((movie) => (
                  <div
                    key={movie.tmdb_id}
                    style={{
                      display: "flex",
                      gap: "12px",
                      padding: "12px 16px",
                      borderBottom: "1px solid var(--color-border-subtle)",
                      cursor: "pointer",
                    }}
                    onClick={() => {
                      // Could navigate to movie detail or add to recommendations
                      setShowSearch(false);
                      setSearchQuery("");
                    }}
                  >
                    <div
                      style={{
                        width: "48px",
                        height: "72px",
                        borderRadius: "6px",
                        overflow: "hidden",
                        flexShrink: 0,
                        background: "var(--color-bg)",
                      }}
                    >
                      {movie.poster_path && (
                        <Image
                          src={posterUrl(movie.poster_path, "w92")}
                          alt={movie.title}
                          width={48}
                          height={72}
                          style={{ objectFit: "cover" }}
                          unoptimized
                        />
                      )}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 600, color: "var(--color-text-primary)", fontSize: "14px" }}>
                        {movie.title}
                      </div>
                      <div style={{ fontSize: "12px", color: "var(--color-text-muted)", marginTop: "2px" }}>
                        {movie.year && <span>{movie.year}</span>}
                        {movie.year && movie.original_language && <span> · </span>}
                        {movie.original_language && <span>{languageLabel(movie.original_language)}</span>}
                      </div>
                      {movie.imdb_rating && (
                        <div style={{ fontSize: "11px", color: "#fbbf24", marginTop: "4px" }}>
                          IMDb {movie.imdb_rating.toFixed(1)} ({movie.imdb_votes?.toLocaleString()} votes)
                        </div>
                      )}
                      {movie.genres && movie.genres.length > 0 && (
                        <div style={{ fontSize: "11px", color: "var(--color-text-muted)", marginTop: "2px" }}>
                          {movie.genres.join(", ")}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
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
              For You
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
                <div style={{ display: "flex", gap: "14px", overflow: "hidden" }}>
                  {Array.from({ length: 6 }).map((_, j) => (
                    <div key={j} style={{ width: "160px", flexShrink: 0 }}>
                      <div
                        className="skeleton-shimmer"
                        style={{
                          aspectRatio: "2 / 3",
                          borderRadius: "var(--radius-poster)",
                        }}
                      />
                      <div
                        className="skeleton-shimmer"
                        style={{
                          height: "12px",
                          width: "80%",
                          borderRadius: "999px",
                          marginTop: "10px",
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Stacks */}
        {!loading && (
          <div style={{ display: "grid", gap: "48px" }}>
            {stacks.map((stack) => (
              <StackRow
                key={stack.id}
                stack={stack}
                disabled={loading}
                onAction={handleAction}
                onOpenDetail={() => setActiveStack(stack.id)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Stack detail overlay */}
      <AnimatePresence>
        {activeStack && (
          <StackDetailView
            stack={stacks.find((s) => s.id === activeStack)!}
            onBack={() => setActiveStack(null)}
            onAction={handleAction}
            disabled={loading}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {showUpdateToast && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            style={{
              position: "fixed",
              inset: 0,
              zIndex: 9999,
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
      </AnimatePresence>

      {showYourLikes && (
        <YourLikesView
          sessionId={session.session_id}
          onClose={() => setShowYourLikes(false)}
        />
      )}

      {showPrefs && (
        <PreferencesModal
          preferences={preferences}
          onUpdate={handlePreferenceUpdate}
          onClose={() => setShowPrefs(false)}
          mode="recommendations"
        />
      )}
    </div>
  );
}

/* ─── Stack Detail (Full-Screen Grid) ──────────── */

function StackDetailView({
  stack,
  onBack,
  onAction,
  disabled,
}: {
  stack: Stack;
  onBack: () => void;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  disabled: boolean;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.25 }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 50,
        background: "var(--color-bg)",
        overflowY: "auto",
        overflowX: "hidden",
      }}
    >
      {/* Detail header */}
      <div
        style={{
          position: "sticky",
          top: 0,
          zIndex: 10,
          background: "var(--color-bg)",
          borderBottom: "1px solid var(--color-border-subtle)",
          padding: "12px 16px",
          display: "flex",
          alignItems: "center",
          gap: "16px",
        }}
      >
        <button
          onClick={onBack}
          style={{
            fontSize: "13px",
            fontWeight: 500,
            color: "var(--color-text-primary)",
            cursor: "pointer",
            background: "var(--color-surface)",
            border: "1px solid var(--color-border-subtle)",
            borderRadius: "var(--radius-pill)",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "6px 14px",
            boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
          }}
        >
          <span style={{ fontSize: "16px", transform: "translateY(-1px)" }}>←</span> Back
        </button>
        <div>
          <h2
            style={{
              fontSize: "16px",
              fontWeight: 600,
              letterSpacing: "-0.02em",
              margin: 0,
              color: "var(--color-text-primary)",
            }}
          >
            {stack.label}
          </h2>
          <p style={{ fontSize: "12px", color: "var(--color-text-muted)", marginTop: "2px" }}>
            {stack.movies.length > 50
              ? `Top 50 of ${stack.movies.length} movies`
              : `${stack.movies.length} movie${stack.movies.length !== 1 ? "s" : ""}`}
          </p>
        </div>
      </div>

      {/* Grid */}
      <div
        className="stack-detail-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
          gap: "16px",
          padding: "16px 16px 40px",
        }}
      >
        <AnimatePresence initial={false}>
          {stack.movies.slice(0, 50).map((movie, index) => (
            <PosterCard
              key={recommendationId(movie)}
              movie={movie}
              disabled={disabled}
              onAction={onAction}
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
}

/* ─── Stack Row — Paged Carousel ───────────────── */

function StackRow({
  stack,
  disabled,
  onAction,
  onOpenDetail,
}: {
  stack: Stack;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  onOpenDetail: () => void;
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
    <section style={{ width: "100%", overflow: "hidden" }}>
      {/* Stack header */}
      <div
        style={{
          padding: "0 20px",
          marginBottom: "12px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <button
          className="stack-name-btn"
          onClick={onOpenDetail}
          style={{ background: "none", border: "none", padding: "10px 8px 10px 0", cursor: "pointer", textAlign: "left" }}
        >
          <h3
            style={{
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.07em",
              textTransform: "uppercase",
              color: "var(--color-text-secondary)",
              margin: 0,
              display: "flex",
              alignItems: "center",
              gap: "4px",
            }}
          >
            {stack.label}
            <svg
              className="chevron-icon"
              width="10"
              height="10"
              viewBox="0 0 14 14"
              fill="none"
              aria-hidden="true"
            >
              <path d="M5 3L9 7L5 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </h3>
        </button>

        <button
          onClick={onOpenDetail}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontSize: "11px",
            fontWeight: 500,
            color: "var(--color-text-muted)",
            padding: "6px 0",
            letterSpacing: "0.02em",
            flexShrink: 0,
          }}
        >
          View All
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
            ‹
          </button>

          {/* Desktop-only right chevron */}
          <button
            className="carousel-btn carousel-btn-right"
            onClick={() => scrollBy("right")}
            aria-label="Next"
            style={{ opacity: canRight ? 1 : 0, pointerEvents: canRight ? "auto" : "none" }}
          >
            ›
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

/* ─── Poster Card with Hover Overlay ───────────── */

function PosterCard({
  movie,
  disabled,
  onAction,
  priority = false,
  showFullInfo = false,
}: {
  movie: Recommendation;
  disabled: boolean;
  onAction: (movie: Recommendation, action: RecommendationAction) => void;
  priority?: boolean;
  showFullInfo?: boolean;
}) {
  const poster = usePoster(movie.poster_path, recommendationId(movie), "w500");
  const [showActions, setShowActions] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const touchStartXRef = useRef<number | null>(null);

  const overlayJustShownRef = useRef(false);

  // Dismiss overlay when tapping/clicking outside this card
  useEffect(() => {
    if (!showActions) return;
    const handler = (e: MouseEvent | TouchEvent) => {
      if (cardRef.current && !cardRef.current.contains(e.target as Node)) {
        setShowActions(false);
      }
    };
    document.addEventListener("mousedown", handler);
    document.addEventListener("touchstart", handler, { passive: true });
    return () => {
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("touchstart", handler);
    };
  }, [showActions]);

  // Touch handler — suppresses ghost click after touchend.
  // Local swipe guard: if horizontal movement > 10px, treat as carousel scroll, not tap.
  const handleContainerTouchStart = (e: React.TouchEvent) => {
    touchStartXRef.current = e.touches[0].clientX;
  };
  const handleContainerTouchEnd = (e: React.TouchEvent) => {
    const dx = touchStartXRef.current !== null
      ? Math.abs(e.changedTouches[0].clientX - touchStartXRef.current)
      : 0;
    touchStartXRef.current = null;
    if (dx > 10) return; // swipe gesture, not a tap
    e.preventDefault();
    if (!showActions) {
      overlayJustShownRef.current = true;
      setShowActions(true);
      setTimeout(() => {
        overlayJustShownRef.current = false;
      }, 400);
    } else {
      setShowActions(false);
    }
  };

  // Mouse-only click toggle (touch path is fully handled by onTouchEnd above)
  const handleContainerClick = () => {
    setShowActions((v) => !v);
  };

  const lang = movie.original_language
    ? languageLabel(movie.original_language)
    : "";
  const imdb = movie.imdb_rating
    ? `IMDb ${movie.imdb_rating.toFixed(1)}`
    : movie.vote_average
      ? `★ ${movie.vote_average.toFixed(1)}`
      : "";
  const genres =
    movie.genres?.slice(0, 3).join(", ") || movie.primary_genre || "";
  const meta = [movie.year, lang, imdb].filter(Boolean).join(" · ");

  return (
    <motion.article
      ref={cardRef}
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.88, transition: { duration: 0.25, ease: "easeIn" } }}
      transition={{
        layout: { type: "spring", stiffness: 400, damping: 36 },
        opacity: { duration: 0.2 },
        scale: { duration: 0.2 },
      }}
      className="poster-card"
      style={{
        width: "min(42vw, 165px)",
        minWidth: "130px",
        flexShrink: 0,
        scrollSnapAlign: "start",
      }}
    >
      {/* Poster with overlays */}
      <div
        className="poster-container"
        onTouchStart={handleContainerTouchStart}
        onTouchEnd={handleContainerTouchEnd}
        onClick={handleContainerClick}
        style={{
          position: "relative",
          aspectRatio: "2 / 3",
          borderRadius: "var(--radius-poster)",
          overflow: "hidden",
          background: "var(--color-surface)",
          cursor: "pointer",
          border: "1px solid transparent",
          transition: "border-color 0.22s ease",
        }}
      >
        <Image
          src={poster}
          alt={movie.title}
          fill
          sizes="(max-width: 640px) 48vw, 180px"
          style={{ objectFit: "cover" }}
          unoptimized
          priority={priority}
        />

        {/* Rating badge top-right */}
        {imdb && (
          <div
            className={`rating-badge ${showActions ? "hidden-by-actions" : ""}`}
            style={{
              position: "absolute",
              top: "6px",
              right: "6px",
              padding: "3px 6px",
              borderRadius: "6px",
              background: "rgba(0,0,0,0.75)",
              backdropFilter: "blur(4px)",
              WebkitBackdropFilter: "blur(4px)",
              fontSize: "10px",
              fontWeight: 600,
              color: "#fbbf24",
              pointerEvents: "none",
            }}
          >
            {imdb}
          </div>
        )}

        {/* Persistent bottom info overlay */}
        <div
          className={`poster-bottom-info ${showActions ? "hidden-by-actions" : ""}`}
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            background: showFullInfo
              ? "linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.7) 50%, transparent 100%)"
              : "linear-gradient(to top, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.5) 55%, transparent 100%)",
            padding: showFullInfo ? "52px 10px 10px" : "40px 8px 8px",
            pointerEvents: "none",
          }}
        >
          <p
            style={{
              fontSize: showFullInfo ? "13px" : "11px",
              fontWeight: 600,
              color: "#fff",
              margin: 0,
              lineHeight: 1.3,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {movie.title}
          </p>
          {showFullInfo ? (
            <>
              {/* Year · Language · IMDb */}
              <p style={{ fontSize: "11px", color: "rgba(255,255,255,0.65)", margin: "4px 0 0", lineHeight: 1.3 }}>
                {[
                  movie.year,
                  lang || null,
                  imdb || null,
                ].filter(Boolean).join(" · ")}
              </p>
              {/* Genre */}
              {genres && (
                <p style={{
                  fontSize: "10px",
                  color: "rgba(255,255,255,0.45)",
                  margin: "2px 0 0",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}>
                  {genres}
                </p>
              )}
            </>
          ) : (
            (movie.year || lang || imdb) && (
              <p style={{ fontSize: "10px", color: "rgba(255,255,255,0.55)", margin: "2px 0 0", lineHeight: 1.3 }}>
                {[movie.year, lang, imdb].filter(Boolean).join(" · ")}
              </p>
            )
          )}
        </div>

        {/* Action overlay */}
        <div
          className={`action-overlay ${showActions ? "active" : ""}`}
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexWrap: "wrap",
            alignContent: "center",
            justifyContent: "center",
            gap: "8px",
            background: "linear-gradient(160deg, rgba(10,10,18,0.72) 0%, rgba(0,0,0,0.92) 100%)",
            backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
            padding: "12px",
          }}
        >
          {CARD_ACTIONS.map((btn) => (
            <button
              key={btn.action}
              // Touch: preventDefault kills ghost-click; guard prevents
              // firing on the same tap that opened the overlay.
              onTouchEnd={(e) => {
                e.preventDefault();
                e.stopPropagation();
                if (disabled || overlayJustShownRef.current) return;
                setShowActions(false);
                onAction(movie, btn.action);
              }}
              onClick={(e) => {
                e.stopPropagation();
                if (disabled) return;
                setShowActions(false);
                onAction(movie, btn.action);
              }}
              disabled={disabled}
              className="action-btn"
              style={{
                width: "calc(50% - 4px)",
                color: btn.color,
                opacity: disabled ? 0.4 : 1,
                cursor: disabled ? "not-allowed" : "pointer",
              }}
            >
              <span style={{ display: "flex", alignItems: "center", justifyContent: "center", lineHeight: 1 }}>
                {btn.icon}
              </span>
              <span style={{ fontSize: "9px", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                {btn.label}
              </span>
            </button>
          ))}
        </div>
      </div>
    </motion.article>
  );
}
