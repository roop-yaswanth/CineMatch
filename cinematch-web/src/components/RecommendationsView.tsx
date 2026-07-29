"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";



import dynamic from "next/dynamic";
import BackButton from "@/components/ui/BackButton";
import HeroFeature from "@/components/dashboard/HeroFeature";
import CompactRail from "@/components/dashboard/CompactRail";
import AppFooter from "@/components/AppFooter";
import { toast } from "@/components/ui/Toast";
import { useSession } from "@/context/SessionContext";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";

import MobileMenu, { DesktopNavTabs } from "@/components/MobileMenu";
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
import MovieDetailModal from "@/components/modals/MovieDetailModal";

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

function readRecsCache(userId: string): RecsCache | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(`${RECS_CACHE_KEY}_${userId}`);
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

function writeRecsCache(userId: string, data: Omit<RecsCache, "ts">) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(
      `${RECS_CACHE_KEY}_${userId}`,
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
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // ── Try to restore from navigation cache on mount ─────────────────────────
  const cachedRef = useRef(readRecsCache(session.user_id));
  const hadCache = cachedRef.current !== null;

  const [stacks, setStacks] = useState<Stack[]>(() => cachedRef.current?.stacks ?? []);
  const [movies, setMovies] = useState<Recommendation[]>(() => cachedRef.current?.movies ?? []);
  const [loading, setLoading] = useState(false);
  const [initialLoad, setInitialLoad] = useState(!hadCache);

  const [showUpdateToast] = useState(false);
  const [activeStack, setActiveStack] = useState<StackId | null>(null);
  // Derived once from the session profile at mount. Preference changes happen
  // via the overlay modal (PreferencesModal), which persists the new profile —
  // so this component remounts and re-derives rather than mutating in place.
  const [preferences] = useState<RecommendationPreferences>(
    () => preferencesFromProfile(session.profile)
  );

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  // Detail Modal state
  const [activeMovie, setActiveMovie] = useState<DetailMovie | null>(null);

  // Route-based navigation for sub-pages
  const openYourLikes = () => router.push("/your-likes");
  const openWatchlist = () => router.push("/your-likes?filter=watchlist");
  const { openPreferences } = useSession();
  const openPrefs = () => openPreferences();


  const [isUpdating, setIsUpdating] = useState(false);


  // Action counter for auto-rerun
  const actionCountRef = useRef({ positive: 0, negative: 0, total: 0 });
  // tmdb_ids that already contributed to actionCountRef this window —
  // re-rating the same movie shouldn't bump the auto-rerun counters again.
  const countedActionsRef = useRef<Set<number>>(new Set());

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
          const rest = results.filter((r) => r.original_language !== langCode);
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
  // Monotonically increasing token stamped on every silentRefresh call.
  // The handler discards a response whose token is older than the current
  // value — without this guard, a slow batch landing after the user has
  // already kept tapping (which advances stacks state) silently overwrites
  // the fresher state with stale data.
  const silentRefreshToken = useRef(0);
  const silentRefresh = useCallback(async (prefs: RecommendationPreferences) => {
    if (silentRefreshInFlight.current) return;
    silentRefreshInFlight.current = true;
    silentRefreshToken.current += 1;
    const myToken = silentRefreshToken.current;
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

      // Drop the response if a newer silentRefresh has been kicked off
      // while we were waiting on the network — the newer one will deliver
      // fresher data.
      if (myToken !== silentRefreshToken.current) return;

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
        actionCountRef.current = { positive: 0, negative: 0, total: 0 }; countedActionsRef.current = new Set();
      }

      await prefetchPosters(totalMovies);
      // Re-check the token after the await — prefetchPosters can be slow
      // on cold caches and another refresh may have started in the meantime.
      if (myToken !== silentRefreshToken.current) return;
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
        actionCountRef.current = { positive: 0, negative: 0, total: 0 }; countedActionsRef.current = new Set();
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

        if (autoRerun) {
          // Keep current stacks visible — the user may still be browsing
          // movies they haven't acted on. Replace the cache entirely with
          // fresh taste-profile-aware results so the NEXT movie pulled in
          // after a swipe comes from the updated batch.
          const filterFresh = (arr: Recommendation[]) =>
            applyFilters(
              arr.filter((m) => !seenIdsRef.current.has(recommendationId(m))),
              nextPreferences
            );

          const newEn = filterFresh(resp.buckets.english || []);
          const newGlob = filterFresh(resp.buckets.global || []);
          const regionalEntries = Object.entries(resp.buckets.regional || {});
          const regionalMerged: Recommendation[] = [];
          if (regionalEntries.length > 0) {
            const buckets = regionalEntries.map(([, arr]) => [...(arr || [])]);
            const cursors = buckets.map(() => 0);
            let added = true;
            while (added) {
              added = false;
              for (let i = 0; i < buckets.length; i++) {
                if (cursors[i] < buckets[i].length) {
                  regionalMerged.push(buckets[i][cursors[i]++]);
                  added = true;
                }
              }
            }
          }
          const newReg = filterFresh(regionalMerged);

          // REPLACE cache with fresh taste-profile-aware results.
          // Old cache items were ranked by the pre-rerun taste profile
          // and sat at the front of the queue — new items were stuck at
          // the back and never reached the user. Now fresh results go
          // directly to the front.
          bucketCacheRef.current = {
            hollywood: newEn,
            matched: newReg,
            other: newGlob,
          };

          // Persist to localStorage so a dashboard remount doesn't lose
          // the freshly-fetched recs (the write effect only fires on
          // stacks/movies changes, not cache-only mutations).
          writeRecsCache(session.user_id, {
            stacks,
            movies,
            bucketCache: bucketCacheRef.current,
            seenIds: Array.from(seenIdsRef.current),
            displayedIds: Array.from(displayedIdsRef.current),
          });

          if (resp.session) onSessionUpdate(resp.session);
        } else {
          applyBucketResponse(resp, nextPreferences);
          if (resp.session) onSessionUpdate(resp.session);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
        setIsUpdating(false);
      }
    },
    [applyBucketResponse, onSessionUpdate, preferences, session.user_id]
  );



  useEffect(() => {
    if (!initialLoad) return;
    void generate(preferences);
    setInitialLoad(false);
  }, [generate, initialLoad, preferences]);

  // Keep navigation cache in sync with optimistic stack updates (actions, etc.)
  useEffect(() => {
    if (stacks.length === 0) return; // Don't cache empty state
    writeRecsCache(session.user_id, {
      stacks,
      movies,
      bucketCache: bucketCacheRef.current,
      seenIds: Array.from(seenIdsRef.current),
      displayedIds: Array.from(displayedIdsRef.current),
    });
  }, [stacks, movies, session.user_id]);

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

      // Per-movie dedup of the rerun counters. Re-rating the same movie
      // (like → dislike → like) used to bump `total` three times even
      // though it represents one piece of feedback. Track the set of
      // tmdb_ids already counted in this refresh window — only the first
      // action per movie contributes.
      const counted = countedActionsRef.current;
      const isFirstForMovie = !counted.has(tmdbId);
      if (isFirstForMovie) {
        counted.add(tmdbId);
        actionCountRef.current.total++;
        if (action === "like" || action === "okay") actionCountRef.current.positive++;
        if (action === "dislike") actionCountRef.current.negative++;
        if (action === "remove" || action === "skip") actionCountRef.current.negative += 0.5;
      }

      const { positive, negative, total } = actionCountRef.current;
      // Match the backend thresholds (all 30). Was 10/10/10, which fired
      // the profile rebuild after a third of the documented count.
      const shouldAutoRerun = negative >= 30 || total >= 30 || positive >= 30;

      if (shouldAutoRerun) {
        actionCountRef.current = { positive: 0, negative: 0, total: 0 }; countedActionsRef.current = new Set();
        setIsUpdating(true);
        try {
          // Fire action and rerun in parallel — the action is a write
          // that also triggers a backend pool rebuild; we don't need its
          // response before fetching fresh multi-bucket recs. Running
          // them concurrently halves the wall-clock wait.
          const actionPromise = apiRecommendationAction(session.session_id, tmdbId, action)
            .catch((err) => console.error("Action during rerun failed:", err));
          await generate(preferences, { autoRerun: true });
          await actionPromise;
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

  // Preference updates are now applied via the overlay PreferencesModal, which
  // persists the new profile (server + cached session). This component re-derives
  // `preferences` from `session.profile` at mount, and the initial-load effect
  // above fires a single generate() against it. That replaces the old
  // sessionStorage-stash / custom-event / visibilitychange relay.

  return (
    <div
      style={{
        // Fill the viewport exactly and own scroll inside this container.
        // Body scroll is locked at the dashboard page level (so the global
        // footer can't slide up *behind* the fixed swipe cards), so all
        // vertical scrolling for rails / extra content has to happen in
        // here instead.
        height: "100dvh",
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-sans)",
        width: "100%",
        // Note: don't use 100vw here — vw units don't scale with html { zoom },
        // so on desktop the page would only fill 85% of physical viewport.
        // width: 100% inherits correctly from the zoomed html element.
        maxWidth: "100%",
        overflowX: "hidden",
        overflowY: "auto",
        WebkitOverflowScrolling: "touch",
        position: "relative",
        background: "var(--color-bg)",
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
              className="dashboard-header-bar"
              style={{
                width: "100%",
                padding: "12px 16px 12px",
                display: "flex",
                alignItems: "center",
                position: "relative",   // needed for absolute title centering
              }}
            >
              {/* Left: Brand logo title */}
              <div style={{ flex: 1, display: "flex", alignItems: "center" }}>
                <h1
                  className="heading-display header-title-brand"
                  style={{
                    fontSize: "21px",
                    fontWeight: 700,
                    letterSpacing: "-0.035em",
                    background: "linear-gradient(180deg, #ffffff 0%, #a0a0a0 100%)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                    margin: 0,
                    whiteSpace: "nowrap",
                    cursor: "pointer",
                  }}
                  onClick={() => router.push("/dashboard")}
                >
                  CineMatch
                </h1>
              </div>

              {/* Center: Desktop Navigation Bar */}
              <DesktopNavTabs onPreferences={openPrefs} onWatchlist={openWatchlist} />

              {/* Right: search + user account menu */}
              <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "8px" }}>
                {/* Desktop mini search box — hidden on mobile via CSS */}
                <button
                  className="header-search-box"
                  onClick={() => router.push("/search")}
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
                  onClick={() => router.push("/search")}
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
            <style>{`
              @media (max-width: 899px) {
                .header-title-brand { display: none !important; }
                .dashboard-header-bar { padding: 8px 16px !important; }
              }
            `}</style>
          </header>

          {/* Content */}
          <div className="app-container" style={{ flex: 1, width: "100%", padding: 0 }}>

            {/* Loading skeleton */}
            {loading && movies.length === 0 && (
              <div style={{ display: "grid", gap: "48px", padding: "24px 20px 0" }}>
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
              <div
                style={{
                  display: "grid",
                  gap: "40px",
                }}
              >
                {stacks.map((stack) => (
                  <StackRow
                    key={stack.id}
                    stack={stack}
                    disabled={loading}
                    onAction={handleAction}
                    onOpenDetail={() => setActiveStack(stack.id)}
                    onMovieClick={(m) => setActiveMovie(toDetailMovie(m))}
                  />
                ))}
                <AppFooter />
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


        {/* Updating-taste-profile indicator. Used to be a full-screen
            blackout with a giant popcorn — that blocked the user from
            doing anything (swipe, tap, watchlist) while a 1–3 second
            rebuild was running. Now it's a small glass pill in the
            bottom-left corner; the rest of the dashboard stays
            interactive throughout the rerun. */}
        {mounted && createPortal(
          <AnimatePresence>
            {(showUpdateToast || isUpdating) && (
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 16 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
                role="status"
                aria-live="polite"
                style={{
                  position: "fixed",
                  // Sit in the bottom-left, well clear of the bottom-nav
                  // pill on the right and the iOS home indicator below.
                  left: "calc(16px + env(safe-area-inset-left))",
                  bottom: "calc(96px + env(safe-area-inset-bottom))",
                  zIndex: 90,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "10px",
                  padding: "10px 14px 10px 12px",
                  borderRadius: "999px",
                  background: "rgba(20, 22, 28, 0.78)",
                  backdropFilter: "blur(28px) saturate(1.6)",
                  WebkitBackdropFilter: "blur(28px) saturate(1.6)",
                  border: "1px solid rgba(255,255,255,0.10)",
                  boxShadow: "0 12px 36px rgba(0,0,0,0.45), 0 1px 0 rgba(255,255,255,0.10) inset",
                  pointerEvents: "none",  // explicitly never block interaction
                }}
              >
                <motion.span
                  animate={{ y: [0, -3, 0] }}
                  transition={{ repeat: Infinity, duration: 0.9, ease: "easeInOut" }}
                  style={{ fontSize: "18px", display: "inline-flex" }}
                >
                  🍿
                </motion.span>
                <span
                  style={{
                    color: "white",
                    fontSize: "13px",
                    fontWeight: 500,
                    letterSpacing: "-0.01em",
                    whiteSpace: "nowrap",
                  }}
                >
                  Updating taste profile…
                </span>
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
        // Solid bg, NOT transparent. Two reasons:
        //  (1) RouteTransition wraps children with a 180ms `transform`
        //      animation. While that transform is active, `position:fixed`
        //      descendants are re-anchored to the transformed ancestor
        //      instead of the viewport — which briefly exposed the global
        //      footer through the transparent grid gaps on this overlay.
        //  (2) Even without that quirk, a transparent full-screen overlay
        //      is fragile: anything painted in the body below (footer,
        //      extra route content) bleeds through whenever the stacking
        //      context changes. Solid bg = bug surface area = 0.
        background: "var(--color-bg)",
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

  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;
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







export function PosterCard({
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
    if (disabled || !isDesktopHoverEnabled()) return;
    if (hoverCloseTimeoutRef.current) {
      clearTimeout(hoverCloseTimeoutRef.current);
      hoverCloseTimeoutRef.current = null;
    }
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    hoverTimeoutRef.current = setTimeout(() => {
      measureAndExpand();
    }, 280);
  }, [disabled, isDesktopHoverEnabled, measureAndExpand]);

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
              <img src={hasBackdrop ? backdrop : poster} alt={movie.title} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover", objectPosition: "center 20%" }} />
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
          <img src={poster} alt={movie.title} loading={priority ? "eager" : "lazy"} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }} />
        </div>
        <div onClick={(e) => { e.stopPropagation(); if (onClick) onClick(); }} style={{ padding: "10px 8px 12px", cursor: "pointer" }}>
          <p
            style={{
              fontSize: "12px",
              fontWeight: 600,
              color: "var(--color-text-primary)",
              margin: 0,
              lineHeight: 1.3,
              // Reserve two lines so the metadata row lines up across every card
              // in the rail whether the title wraps to one line or two — this is
              // what fixes the ragged look where short titles sat higher.
              minHeight: "2.6em",
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {movie.title}
          </p>
          {/* One tidy metadata row — matches the compact rail cards: "year · lang"
              muted on the left, a single gold rating chip pushed to the right. */}
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "5px", minHeight: "18px" }}>
            {(movie.year || lang) && (
              <span
                style={{
                  fontSize: "10.5px",
                  color: "var(--color-text-muted)",
                  lineHeight: 1.2,
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  minWidth: 0,
                }}
              >
                {[movie.year, lang].filter(Boolean).join(" · ")}
              </span>
            )}
            {imdb && (
              <span
                style={{
                  marginLeft: "auto",
                  flexShrink: 0,
                  padding: "2px 6px",
                  borderRadius: "6px",
                  background: "rgba(232,200,74,0.15)",
                  color: "#e8c84a",
                  fontSize: "10px",
                  fontWeight: 700,
                  whiteSpace: "nowrap",
                }}
              >
                {movie.imdb_rating ? `IMDb ${imdb}` : `★ ${imdb}`}
              </span>
            )}
          </div>
        </div>
      </div>
      {portalElement}
    </motion.article>
  );
}
