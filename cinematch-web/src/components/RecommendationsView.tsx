"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";

import HeroCarousel from "@/components/dashboard/HeroCarousel";
import { ShelfRow, type QuickAction } from "@/components/dashboard/ShelfRow";
import CollectionOverlay, { type Collection } from "@/components/dashboard/CollectionOverlay";
import { buildShelves, type Shelf } from "@/components/dashboard/shelves";
import EmptyState from "@/components/ui/EmptyState";
import { useSession } from "@/context/SessionContext";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";
import { prefetchBackdrops } from "@/lib/usePoster";

import MobileMenu, { DesktopNavTabs } from "@/components/MobileMenu";
import {
  apiMultiRecommendations,
  apiRecommendationAction,
  invalidateHistoryCache,
  isSessionExpiredError,
  languageLabel,
  prefetchPosters,
  preferencesFromProfile,
  recommendationId,
  type MultiBucketResponse,
  type Recommendation,
  type RecommendationPreferences,
  type UserSession,
} from "@/lib/api";
import { useMounted } from "@/lib/useMounted";
import MovieDetailModal from "@/components/modals/MovieDetailModal";

interface Props {
  session: UserSession;
  onSessionUpdate: (s: UserSession) => void;
  onBackToOnboarding: () => void;
  onLogout: () => void;
}

type RecommendationAction = "love" | "like" | "dislike" | "remove" | "watchlist" | "skip";
type StackId = "hollywood" | "matched" | "other";

interface Stack {
  id: StackId;
  label: string;
  subtitle: string;
  movies: Recommendation[];
  byLanguage?: Record<string, Recommendation[]>;
}

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
  const seenRegional = new Set<number>();
  if (hasRegional) {
    const buckets = regionalEntries.map(([, arr]) => [...arr]);
    const cursors = buckets.map(() => 0);
    let added = true;
    while (added) {
      added = false;
      for (let i = 0; i < buckets.length; i++) {
        if (cursors[i] < buckets[i].length) {
          const m = buckets[i][cursors[i]];
          cursors[i]++;
          added = true;
          if (!seenRegional.has(recommendationId(m))) {
            seenRegional.add(recommendationId(m));
            regionalMerged.push(m);
          }
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
    const byLanguage: Record<string, Recommendation[]> = {};
    for (const [lang, arr] of regionalEntries) {
      if (arr && arr.length > 0) {
        byLanguage[lang] = arr;
      }
    }
    result.push({
      id: "matched",
      label: matchedLabel ? `${matchedLabel} Cinema` : "Regional Favorites",
      subtitle: "",
      movies,
      byLanguage,
    });
  }

  if ((english || []).length > 0) {
    result.push({
      id: "hollywood",
      label: "Hollywood",
      subtitle: "",
      movies: english || [],
    });
  }

  if ((globalMovies || []).length > 0) {
    result.push({
      id: "other",
      label: "Global Cinema",
      subtitle: "",
      movies: globalMovies || [],
    });
  }

  const allSeen = new Set<number>();
  const allMovies: Recommendation[] = [];
  for (const m of [
    ...(hasRegional ? regionalMerged : (regional._merged || [])),
    ...(english || []),
    ...(globalMovies || []),
  ]) {
    if (!allSeen.has(recommendationId(m))) {
      allSeen.add(recommendationId(m));
      allMovies.push(m);
    }
  }

  return { stacks: result, allMovies };
}

// Deep buckets: the dashboard cuts 8–14 shelves from these pools, and every
// shelf must draw DISJOINT items (no movie on two rails). ~150/bucket gives
// the preferred-language pool enough unique depth to fill all of its shelves
// (backend caps per_bucket_k at 200).
const BUCKET_DISPLAY = 150;
const BUCKET_FETCH = 150;

// Latency scales ~0.7s per bucket server-side and the proxy aborts slow
// upstreams, so depth is scaled down when many language buckets are requested:
// identity rails only need ~36 cards per language — only the MERGED pool that
// feeds the theme rails needs ~150 unique items in total.
function bucketFetchK(preferences: RecommendationPreferences): number {
  const langs = preferences.languages ?? [];
  const regionalCount = langs.filter((l) => l && l.toLowerCase() !== "en").length;
  const includesEn = regionalCount === 0 || langs.some((l) => l.toLowerCase() === "en");
  const buckets = regionalCount + (includesEn ? 1 : 0);
  if (buckets <= 3) return BUCKET_FETCH;
  return Math.max(60, Math.min(BUCKET_FETCH, Math.ceil(450 / buckets)));
}
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
  const mounted = useMounted();

  const [initialCache] = useState(() => readRecsCache(session.user_id));
  const hadCache = initialCache !== null;

  const [stacks, setStacks] = useState<Stack[]>(() => initialCache?.stacks ?? []);
  const [movies, setMovies] = useState<Recommendation[]>(() => initialCache?.movies ?? []);
  const [loading, setLoading] = useState(!hadCache);
  const [initialLoad, setInitialLoad] = useState(!hadCache);

  // Track the OPEN shelf by id, not by snapshot: the collection is derived
  // from live shelves below, so a skip/like inside the detail modal (which
  // removes the movie from stacks) updates the overlay grid in real time


  const [preferences, setPreferences] = useState<RecommendationPreferences>(
    () => preferencesFromProfile(session.profile)
  );

  // Detail Modal state
  const [activeMovie, setActiveMovie] = useState<DetailMovie | null>(null);
  const [activeCollection, setActiveCollection] = useState<Collection | null>(null);

  // Route-based navigation for sub-pages
  const openYourLikes = () => router.push("/your-likes");
  const openWatchlist = () => router.push("/your-likes?filter=watchlist");
  const { openPreferences } = useSession();
  const openPrefs = () => openPreferences();


  const [isUpdating, setIsUpdating] = useState(false);
  const [headerHidden, setHeaderHidden] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    let lastY = window.scrollY;
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const y = window.scrollY;
        const dy = y - lastY;
        const max = document.documentElement.scrollHeight - window.innerHeight;
        setScrollProgress(max > 40 ? Math.min(y / max, 1) : 0);
        if (Math.abs(dy) > 6) {
          if (y < 60) setHeaderHidden(false);
          else if (dy > 0) setHeaderHidden(true);
          else setHeaderHidden(false);
          lastY = y;
        }
        ticking = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);


  // Action counter for auto-rerun
  const actionCountRef = useRef({ positive: 0, negative: 0, total: 0 });
  // tmdb_ids that already contributed to actionCountRef this window —
  // re-rating the same movie shouldn't bump the auto-rerun counters again.
  const countedActionsRef = useRef<Set<number>>(new Set());

  // Every movie ID the user has acted on — prevents re-showing after refresh
  const seenIdsRef = useRef<Set<number>>(
    new Set(initialCache?.seenIds ?? [])
  );

  // All movie IDs ever displayed — sent to backend on auto-rerun so it generates truly new movies
  const displayedIdsRef = useRef<Set<number>>(
    new Set(initialCache?.displayedIds ?? [])
  );


  const bucketCacheRef = useRef<StackCache>(
    initialCache?.bucketCache ?? EMPTY_CACHE()
  );

  // Render-time mirrors of the latest stacks/movies. Used by async handlers
  // (e.g. the post-action refill check) to read current state without
  // stuffing side effects inside setState updaters.
  const stacksRef = useRef<Stack[]>(initialCache?.stacks ?? []);
  const moviesRef = useRef<Recommendation[]>(initialCache?.movies ?? []);
  useEffect(() => { stacksRef.current = stacks; }, [stacks]);
  useEffect(() => { moviesRef.current = movies; }, [movies]);



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

      // Filter each regional language bucket individually so per-language lists are preserved
      const regionalEntries = Object.entries(resp.buckets.regional || {}).filter(([k]) => k !== "_merged");
      const regionalByLangDisplay: Record<string, Recommendation[]> = {};
      const regionalMergedRaw: Recommendation[] = [];
      const seenRegRaw = new Set<number>();

      if (regionalEntries.length > 0) {
        for (const [lang, arr] of regionalEntries) {
          const filtered = filterBucket(arr || []);
          regionalByLangDisplay[lang] = filtered.slice(0, BUCKET_DISPLAY);
        }
        // Interleave regional languages for the merged rail
        const buckets = Object.values(regionalByLangDisplay).map((arr) => [...arr]);
        const cursors = buckets.map(() => 0);
        let added = true;
        while (added) {
          added = false;
          for (let i = 0; i < buckets.length; i++) {
            if (cursors[i] < buckets[i].length) {
              const m = buckets[i][cursors[i]++];
              added = true;
              if (!seenRegRaw.has(recommendationId(m))) {
                seenRegRaw.add(recommendationId(m));
                regionalMergedRaw.push(m);
              }
            }
          }
        }
      }
      const displayReg = regionalMergedRaw;

      // Split each into display slice and cache reserve
      const displayEn = fEn.slice(0, BUCKET_DISPLAY);
      const displayGlob = fGlob.slice(0, BUCKET_DISPLAY);

      bucketCacheRef.current = {
        hollywood: fEn.slice(BUCKET_DISPLAY),
        matched: displayReg.slice(BUCKET_DISPLAY),
        other: fGlob.slice(BUCKET_DISPLAY),
      };

      const displayResp: MultiBucketResponse = {
        ...resp,
        buckets: {
          english: displayEn,
          regional: regionalEntries.length > 0 ? { ...regionalByLangDisplay, _merged: displayReg } : {},
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

  const silentRefreshToken = useRef(0);
  const silentRefresh = useCallback(async (prefs: RecommendationPreferences) => {
    if (silentRefreshInFlight.current) return;
    silentRefreshInFlight.current = true;
    silentRefreshToken.current += 1;
    const myToken = silentRefreshToken.current;
    if (!session?.session_id) return; // no hydrated session — skip instead of 422 spam
    try {
      const resp = await apiMultiRecommendations(session.session_id, {
        languages: prefs.languages,
        genres: prefs.genres,
        age_group: prefs.age_group,
        region: prefs.region,
        include_classics: prefs.include_classics,
        semantic_index: prefs.semantic_index,
        per_bucket_k: bucketFetchK(prefs),
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
      if (isSessionExpiredError(err)) {
        onLogout();
        return;
      }
      console.error("[silentRefresh] Failed:", err);
    } finally {
      silentRefreshInFlight.current = false;
    }
  }, [applyBucketResponse, onLogout, onSessionUpdate, session.session_id]);

  // Monotonic token for generate(). Preference saves can now trigger a
  // regeneration while a previous one is still in flight; without this
  // guard a slow older response could land last and overwrite the fresher
  // slate with stale data.
  const generateTokenRef = useRef(0);
  const generate = useCallback(
    async (
      nextPreferences: RecommendationPreferences = preferences,
      { autoRerun = false }: { autoRerun?: boolean } = {}
    ) => {
      const myToken = ++generateTokenRef.current;
      if (autoRerun) {
        setIsUpdating(true);
      } else {
        setLoading(true);
        setActiveCollection(null);
        setStacks([]);
        setMovies([]);
        bucketCacheRef.current = EMPTY_CACHE();
        seenIdsRef.current = new Set();
        displayedIdsRef.current = new Set();
        actionCountRef.current = { positive: 0, negative: 0, total: 0 }; countedActionsRef.current = new Set();
      }
      if (!session?.session_id) return; // no hydrated session — skip instead of 422 spam
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
          per_bucket_k: bucketFetchK(nextPreferences),
          exclude_ids: excludeIds,
        });

        const allMovies = [
          ...(resp.buckets.english || []),
          ...Object.values(resp.buckets.regional || {}).flat(),
          ...(resp.buckets.global || []),
        ];
        await prefetchPosters(allMovies);

        // A newer generate() superseded this one (e.g. the user saved
        // preferences twice in a row) — discard the stale response.
        if (myToken !== generateTokenRef.current) return;

        if (autoRerun) {
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
          bucketCacheRef.current = {
            hollywood: newEn,
            matched: newReg,
            other: newGlob,
          };

          writeRecsCache(session.user_id, {
            stacks: stacksRef.current,
            movies: moviesRef.current,
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
        if (isSessionExpiredError(err)) {
          onLogout();
          return;
        }
        console.error(err);
        const msg = err instanceof Error ? err.message : "";
        if (
          (err && typeof err === "object" && "isServerSleeping" in err) ||
          msg.includes("500") ||
          msg.includes("ServerSleeping") ||
          msg.includes("exceeded") ||
          msg.includes("SERVER_SLEEPING")
        ) {
          if (typeof window !== "undefined") {
            window.location.href = "/500";
          }
        }
      } finally {
        // Only the newest generation may clear the loading flags — an older
        // superseded call finishing late must not hide the newer one's
        // skeleton/pill mid-flight.
        if (myToken === generateTokenRef.current) {
          setLoading(false);
          setIsUpdating(false);
        }
      }
    },
    [
      applyBucketResponse,
      applyFilters,
      onLogout,
      onSessionUpdate,
      preferences,
      session.session_id,
      session.user_id,
    ]
  );



  useEffect(() => {
    if (!initialLoad) return;

    const t = setTimeout(() => {
      void generate(preferences);
      setInitialLoad(false);
    }, 0);
    return () => clearTimeout(t);
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


      const counted = countedActionsRef.current;
      const isFirstForMovie = !counted.has(tmdbId);
      if (isFirstForMovie) {
        counted.add(tmdbId);
        actionCountRef.current.total++;
        if (action === "love" || action === "like") actionCountRef.current.positive++;
        if (action === "dislike") actionCountRef.current.negative++;
        if (action === "remove" || action === "skip") actionCountRef.current.negative += 0.5;
      }

      const { positive, negative, total } = actionCountRef.current;
      const shouldAutoRerun = negative >= 30 || total >= 30 || positive >= 30;

      if (shouldAutoRerun) {
        actionCountRef.current = { positive: 0, negative: 0, total: 0 }; countedActionsRef.current = new Set();
        setIsUpdating(true);
        try {

          const actionPromise = apiRecommendationAction(session.session_id, tmdbId, action)
            .then((result) => {
              invalidateHistoryCache(session.session_id);
              onSessionUpdate(result.session);
            })
            .catch((err) => {
              if (isSessionExpiredError(err)) { onLogout(); return; }
              console.error("Action during rerun failed:", err);
            });
          await generate(preferences, { autoRerun: true });
          await actionPromise;
        } catch (err) {
          if (isSessionExpiredError(err)) { onLogout(); return; }
          console.error("Taste profile update failed:", err);
          try { await generate(preferences, { autoRerun: true }); } catch { setIsUpdating(false); }
        }
        return;
      }

      apiRecommendationAction(session.session_id, tmdbId, action)
        .then((result) => {
          onSessionUpdate(result.session);
          invalidateHistoryCache(session.session_id);

          if (targetStackId) {
            const st = stacksRef.current.find(s => s.id === targetStackId);
            const cacheRemaining = bucketCacheRef.current[targetStackId]?.length || 0;
            if (st && st.movies.length < CACHE_REFETCH_THRESHOLD && cacheRemaining === 0) {
              void silentRefresh(preferences);
            }
          }
        })
        .catch((err) => {
          if (isSessionExpiredError(err)) { onLogout(); return; }
          console.error("Recommendation action failed:", err);
        });
    },
    [generate, onLogout, onSessionUpdate, preferences, session.session_id, silentRefresh]
  );

  const appliedPrefsKeyRef = useRef<string>(JSON.stringify(preferences));
  useEffect(() => {
    const nextPrefs = preferencesFromProfile(session.profile);
    const nextKey = JSON.stringify(nextPrefs);
    if (nextKey === appliedPrefsKeyRef.current) return;
    appliedPrefsKeyRef.current = nextKey;
    setPreferences(nextPrefs);
    void generate(nextPrefs);
  }, [session.profile, generate]);


  const { shelves, heroMovies } = useMemo(() => buildShelves(stacks, preferences), [stacks, preferences]);

  // Eagerly prefetch horizontal backdrops for the Hero Carousel
  useEffect(() => {
    if (heroMovies.length) {
      prefetchBackdrops(heroMovies);
    }
  }, [heroMovies]);

  const openMovieDetail = useCallback(
    (m: Recommendation) => setActiveMovie(toDetailMovie(m)),
    []
  );

  // ── Infinite collection loader ────────────────────────────────────────────
  // Feeds the open overlay indefinitely: bucket-cache reserves first (instant),
  // then a fresh multi request excluding everything ever displayed. Returns
  // null when no new items exist — the overlay then shows its end-marker.
  const loadMoreForCollection = useCallback(
    async (col: Collection): Promise<Recommendation[] | null> => {
      if (!session?.session_id) return null;
      const existing = new Set(col.movies.map(recommendationId));
      const take = (pool: Recommendation[], n: number) => {
        const out: Recommendation[] = [];
        while (pool.length && out.length < n) {
          const m = pool.shift()!;
          if (!existing.has(recommendationId(m)) &&
              !seenIdsRef.current.has(recommendationId(m))) out.push(m);
        }
        return out;
      };
      const preferred = (preferences.languages ?? []).length === 0 ||
        (preferences.languages ?? []).some((l) => l.toLowerCase() === "en");
      const keys: StackId[] =
        col.id === "hollywood" ? ["hollywood"]
        : col.id === "world" ? ["other"]
        : col.id === "matched" || col.id.startsWith("matched-") ? ["matched"]
        : preferred ? ["matched", "hollywood"] : ["matched"];

      // 1) instant: cache reserves
      const batch: Recommendation[] = [];
      for (const k of keys) {
        const cache = bucketCacheRef.current[k];
        if (cache?.length) batch.push(...take(cache, 40 - batch.length));
        if (batch.length >= 30) break;
      }
      if (batch.length >= 10) return batch;

      // 2) network: fresh batch excluding everything shown
      const exclude = new Set<number>([
        ...col.movies.map(recommendationId),
        ...displayedIdsRef.current,
      ]);
      try {
        const resp = await apiMultiRecommendations(session.session_id, {
          languages: preferences.languages,
          genres: preferences.genres,
          age_group: preferences.age_group,
          region: preferences.region,
          include_classics: preferences.include_classics,
          semantic_index: preferences.semantic_index,
          per_bucket_k: bucketFetchK(preferences),
          exclude_ids: Array.from(exclude),
        });
        const pools: Recommendation[][] = [];
        if (keys.includes("matched"))
          pools.push(Object.entries(resp.buckets.regional)
            .filter(([k]) => k !== "_merged").flatMap(([, v]) => v));
        if (keys.includes("hollywood")) pools.push(resp.buckets.english || []);
        if (keys.includes("other")) pools.push(resp.buckets.global || []);
        for (const pool of pools) batch.push(...take([...pool], 40 - batch.length));
        for (const m of batch) displayedIdsRef.current.add(recommendationId(m));
        return batch.length ? batch : null;
      } catch {
        return null;
      }
    },
    [session.session_id, preferences]
  );

  const openShelfCollection = useCallback((shelf: Shelf) => {
    setActiveCollection({
      id: shelf.id,
      title: shelf.title,
      subtitle: shelf.subtitle,
      movies: shelf.fullMovies ?? shelf.movies,
    });
  }, [setActiveCollection]);

  // Shelf cards expose only like/watchlist; the wider RecommendationAction
  // union on handleAction makes it directly assignable here.
  const handleQuickAction = useCallback(
    (movie: Recommendation, action: QuickAction) => {
      void handleAction(movie, action);
    },
    [handleAction]
  );


  // ── Rail analytics (#10): buffered exposure/action telemetry per shelf.
  // Batched flush every 10 events or on tab-hide; fire-and-forget — analytics
  // must never block or break the recommendation path.
  const railEventsRef = useRef<
    { shelf_id: string; tmdb_id: number; action: string; position: number; ts: number }[]
  >([]);
  const sessionSid = session?.session_id ?? "";
  const flushRailEvents = useCallback(() => {
    if (!railEventsRef.current.length || !sessionSid) return;
    const events = railEventsRef.current.splice(0);
    try {
      // keepalive lets the batch survive tab close; same-origin → Next proxy.
      void fetch("/api/analytics/rail", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionSid, events }),
        keepalive: true,
      }).catch(() => {});
    } catch { /* analytics is best-effort */ }
  }, [sessionSid]);
  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === "hidden") flushRailEvents();
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [flushRailEvents]);
  const trackRailEvent = useCallback(
    (shelf: Shelf, movie: Recommendation, action: string) => {
      railEventsRef.current.push({
        shelf_id: shelf.id,
        tmdb_id: recommendationId(movie),
        action,
        position: shelf.movies.findIndex((m) => recommendationId(m) === recommendationId(movie)),
        ts: Date.now() / 1000,
      });
      if (railEventsRef.current.length >= 10) flushRailEvents();
    },
    [flushRailEvents]
  );

  const showSkeleton = loading && movies.length === 0;
  const showEmpty = !loading && movies.length === 0;
  const scrolled = scrollProgress > 0.002;

  const accountMenu = (
    <MobileMenu
      onLogout={onLogout}
      onReset={onBackToOnboarding}
      onPreferences={openPrefs}
      onYourLikes={openYourLikes}
      onWatchlist={openWatchlist}
    />
  );

  return (
    <div
      className="dash-root"
      style={{
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        fontFamily: "var(--font-sans)",
      }}
    >
      <header
        className={`dash-topbar${scrolled ? " is-scrolled" : ""}`}
        style={{
          transform: headerHidden ? "translateY(-110%)" : "translateY(0)",
        }}
      >
        <div
          className="dash-topbar-progress"
          style={{ transform: `scaleX(${scrollProgress})` }}
          aria-hidden
        />
        <div className="dash-topbar-inner">
          <h1
            className="heading-display dash-brand"
            onClick={() => router.push("/dashboard")}
          >
            CineMatch
          </h1>

          <span className="desktop-only dash-topbar-nav">
            <DesktopNavTabs onPreferences={openPrefs} onWatchlist={openWatchlist} />
          </span>

          <div className="dash-topbar-right">
            <button
              type="button"
              className="dash-search desktop-only"
              onClick={() => router.push("/search")}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              Search movies…
            </button>
            {accountMenu}
          </div>
        </div>
      </header>

      {showSkeleton ? (
        /* ── Loading skeleton — mirrors the real layout so the swap-in
              doesn't shift the eye. ── */
        <div aria-busy="true" aria-label="Loading recommendations">
          <div className="skeleton-shimmer dash-hero-skel" />
          {[0, 1, 2, 3].map((i) => (
            <section key={i} className="shelf-section">
              <div className="shelf-header">
                <div>
                  <div className="skeleton-shimmer" style={{ height: 11, width: 110, borderRadius: 999, marginBottom: 8 }} />
                  <div className="skeleton-shimmer" style={{ height: 22, width: i === 0 ? 210 : i === 1 ? 180 : 160, borderRadius: 999 }} />
                </div>
              </div>
              <div className="hide-scrollbar" style={{ display: "flex", gap: "var(--s-card-gap)", overflow: "hidden", padding: "6px var(--rail-x) 16px" }}>
                {Array.from({ length: 9 }).map((_, j) => (
                  <div key={j} className="dash-skel-card" style={{ width: "var(--poster-w)" }}>
                    <div className="skeleton-shimmer skeleton-grain" style={{ aspectRatio: "2 / 3", borderRadius: "var(--radius-poster)" }} />
                    <div style={{ marginTop: 14 }}>
                      <div className="skeleton-shimmer" style={{ height: 14, width: "85%", borderRadius: 4, marginBottom: 6 }} />
                      <div className="skeleton-shimmer" style={{ height: 11, width: "55%", borderRadius: 4 }} />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      ) : (
        <>
          {!showEmpty && (
            <HeroCarousel
              movies={heroMovies}
              onOpenDetail={openMovieDetail}
              onWatchlist={(m) => handleAction(m, "watchlist")}
              onLike={(m) => handleAction(m, "like")}
              onAction={(m, action) => handleAction(m, action)}
            />
          )}

          <main
            className="app-container"
            style={{
              width: "100%",
              paddingBottom: 16,
              position: "relative",
              zIndex: 1,
            }}
          >
            {shelves.map((shelf, i) => (
              <ShelfRow
                key={shelf.id}
                shelf={shelf}
                index={i}
                priority={i === 0}
                onOpenMovie={(m) => {
                  trackRailEvent(shelf, m, "open");
                  openMovieDetail(m);
                }}
                onOpenShelf={openShelfCollection}
                onQuickAction={(m, a) => {
                  trackRailEvent(shelf, m, a);
                  handleQuickAction(m, a);
                }}
              />
            ))}

            {showEmpty && (
              <div style={{ padding: "10dvh var(--rail-x)" }}>
                <EmptyState
                  title={initialLoad ? "Warming up your slate…" : "Nothing to show right now"}
                  description={
                    initialLoad
                      ? "We're assembling recommendations tuned to your taste."
                      : "Your rails ran dry or something hiccuped upstream. A retry usually fixes it."
                  }
                  cta={{
                    kind: "button",
                    label: "Reload recommendations",
                    onClick: () => void generate(preferences),
                  }}
                />
              </div>
            )}

          </main>
        </>
      )}

      {/* See-all overlay */}
      <AnimatePresence>
        {activeCollection && (
          <CollectionOverlay
            key={"collection-" + activeCollection.id}
            collection={activeCollection}
            onBack={() => setActiveCollection(null)}
            onMovieClick={openMovieDetail}
            onQuickAction={handleAction}
            onLoadMore={() => loadMoreForCollection(activeCollection)}
          />
        )}
      </AnimatePresence>

      {/* Updating-taste-profile indicator — small non-blocking glass pill;
          the dashboard stays fully interactive during a rebuild. */}
      {mounted && createPortal(
        <AnimatePresence>
          {isUpdating && (
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 16 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
              role="status"
              aria-live="polite"
              style={{
                position: "fixed",
                left: "50%",
                bottom: "calc(96px + env(safe-area-inset-bottom))",
                translate: "-50% 0",
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
                pointerEvents: "none", // explicitly never block interaction
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
    </div>
  );
}
