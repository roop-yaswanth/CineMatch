"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  apiMultiRecommendations,
  apiRecommendationAction,
  apiTrendingHero,
  invalidateHistoryCache,
  isSessionExpiredError,
  languageLabel,
  prefetchPosters,
  type MultiBucketResponse,
  type Recommendation,
  type RecommendationPreferences,
  type TrendingHeroResponse,
  type UserSession,
} from "@/lib/api";
import { preferencesFromProfile, recommendationId } from "@/domain/types/movie";
import type { Collection } from "@/components/dashboard/CollectionOverlay";
import type { DetailMovie } from "@/components/modals/MovieDetailModal";

// Exported types from the hook
export type RecommendationAction = "love" | "like" | "dislike" | "remove" | "watchlist" | "skip";
export type StackId = "hollywood" | "matched" | "other" | string;

export interface Stack {
  id: StackId;
  label: string;
  subtitle: string;
  movies: Recommendation[];
  byLanguage?: Record<string, Recommendation[]>;
}

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

const BUCKET_DISPLAY = 150;
const BUCKET_FETCH = 150;
const CACHE_REFETCH_THRESHOLD = 20;

function bucketFetchK(preferences: RecommendationPreferences): number {
  const langs = preferences.languages ?? [];
  const regionalCount = langs.filter((l) => l && l.toLowerCase() !== "en").length;
  const includesEn = regionalCount === 0 || langs.some((l) => l.toLowerCase() === "en");
  const buckets = regionalCount + (includesEn ? 1 : 0);
  if (buckets <= 3) return BUCKET_FETCH;
  return Math.max(60, Math.min(BUCKET_FETCH, Math.ceil(450 / buckets)));
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
  } catch { /* localStorage full */ }
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

export function useRecommendations(
  session: UserSession,
  onSessionUpdate: (s: UserSession) => void,
  onLogout: () => void
) {
  const [initialCache] = useState(() => readRecsCache(session.user_id));
  const hadCache = initialCache !== null;

  const [stacks, setStacks] = useState<Stack[]>(() => initialCache?.stacks ?? []);
  const [movies, setMovies] = useState<Recommendation[]>(() => initialCache?.movies ?? []);
  const [loading, setLoading] = useState(!hadCache);
  const [initialLoad, setInitialLoad] = useState(!hadCache);
  const [isUpdating, setIsUpdating] = useState(false);

  const [preferences, setPreferences] = useState<RecommendationPreferences>(
    () => preferencesFromProfile(session.profile)
  );

  const [trendingHero, setTrendingHero] = useState<TrendingHeroResponse | null>(null);

  const actionCountRef = useRef({ positive: 0, negative: 0, total: 0 });
  const countedActionsRef = useRef<Set<number>>(new Set());

  const seenIdsRef = useRef<Set<number>>(new Set(initialCache?.seenIds ?? []));
  const [seenMovieIds, setSeenMovieIds] = useState<Set<number>>(() => new Set(initialCache?.seenIds ?? []));

  const displayedIdsRef = useRef<Set<number>>(new Set(initialCache?.displayedIds ?? []));
  const bucketCacheRef = useRef<StackCache>(initialCache?.bucketCache ?? EMPTY_CACHE());

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

      const regionalEntries = Object.entries(resp.buckets.regional || {}).filter(([k]) => k !== "_merged");
      const regionalByLangDisplay: Record<string, Recommendation[]> = {};
      const regionalMergedRaw: Recommendation[] = [];
      const seenRegRaw = new Set<number>();

      if (regionalEntries.length > 0) {
        for (const [lang, arr] of regionalEntries) {
          const filtered = filterBucket(arr || []);
          regionalByLangDisplay[lang] = filtered.slice(0, BUCKET_DISPLAY);
        }
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
    if (!session?.session_id) return;
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
        setSeenMovieIds(new Set());
        actionCountRef.current = { positive: 0, negative: 0, total: 0 }; countedActionsRef.current = new Set();
      }

      await prefetchPosters(totalMovies);
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
        setStacks([]);
        setMovies([]);
        bucketCacheRef.current = EMPTY_CACHE();
        seenIdsRef.current = new Set();
        setSeenMovieIds(new Set());
        displayedIdsRef.current = new Set();
        actionCountRef.current = { positive: 0, negative: 0, total: 0 }; countedActionsRef.current = new Set();
      }
      if (!session?.session_id) return;
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

  // Keep navigation cache in sync with optimistic stack updates
  useEffect(() => {
    if (stacks.length === 0) return;
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
      setSeenMovieIds((prev) => {
        const next = new Set(prev);
        next.add(tmdbId);
        return next;
      });

      // Remove from trendingHero immediately so hero rotates to next fresh title
      setTrendingHero((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          results: prev.results.filter((m) => recommendationId(m) !== tmdbId),
        };
      });

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

  useEffect(() => {
    if (!preferences.languages) return;
    let cancelled = false;
    apiTrendingHero(preferences.languages, preferences.genres, preferences.region)
      .then((data) => {
        if (!cancelled) setTrendingHero(data);
      });
    return () => { cancelled = true; };
  }, [preferences.languages, preferences.genres, preferences.region]);

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

      const batch: Recommendation[] = [];
      for (const k of keys) {
        const cache = bucketCacheRef.current[k];
        if (cache?.length) batch.push(...take(cache, 40 - batch.length));
        if (batch.length >= 30) break;
      }
      if (batch.length >= 10) return batch;

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

  return {
    stacks,
    movies,
    loading,
    initialLoad,
    isUpdating,
    preferences,
    trendingHero,
    seenMovieIds,
    handleAction,
    loadMoreForCollection,
    refresh: () => void generate(preferences),
  };
}
