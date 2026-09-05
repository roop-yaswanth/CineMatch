"use client";

/**
 * useMovieActions — Presentation hook that composes domain services.
 * The concrete HttpRecommendationRepository is injected via the composition root
 * (data/repositories/HttpMovieRepository). Swap the adapter, no UI change.
 *
 * SRP: this hook has one reason to change — the movie-action user flow.
 * It delegates to injected domain services and repositories.
 */

import { useCallback, useState } from "react";
import { isSessionExpiredError, type Recommendation, type ExploreMovie, type Movie } from "@/lib/api";
import { useSession } from "@/context/SessionContext";
// New layered imports — presentation depends on domain abstractions, not data concretions
import { movieRepositories } from "@/data/repositories/HttpMovieRepository";
import { localStore } from "@/infrastructure/storage/StorageService";
import { executeMovieAction } from "@/domain/services/movieActionService";

export type MovieLike = Movie | Recommendation | ExploreMovie;
export type ActionType = "love" | "like" | "dislike" | "watchlist" | "skip" | "remove";

export type UserReactionEntry = {
  rating?: "love" | "like" | "dislike" | null;
  watchlist?: boolean;
};

/**
 * Unified movie-action handler.
 * Single place that owns:
 * - API call
 * - history-cache invalidation
 * - session-expiry handling
 * - optimistic reaction map updates
 */
export function useMovieActions(opts?: {
  onSessionExpired?: () => void;
  onSuccess?: (movie: MovieLike, action: ActionType) => void;
}) {
  const { session, updateSession } = useSession();
  const [pendingIds, setPendingIds] = useState<Set<number>>(new Set());

  const act = useCallback(
    async (movie: MovieLike, action: ActionType) => {
      if (!session?.session_id) return;
      const tmdbId = (movie as Recommendation).tmdb_id ?? (movie as Movie).id;
      if (!tmdbId) return;

      setPendingIds((s) => new Set(s).add(tmdbId));

      // Delegates to pure domain service with explicit deps (no hidden globals)
      const result = await executeMovieAction(movie, action, {
        recommendationRepo: movieRepositories.recommendations,
        historyCache: localStore,
        sessionId: session.session_id,
        onSessionRefresh: (s) => updateSession(s),
        onSessionExpired: opts?.onSessionExpired,
      });

      if (result.ok) opts?.onSuccess?.(movie, action);
      else if (!isSessionExpiredError(result.error)) {
        console.error("[useMovieActions] action failed", result.error);
      }

      setPendingIds((s) => {
        const n = new Set(s);
        n.delete(tmdbId);
        return n;
      });
    },
    [session, updateSession, opts]
  );

  const isPending = useCallback((movie: MovieLike) => {
    const id = (movie as Recommendation).tmdb_id ?? (movie as Movie).id;
    return pendingIds.has(id);
  }, [pendingIds]);

  return { act, isPending, pendingIds };
}

/** Build a lookup map from HistoryItem[] -> tmdb_id -> reaction */
export function buildReactionsMap(items: Array<{ tmdb_id: number; rating?: string | null }>): Record<number, UserReactionEntry> {
  const map: Record<number, UserReactionEntry> = {};
  for (const item of items) {
    if (!map[item.tmdb_id]) map[item.tmdb_id] = {};
    const r = (item.rating || "").toLowerCase();
    if (r === "love" || r === "like" || r === "dislike") {
      map[item.tmdb_id].rating = r as UserReactionEntry["rating"];
    } else if (r === "watchlist") {
      map[item.tmdb_id].watchlist = true;
    }
  }
  return map;
}

/** Hook wrapper that fetches history + maintains reaction map */
export function useHistoryReactions(sessionId?: string | null) {
  const [reactions, setReactions] = useState<Record<number, UserReactionEntry>>({});
  const refresh = useCallback(async () => {
    if (!sessionId) return;
    try {
      const { apiGetHistory, writeHistoryCache, readHistoryCache } = await import("@/lib/api");
      // seed from cache for instant paint
      const cached = readHistoryCache(sessionId);
      if (cached) setReactions(buildReactionsMap(cached as unknown as Array<{ tmdb_id: number; rating?: string | null }>));
      const history = await apiGetHistory(sessionId);
      if (Array.isArray(history)) {
        writeHistoryCache(sessionId, history);
        setReactions(buildReactionsMap(history as unknown as Array<{ tmdb_id: number; rating?: string | null }>));
      }
    } catch { /* best-effort */ }
  }, [sessionId]);

  return { reactions, setReactions, refresh, buildReactionsMap };
}
