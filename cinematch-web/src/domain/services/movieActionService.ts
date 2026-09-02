import type { MovieLike, ActionType } from "@/hooks/useMovieActions";
import type { RecommendationRepository } from "../repositories/MovieRepository";
import type { HapticsPort } from "@/infrastructure/haptics/HapticsService";
import type { StorageService } from "@/infrastructure/storage/StorageService";

export interface MovieActionDeps {
  recommendationRepo: RecommendationRepository;
  historyCache: StorageService;
  haptics: HapticsPort;
  sessionId: string;
  onSessionRefresh?: (session: import("../types/movie").UserSession) => void;
  onSessionExpired?: () => void;
}

export interface MovieActionResult {
  ok: boolean;
  error?: unknown;
}

export async function executeMovieAction(
  movie: MovieLike,
  action: ActionType,
  deps: MovieActionDeps
): Promise<MovieActionResult> {
  const tmdbId = (movie as { tmdb_id?: number; id: number }).tmdb_id ?? (movie as { id: number }).id;
  if (!tmdbId) return { ok: false, error: new Error("Missing tmdbId") };

  deps.haptics.trigger(action as never);

  try {
    const res = await deps.recommendationRepo.submitAction(deps.sessionId, tmdbId, action);
    deps.historyCache.remove(`history_cache_${deps.sessionId}`);
    if (res?.session) deps.onSessionRefresh?.(res.session);
    return { ok: true };
  } catch (err) {
    // Session expiry is a domain event, not a fetch detail — let the caller decide (e.g., redirect to /login)
    const isExpired =
      err instanceof Error &&
      (err.message.includes("Session not found") ||
        err.message.includes("session expired") ||
        (err as unknown as { isSessionExpired?: boolean }).isSessionExpired === true);
    if (isExpired) deps.onSessionExpired?.();
    return { ok: false, error: err };
  }
}

export function optimisticReactionUpdate(
  prev: Record<number, { rating?: string | null; watchlist?: boolean }>,
  tmdbId: number,
  action: ActionType
): Record<number, { rating?: string | null; watchlist?: boolean }> {
  const cur = prev[tmdbId] ?? {};
  if (action === "watchlist") return { ...prev, [tmdbId]: { ...cur, watchlist: !cur.watchlist } };
  if (action === "skip") return prev; // no optimistic change for skip
  return { ...prev, [tmdbId]: { ...cur, rating: action } };
}
