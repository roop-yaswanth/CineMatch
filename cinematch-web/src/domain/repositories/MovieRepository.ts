/**
 * MovieRepository — abstraction (port) for movie data.
 * High-level domain depends on this interface, not on concrete HttpMovieRepository.
 * Change the data source (e.g., from backend API to local cache) by swapping the adapter,
 * without touching a single component or service.
 *
 * ISP: small, focused interfaces — not one bloated repository.
 */

import type {
  Recommendation,
  HistoryItem,
  MultiBucketResponse,
  RecommendationPreferences,
} from "../types/movie";
import type { MultiSearchResponse, ExploreCategory, DiscoverFilters, ExploreResponse } from "../types/search";

export interface RecommendationRepository {
  getMultiBuckets(
    sessionId: string,
    prefs: RecommendationPreferences & { per_bucket_k?: number; exclude_ids?: number[] }
  ): Promise<MultiBucketResponse>;
  submitAction(
    sessionId: string,
    tmdbId: number,
    action: string,
    dwellMs?: number
  ): Promise<{ session: import("../types/movie").UserSession }>;
  getHistory(sessionId: string): Promise<HistoryItem[]>;
}

export interface SearchRepository {
  searchMulti(query: string): Promise<MultiSearchResponse>;
}

export interface ExploreRepository {
  explore(
    category: ExploreCategory,
    page?: number,
    region?: string,
    lang?: string,
    genre?: string,
    sortBy?: string
  ): Promise<ExploreResponse>;
  discover(filters: DiscoverFilters): Promise<ExploreResponse>;
}

export interface CreditsRepository {
  getCredits(tmdbId: number, kind?: "movie" | "tv"): Promise<import("@/lib/api").CreditsResponse>;
  getSimilar(
    tmdbId: number,
    sessionId?: string | null,
    n?: number
  ): Promise<Recommendation[]>;
}
