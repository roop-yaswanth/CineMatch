/**
 * HttpMovieRepository — concrete adapter for MovieRepository ports.
 * Single Responsibility: translate domain calls into HTTP requests.
 * Depends on HttpClient abstraction, not on `fetch` directly.
 * High-level services depend on the `MovieRepository` interface, not this class.
 */

import { httpRequest } from "@/infrastructure/http/HttpClient";
import type {
  RecommendationRepository,
  SearchRepository,
  ExploreRepository,
} from "@/domain/repositories/MovieRepository";
import type { MultiBucketResponse, RecommendationPreferences, HistoryItem } from "@/domain/types/movie";
import type { MultiSearchResponse, ExploreCategory, ExploreResponse, DiscoverFilters } from "@/domain/types/search";

export class HttpRecommendationRepository implements RecommendationRepository {
  async getMultiBuckets(
    sessionId: string,
    prefs: RecommendationPreferences & { per_bucket_k?: number; exclude_ids?: number[] }
  ): Promise<MultiBucketResponse> {
    return httpRequest<MultiBucketResponse>("/api/recommendations/multi", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, ...prefs }),
    });
  }
  async submitAction(sessionId: string, tmdbId: number, action: string, dwellMs = 0) {
    return httpRequest<{ session: import("@/domain/types/movie").UserSession }>("/api/recommendations/action", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, tmdb_id: tmdbId, action, dwell_ms: dwellMs }),
    });
  }
  async getHistory(sessionId: string): Promise<HistoryItem[]> {
    return httpRequest<HistoryItem[]>(`/api/history?session_id=${sessionId}`);
  }
}

export class HttpSearchRepository implements SearchRepository {
  private cache = new Map<string, MultiSearchResponse>();
  private max = 30;
  async searchMulti(query: string): Promise<MultiSearchResponse> {
    const key = query.trim().toLowerCase();
    const hit = this.cache.get(key);
    if (hit) {
      this.cache.delete(key);
      this.cache.set(key, hit);
      return hit;
    }
    const res = await fetch(`/api/search/multi?q=${encodeURIComponent(query)}`);
    if (!res.ok) return { movies: [], tv: [], people: [] };
    const data: MultiSearchResponse = await res.json();
    this.cache.set(key, data);
    if (this.cache.size > this.max) {
      const oldest = this.cache.keys().next().value;
      if (oldest) this.cache.delete(oldest);
    }
    return data;
  }
}

export class HttpExploreRepository implements ExploreRepository {
  async explore(category: ExploreCategory, page = 1, region?: string, lang?: string, genre?: string, sortBy?: string): Promise<ExploreResponse> {
    const params = new URLSearchParams({ category, page: String(page) });
    if (region) params.set("region", region);
    if (lang) params.set("with_original_language", lang);
    if (genre) params.set("with_genres", genre);
    if (sortBy) params.set("sort_by", sortBy);
    const res = await fetch(`/api/tmdb/explore?${params.toString()}`);
    if (!res.ok) throw new Error(`Explore fetch failed: ${res.status}`);
    return res.json();
  }
  async discover(filters: DiscoverFilters): Promise<ExploreResponse> {
    const params = new URLSearchParams();
    if (filters.sort_by) params.set("sort_by", filters.sort_by);
    if (filters.with_genres?.length) params.set("with_genres", filters.with_genres.join(","));
    if (filters.year_from) params.set("year_from", String(filters.year_from));
    if (filters.year_to) params.set("year_to", String(filters.year_to));
    if (filters.with_original_language) params.set("with_original_language", filters.with_original_language);
    if (filters.region) params.set("region", filters.region);
    params.set("page", String(filters.page ?? 1));
    const res = await fetch(`/api/tmdb/discover?${params.toString()}`);
    if (!res.ok) throw new Error(`Discover fetch failed: ${res.status}`);
    return res.json();
  }
}

// Singleton composition root — the only place that knows which concrete adapter is used.
// Presentation imports `movieRepositories` (the abstraction), not `fetch`.
export const movieRepositories = {
  recommendations: new HttpRecommendationRepository(),
  search: new HttpSearchRepository(),
  explore: new HttpExploreRepository(),
};
