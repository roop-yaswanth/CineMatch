/**
 * HttpMovieRepository — concrete adapter for MovieRepository ports.
 * Single Responsibility: translate domain calls into HTTP requests.
 * Depends on HttpClient abstraction, not on `fetch` directly.
 * High-level services depend on the `MovieRepository` interface, not this class.
 */

import { httpRequest } from "@/infrastructure/http/HttpClient";

// P1: bounded timeout for same-origin GETs (was bare fetch() with no timeout).
// httpRequest throws on !ok — callers below preserve their old empty/throw contract.
async function boundedGet<T>(path: string, timeoutMs = 10_000): Promise<T> {
  return httpRequest<T>(path, { timeout: timeoutMs });
}
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
    // P1: /multi can take 12-14s+ on a cold backend — must outlive the 30s proxy.
    return httpRequest<MultiBucketResponse>("/api/recommendations/multi", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, ...prefs }),
      timeout: 30000,
    });
  }
  async submitAction(sessionId: string, tmdbId: number, action: string, dwellMs = 0) {
    // P1: /action can trigger a full pool rebuild — 6s default aborted the
    // request while the server still applied it (phantom failures).
    return httpRequest<{ session: import("@/domain/types/movie").UserSession }>("/api/recommendations/action", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, tmdb_id: tmdbId, action, dwell_ms: dwellMs }),
      timeout: 30000,
    });
  }
  async getHistory(sessionId: string): Promise<HistoryItem[]> {
    // P1: header-only — never put the session id in the URL (leaks to
    // access logs, proxies, and shared-link referrers). Backend still accepts
    // ?session_id for backward compat, but clients must not send it.
    return httpRequest<HistoryItem[]>(`/api/history`, {
      headers: { "X-Session-Id": sessionId },
    });
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
    let data: MultiSearchResponse;
    try {
      data = await boundedGet<MultiSearchResponse>(`/api/search/multi?q=${encodeURIComponent(query)}`);
    } catch {
      return { movies: [], tv: [], people: [] };
    }
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
    return boundedGet<ExploreResponse>(`/api/tmdb/explore?${params.toString()}`);
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
    return boundedGet<ExploreResponse>(`/api/tmdb/discover?${params.toString()}`);
  }
}

// Singleton composition root — the only place that knows which concrete adapter is used.
// Presentation imports `movieRepositories` (the abstraction), not `fetch`.
export const movieRepositories = {
  recommendations: new HttpRecommendationRepository(),
  search: new HttpSearchRepository(),
  explore: new HttpExploreRepository(),
};
