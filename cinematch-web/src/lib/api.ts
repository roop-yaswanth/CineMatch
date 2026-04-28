/* ─── CineMatch API Client ─────────────────────────────────────
 * ──────────────────────────────────────────────────────────── */

const API_BASE = "";
const REQUEST_TIMEOUT_MS = 30_000;
const MAX_RETRIES = 2;

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${path}`;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const res = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      // Retry on transient server errors
      if ((res.status === 502 || res.status === 503 || res.status === 504) && attempt < MAX_RETRIES) {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
        continue;
      }

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API ${res.status}: ${text}`);
      }
      return res.json();
    } catch (err) {
      clearTimeout(timeoutId);
      if (err instanceof DOMException && err.name === "AbortError") {
        if (attempt < MAX_RETRIES) {
          await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
          continue;
        }
        throw new Error("Request timed out. Please try again.");
      }
      throw err;
    }
  }

  throw new Error("Request failed after retries.");
}

/* ─── Types ─────────────────────────────────────────────────── */

export interface Movie {
  id: number;
  tmdb_id?: number;
  title: string;
  original_title?: string;
  year?: number;
  poster_path?: string;
  backdrop_path?: string;
  overview?: string;
  original_language?: string;
  genres?: string[];
  primary_genre?: string;
  vote_average?: number;
  vote_count?: number;
  director?: string;
  imdb_rating?: number;
  runtime?: number;
}

export interface UserProfile {
  preferred_languages?: string[];
  preferred_genres?: string[];
  genre_picks?: string[];
  include_classics?: boolean;
  age_group?: string;
  region?: string;
}

export interface UserSession {
  session_id: string;
  user_id: string;
  identifier: string;
  is_returning: boolean;
  profile: UserProfile;
  onboarding_complete: boolean;
  onboarding_index: number;
  onboarding_total: number;
  onboarding_likes: number;
  min_likes_needed: number;
  has_recommendations: boolean;
}

export interface OnboardingState {
  session: UserSession;
  movie: Movie | null;
  feedback_counts: Record<string, number>;
  is_complete: boolean;
  is_ready: boolean;
}

export interface Recommendation {
  id: number;
  tmdb_id?: number;
  title: string;
  year?: number;
  poster_path?: string;
  backdrop_path?: string;
  overview?: string;
  genres?: string[];
  vote_average?: number;
  score?: number;
  reason?: string;
  original_language?: string;
  director?: string;
  imdb_rating?: number;
  imdb_votes?: number;
  vote_count?: number;
  runtime?: number;
  primary_genre?: string;
}

export interface RecommendationPage {
  session: UserSession;
  movies: Recommendation[];
  status: string;
  total_pool_size: number;
}

/** Response shape for /api/recommendations/multi */
export interface MultiBucketResponse {
  session: UserSession;
  buckets: {
    english: Recommendation[];
    /** lang code → movies (e.g. { te: [...], hi: [...] }) */
    regional: Record<string, Recommendation[]>;
    global: Recommendation[];
  };
  total_pool_size: number;
  status: string;
  errors?: Record<string, string> | null;
}



export interface RecommendationPreferences {
  languages: string[];
  genres: string[];
  semantic_index: string;
  include_classics: boolean;
  age_group: string;
  region: string;
}

export interface HistoryItem {
  tmdb_id: number;
  title: string;
  poster_path?: string;
  rating: string;
  context: "onboarding" | "recommendation";
  year?: number;
  original_language?: string;
  primary_genre?: string;
}

/* ─── Constants ─────────────────────────────────────────────── */

export const REGION_OPTIONS = [
  "India", "USA", "Canada", "UK", "Europe", "Latin-America",
  "East Asia", "South-East Asia", "Middle-East", "Africa", "Other",
] as const;

export const AGE_GROUP_OPTIONS = [
  "18-24", "25-34", "35-44", "45-54", "55+", "Prefer not to say",
] as const;

export const REGION_LANGUAGE_MAP: Record<string, string[]> = {
  India: ["hi", "te", "ta", "ml", "kn"],
  USA: ["en"],
  Canada: ["en", "fr"],
  UK: ["en"],
  Europe: ["fr", "de", "it", "es"],
  "Latin-America": ["es", "pt"],
  "East Asia": ["ja", "ko", "zh", "cn"],
  "South-East Asia": ["th", "id"],
  "Middle-East": ["ar", "fa", "tr"],
  Africa: ["ar", "en", "fr"],
  Other: ["en"],
};

export const LANGUAGE_LABELS: Record<string, string> = {
  ar: "Arabic", bn: "Bengali", cn: "Cantonese", da: "Danish",
  de: "German", el: "Greek", en: "English", es: "Spanish",
  fa: "Persian", fi: "Finnish", fr: "French", he: "Hebrew",
  hi: "Hindi", id: "Indonesian", it: "Italian", ja: "Japanese",
  kn: "Kannada", ko: "Korean", ml: "Malayalam", mr: "Marathi",
  nl: "Dutch", no: "Norwegian", pl: "Polish", pt: "Portuguese",
  ro: "Romanian", ru: "Russian", sv: "Swedish", ta: "Tamil",
  te: "Telugu", th: "Thai", tr: "Turkish", uk: "Ukrainian",
  ur: "Urdu", zh: "Mandarin",
};

export function languageLabel(code: string): string {
  if (!code) return "Unknown";
  return LANGUAGE_LABELS[code.toLowerCase()] || code.toUpperCase();
}

export function recommendationId(
  movie: Pick<Movie, "id" | "tmdb_id"> | Pick<Recommendation, "id" | "tmdb_id">
): number {
  return movie.tmdb_id ?? movie.id;
}

export function preferencesFromProfile(
  profile?: UserProfile | null
): RecommendationPreferences {
  const savedGenres = profile?.preferred_genres ?? profile?.genre_picks ?? [];
  return {
    languages: profile?.preferred_languages?.filter(Boolean) ?? [],
    genres: savedGenres.filter(Boolean),
    semantic_index: "tmdb_bge",
    include_classics: profile?.include_classics ?? false,
    age_group: profile?.age_group ?? "25-34",
    region: profile?.region ?? "USA",
  };
}

export function regionLanguages(region?: string): string[] {
  return REGION_LANGUAGE_MAP[region || "Other"] ?? ["en"];
}

/* ─── Endpoints ─────────────────────────────────────────────── */

export async function apiLogin(email: string): Promise<UserSession> {
  return request<UserSession>("/api/login", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export async function apiBuildSlate(
  sessionId: string,
  preferences: {
    age_group?: string;
    region?: string;
    languages?: string[];
    genres?: string[];
    include_classics?: boolean;
    semantic_index?: string;
  }
): Promise<OnboardingState> {
  return request<OnboardingState>("/api/onboarding/slate", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, ...preferences }),
  });
}

export async function apiRateOnboarding(
  sessionId: string,
  tmdbId: number,
  rating: string,
  dwellMs: number = 0
): Promise<OnboardingState> {
  return request<OnboardingState>("/api/onboarding/rate", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      tmdb_id: tmdbId,
      rating,
      dwell_ms: dwellMs,
    }),
  });
}

export async function apiOnboardingNav(
  sessionId: string,
  direction: "prev" | "next"
): Promise<OnboardingState> {
  return request<OnboardingState>("/api/onboarding/nav", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, direction }),
  });
}

export async function apiUndoOnboarding(
  sessionId: string
): Promise<OnboardingState> {
  return request<OnboardingState>("/api/onboarding/undo", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function apiEscapeObscure(
  sessionId: string
): Promise<OnboardingState> {
  return request<OnboardingState>("/api/onboarding/escape_obscure", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function apiGenerateRecommendations(
  sessionId: string,
  preferences?: {
    age_group?: string;
    region?: string;
    languages?: string[];
    genres?: string[];
    include_classics?: boolean;
    semantic_index?: string;
    update_profile?: boolean;
  }
): Promise<RecommendationPage> {
  return request<RecommendationPage>("/api/recommendations", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, ...preferences }),
  });
}

export async function apiMultiRecommendations(
  sessionId: string,
  preferences: {
    languages: string[];
    genres: string[];
    age_group: string;
    region: string;
    include_classics: boolean;
    semantic_index: string;
    per_bucket_k?: number;
    exclude_ids?: number[];
  }
): Promise<MultiBucketResponse> {
  return request<MultiBucketResponse>("/api/recommendations/multi", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, ...preferences }),
  });
}



export async function apiRecommendationAction(
  sessionId: string,
  tmdbId: number,
  action: string,
  dwellMs: number = 0
): Promise<RecommendationPage> {
  return request<RecommendationPage>("/api/recommendations/action", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      tmdb_id: tmdbId,
      action,
      dwell_ms: dwellMs,
    }),
  });
}

export async function apiGetHistory(
  sessionId: string
): Promise<HistoryItem[]> {
  return request<HistoryItem[]>(`/api/history?session_id=${sessionId}`);
}

export interface SearchResult {
  tmdb_id: number;
  title: string;
  year?: number;
  original_language: string;
  poster_path?: string;
  backdrop_path?: string;
  imdb_rating?: number;
  imdb_votes?: number;
  genres: string[];
  overview?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
}

export interface MultiSearchMovie extends SearchResult {
  primary_genre?: string;
  vote_average?: number;
  source: "db" | "tmdb";
}
export interface MultiSearchTv {
  tmdb_id: number;
  name: string;
  year?: number;
  original_language?: string;
  poster_path?: string;
  backdrop_path?: string;
  overview?: string;
  genres: string[];
  vote_average?: number;
}
export interface MultiSearchPerson {
  tmdb_id: number;
  name: string;
  profile_path?: string;
  known_for_department?: string;
  popularity?: number;
  known_for: Array<{ id: number; title: string; media_type?: string; poster_path?: string }>;
}
export interface MultiSearchResponse {
  movies: MultiSearchMovie[];
  tv: MultiSearchTv[];
  people: MultiSearchPerson[];
}

// Tiny LRU keyed by query string. Multi-search is heavy (5 parallel TMDB calls
// per request); when the user types-deletes-types the same query we want an
// instant response. Cap is small because each entry can hold ~80 result objects.
const MULTI_SEARCH_CACHE_MAX = 30;
const multiSearchCache = new Map<string, MultiSearchResponse>();

/** Synchronous read of the multi-search cache. Returns null on miss.
 *  Used by the search page to render instantly when the user re-types a
 *  recent query — no async boundary, no flash of empty state. */
export function peekMultiSearchCache(query: string): MultiSearchResponse | null {
  const key = query.trim().toLowerCase();
  return multiSearchCache.get(key) ?? null;
}

export async function apiSearchMulti(query: string): Promise<MultiSearchResponse> {
  const key = query.trim().toLowerCase();
  const hit = multiSearchCache.get(key);
  if (hit) {
    // Refresh recency by re-inserting (Map preserves insertion order).
    multiSearchCache.delete(key);
    multiSearchCache.set(key, hit);
    return hit;
  }
  const res = await fetch(`/api/search/multi?q=${encodeURIComponent(query)}`);
  if (!res.ok) return { movies: [], tv: [], people: [] };
  const data: MultiSearchResponse = await res.json();
  multiSearchCache.set(key, data);
  if (multiSearchCache.size > MULTI_SEARCH_CACHE_MAX) {
    // Evict oldest entry.
    const oldest = multiSearchCache.keys().next().value;
    if (oldest !== undefined) multiSearchCache.delete(oldest);
  }
  return data;
}

export async function apiSearchMovies(
  query: string,
  limit: number = 20
): Promise<SearchResponse> {
  return request<SearchResponse>(
    `/api/search?q=${encodeURIComponent(query)}&limit=${limit}`
  );
}

export async function apiUpdatePreferences(
  sessionId: string,
  preferences: {
    languages?: string[];
    genres?: string[];
    semantic_index?: string;
  }
): Promise<UserSession> {
  return request<UserSession>("/api/preferences", {
    method: "PUT",
    body: JSON.stringify({ session_id: sessionId, ...preferences }),
  });
}

// Similar-movies cache. Keyed by (tmdbId, sessionId or "anon", n) since results
// are personalized when a session is provided.
const SIMILAR_CACHE_MAX = 200;
const similarCache = new Map<string, Recommendation[]>();
const similarInflight = new Map<string, Promise<Recommendation[]>>();

export async function apiSimilarMovies(
  tmdbId: number,
  sessionId?: string | null,
  n = 10
): Promise<Recommendation[]> {
  const key = `${tmdbId}:${sessionId ?? "anon"}:${n}`;
  const cached = similarCache.get(key);
  if (cached) return cached;
  const inflight = similarInflight.get(key);
  if (inflight) return inflight;
  const p = (async () => {
    try {
      const params = new URLSearchParams({ tmdb_id: String(tmdbId), n: String(n) });
      if (sessionId) params.set("session_id", sessionId);
      const data = await request<{ results: Recommendation[] }>(
        `/api/movies/similar?${params.toString()}`
      );
      const results = data.results ?? [];
      similarCache.set(key, results);
      if (similarCache.size > SIMILAR_CACHE_MAX) {
        const oldest = similarCache.keys().next().value;
        if (oldest !== undefined) similarCache.delete(oldest);
      }
      return results;
    } finally {
      similarInflight.delete(key);
    }
  })();
  similarInflight.set(key, p);
  return p;
}

export type ExploreCategory =
  | "trending_day"
  | "trending_week"
  | "popular"
  | "top_rated"
  | "now_playing"
  | "upcoming";

export interface ExploreMovie {
  id: number;
  tmdb_id: number;
  title: string;
  original_title?: string;
  year?: number;
  release_date?: string | null;
  poster_path?: string;
  backdrop_path?: string;
  overview?: string;
  original_language?: string;
  vote_average?: number;
  vote_count?: number;
  genres?: string[];
  primary_genre?: string;
}

export interface ExploreResponse {
  results: ExploreMovie[];
  page: number;
  total_pages: number;
}

export type DiscoverSort =
  | "popularity.desc"
  | "popularity.asc"
  | "vote_average.desc"
  | "vote_average.asc"
  | "primary_release_date.desc"
  | "primary_release_date.asc"
  | "revenue.desc"
  | "title.asc";

export interface DiscoverFilters {
  sort_by?: DiscoverSort;
  with_genres?: number[];
  year_from?: number;
  year_to?: number;
  with_original_language?: string;
  vote_average_gte?: number;
  vote_count_gte?: number;
  region?: string;
  page?: number;
}

export async function apiDiscover(filters: DiscoverFilters): Promise<ExploreResponse> {
  const params = new URLSearchParams();
  if (filters.sort_by) params.set("sort_by", filters.sort_by);
  if (filters.with_genres?.length) params.set("with_genres", filters.with_genres.join(","));
  if (filters.year_from) params.set("year_from", String(filters.year_from));
  if (filters.year_to) params.set("year_to", String(filters.year_to));
  if (filters.with_original_language) params.set("with_original_language", filters.with_original_language);
  if (filters.vote_average_gte != null) params.set("vote_average_gte", String(filters.vote_average_gte));
  if (filters.vote_count_gte != null) params.set("vote_count_gte", String(filters.vote_count_gte));
  if (filters.region) params.set("region", filters.region);
  params.set("page", String(filters.page ?? 1));
  const res = await fetch(`/api/tmdb/discover?${params.toString()}`);
  if (!res.ok) throw new Error(`Discover fetch failed: ${res.status}`);
  return res.json();
}

export interface CastMember {
  id: number;
  name: string;
  character?: string | null;
  profile_path?: string | null;
}
export interface CrewMember {
  id: number;
  name: string;
  job?: string;
  profile_path?: string | null;
}
export interface CreditsResponse {
  cast: CastMember[];
  directors: CrewMember[];
  writers: CrewMember[];
}

export interface PersonCredit {
  id: number;
  tmdb_id: number;
  title: string;
  year?: number;
  release_date?: string | null;
  poster_path?: string;
  media_type: "movie" | "tv";
  character?: string | null;
  job?: string | null;
  department?: string | null;
  vote_average?: number;
  popularity?: number;
  overview?: string;
}

export interface PersonDetail {
  id: number;
  name: string;
  biography: string;
  birthday: string | null;
  deathday: string | null;
  place_of_birth: string | null;
  gender: number;
  known_for_department: string | null;
  profile_path: string | null;
  also_known_as: string[];
  homepage: string | null;
  imdb_id: string | null;
  cast: PersonCredit[];
  crew: PersonCredit[];
  known_for: PersonCredit[];
}

export async function apiPerson(personId: number): Promise<PersonDetail | null> {
  const res = await fetch(`/api/tmdb/person?id=${personId}`);
  if (!res.ok) return null;
  return res.json();
}

// Per-movie credits cache. Server already caches at the edge; this avoids
// re-parsing the JSON on repeat opens of the same movie modal.
const CREDITS_CACHE_MAX = 200;
const creditsCache = new Map<string, CreditsResponse>();
const creditsInflight = new Map<string, Promise<CreditsResponse>>();

export async function apiCredits(tmdbId: number, kind: "movie" | "tv" = "movie"): Promise<CreditsResponse> {
  const key = `${kind}:${tmdbId}`;
  const cached = creditsCache.get(key);
  if (cached) return cached;
  const inflight = creditsInflight.get(key);
  if (inflight) return inflight;
  const p = (async () => {
    try {
      const res = await fetch(`/api/tmdb/credits?id=${tmdbId}&kind=${kind}`);
      if (!res.ok) return { cast: [], directors: [], writers: [] };
      const data: CreditsResponse = await res.json();
      creditsCache.set(key, data);
      if (creditsCache.size > CREDITS_CACHE_MAX) {
        const oldest = creditsCache.keys().next().value;
        if (oldest !== undefined) creditsCache.delete(oldest);
      }
      return data;
    } finally {
      creditsInflight.delete(key);
    }
  })();
  creditsInflight.set(key, p);
  return p;
}

/**
 * Fire-and-forget warmer used on hover. Kicks off the credits + similar fetches
 * so the modal opens with data ready. Safe to call repeatedly — both targets
 * dedupe in-flight requests.
 */
export function prefetchMovieDetails(tmdbId: number): void {
  if (!tmdbId) return;
  // No await: we just want the underlying cache to fill in the background.
  void apiCredits(tmdbId, "movie").catch(() => {});
  void apiSimilarMovies(tmdbId, null, 20).catch(() => {});
}

export interface TmdbGenre { id: number; name: string }
let genreCache: TmdbGenre[] | null = null;
export async function apiGenres(): Promise<TmdbGenre[]> {
  if (genreCache) return genreCache;
  const res = await fetch("/api/tmdb/genres");
  if (!res.ok) return [];
  const data = await res.json();
  genreCache = data.genres || [];
  return genreCache!;
}

export async function apiExplore(
  category: ExploreCategory,
  page: number = 1,
  region?: string
): Promise<ExploreResponse> {
  const params = new URLSearchParams({ category, page: String(page) });
  if (region) params.set("region", region);
  const res = await fetch(`/api/tmdb/explore?${params.toString()}`);
  if (!res.ok) throw new Error(`Explore fetch failed: ${res.status}`);
  return res.json();
}

export function posterUrl(path: string | null | undefined, size = "w500"): string {
  if (!path) return "/poster_placeholder.svg";
  if (path.startsWith("http")) return path;
  return `https://image.tmdb.org/t/p/${size}${path}`;
}

/* ─── TMDB Poster Fallback ──────────────────────── */

// LRU-capped poster cache. A long browse session (Explore + Discover paging)
// can easily touch 500+ ids; without a cap this grows unbounded for the page
// lifetime. Map preserves insertion order, which we use as recency.
const POSTER_CACHE_MAX = 500;
const posterCache = new Map<number, string | null>();
const pendingFetches = new Map<number, Promise<string | null>>();

function rememberPoster(tmdbId: number, path: string | null): void {
  posterCache.set(tmdbId, path);
  if (posterCache.size > POSTER_CACHE_MAX) {
    const oldest = posterCache.keys().next().value;
    if (oldest !== undefined) posterCache.delete(oldest);
  }
}

export async function fetchTmdbPoster(tmdbId: number): Promise<string | null> {
  if (posterCache.has(tmdbId)) {
    const v = posterCache.get(tmdbId) ?? null;
    // Touch for recency.
    posterCache.delete(tmdbId);
    posterCache.set(tmdbId, v);
    return v;
  }
  const inflight = pendingFetches.get(tmdbId);
  if (inflight) return inflight;

  const promise = (async () => {
    try {
      const res = await fetch(`/api/tmdb?id=${tmdbId}`);
      if (!res.ok) {
        rememberPoster(tmdbId, null);
        return null;
      }
      const data = await res.json();
      const path = data.poster_path || null;
      rememberPoster(tmdbId, path);
      return path;
    } catch {
      rememberPoster(tmdbId, null);
      return null;
    } finally {
      pendingFetches.delete(tmdbId);
    }
  })();

  pendingFetches.set(tmdbId, promise);
  return promise;
}

/* ─── Poster Prefetch ──────────────────────────── */

export async function prefetchPosters(
  movies: Array<{ poster_path?: string; id: number; tmdb_id?: number }>
): Promise<void> {
  const missing = movies.filter((m) => !m.poster_path);
  await Promise.allSettled(
    missing.map((m) => fetchTmdbPoster(m.tmdb_id ?? m.id))
  );
}
