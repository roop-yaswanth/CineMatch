/**
 * Domain types 
 * Single Reason to Change: domain model evolves (e.g., new field on Recommendation).
 * No reason to change because fetch, localStorage, or UI changed.
 */

// --- Value Objects ---
export type RatingValue = "love" | "like" | "dislike" | "skip" | "watchlist" | "not_watched";
export type LanguageCode = string;
export type GenreName = string;

// --- Entities ---
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
  imdb_id?: string;
  runtime?: number;
  release_date?: string | null;
  status?: string | null;
  certification?: string | null;
}

export interface Recommendation extends Movie {
  trend_score?: number;
  popularity?: number;
  score?: number;
  reason?: string;
  imdb_votes?: number;
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

// --- Aggregates ---
export interface UserProfile {
  preferred_languages?: string[];
  preferred_genres?: string[];
  genre_picks?: string[];
  include_classics?: boolean;
  age_group?: string;
  region?: string;
  name?: string;
}

export interface UserSession {
  session_id: string;
  user_id: string;
  identifier: string;
  name?: string;
  picture?: string;
  is_returning: boolean;
  profile: UserProfile;
  onboarding_complete: boolean;
  onboarding_index: number;
  onboarding_total: number;
  onboarding_likes: number;
  min_likes_needed: number;
  has_recommendations: boolean;
  auth_token?: string;
}

export interface RecommendationPreferences {
  languages: string[];
  genres: string[];
  semantic_index: string;
  include_classics: boolean;
  age_group: string;
  region: string;
}

export interface MultiBucketResponse {
  session: UserSession;
  buckets: {
    english: Recommendation[];
    regional: Record<string, Recommendation[]>;
    global: Recommendation[];
  };
  total_pool_size: number;
  status: string;
  errors?: Record<string, string> | null;
}

// --- Helpers (pure, explicit inputs/outputs, no hidden globals) ---
export function recommendationId(movie: Pick<Movie, "id" | "tmdb_id">): number {
  return movie.tmdb_id ?? movie.id;
}

export function preferencesFromProfile(profile?: UserProfile | null): RecommendationPreferences {
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
// --- Presentation-adjacent mappers ---

/**
 * toDetailMovie — canonical mapper from Recommendation/HistoryItem to DetailMovie shape.
 */
export interface DetailMovieShape {
  id: number;
  tmdb_id?: number;
  title: string;
  poster_path?: string;
  year?: number;
  original_language?: string;
  primary_genre?: string;
  genres?: string[];
  overview?: string;
  backdrop_path?: string;
  imdb_rating?: number;
  director?: string;
}

export function toDetailMovie(m: Partial<Movie> & { tmdb_id?: number; id?: number; title: string }): DetailMovieShape {
  return {
    id: m.tmdb_id ?? m.id ?? 0,
    tmdb_id: m.tmdb_id ?? m.id,
    title: m.title,
    poster_path: m.poster_path,
    year: m.year,
    original_language: m.original_language,
    primary_genre: m.primary_genre,
    genres: m.genres,
    overview: m.overview,
    backdrop_path: m.backdrop_path,
    imdb_rating: m.imdb_rating,
    director: m.director,
  };
}
