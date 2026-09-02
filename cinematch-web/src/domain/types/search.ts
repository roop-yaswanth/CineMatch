import type { Movie } from "./movie";

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

export interface MultiSearchMovie extends SearchResult {
  primary_genre?: string;
  vote_average?: number;
  source: "db" | "tmdb" | "imdb";
  imdb_url?: string;
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
export type MultiSearchTopItem =
  | ({ media_type: "movie" } & MultiSearchMovie)
  | ({ media_type: "tv" } & MultiSearchTv)
  | ({ media_type: "person" } & MultiSearchPerson);

export interface MultiSearchResponse {
  movies: MultiSearchMovie[];
  tv: MultiSearchTv[];
  people: MultiSearchPerson[];
  top?: MultiSearchTopItem[];
}

export interface ExploreMovie extends Movie {
  tmdb_id: number;
}

export type ExploreCategory = "trending_day" | "trending_week" | "popular" | "top_rated" | "now_playing" | "upcoming";

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
