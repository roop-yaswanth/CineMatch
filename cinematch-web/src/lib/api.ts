/* ─── CineMatch API Client ─────────────────────────────────────
 *  Talks to the FastAPI backend running on Cloud via the
 *  Next.js proxy (/api/* → backend). This avoids CORS entirely
 *  because the browser only ever talks to localhost.
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
  overview?: string;
  genres?: string[];
  vote_average?: number;
  score?: number;
  reason?: string;
  original_language?: string;
  director?: string;
  imdb_rating?: number;
  primary_genre?: string;
}

export interface RecommendationPage {
  session: UserSession;
  movies: Recommendation[];
  status: string;
  total_pool_size: number;
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
  "East Asia": ["ja", "ko", "zh"],
  "South-East Asia": ["th", "id"],
  "Middle-East": ["ar", "fa", "tr"],
  Africa: ["ar", "en", "fr"],
  Other: ["en"],
};

export const LANGUAGE_LABELS: Record<string, string> = {
  ar: "Arabic", bn: "Bengali", cn: "Chinese", da: "Danish",
  de: "German", el: "Greek", en: "English", es: "Spanish",
  fa: "Persian", fi: "Finnish", fr: "French", he: "Hebrew",
  hi: "Hindi", id: "Indonesian", it: "Italian", ja: "Japanese",
  kn: "Kannada", ko: "Korean", ml: "Malayalam", mr: "Marathi",
  nl: "Dutch", no: "Norwegian", pl: "Polish", pt: "Portuguese",
  ro: "Romanian", ru: "Russian", sv: "Swedish", ta: "Tamil",
  te: "Telugu", th: "Thai", tr: "Turkish", uk: "Ukrainian",
  ur: "Urdu", zh: "Chinese",
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
  return {
    languages:
      profile?.preferred_languages?.filter(Boolean) && profile.preferred_languages.length > 0
        ? profile.preferred_languages
        : ["en"],
    genres: profile?.preferred_genres ?? [],
    semantic_index: "tmdb_bge",
    include_classics: profile?.include_classics ?? false,
    age_group: profile?.age_group ?? "18-24",
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
  rating: string
): Promise<OnboardingState> {
  return request<OnboardingState>("/api/onboarding/rate", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      tmdb_id: tmdbId,
      rating,
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

export async function apiGenerateRecommendations(
  sessionId: string,
  preferences?: {
    age_group?: string;
    region?: string;
    languages?: string[];
    genres?: string[];
    include_classics?: boolean;
    semantic_index?: string;
  }
): Promise<RecommendationPage> {
  return request<RecommendationPage>("/api/recommendations", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, ...preferences }),
  });
}

export async function apiRecommendationAction(
  sessionId: string,
  tmdbId: number,
  action: string
): Promise<RecommendationPage> {
  return request<RecommendationPage>("/api/recommendations/action", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      tmdb_id: tmdbId,
      action,
    }),
  });
}

export async function apiGetHistory(
  sessionId: string
): Promise<HistoryItem[]> {
  return request<HistoryItem[]>(`/api/history?session_id=${sessionId}`);
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

export function posterUrl(path: string | null | undefined, size = "w500"): string {
  if (!path) return "/poster_placeholder.svg";
  if (path.startsWith("http")) return path;
  return `https://image.tmdb.org/t/p/${size}${path}`;
}

/* ─── TMDB Poster Fallback ──────────────────────── */

const posterCache: Record<number, string | null> = {};
const pendingFetches: Record<number, Promise<string | null>> = {};

export async function fetchTmdbPoster(tmdbId: number): Promise<string | null> {
  if (tmdbId in posterCache) return posterCache[tmdbId];
  if (tmdbId in pendingFetches) return pendingFetches[tmdbId];

  const promise = (async () => {
    try {
      const res = await fetch(`/api/tmdb?id=${tmdbId}`);
      if (!res.ok) {
        posterCache[tmdbId] = null;
        return null;
      }
      const data = await res.json();
      const path = data.poster_path || null;
      posterCache[tmdbId] = path;
      return path;
    } catch {
      posterCache[tmdbId] = null;
      return null;
    } finally {
      delete pendingFetches[tmdbId];
    }
  })();

  pendingFetches[tmdbId] = promise;
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
  // Preload images into browser cache
  for (const m of movies) {
    const path = m.poster_path || posterCache[m.tmdb_id ?? m.id];
    if (path) {
      const img = new window.Image();
      img.src = posterUrl(path, "w500");
    }
  }
}
