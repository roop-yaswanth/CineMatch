/* ─── CineMatch API Client ─────────────────────────────────────
 *  Talks to the FastAPI backend running on Colab via the
 *  Next.js proxy (/api/* → backend). This avoids CORS entirely
 *  because the browser only ever talks to localhost.
 * ──────────────────────────────────────────────────────────── */

// Always relative — requests are proxied by next.config.ts rewrites
const API_BASE = "";

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

/* ─── Types ─────────────────────────────────────────────────── */

export interface Movie {
  id: number;
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

export interface UserSession {
  session_id: string;
  user_id: string;
  identifier: string;
  is_returning: boolean;
  profile: Record<string, unknown>;
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
    languages?: string[];
    genres?: string[];
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

export async function fetchTmdbPoster(tmdbId: number): Promise<string | null> {
  if (tmdbId in posterCache) return posterCache[tmdbId];
  try {
    const res = await fetch(`/api/tmdb?id=${tmdbId}`);
    if (!res.ok) {
      posterCache[tmdbId] = null;
      return null;
    }
    const data = await res.json();
    posterCache[tmdbId] = data.poster_path || null;
    return data.poster_path || null;
  } catch {
    posterCache[tmdbId] = null;
    return null;
  }
}

