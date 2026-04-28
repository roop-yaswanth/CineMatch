import { NextRequest, NextResponse } from "next/server";
import { getGenreMap, sanitizeQuery, tmdbCacheHeaders } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const HF_API_URL = process.env.HF_API_URL ?? "http://localhost:8000";
const HF_TOKEN = process.env.HF_TOKEN ?? "";

const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

// ── Language keyword detection ────────────────────────────────────────────────
// Maps common language names / demonyms to TMDB ISO 639-1 codes.
const LANG_KEYWORD_MAP: Record<string, string> = {
  telugu: "te",
  hindi: "hi",
  tamil: "ta",
  malayalam: "ml",
  kannada: "kn",
  marathi: "mr",
  bengali: "bn",
  gujarati: "gu",
  punjabi: "pa",
  urdu: "ur",
  korean: "ko",
  japanese: "ja",
  chinese: "zh",
  mandarin: "zh",
  cantonese: "cn",
  french: "fr",
  spanish: "es",
  german: "de",
  italian: "it",
  portuguese: "pt",
  russian: "ru",
  arabic: "ar",
  turkish: "tr",
  thai: "th",
  indonesian: "id",
  persian: "fa",
  farsi: "fa",
  swedish: "sv",
  danish: "da",
  dutch: "nl",
  polish: "pl",
  ukrainian: "uk",
  greek: "el",
  hebrew: "he",
  english: "en",
};

/**
 * Detects a language keyword and/or 4-digit year anywhere in the query string.
 * Returns { cleanQuery, langCode, year } — cleanQuery has keywords/year removed.
 */
function extractLangFromQuery(raw: string): { cleanQuery: string; langCode: string | null; year: number | null } {
  let working = raw;
  let langCode: string | null = null;
  let year: number | null = null;

  // Language keyword detection
  const lower = working.toLowerCase();
  const tokens = lower.split(/\s+/);
  for (const token of tokens) {
    const code = LANG_KEYWORD_MAP[token];
    if (code) {
      langCode = code;
      working = working.replace(new RegExp(`\\b${token}\\b`, "gi"), "").replace(/\s{2,}/g, " ").trim();
      break;
    }
  }

  // Year detection (1888–2099)
  const yearMatch = working.match(/\b(18[89]\d|19\d{2}|20\d{2})\b/);
  if (yearMatch) {
    year = parseInt(yearMatch[1], 10);
    working = working.replace(yearMatch[0], "").replace(/\s{2,}/g, " ").trim();
  }

  return { cleanQuery: working || raw, langCode, year };
}

// ─────────────────────────────────────────────────────────────────────────────

interface DbMovieResult {
  tmdb_id: number;
  title: string;
  year?: number;
  original_language?: string;
  poster_path?: string;
  backdrop_path?: string;
  imdb_rating?: number;
  imdb_votes?: number;
  genres?: string[];
  overview?: string;
}

interface TmdbMovie {
  id: number;
  title?: string;
  original_title?: string;
  release_date?: string;
  poster_path?: string | null;
  backdrop_path?: string | null;
  overview?: string;
  original_language?: string;
  vote_average?: number;
  vote_count?: number;
  genre_ids?: number[];
}

interface TmdbTv {
  id: number;
  name?: string;
  original_name?: string;
  first_air_date?: string;
  poster_path?: string | null;
  backdrop_path?: string | null;
  overview?: string;
  original_language?: string;
  vote_average?: number;
  vote_count?: number;
  genre_ids?: number[];
}

interface TmdbPerson {
  id: number;
  name: string;
  profile_path?: string | null;
  known_for_department?: string;
  popularity?: number;
  known_for?: Array<{ id: number; title?: string; name?: string; media_type?: string; poster_path?: string | null }>;
}

async function searchDbMovies(q: string, limit: number, langCode?: string | null): Promise<DbMovieResult[]> {
  try {
    const params = new URLSearchParams({ q, limit: String(limit) });
    if (langCode) params.set("language", langCode);
    const url = `${HF_API_URL}/api/search?${params.toString()}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(HF_TOKEN ? { Authorization: `Bearer ${HF_TOKEN}` } : {}),
    };
    const res = await fetch(url, { headers });
    if (!res.ok) return [];
    const data = await res.json();
    return (data.results || []) as DbMovieResult[];
  } catch {
    return [];
  }
}

async function searchTmdb(kind: "movie" | "tv" | "person", q: string, langCode?: string | null, year?: number | null) {
  if (!TMDB_BEARER) return [];
  try {
    const params = new URLSearchParams({
      query: q,
      include_adult: "false",
      language: "en-US",
      page: "1",
    });
    // TMDB supports with_original_language for movie/tv searches
    if (langCode && kind !== "person") {
      params.set("with_original_language", langCode);
    }
    // TMDB supports year filtering for movies, first_air_date_year for TV
    if (year && kind === "movie") params.set("year", String(year));
    if (year && kind === "tv") params.set("first_air_date_year", String(year));
    const res = await fetch(
      `https://api.themoviedb.org/3/search/${kind}?${params.toString()}`,
      { headers: TMDB_HEADERS, next: { revalidate: 600 } }
    );
    if (!res.ok) return [];
    const data = await res.json();
    return (data.results || []) as unknown[];
  } catch {
    return [];
  }
}

export async function GET(req: NextRequest) {
  const rawQ = sanitizeQuery(req.nextUrl.searchParams.get("q"), 500);
  if (!rawQ) return NextResponse.json({ movies: [], tv: [], people: [] });

  // Detect and strip language keyword + year (e.g. "rebel telugu 2012" → query="rebel", lang="te", year=2012)
  const { cleanQuery: q, langCode, year } = extractLangFromQuery(rawQ);

  const [dbMovies, tmdbMovies, tmdbTv, tmdbPeople, genreMap] = await Promise.all([
    searchDbMovies(q, 20, langCode),
    searchTmdb("movie", q, langCode, year),
    searchTmdb("tv", q, langCode, year),
    // People search: still use full raw query so language/year words don't break actor lookups
    searchTmdb("person", rawQ),
    getGenreMap(),
  ]);

  // Merge movies: DB first, supplement with TMDB hits not already present
  const seen = new Set<number>(dbMovies.map((m) => m.tmdb_id));
  const fallback = (tmdbMovies as TmdbMovie[])
    .filter((m) => !seen.has(m.id))
    .map((m) => {
      const dateStr = m.release_date || "";
      const year = dateStr ? parseInt(dateStr.slice(0, 4), 10) || undefined : undefined;
      const gn = (m.genre_ids || []).map((id) => genreMap[id]).filter((n): n is string => Boolean(n));
      return {
        tmdb_id: m.id,
        title: m.title || m.original_title || "",
        year,
        original_language: m.original_language,
        poster_path: m.poster_path || undefined,
        backdrop_path: m.backdrop_path || undefined,
        overview: m.overview,
        genres: gn,
        primary_genre: gn[0],
        vote_average: m.vote_average,
        source: "tmdb" as const,
      };
    });

  let mergedMovies = [
    ...dbMovies.map((m) => ({ ...m, source: "db" as const })),
    ...fallback,
  ];

  // If a language was detected, boost same-language results to the top.
  // Results that DON'T match the target language are demoted to the end.
  if (langCode) {
    const matching = mergedMovies.filter((m) => m.original_language === langCode);
    const others = mergedMovies.filter((m) => m.original_language !== langCode);
    mergedMovies = [...matching, ...others];
  }
  // Year boost: float ±1 year matches to the very top within language group
  if (year) {
    const exact = mergedMovies.filter((m) => m.year && Math.abs(m.year - year) <= 1);
    const other = mergedMovies.filter((m) => !m.year || Math.abs(m.year - year) > 1);
    mergedMovies = [...exact, ...other];
  }

  const tv = (tmdbTv as TmdbTv[]).map((t) => {
    const dateStr = t.first_air_date || "";
    const year = dateStr ? parseInt(dateStr.slice(0, 4), 10) || undefined : undefined;
    const gn = (t.genre_ids || []).map((id) => genreMap[id]).filter((n): n is string => Boolean(n));
    return {
      tmdb_id: t.id,
      name: t.name || t.original_name || "",
      year,
      original_language: t.original_language,
      poster_path: t.poster_path || undefined,
      backdrop_path: t.backdrop_path || undefined,
      overview: t.overview,
      genres: gn,
      vote_average: t.vote_average,
    };
  });

  const people = (tmdbPeople as TmdbPerson[]).map((p) => ({
    tmdb_id: p.id,
    name: p.name,
    profile_path: p.profile_path || undefined,
    known_for_department: p.known_for_department,
    popularity: p.popularity,
    known_for: (p.known_for || []).slice(0, 4).map((k) => ({
      id: k.id,
      title: k.title || k.name || "",
      media_type: k.media_type,
      poster_path: k.poster_path || undefined,
    })),
  }));

  return NextResponse.json(
    { movies: mergedMovies, tv, people },
    { headers: tmdbCacheHeaders(600) }
  );
}
