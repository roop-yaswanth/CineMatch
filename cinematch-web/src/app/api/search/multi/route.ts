import { NextRequest, NextResponse } from "next/server";
import { getGenreMap, sanitizeQuery, tmdbCacheHeaders } from "@/lib/tmdb-server";
import { clientIp, createRateLimiter } from "@/lib/rate-limit";

// Multi-search fans out to 5 parallel upstream calls per request, so we cap
// it tightly. 30 requests / minute (0.5/s sustained) with a small burst.
const limiter = createRateLimiter(15, 0.5);

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

interface ImdbSearchResult {
  imdb_id: string;
  title: string;
  year?: number;
  type?: string;
  image?: string;
  imdb_url?: string;
}

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

// Unified shape for a movie from any source (db, tmdb, or imdb).
interface MergedMovieItem {
  tmdb_id: number;
  title: string;
  year?: number;
  original_language?: string;
  poster_path?: string;
  backdrop_path?: string;
  overview?: string;
  genres?: string[];
  primary_genre?: string;
  vote_average?: number;
  imdb_rating?: number;
  imdb_votes?: number;
  source: "db" | "tmdb" | "imdb";
  imdb_id?: string;
  imdb_url?: string;
}

async function searchImdb(q: string): Promise<ImdbSearchResult[]> {
  try {
    const url = `${HF_API_URL}/api/imdb/search?q=${encodeURIComponent(q)}&limit=8`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(HF_TOKEN ? { Authorization: `Bearer ${HF_TOKEN}` } : {}),
    };
    const res = await fetch(url, { headers });
    if (!res.ok) return [];
    const data = await res.json();
    return (data.results || []) as ImdbSearchResult[];
  } catch {
    return [];
  }
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

/**
 * Query↔title relevance in [0, ~4.5]. Exact and prefix matches dominate;
 * token overlap catches word-order variants; a small popularity term breaks
 * ties. Used to rank ALL sources on equal footing — a junk fuzzy match from
 * the library CSV must never outrank an exact TMDB movie/TV hit.
 */
function relevanceScore(query: string, title: string, votes = 0, extra = 0): number {
  const q = query.toLowerCase().trim();
  const t = (title || "").toLowerCase().trim();
  if (!q || !t) return 0;
  let score = 0;
  if (t === q) score = 3;
  else if (t.startsWith(q) || q.startsWith(t)) score = 2.2;
  else if (t.includes(q)) score = 1.6;
  else {
    const qTokens = new Set(q.split(/\s+/));
    const tTokens = new Set(t.split(/\s+/));
    let hits = 0;
    for (const tok of qTokens) if (tTokens.has(tok)) hits++;
    score = qTokens.size ? (hits / qTokens.size) * 1.4 : 0;
  }
  // log-scaled vote count tie-break, capped at 0.5
  score += Math.min(0.5, Math.log1p(votes) / Math.log1p(500000) * 0.5);
  return score + extra;
}

export async function GET(req: NextRequest) {
  const rl = limiter.check(clientIp(req));
  if (!rl.allowed) {
    return NextResponse.json(
      { error: "Too many requests" },
      { status: 429, headers: { "Retry-After": String(rl.retryAfterSeconds) } }
    );
  }

  const rawQ = sanitizeQuery(req.nextUrl.searchParams.get("q"), 500);
  if (!rawQ) return NextResponse.json({ movies: [], tv: [], people: [] });

  // Detect and strip language keyword + year (e.g. "rebel telugu 2012" → query="rebel", lang="te", year=2012)
  const { cleanQuery: q, langCode, year } = extractLangFromQuery(rawQ);

  const [dbMovies, tmdbMovies, tmdbTv, tmdbPeople, genreMap, imdbMovies] = await Promise.all([
    searchDbMovies(q, 20, langCode),
    searchTmdb("movie", q, langCode, year),
    searchTmdb("tv", q, langCode, year),
    // People search: still use full raw query so language/year words don't break actor lookups
    searchTmdb("person", rawQ),
    getGenreMap(),
    searchImdb(q),
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

  let mergedMovies: MergedMovieItem[] = [
    ...dbMovies.map((m) => ({ ...m, source: "db" as const })),
    ...fallback,
  ];

  // Unified relevance ranking across sources. Library entries get a small
  // tie-break bonus (they carry richer metadata + IMDB ratings), and the
  // detected language / year hints act as score boosts rather than hard
  // reorderings so a strong title match still wins.
  const movieScore = (m: MergedMovieItem): number =>
    relevanceScore(
      q,
      m.title,
      m.imdb_votes ?? 0,
      (m.source === "db" ? 0.25 : 0)
        + (langCode && m.original_language === langCode ? 1.2 : 0)
        + (year && m.year && Math.abs(m.year - year) <= 1 ? 1.0 : 0)
    );
  mergedMovies.sort((a, b) => movieScore(b) - movieScore(a));

  // Merge IMDb-only results: append titles not already present by imdb_id or by title+year match.
  // IMDb results only have title/year/image — no TMDB metadata — so they open to imdb_url.
  const seenImdbIds = new Set<string>(
    dbMovies.map((m) => (m as DbMovieResult & { imdb_id?: string }).imdb_id).filter(Boolean) as string[]
  );
  const seenTitlesYears = new Set<string>(
    mergedMovies.map((m) => `${(m.title || "").toLowerCase().trim()}:${m.year ?? ""}`)
  );
  const imdbFallback = imdbMovies
    .filter((r) => {
      if (r.imdb_id && seenImdbIds.has(r.imdb_id)) return false;
      const key = `${(r.title || "").toLowerCase().trim()}:${r.year ?? ""}`;
      return !seenTitlesYears.has(key);
    })
    .map((r) => ({
      tmdb_id: 0, // no TMDB id — signals frontend to use imdb_url
      title: r.title || "",
      year: r.year,
      original_language: undefined,
      poster_path: r.image,  // Amazon image URL — posterUrl() will detect the http prefix
      backdrop_path: undefined,
      overview: undefined,
      genres: [] as string[],
      primary_genre: undefined,
      vote_average: undefined,
      imdb_id: r.imdb_id,
      imdb_url: r.imdb_url,
      source: "imdb" as const,
    }));

  mergedMovies = [...mergedMovies, ...imdbFallback];

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
      vote_count: t.vote_count,
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

  // Mixed "top results": movies, TV, and people compete on the same
  // relevance scale. This is what lets a TV-only title (absent from the
  // movie CSV) lead the page instead of drowning under fuzzy CSV matches.
  const top = [
    ...mergedMovies.slice(0, 8).map((m) => ({
      media_type: "movie" as const,
      score: movieScore(m),
      item: m as unknown as Record<string, unknown>,
    })),
    ...tv.slice(0, 6).map((t) => ({
      media_type: "tv" as const,
      score: relevanceScore(q, t.name, t.vote_count ?? 0),
      item: t as Record<string, unknown>,
    })),
    ...people.slice(0, 3).map((p) => ({
      media_type: "person" as const,
      score: relevanceScore(rawQ, p.name, (p.popularity ?? 0) * 1000),
      item: p as Record<string, unknown>,
    })),
  ]
    .sort((a, b) => b.score - a.score)
    .slice(0, 8)
    .map(({ media_type, item }) => ({ media_type, ...item }));

  return NextResponse.json(
    { movies: mergedMovies, tv, people, top },
    { headers: tmdbCacheHeaders(600) }
  );
}
