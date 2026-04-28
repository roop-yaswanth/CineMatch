const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

/**
 * Strict positive-integer parser for IDs coming off the URL.
 * Accepts "12345" only — rejects "12.5", "0x10", "1e5", "-3", "Infinity", "".
 * Returns null on any invalid input so callers can 400 cleanly.
 */
export function parseTmdbId(raw: string | null): number | null {
  if (!raw) return null;
  if (!/^[1-9][0-9]{0,9}$/.test(raw)) return null;
  const n = Number(raw);
  if (!Number.isSafeInteger(n) || n <= 0) return null;
  return n;
}

/**
 * Bound-checked positive integer (e.g. for `page` params).
 */
export function parseBoundedInt(raw: string | null, min: number, max: number, fallback: number): number {
  if (!raw) return fallback;
  if (!/^[0-9]{1,5}$/.test(raw)) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < min || n > max) return fallback;
  return n;
}

/**
 * Trim + length-cap a free-text query parameter. Strips control chars.
 */
export function sanitizeQuery(raw: string | null, maxLen = 200): string {
  if (!raw) return "";
  // eslint-disable-next-line no-control-regex
  return raw.replace(/[\x00-\x1F\x7F]/g, "").trim().slice(0, maxLen);
}

/**
 * Build a Cache-Control header for proxied TMDB responses.
 * - `s-maxage`: edge/CDN cache TTL
 * - `stale-while-revalidate`: serve stale while async refresh
 *
 * Browsers also honor s-maxage as max-age for shared caches; we explicitly set
 * `public` so Vercel's edge can cache user-agnostic data.
 */
export function tmdbCacheHeaders(sMaxAgeSeconds: number, swrSeconds = sMaxAgeSeconds * 24): HeadersInit {
  return {
    "Cache-Control": `public, s-maxage=${sMaxAgeSeconds}, stale-while-revalidate=${swrSeconds}`,
  };
}

export const REGION_TO_ISO: Record<string, string> = {
  India: "IN",
  USA: "US",
  Canada: "CA",
  UK: "GB",
};

interface TmdbGenre { id: number; name: string }

interface GenreCache { map: Record<number, string>; expires: number }

const GENRE_TTL_MS = 24 * 60 * 60 * 1000;
let movieGenreCache: GenreCache | null = null;
let movieGenrePromise: Promise<Record<number, string>> | null = null;

export async function getGenreMap(kind: "movie" | "tv" = "movie"): Promise<Record<number, string>> {
  if (kind !== "movie") return {}; // extend later
  const now = Date.now();
  if (movieGenreCache && movieGenreCache.expires > now) return movieGenreCache.map;
  if (movieGenrePromise) return movieGenrePromise;
  if (!TMDB_BEARER) return {};

  movieGenrePromise = (async () => {
    try {
      const res = await fetch(
        "https://api.themoviedb.org/3/genre/movie/list?language=en",
        { headers: TMDB_HEADERS, next: { revalidate: 86400 } }
      );
      if (!res.ok) return {};
      const data = await res.json();
      const map: Record<number, string> = {};
      for (const g of (data.genres || []) as TmdbGenre[]) map[g.id] = g.name;
      movieGenreCache = { map, expires: now + GENRE_TTL_MS };
      return map;
    } catch {
      return {};
    } finally {
      movieGenrePromise = null;
    }
  })();
  return movieGenrePromise;
}

export async function getGenreList(kind: "movie" | "tv" = "movie"): Promise<TmdbGenre[]> {
  const map = await getGenreMap(kind);
  return Object.entries(map).map(([id, name]) => ({ id: Number(id), name }));
}
