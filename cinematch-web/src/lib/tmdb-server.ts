const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

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
