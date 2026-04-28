import { NextRequest, NextResponse } from "next/server";
import { getGenreMap } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const HF_API_URL = process.env.HF_API_URL ?? "http://localhost:8000";
const HF_TOKEN = process.env.HF_TOKEN ?? "";

const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

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

async function searchDbMovies(q: string, limit: number): Promise<DbMovieResult[]> {
  try {
    const url = `${HF_API_URL}/api/search?q=${encodeURIComponent(q)}&limit=${limit}`;
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

async function searchTmdb(kind: "movie" | "tv" | "person", q: string) {
  if (!TMDB_BEARER) return [];
  try {
    const params = new URLSearchParams({
      query: q,
      include_adult: "false",
      language: "en-US",
      page: "1",
    });
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
  const q = (req.nextUrl.searchParams.get("q") || "").trim();
  if (!q) return NextResponse.json({ movies: [], tv: [], people: [] });

  const [dbMovies, tmdbMovies, tmdbTv, tmdbPeople, genreMap] = await Promise.all([
    searchDbMovies(q, 20),
    searchTmdb("movie", q),
    searchTmdb("tv", q),
    searchTmdb("person", q),
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
  const mergedMovies = [
    ...dbMovies.map((m) => ({ ...m, source: "db" as const })),
    ...fallback,
  ];

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

  return NextResponse.json({ movies: mergedMovies, tv, people });
}
