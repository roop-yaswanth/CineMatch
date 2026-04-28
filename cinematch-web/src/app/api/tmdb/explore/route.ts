import { NextRequest, NextResponse } from "next/server";
import { getGenreMap, parseBoundedInt, REGION_TO_ISO, tmdbCacheHeaders } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

const CATEGORY_PATHS: Record<string, string> = {
  popular: "/movie/popular",
  top_rated: "/movie/top_rated",
  now_playing: "/movie/now_playing",
  upcoming: "/movie/upcoming",
  trending_day: "/trending/movie/day",
  trending_week: "/trending/movie/week",
};

interface TmdbMovie {
  id: number;
  title?: string;
  name?: string;
  original_title?: string;
  release_date?: string;
  first_air_date?: string;
  poster_path?: string | null;
  backdrop_path?: string | null;
  overview?: string;
  original_language?: string;
  vote_average?: number;
  vote_count?: number;
  genre_ids?: number[];
}

export async function GET(req: NextRequest) {
  const category = req.nextUrl.searchParams.get("category") || "popular";
  const page = String(parseBoundedInt(req.nextUrl.searchParams.get("page"), 1, 500, 1));
  // Region: only allow a tight allow-list of keys (REGION_TO_ISO). Anything
  // else is silently treated as no-region.
  const regionParam = req.nextUrl.searchParams.get("region") || "";
  const region = regionParam in REGION_TO_ISO ? regionParam : "";

  const path = CATEGORY_PATHS[category];
  if (!path) {
    return NextResponse.json({ error: "Invalid category" }, { status: 400 });
  }

  if (!TMDB_BEARER) {
    return NextResponse.json({ results: [], total_pages: 0, page: 1 });
  }

  const iso = REGION_TO_ISO[region];

  // For "upcoming", use /discover with release_date filter so we only get
  // movies that are actually still upcoming (TMDB's /movie/upcoming includes
  // some recently-released titles).
  let url: string;
  if (category === "upcoming") {
    const today = new Date().toISOString().slice(0, 10);
    const upcomingParams = new URLSearchParams({
      language: "en-US",
      page,
      sort_by: "popularity.desc",
      include_adult: "false",
      include_video: "false",
      "primary_release_date.gte": today,
      "with_release_type": "2|3",
    });
    if (iso) upcomingParams.set("region", iso);
    url = `https://api.themoviedb.org/3/discover/movie?${upcomingParams.toString()}`;
  } else {
    const params = new URLSearchParams({ language: "en-US", page });
    if (iso && (category === "now_playing" || category === "popular")) {
      params.set("region", iso);
    }
    url = `https://api.themoviedb.org/3${path}?${params.toString()}`;
  }

  try {
    const [res, genreMap] = await Promise.all([
      fetch(url, { headers: TMDB_HEADERS, next: { revalidate: 3600 } }),
      getGenreMap(),
    ]);
    if (!res.ok) {
      return NextResponse.json({ error: "TMDB fetch failed" }, { status: 502 });
    }
    const data = await res.json();
    const results = (data.results || []).map((m: TmdbMovie) => {
      const dateStr = m.release_date || m.first_air_date || "";
      const year = dateStr ? parseInt(dateStr.slice(0, 4), 10) || undefined : undefined;
      const genres = (m.genre_ids || [])
        .map((id) => genreMap[id])
        .filter((n): n is string => Boolean(n));
      return {
        id: m.id,
        tmdb_id: m.id,
        title: m.title || m.name || "",
        original_title: m.original_title,
        year,
        release_date: m.release_date || null,
        poster_path: m.poster_path || undefined,
        backdrop_path: m.backdrop_path || undefined,
        overview: m.overview,
        original_language: m.original_language,
        vote_average: m.vote_average,
        vote_count: m.vote_count,
        genres,
        primary_genre: genres[0],
      };
    });
    return NextResponse.json(
      { results, page: data.page, total_pages: data.total_pages },
      { headers: tmdbCacheHeaders(3600) }
    );
  } catch {
    return NextResponse.json({ error: "TMDB fetch failed" }, { status: 502 });
  }
}
