import { NextRequest, NextResponse } from "next/server";
import { getGenreMap, parseBoundedInt, REGION_TO_ISO, tmdbCacheHeaders } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

const ALLOWED_SORTS = new Set([
  "popularity.desc",
  "popularity.asc",
  "vote_average.desc",
  "vote_average.asc",
  "primary_release_date.desc",
  "primary_release_date.asc",
  "revenue.desc",
  "title.asc",
]);

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

export async function GET(req: NextRequest) {
  if (!TMDB_BEARER) {
    return NextResponse.json({ results: [], total_pages: 0, page: 1 });
  }
  const sp = req.nextUrl.searchParams;

  const sort = sp.get("sort_by") || "popularity.desc";
  if (!ALLOWED_SORTS.has(sort)) {
    return NextResponse.json({ error: "Invalid sort_by" }, { status: 400 });
  }

  const params = new URLSearchParams({
    language: "en-US",
    page: String(parseBoundedInt(sp.get("page"), 1, 500, 1)),
    sort_by: sort,
    include_adult: "false",
    include_video: "false",
  });

  // with_genres: comma-separated list of TMDB genre ids; allow only digits + commas.
  const genres = sp.get("with_genres");
  if (genres && /^[0-9]+(,[0-9]+){0,30}$/.test(genres)) params.set("with_genres", genres);

  const year = sp.get("year");
  if (year && /^\d{4}$/.test(year)) params.set("primary_release_year", year);

  const yearFrom = sp.get("year_from");
  const yearTo = sp.get("year_to");
  if (yearFrom && /^\d{4}$/.test(yearFrom)) params.set("primary_release_date.gte", `${yearFrom}-01-01`);
  if (yearTo && /^\d{4}$/.test(yearTo)) params.set("primary_release_date.lte", `${yearTo}-12-31`);

  // Language code: 2-letter ISO 639-1 only.
  const lang = sp.get("with_original_language");
  if (lang && /^[a-z]{2}$/.test(lang)) params.set("with_original_language", lang);

  // Numeric thresholds: bounded floats / ints to prevent injection.
  const minVoteAvg = sp.get("vote_average_gte");
  if (minVoteAvg && /^[0-9]+(\.[0-9])?$/.test(minVoteAvg)) {
    const v = Number(minVoteAvg);
    if (v >= 0 && v <= 10) params.set("vote_average.gte", String(v));
  }

  const minVoteCount = sp.get("vote_count_gte");
  if (minVoteCount && /^[0-9]{1,7}$/.test(minVoteCount)) {
    params.set("vote_count.gte", minVoteCount);
  } else if (sort.startsWith("vote_average")) {
    params.set("vote_count.gte", "300"); // sane default
  }

  const region = sp.get("region") || "";
  const iso = REGION_TO_ISO[region];
  if (iso) params.set("region", iso);

  try {
    const [res, genreMap] = await Promise.all([
      fetch(`https://api.themoviedb.org/3/discover/movie?${params.toString()}`, {
        headers: TMDB_HEADERS,
        next: { revalidate: 1800 },
      }),
      getGenreMap(),
    ]);
    if (!res.ok) {
      return NextResponse.json({ error: "TMDB fetch failed" }, { status: 502 });
    }
    const data = await res.json();
    const results = (data.results || []).map((m: TmdbMovie) => {
      const dateStr = m.release_date || "";
      const year = dateStr ? parseInt(dateStr.slice(0, 4), 10) || undefined : undefined;
      const gn = (m.genre_ids || [])
        .map((id) => genreMap[id])
        .filter((n): n is string => Boolean(n));
      return {
        id: m.id,
        tmdb_id: m.id,
        title: m.title || "",
        original_title: m.original_title,
        year,
        release_date: m.release_date || null,
        poster_path: m.poster_path || undefined,
        backdrop_path: m.backdrop_path || undefined,
        overview: m.overview,
        original_language: m.original_language,
        vote_average: m.vote_average,
        vote_count: m.vote_count,
        genres: gn,
        primary_genre: gn[0],
      };
    });
    return NextResponse.json(
      { results, page: data.page, total_pages: data.total_pages },
      { headers: tmdbCacheHeaders(1800) }
    );
  } catch {
    return NextResponse.json({ error: "TMDB fetch failed" }, { status: 502 });
  }
}
