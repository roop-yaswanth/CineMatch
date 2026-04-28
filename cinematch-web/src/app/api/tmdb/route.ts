import { NextRequest, NextResponse } from "next/server";
import { parseTmdbId, tmdbCacheHeaders } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

async function fetchTmdbEntity(kind: "movie" | "tv", tmdbId: string) {
  const response = await fetch(
    `https://api.themoviedb.org/3/${kind}/${tmdbId}?language=en-US`,
    {
      headers: TMDB_HEADERS,
      next: { revalidate: 86400 },
    }
  );

  if (!response.ok) return null;
  return response.json();
}

export async function GET(req: NextRequest) {
  const tmdbIdNum = parseTmdbId(req.nextUrl.searchParams.get("id"));
  if (!tmdbIdNum) {
    return NextResponse.json({ error: "Invalid id" }, { status: 400 });
  }
  const tmdbId = String(tmdbIdNum);

  if (!TMDB_BEARER) {
    return NextResponse.json({ poster_path: null }, { status: 200 });
  }

  try {
    const movie = await fetchTmdbEntity("movie", tmdbId);
    const tv = movie?.poster_path ? null : await fetchTmdbEntity("tv", tmdbId);
    const data = movie?.poster_path ? movie : tv ?? movie;

    if (!data) {
      return NextResponse.json({ poster_path: null }, { status: 200 });
    }

    return NextResponse.json(
      {
        poster_path: data.poster_path || null,
        backdrop_path: data.backdrop_path || null,
        overview: data.overview || null,
        original_language: data.original_language || null,
        genres: (data.genres || []).map((g: { name: string }) => g.name),
        vote_average: data.vote_average || null,
        imdb_id: data.imdb_id || null,
      },
      { headers: tmdbCacheHeaders(86400) }
    );
  } catch {
    return NextResponse.json({ poster_path: null }, { status: 200 });
  }
}
