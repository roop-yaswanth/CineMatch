import { NextRequest, NextResponse } from "next/server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";

export async function GET(req: NextRequest) {
  const tmdbId = req.nextUrl.searchParams.get("id");
  if (!tmdbId) {
    return NextResponse.json({ error: "Missing id" }, { status: 400 });
  }

  if (!TMDB_BEARER) {
    return NextResponse.json({ poster_path: null }, { status: 200 });
  }

  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/movie/${tmdbId}?language=en-US`,
      {
        headers: {
          Authorization: `Bearer ${TMDB_BEARER}`,
          accept: "application/json",
        },
        next: { revalidate: 86400 }, // Cache for 24 hours
      }
    );

    if (!res.ok) {
      return NextResponse.json({ poster_path: null }, { status: 200 });
    }

    const data = await res.json();
    return NextResponse.json({
      poster_path: data.poster_path || null,
      backdrop_path: data.backdrop_path || null,
      overview: data.overview || null,
      original_language: data.original_language || null,
      genres: (data.genres || []).map((g: { name: string }) => g.name),
      vote_average: data.vote_average || null,
      imdb_id: data.imdb_id || null,
    });
  } catch {
    return NextResponse.json({ poster_path: null }, { status: 200 });
  }
}
