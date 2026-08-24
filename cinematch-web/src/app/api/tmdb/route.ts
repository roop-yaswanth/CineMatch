import { NextRequest, NextResponse } from "next/server";
import { parseTmdbId, tmdbCacheHeaders } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

async function fetchTmdbEntity(kind: "movie" | "tv", tmdbId: string) {
  const response = await fetch(
    `https://api.themoviedb.org/3/${kind}/${tmdbId}?append_to_response=images&include_image_language=en,null`,
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
    return NextResponse.json({ poster_path: null, logo_path: null }, { status: 200 });
  }

  try {
    const movie = await fetchTmdbEntity("movie", tmdbId);
    const tv = movie?.poster_path ? null : await fetchTmdbEntity("tv", tmdbId);
    const data = movie?.poster_path ? movie : tv ?? movie;

    if (!data) {
      return NextResponse.json({ poster_path: null, logo_path: null }, { status: 200 });
    }

    const logos: Array<{ file_path: string; iso_639_1?: string | null }> = data.images?.logos || [];
    const englishLogo = logos.find((l) => l.iso_639_1 === "en") || logos[0];
    const logo_path = englishLogo?.file_path || null;

    // Prefer an English-language poster even for non-English movies (TMDB
    // often hosts localized artwork variants); fall back to the default
    // primary poster when no English variant exists.
    const posters: Array<{ file_path: string; iso_639_1?: string | null }> = data.images?.posters || [];
    const englishPoster = posters.find((p) => p.iso_639_1 === "en") || null;
    const poster_path = englishPoster?.file_path || data.poster_path || null;

    const backdrops: Array<{ file_path: string; iso_639_1?: string | null; width?: number }> = data.images?.backdrops || [];
    // Prioritize highest resolution (4K / 3840px+) backdrops with clean or English art
    const sortedBackdrops = [...backdrops].sort((a, b) => (b.width || 0) - (a.width || 0));
    const bestBackdrop = sortedBackdrops.find((b) => b.iso_639_1 === null || b.iso_639_1 === "en") || sortedBackdrops[0];
    const backdrop_path = bestBackdrop?.file_path || data.backdrop_path || null;

    return NextResponse.json(
      {
        poster_path,
        poster_is_english: Boolean(englishPoster),
        backdrop_path,
        logo_path,
        overview: data.overview || null,
        original_language: data.original_language || null,
        genres: (data.genres || []).map((g: { name: string }) => g.name),
        vote_average: data.vote_average || null,
        imdb_id: data.imdb_id || null,
      },
      { headers: tmdbCacheHeaders(86400) }
    );
  } catch {
    return NextResponse.json({ poster_path: null, logo_path: null }, { status: 200 });
  }
}
