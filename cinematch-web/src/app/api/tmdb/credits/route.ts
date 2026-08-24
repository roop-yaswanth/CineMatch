import { NextRequest, NextResponse } from "next/server";
import { parseTmdbId, tmdbCacheHeaders } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

interface TmdbCast {
  id: number;
  name: string;
  character?: string;
  profile_path?: string | null;
  order?: number;
}

interface TmdbCrew {
  id: number;
  name: string;
  job?: string;
  department?: string;
  profile_path?: string | null;
}

export async function GET(req: NextRequest) {
  const id = parseTmdbId(req.nextUrl.searchParams.get("id"));
  const kind = req.nextUrl.searchParams.get("kind") === "tv" ? "tv" : "movie";
  if (!id) return NextResponse.json({ error: "Invalid id" }, { status: 400 });
  if (!TMDB_BEARER) return NextResponse.json({ cast: [], directors: [], writers: [], logo_path: null, poster_path: null });

  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/${kind}/${id}?append_to_response=credits,images,release_dates,content_ratings&include_image_language=en,null`,
      { headers: TMDB_HEADERS, next: { revalidate: 86400 } }
    );
    if (!res.ok) return NextResponse.json({ cast: [], directors: [], writers: [], logo_path: null, poster_path: null, runtime: null, certification: null, status: null, tagline: null });
    const data = await res.json();

    const creditsData = data.credits || {};
    const cast = ((creditsData.cast || []) as TmdbCast[])
      .slice(0, 15)
      .map((c) => ({
        id: c.id,
        name: c.name,
        character: c.character || null,
        profile_path: c.profile_path || null,
      }));

    const crew = (creditsData.crew || []) as TmdbCrew[];
    const directors = crew
      .filter((c) => c.job === "Director")
      .map((c) => ({ id: c.id, name: c.name, profile_path: c.profile_path || null }));
    const writers = crew
      .filter((c) => c.department === "Writing" && (c.job === "Writer" || c.job === "Screenplay" || c.job === "Story"))
      .map((c) => ({ id: c.id, name: c.name, job: c.job, profile_path: c.profile_path || null }));

    // Extract official title logo PNG from images.logos
    const logos: Array<{ file_path: string; iso_639_1?: string | null; aspect_ratio?: number }> = data.images?.logos || [];
    const englishLogo = logos.find((l) => l.iso_639_1 === "en") || logos[0];
    const logo_path = englishLogo?.file_path || null;
    const logo_aspect_ratio = englishLogo?.aspect_ratio || null;

    // English-preferred poster (same images payload — no extra API cost):
    // fall back to the default primary poster when no 'en' variant exists.
    const posters: Array<{ file_path: string; iso_639_1?: string | null }> = data.images?.posters || [];
    const englishPoster = posters.find((p) => p.iso_639_1 === "en") || null;
    const poster_path = englishPoster?.file_path || data.poster_path || null;

    // Extract age rating / certification (movies: release_dates, TV: content_ratings)
    const releaseDates = (data.release_dates?.results || []) as Array<{
      iso_3166_1: string;
      release_dates: Array<{ certification?: string }>;
    }>;
    const tvRatings = (data.content_ratings?.results || []) as Array<{
      iso_3166_1: string;
      rating?: string;
    }>;

    let cert: string | null = null;

    if (kind === "tv" && tvRatings.length > 0) {
      cert =
        tvRatings.find((r) => r.iso_3166_1 === "US")?.rating?.trim() ||
        tvRatings.find((r) => r.iso_3166_1 === "GB")?.rating?.trim() ||
        tvRatings.find((r) => r.iso_3166_1 === "IN")?.rating?.trim() ||
        tvRatings.find((r) => r.rating?.trim())?.rating?.trim() ||
        null;
    } else if (releaseDates.length > 0) {
      // 1. Try US release date certifications
      cert = releaseDates.find((r) => r.iso_3166_1 === "US")?.release_dates?.find((d) => d.certification?.trim())?.certification?.trim() || null;
      // 2. Try GB (e.g. Harry Potter, UK productions)
      if (!cert) {
        cert = releaseDates.find((r) => r.iso_3166_1 === "GB")?.release_dates?.find((d) => d.certification?.trim())?.certification?.trim() || null;
      }
      // 3. Try IN (e.g. Indian cinema like Ala Vaikunthapurramuloo, RRR, etc.)
      if (!cert) {
        cert = releaseDates.find((r) => r.iso_3166_1 === "IN")?.release_dates?.find((d) => d.certification?.trim())?.certification?.trim() || null;
      }
      // 4. Fallback: Search all countries for any non-empty certification
      if (!cert) {
        for (const r of releaseDates) {
          const found = r.release_dates?.find((d) => d.certification?.trim())?.certification?.trim();
          if (found) {
            cert = found;
            break;
          }
        }
      }
    }

    const certification = cert || null;

    const runtime = typeof data.runtime === "number" && data.runtime > 0 ? data.runtime : null;
    const status = data.status || null;
    const tagline = data.tagline || null;

    return NextResponse.json(
      {
        cast,
        directors,
        writers,
        logo_path,
        logo_aspect_ratio,
        poster_path,
        runtime,
        certification,
        status,
        tagline,
      },
      { headers: tmdbCacheHeaders(86400) }
    );
  } catch {
    return NextResponse.json({ cast: [], directors: [], writers: [], logo_path: null, poster_path: null, runtime: null, certification: null, status: null, tagline: null });
  }
}
