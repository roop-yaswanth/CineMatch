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
  if (!TMDB_BEARER) return NextResponse.json({ cast: [], directors: [], writers: [] });

  try {
    const res = await fetch(
      `https://api.themoviedb.org/3/${kind}/${id}/credits?language=en-US`,
      { headers: TMDB_HEADERS, next: { revalidate: 86400 } }
    );
    if (!res.ok) return NextResponse.json({ cast: [], directors: [], writers: [] });
    const data = await res.json();

    const cast = ((data.cast || []) as TmdbCast[])
      .slice(0, 15)
      .map((c) => ({
        id: c.id,
        name: c.name,
        character: c.character || null,
        profile_path: c.profile_path || null,
      }));

    const crew = (data.crew || []) as TmdbCrew[];
    const directors = crew
      .filter((c) => c.job === "Director")
      .map((c) => ({ id: c.id, name: c.name, profile_path: c.profile_path || null }));
    const writers = crew
      .filter((c) => c.department === "Writing" && (c.job === "Writer" || c.job === "Screenplay" || c.job === "Story"))
      .map((c) => ({ id: c.id, name: c.name, job: c.job, profile_path: c.profile_path || null }));

    return NextResponse.json(
      { cast, directors, writers },
      { headers: tmdbCacheHeaders(86400) }
    );
  } catch {
    return NextResponse.json({ cast: [], directors: [], writers: [] });
  }
}
