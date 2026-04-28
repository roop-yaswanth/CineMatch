import { NextRequest, NextResponse } from "next/server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

interface TmdbCredit {
  id: number;
  title?: string;
  name?: string;
  release_date?: string;
  first_air_date?: string;
  poster_path?: string | null;
  character?: string;
  job?: string;
  department?: string;
  vote_average?: number;
  media_type?: string;
  popularity?: number;
  overview?: string;
}

export async function GET(req: NextRequest) {
  const id = req.nextUrl.searchParams.get("id");
  if (!id) return NextResponse.json({ error: "Missing id" }, { status: 400 });
  if (!TMDB_BEARER) return NextResponse.json({ error: "TMDB not configured" }, { status: 503 });

  try {
    const [personRes, creditsRes] = await Promise.all([
      fetch(`https://api.themoviedb.org/3/person/${id}?language=en-US`, {
        headers: TMDB_HEADERS,
        next: { revalidate: 86400 },
      }),
      fetch(`https://api.themoviedb.org/3/person/${id}/combined_credits?language=en-US`, {
        headers: TMDB_HEADERS,
        next: { revalidate: 86400 },
      }),
    ]);
    if (!personRes.ok) return NextResponse.json({ error: "Not found" }, { status: 404 });

    const person = await personRes.json();
    const credits = creditsRes.ok ? await creditsRes.json() : { cast: [], crew: [] };

    const mapCredit = (c: TmdbCredit) => {
      const dateStr = c.release_date || c.first_air_date || "";
      const year = dateStr ? parseInt(dateStr.slice(0, 4), 10) || undefined : undefined;
      return {
        id: c.id,
        tmdb_id: c.id,
        title: c.title || c.name || "",
        year,
        release_date: c.release_date || c.first_air_date || null,
        poster_path: c.poster_path || undefined,
        media_type: c.media_type === "tv" ? "tv" : "movie",
        character: c.character || null,
        job: c.job || null,
        department: c.department || null,
        vote_average: c.vote_average,
        popularity: c.popularity,
        overview: c.overview || undefined,
      };
    };

    const cast = ((credits.cast || []) as TmdbCredit[]).map(mapCredit);
    const crew = ((credits.crew || []) as TmdbCredit[]).map(mapCredit);

    // Known for: top by popularity (cast roles)
    const knownFor = [...cast]
      .sort((a, b) => (b.popularity || 0) - (a.popularity || 0))
      .slice(0, 12);

    return NextResponse.json({
      id: person.id,
      name: person.name,
      biography: person.biography || "",
      birthday: person.birthday || null,
      deathday: person.deathday || null,
      place_of_birth: person.place_of_birth || null,
      gender: person.gender,
      known_for_department: person.known_for_department || null,
      profile_path: person.profile_path || null,
      also_known_as: person.also_known_as || [],
      homepage: person.homepage || null,
      imdb_id: person.imdb_id || null,
      cast,
      crew,
      known_for: knownFor,
    });
  } catch {
    return NextResponse.json({ error: "TMDB fetch failed" }, { status: 502 });
  }
}
