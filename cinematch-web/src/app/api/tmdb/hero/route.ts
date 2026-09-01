import { NextRequest, NextResponse } from "next/server";
import { getGenreMap, tmdbCacheHeaders } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

/** 12-hour epoch — deterministic rotation seed for hero selection.
 *  Every client that calls within the same 12h window gets the same epoch,
 *  so shelf logic can seed a stable shuffle without extra state. */
function heroEpoch(): number {
  return Math.floor(Date.now() / (12 * 3600 * 1000));
}

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
  popularity?: number;
}

/**
 * GET /api/tmdb/hero
 *
 * Fetches TMDB trending + now_playing movies, optionally filtered by the user's
 * preferred languages. Returns up to 20 candidates ranked by popularity, plus
 * a 12-hour epoch for deterministic client-side rotation.
 *
 * Query params:
 *   languages — comma-separated ISO 639-1 codes (e.g. "te,hi,en")
 *   region    — ISO 3166-1 country code (e.g. "IN", "US")
 */
export async function GET(req: NextRequest) {
  if (!TMDB_BEARER) {
    return NextResponse.json({ results: [], epoch: heroEpoch() });
  }

  const sp = req.nextUrl.searchParams;
  const languages = (sp.get("languages") || "")
    .split(",")
    .map((l) => l.trim().toLowerCase())
    .filter((l) => /^[a-z]{2}$/.test(l));
  const region = sp.get("region") || "";

  try {
    // Fetch trending (week) pages 1–2 + now_playing page 1 in parallel.
    // Week window gives broader cultural coverage than day; now_playing
    // captures recent OTT/theatrical releases that haven't trended yet.
    const regionParam = /^[A-Z]{2}$/.test(region) ? `&region=${region}` : "";
    const fetches = [
      fetch(
        `https://api.themoviedb.org/3/trending/movie/week?language=en-US&page=1`,
        { headers: TMDB_HEADERS, next: { revalidate: 43200 } }
      ),
      fetch(
        `https://api.themoviedb.org/3/trending/movie/week?language=en-US&page=2`,
        { headers: TMDB_HEADERS, next: { revalidate: 43200 } }
      ),
      fetch(
        `https://api.themoviedb.org/3/movie/now_playing?language=en-US&page=1${regionParam}`,
        { headers: TMDB_HEADERS, next: { revalidate: 43200 } }
      ),
    ];

    const [trendRes1, trendRes2, nowPlayingRes] = await Promise.all(fetches);
    const genreMap = await getGenreMap();

    const parseResults = async (res: Response): Promise<TmdbMovie[]> => {
      if (!res.ok) return [];
      const data = await res.json();
      return data.results || [];
    };

    const allRaw: TmdbMovie[] = [
      ...(await parseResults(trendRes1)),
      ...(await parseResults(trendRes2)),
      ...(await parseResults(nowPlayingRes)),
    ];

    // Deduplicate by TMDB id
    const seen = new Set<number>();
    const deduped: TmdbMovie[] = [];
    for (const m of allRaw) {
      if (seen.has(m.id)) continue;
      seen.add(m.id);
      // Require a backdrop for hero visual quality
      if (!m.backdrop_path) continue;
      // Require a poster
      if (!m.poster_path) continue;
      deduped.push(m);
    }

    // Filter by user's preferred languages (if provided).
    // If no languages are specified or filtering yields too few results (<5),
    // fall back to the full pool — better to show global trending than nothing.
    let filtered = deduped;
    if (languages.length > 0) {
      const langSet = new Set(languages);
      const langFiltered = deduped.filter(
        (m) => m.original_language && langSet.has(m.original_language)
      );
      if (langFiltered.length >= 5) {
        filtered = langFiltered;
      }
    }

    // Sort by TMDB popularity (trending signal) descending
    filtered.sort((a, b) => (b.popularity ?? 0) - (a.popularity ?? 0));

    // Take top 20 candidates — the client picks 3 from this pool
    const top = filtered.slice(0, 20);

    const results = top.map((m) => {
      const dateStr = m.release_date || m.first_air_date || "";
      const year = dateStr
        ? parseInt(dateStr.slice(0, 4), 10) || undefined
        : undefined;
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
        popularity: m.popularity,
        // Flag so the client can distinguish trending from personal picks
        _trending: true,
      };
    });

    return NextResponse.json(
      { results, epoch: heroEpoch() },
      // 12-hour edge cache per unique query string (languages × region).
      // stale-while-revalidate lets the edge serve stale for 24h while refreshing.
      { headers: tmdbCacheHeaders(43200, 86400) }
    );
  } catch {
    return NextResponse.json({ results: [], epoch: heroEpoch() });
  }
}
