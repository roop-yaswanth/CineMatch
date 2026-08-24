import { NextRequest, NextResponse } from "next/server";
import { getGenreMap, parseTmdbId, tmdbCacheHeaders } from "@/lib/tmdb-server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

interface TmdbItem {
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
  adult?: boolean;
}

const HORROR_GENRE_ID = 27;
const THRILLER_GENRE_ID = 53;
const ANIMATION_GENRE_ID = 16;
const FAMILY_GENRE_ID = 10751;

export async function GET(req: NextRequest) {
  const id = parseTmdbId(req.nextUrl.searchParams.get("id"));
  if (!id) return NextResponse.json({ error: "Invalid id" }, { status: 400 });
  if (!TMDB_BEARER) return NextResponse.json({ results: [] }, { status: 200 });

  try {
    const [seedRes, recRes, genreMap] = await Promise.all([
      fetch(`https://api.themoviedb.org/3/movie/${id}?language=en-US`, {
        headers: TMDB_HEADERS,
        next: { revalidate: 86400 },
      }).catch(() => null),
      fetch(`https://api.themoviedb.org/3/movie/${id}/recommendations?language=en-US&page=1`, {
        headers: TMDB_HEADERS,
        next: { revalidate: 86400 },
      }),
      getGenreMap(),
    ]);

    let seedGenreIds: number[] = [];
    if (seedRes && seedRes.ok) {
      const seedData = await seedRes.json();
      seedGenreIds = (seedData.genres || []).map((g: { id: number }) => g.id);
    }

    const isFamilySeed = seedGenreIds.includes(ANIMATION_GENRE_ID) || seedGenreIds.includes(FAMILY_GENRE_ID);

    let rawItems: TmdbItem[] = [];
    if (recRes.ok) {
      const recData = await recRes.json();
      rawItems = recData.results || [];
    }

    const existingIds = new Set<number>([id]);

    // Filter helper to ensure family friendly matching
    const isSafeMatch = (item: TmdbItem) => {
      if (existingIds.has(item.id)) return false;
      if (item.adult) return false;
      const gids = item.genre_ids || [];
      if (isFamilySeed) {
        // Exclude horror completely
        if (gids.includes(HORROR_GENRE_ID)) return false;
        // Exclude standalone thriller if not also family/animation
        if (gids.includes(THRILLER_GENRE_ID) && !gids.includes(ANIMATION_GENRE_ID) && !gids.includes(FAMILY_GENRE_ID)) {
          return false;
        }
      }
      return true;
    };

    const filteredItems = rawItems.filter(isSafeMatch);
    filteredItems.forEach((m) => existingIds.add(m.id));

    // If we have fewer than 20 items, query discover by matching seed genres
    if (filteredItems.length < 20 && seedGenreIds.length > 0) {
      try {
        const withGenres = seedGenreIds.slice(0, 3).join(",");
        const withoutGenres = isFamilySeed ? String(HORROR_GENRE_ID) : "";
        const discParams = new URLSearchParams({
          language: "en-US",
          page: "1",
          sort_by: "popularity.desc",
          include_adult: "false",
          "vote_count.gte": "800",
          with_genres: withGenres,
        });
        if (withoutGenres) discParams.set("without_genres", withoutGenres);

        const discRes = await fetch(`https://api.themoviedb.org/3/discover/movie?${discParams.toString()}`, {
          headers: TMDB_HEADERS,
          next: { revalidate: 86400 },
        });
        if (discRes.ok) {
          const discData = await discRes.json();
          for (const item of discData.results || []) {
            if (isSafeMatch(item)) {
              filteredItems.push(item);
              existingIds.add(item.id);
            }
            if (filteredItems.length >= 20) break;
          }
        }
      } catch {
        /* non-critical fallback */
      }
    }

    // Secondary fallback: /similar with strict kid-friendly filter
    if (filteredItems.length < 15) {
      try {
        const simRes = await fetch(`https://api.themoviedb.org/3/movie/${id}/similar?language=en-US&page=1`, {
          headers: TMDB_HEADERS,
          next: { revalidate: 86400 },
        });
        if (simRes.ok) {
          const simData = await simRes.json();
          for (const item of simData.results || []) {
            if (isSafeMatch(item)) {
              filteredItems.push(item);
              existingIds.add(item.id);
            }
            if (filteredItems.length >= 20) break;
          }
        }
      } catch {
        /* non-critical fallback */
      }
    }

    const results = filteredItems.slice(0, 20).map((m: TmdbItem) => {
      const dateStr = m.release_date || "";
      const year = dateStr ? parseInt(dateStr.slice(0, 4), 10) || undefined : undefined;
      const gn = (m.genre_ids || [])
        .map((gid) => genreMap[gid])
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
        imdb_rating: m.vote_average,
        genres: gn,
        primary_genre: gn[0],
      };
    });

    return NextResponse.json({ results }, { headers: tmdbCacheHeaders(86400) });
  } catch {
    return NextResponse.json({ results: [] }, { status: 200 });
  }
}
