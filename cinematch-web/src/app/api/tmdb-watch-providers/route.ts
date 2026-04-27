import { NextRequest, NextResponse } from "next/server";

const TMDB_BEARER = process.env.TMDB_BEARER_TOKEN || "";
const TMDB_HEADERS = {
  Authorization: `Bearer ${TMDB_BEARER}`,
  accept: "application/json",
};

export type WatchProvider = {
  provider_id: number;
  provider_name: string;
  logo_path: string;
  display_priority?: number;
};

export type CountryProviders = {
  link?: string;
  flatrate?: WatchProvider[];
  free?: WatchProvider[];
  ads?: WatchProvider[];
  rent?: WatchProvider[];
  buy?: WatchProvider[];
};

export type WatchProvidersResponse = {
  id: number;
  results: Record<string, CountryProviders>;
};

export async function GET(req: NextRequest): Promise<NextResponse<WatchProvidersResponse>> {
  const tmdbId = req.nextUrl.searchParams.get("id");
  if (!tmdbId) {
    return NextResponse.json({ id: 0, results: {} }, { status: 400 });
  }
  if (!TMDB_BEARER) {
    return NextResponse.json({ id: Number(tmdbId) || 0, results: {} });
  }

  const tryFetch = async (kind: "movie" | "tv"): Promise<WatchProvidersResponse | null> => {
    const res = await fetch(
      `https://api.themoviedb.org/3/${kind}/${tmdbId}/watch/providers`,
      // 12-hour cache: provider availability changes occasionally but not minute-to-minute.
      { headers: TMDB_HEADERS, next: { revalidate: 43200 } }
    );
    if (!res.ok) return null;
    const d = await res.json();
    return { id: d.id ?? Number(tmdbId), results: d.results ?? {} };
  };

  try {
    const data = (await tryFetch("movie")) ?? (await tryFetch("tv")) ?? {
      id: Number(tmdbId) || 0,
      results: {},
    };
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ id: Number(tmdbId) || 0, results: {} });
  }
}
