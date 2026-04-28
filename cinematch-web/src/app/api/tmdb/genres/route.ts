import { NextResponse } from "next/server";
import { getGenreList, tmdbCacheHeaders } from "@/lib/tmdb-server";

export async function GET() {
  const genres = await getGenreList("movie");
  return NextResponse.json({ genres }, { headers: tmdbCacheHeaders(86400) });
}
