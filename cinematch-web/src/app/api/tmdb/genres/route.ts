import { NextResponse } from "next/server";
import { getGenreList } from "@/lib/tmdb-server";

export async function GET() {
  const genres = await getGenreList("movie");
  return NextResponse.json({ genres });
}
