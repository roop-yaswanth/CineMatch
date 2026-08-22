"use client";

import { useState, useEffect } from "react";
import { posterUrl, fetchTmdbPoster } from "@/lib/api";

/**
 * Hook that returns a poster URL.
 *
 * 1. If the movie has no poster_path, fetches one from TMDB (via /api/tmdb).
 * 2. If the movie is non-English, swaps in TMDB's English-language poster
 *    variant when one exists (/api/tmdb returns an English-preferred
 *    poster_path); if no English artwork exists the original poster stays —
 *    results are cached client-side (LRU) and server-side (24h data cache),
 *    so grids of regional movies cost at most one lookup per title.
 */
export function usePoster(
  posterPath: string | null | undefined,
  tmdbId: number,
  size = "w500",
  originalLanguage?: string | null
): string {
  const [resolved, setResolved] = useState<{
    tmdbId: number;
    path: string | null;
  }>({ tmdbId, path: null });

  const wantsEnglishArt = originalLanguage != null && originalLanguage !== "en";

  useEffect(() => {
    const needsLookup = !posterPath || wantsEnglishArt;
    if (!needsLookup) {
      return;
    }

    let cancelled = false;
    fetchTmdbPoster(tmdbId).then((path) => {
      if (!cancelled) {
        setResolved({ tmdbId, path });
      }
    });
    return () => { cancelled = true; };
  }, [posterPath, tmdbId, wantsEnglishArt]);

  let resolvedPath: string | null | undefined = posterPath;
  if (resolved.tmdbId === tmdbId && resolved.path) {
    // Only override when a lookup was actually wanted; keeps English movies
    // and pre-resolved posters untouched when no better art exists.
    if (!posterPath || wantsEnglishArt) {
      resolvedPath = resolved.path;
    }
  }

  return posterUrl(resolvedPath, size);
}
