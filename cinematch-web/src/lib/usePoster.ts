"use client";

import { useState, useEffect } from "react";
import { posterUrl, fetchTmdbPoster } from "@/lib/api";

/**
 * Hook that returns a poster URL. If the movie has no poster_path,
 * it automatically fetches from TMDB and returns the resolved URL.
 */
export function usePoster(
  posterPath: string | null | undefined,
  tmdbId: number,
  size = "w500"
): string {
  const [resolvedPath, setResolvedPath] = useState(posterPath);

  useEffect(() => {
    if (posterPath) {
      setResolvedPath(posterPath);
      return;
    }

    // No poster_path — try TMDB API
    let cancelled = false;
    fetchTmdbPoster(tmdbId).then((path) => {
      if (!cancelled && path) {
        setResolvedPath(path);
      }
    });
    return () => { cancelled = true; };
  }, [posterPath, tmdbId]);

  return posterUrl(resolvedPath, size);
}
