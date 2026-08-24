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

const backdropCache = new Map<number, string | null>();
const backdropInflight = new Map<number, Promise<string | null>>();

export function setBackdropCache(tmdbId: number, path: string | null): void {
  backdropCache.set(tmdbId, path);
}

export function fetchBackdrop(tmdbId: number): Promise<string | null> {
  const hit = backdropCache.get(tmdbId);
  if (hit !== undefined) return Promise.resolve(hit);
  const inflight = backdropInflight.get(tmdbId);
  if (inflight) return inflight;
  const p = fetch(`/api/tmdb?id=${tmdbId}`)
    .then((res) => (res.ok ? res.json() : null))
    .then((data): string | null => {
      const path: string | null = data?.backdrop_path ?? null;
      backdropCache.set(tmdbId, path);
      return path;
    })
    .catch(() => {
      // Don't poison the cache on transient failures — retry next mount.
      return null;
    })
    .finally(() => backdropInflight.delete(tmdbId));
  backdropInflight.set(tmdbId, p);
  return p;
}

export async function prefetchBackdrops(
  movies: Array<{ backdrop_path?: string; id: number; tmdb_id?: number }>
): Promise<void> {
  const missing = movies.filter((m) => !m.backdrop_path || !m.backdrop_path.trim());
  await Promise.allSettled(
    missing.map((m) => fetchBackdrop(m.tmdb_id ?? m.id))
  );
}

export function useBackdrop(
  backdropPath: string | null | undefined,
  tmdbId: number,
  size: "w780" | "w1280" | "original" = "original"
): { src: string | null; loading: boolean } {
  const initialPath = backdropPath && backdropPath.trim() ? backdropPath.trim() : null;
  const [asyncPath, setAsyncPath] = useState<{ tmdbId: number; path: string | null }>({
    tmdbId,
    path: null,
  });

  const cached = backdropCache.get(tmdbId);
  const effectivePath = initialPath || cached || (asyncPath.tmdbId === tmdbId ? asyncPath.path : null);

  useEffect(() => {
    if (initialPath || cached !== undefined) return;
    let cancelled = false;
    fetchBackdrop(tmdbId).then((path) => {
      if (!cancelled) setAsyncPath({ tmdbId, path });
    });
    return () => {
      cancelled = true;
    };
  }, [initialPath, tmdbId, cached]);

  return {
    src: effectivePath ? posterUrl(effectivePath, size) : null,
    loading: !initialPath && cached === undefined && asyncPath.tmdbId !== tmdbId,
  };
}
