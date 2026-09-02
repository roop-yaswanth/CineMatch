/**
 * Pure Movie Helpers — explicit inputs/outputs, no hidden globals, no fetch, no localStorage.
 * Single Reason to Change: movie display logic changes.
 * Testable in isolation, movable to any layer.
 */

import type { Movie, Recommendation } from "../types/movie";

type MovieLike = Movie | Recommendation;

export function formatRuntime(runtime?: number | string | null): string {
  if (!runtime) return "";
  const mins = typeof runtime === "string" ? parseInt(runtime, 10) : runtime;
  if (!mins || Number.isNaN(mins) || mins <= 0) return "";
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

export function yearLabel(movie: Pick<MovieLike, "year" | "release_date">): string {
  if (movie.year) return String(movie.year);
  if (movie.release_date) {
    const d = new Date(movie.release_date);
    if (!Number.isNaN(d.getTime())) return String(d.getFullYear());
  }
  return "";
}

export function languageLabelPure(code: string, labels: Record<string, string>): string {
  if (!code) return "Unknown";
  return labels[code.toLowerCase()] || code.toUpperCase();
}

// Pure prominence score — same as shelves.ts but explicit deps (no import of languageLabel)
export function prominenceScorePure(rating: number, votes: number): number {
  if (rating <= 0) return 0;
  const bayes = (votes * rating + 1000 * 6.5) / (votes + 1000);
  return bayes * Math.log10(Math.max(votes, 10));
}
