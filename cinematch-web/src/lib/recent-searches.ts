/**
 * Tiny localStorage-backed list of the user's most recent search queries.
 * Surfaced when the search input is empty so re-running a recent search is
 * one tap away. Capped to 8 entries so the chip row stays readable.
 */

const KEY = "cinematch_recent_searches";
const MAX = 8;
const MIN_LEN = 2;

export function getRecentSearches(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((q): q is string => typeof q === "string" && q.length >= MIN_LEN)
      .slice(0, MAX);
  } catch {
    return [];
  }
}

export function rememberRecentSearch(query: string): void {
  if (typeof window === "undefined") return;
  const q = query.trim();
  if (q.length < MIN_LEN) return;
  try {
    const existing = getRecentSearches();
    // De-dup case-insensitively; keep the user's casing of the most recent.
    const lower = q.toLowerCase();
    const next = [q, ...existing.filter((e) => e.toLowerCase() !== lower)].slice(0, MAX);
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch { /* storage full — non-critical */ }
}

export function clearRecentSearches(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(KEY);
  } catch { /* ignore */ }
}
