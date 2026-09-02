/**
 * StorageService — Encapsulated wrapper around Web Storage.
 * Domain never touches `localStorage`/`sessionStorage` directly.
 * Replace with IndexedDB, cookies, or in-memory store by swapping this adapter.
 *
 * Single Responsibility: persistence only. No formatting, no API calls.
 * Explicit I/O: every method takes a key + value, returns a value — no hidden globals beyond the injected `storage`.
 */

export interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export class StorageService {
  constructor(private readonly storage: KeyValueStorage | null) {}

  get<T>(key: string): T | null {
    try {
      if (!this.storage) return null;
      const raw = this.storage.getItem(key);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as { data?: unknown; ts?: number };
      // Support both {data, ts} wrapper and raw values
      if (parsed && typeof parsed === "object" && "data" in parsed) {
        return (parsed.data as T) ?? null;
      }
      return parsed as T;
    } catch { return null; }
  }

  set<T>(key: string, data: T): void {
    try { this.storage?.setItem(key, JSON.stringify({ data, ts: Date.now() })); } catch {}
  }

  remove(key: string): void {
    try { this.storage?.removeItem(key); } catch {}
  }

  getRaw(key: string): string | null {
    try { return this.storage?.getItem(key) ?? null; } catch { return null; }
  }
}

// Lazy, safe singletons — no side effect at import time (window may be undefined during SSR)
export const localStore = new StorageService(typeof window !== "undefined" ? window.localStorage : null);
export const sessionStore = new StorageService(typeof window !== "undefined" ? window.sessionStorage : null);

/**
 * clearRecsCache — Law of Demeter helper.
 * Pages must NOT construct `cinematch_recs_cache_${userId}` themselves.
 * Call this instead; if the key format changes, only this file needs updating.
 */
export function clearRecsCache(userId: string): void {
  localStore.remove(`cinematch_recs_cache_${userId}`);
}
