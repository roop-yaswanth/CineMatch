"use client";

import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from "react";
import { apiLogin, apiLogout, type UserSession } from "@/lib/api";

interface SessionContextType {
  session: UserSession | null;
  isLoading: boolean;
  login: (email: string) => Promise<UserSession>;
  logout: () => void;
  updateSession: (session: UserSession) => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);
const STORAGE_KEY = "cinematch_email";
const SESSION_CACHE_KEY = "cinematch_session";
const ACTIVITY_KEY = "cinematch_last_activity";

// Stay logged in until the user explicitly logs out or a full month passes with
// no activity. We persist in localStorage (shared across tabs + survives browser
// restarts) rather than sessionStorage (per-tab) — so opening the app in a new
// tab no longer forces a re-login.
const INACTIVITY_LIMIT_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

/** Safely read a cached session from localStorage */
function readCachedSession(): UserSession | null {
  try {
    const raw = localStorage.getItem(SESSION_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    // Basic shape check
    if (parsed && typeof parsed.session_id === "string" && typeof parsed.identifier === "string") {
      return parsed as UserSession;
    }
    return null;
  } catch {
    return null;
  }
}

/** Persist the full session object alongside the email identifier */
function persistSession(s: UserSession) {
  try {
    localStorage.setItem(STORAGE_KEY, s.identifier);
    localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(s));
    markActivity();
  } catch { /* storage full — non-critical */ }
}

/** Clear all session data from localStorage */
function clearStoredSession() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(SESSION_CACHE_KEY);
  localStorage.removeItem(ACTIVITY_KEY);
}

/** Timestamp (ms epoch) of the user's last recorded activity; 0 if unknown. */
function readLastActivity(): number {
  try {
    const raw = localStorage.getItem(ACTIVITY_KEY);
    const n = raw ? parseInt(raw, 10) : 0;
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

/** Record "user is active now" — drives the one-month inactivity logout. */
function markActivity() {
  try {
    localStorage.setItem(ACTIVITY_KEY, String(Date.now()));
  } catch { /* ignore */ }
}

/** True once a month has elapsed since the last recorded activity. */
function inactivityExpired(): boolean {
  const last = readLastActivity();
  return last > 0 && Date.now() - last > INACTIVITY_LIMIT_MS;
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<UserSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const clearSession = useCallback(() => {
    clearStoredSession();
    setSession(null);
  }, []);

  // Restore: instant from cache, then silently validate with backend
  const restoreSession = useCallback(async () => {
    // Enforce the one-month inactivity window across browser restarts: if the
    // last recorded activity is older than the limit, clear and require login.
    if (inactivityExpired()) {
      clearStoredSession();
      setSession(null);
      setIsLoading(false);
      return;
    }

    const cached = readCachedSession();
    const savedIdentifier = localStorage.getItem(STORAGE_KEY);

    if (!cached && !savedIdentifier) {
      // Never logged in
      setSession(null);
      setIsLoading(false);
      return;
    }

    // Instant restore from cache — no API needed, no loading flicker
    if (cached) {
      markActivity(); // opening the app refreshes the inactivity window
      setSession(cached);
      setIsLoading(false);

      // Silently re-validate with the backend in the background
      try {
        const fresh = await apiLogin(cached.identifier);
        setSession(fresh);
        persistSession(fresh);
      } catch {
        // Backend unavailable — keep the cached session, don't log out
        console.warn("[SessionProvider] Background re-validation failed; using cached session.");
      }
      return;
    }

    // Fallback: have an identifier but no cached session (e.g. old storage format)
    try {
      const restored = await apiLogin(savedIdentifier!);
      setSession(restored);
      persistSession(restored);
    } catch {
      // Can't reach backend and no cached session — show login, but keep identifier
      // so next refresh can try again
      setSession(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void restoreSession();
  }, [restoreSession]);

  const login = useCallback(async (email: string) => {
    const newSession = await apiLogin(email);
    setSession(newSession);
    persistSession(newSession);
    return newSession;
  }, []);

  const logout = useCallback(() => {
    // Best-effort server-side invalidation, then clear locally regardless.
    const sid = session?.session_id;
    clearSession();
    if (sid) void apiLogout(sid).catch(() => { /* non-fatal */ });
  }, [clearSession, session]);

  const updateSession = useCallback((newSession: UserSession) => {
    setSession(newSession);
    persistSession(newSession);
  }, []);

  // Mirror login/logout that happened in another tab (localStorage `storage`
  // events fire in every *other* tab). Keeps all tabs in sync: log out in one,
  // they all log out; log in in one, the others pick it up.
  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY || event.key === SESSION_CACHE_KEY) {
        if (event.key === STORAGE_KEY && event.newValue === null) {
          setSession(null);
        } else {
          setSession(readCachedSession());
        }
      }
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  // Inactivity logout: log out only after a FULL MONTH with no user interaction.
  // We record the last-activity timestamp and poll, rather than using a single
  // setTimeout — a 30-day delay overflows setTimeout's 32-bit millisecond range
  // and would fire almost immediately. Crucially we do NOT log out just because
  // the tab is hidden/backgrounded: switching tabs or apps keeps you logged in.
  useEffect(() => {
    if (!session) return; // Only track when logged in

    markActivity(); // using the app counts as activity

    let lastWrite = Date.now();
    const onActivity = () => {
      const now = Date.now();
      // Throttle localStorage writes to at most once a minute.
      if (now - lastWrite > 60 * 1000) {
        lastWrite = now;
        markActivity();
      }
    };

    const events = ["mousemove", "keydown", "wheel", "touchstart", "click"];
    events.forEach((event) => window.addEventListener(event, onActivity, { passive: true }));

    const interval = setInterval(() => {
      if (inactivityExpired()) logout();
    }, 5 * 60 * 1000); // re-check every 5 minutes

    return () => {
      events.forEach((event) => window.removeEventListener(event, onActivity));
      clearInterval(interval);
    };
  }, [session, logout]);

  return (
    <SessionContext.Provider value={{ session, isLoading, login, logout, updateSession }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error("useSession must be used within SessionProvider");
  }
  return context;
}
