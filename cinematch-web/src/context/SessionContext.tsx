"use client";

import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from "react";
import { apiLogin, type UserSession } from "@/lib/api";

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
  } catch { /* storage full — non-critical */ }
}

/** Clear all session data from localStorage */
function clearStoredSession() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(SESSION_CACHE_KEY);
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
    clearSession();
  }, [clearSession]);

  const updateSession = useCallback((newSession: UserSession) => {
    setSession(newSession);
    persistSession(newSession);
  }, []);

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY && event.newValue === null) {
        setSession(null);
      }
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  // Session inactivity / out-of-focus logout.
  // Logs out after 10 minutes of either: no user interaction, or document hidden.
  useEffect(() => {
    if (!session) return; // Only track when logged in

    const INACTIVITY_TIMEOUT = 10 * 60 * 1000; // 10 minutes

    let activityTimeoutId: ReturnType<typeof setTimeout> | null = null;
    let hiddenTimeoutId: ReturnType<typeof setTimeout> | null = null;

    const resetActivityTimer = () => {
      if (activityTimeoutId) clearTimeout(activityTimeoutId);
      activityTimeoutId = setTimeout(() => logout(), INACTIVITY_TIMEOUT);
    };

    const handleVisibility = () => {
      if (document.hidden) {
        // Tab/window hidden — start a separate timer; logout if still hidden after 10 min.
        if (hiddenTimeoutId) clearTimeout(hiddenTimeoutId);
        hiddenTimeoutId = setTimeout(() => logout(), INACTIVITY_TIMEOUT);
      } else {
        // Back in focus — cancel the hidden timer and reset the activity timer.
        if (hiddenTimeoutId) {
          clearTimeout(hiddenTimeoutId);
          hiddenTimeoutId = null;
        }
        resetActivityTimer();
      }
    };

    resetActivityTimer();

    const events = ["mousemove", "keydown", "wheel", "touchstart", "click"];
    events.forEach((event) => window.addEventListener(event, resetActivityTimer, { passive: true }));
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      if (activityTimeoutId) clearTimeout(activityTimeoutId);
      if (hiddenTimeoutId) clearTimeout(hiddenTimeoutId);
      events.forEach((event) => window.removeEventListener(event, resetActivityTimer));
      document.removeEventListener("visibilitychange", handleVisibility);
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
