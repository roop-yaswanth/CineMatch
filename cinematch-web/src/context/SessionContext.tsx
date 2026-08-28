"use client";

import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import {
  apiAuthRefresh,
  apiLogout,
  apiUpdatePreferences,
  isSessionExpiredError,
  preferencesFromProfile,
  type UserSession,
  type RecommendationPreferences,
} from "@/lib/api";
import PreferencesModal from "@/components/PreferencesModal";
import TutorialOverlay from "@/components/TutorialOverlay";

interface SessionContextType {
  session: UserSession | null;
  isLoading: boolean;
  logout: () => void;
  updateSession: (session: UserSession) => void;
  openPreferences: () => void;
  closePreferences: () => void;
  isPreferencesOpen: boolean;
  isTutorialOpen: boolean;
  openTutorial: () => void;
  closeTutorial: () => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);
const STORAGE_KEY = "cinematch_email";
const SESSION_CACHE_KEY = "cinematch_session";
const ACTIVITY_KEY = "cinematch_last_activity";
const PROFILE_OPTIMISTIC_KEY = "cinematch_profile_optimistic_until";

const INACTIVITY_LIMIT_MS = 30 * 24 * 60 * 60 * 1000; // 30 days

function readCachedSession(): UserSession | null {
  try {
    const raw = localStorage.getItem(SESSION_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (isValidUserSession(parsed)) return parsed;
    return null;
  } catch {
    return null;
  }
}

/**
 * Runtime guard against malformed session objects. One poisoned response used
 * to reach setSession() and linger in memory (session_id: null), making every
 * subsequent dashboard poll POST a 422 until the user reloaded.
 */
function isValidUserSession(s: unknown): s is UserSession {
  return Boolean(
    s &&
    typeof s === "object" &&
    typeof (s as UserSession).session_id === "string" &&
    (s as UserSession).session_id.length > 0 &&
    typeof (s as UserSession).identifier === "string"
  );
}

// Lightweight signed-in hint for middleware.ts. Carries no secrets (real auth
// stays in localStorage + bearer token) — it only lets the server redirect
// logged-out visitors away from app routes before any page JS loads.
function setAuthHintCookie(on: boolean) {
  try {
    if (typeof document === "undefined") return;
    if (on) {
      document.cookie = `cm_auth=1; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax${typeof location !== "undefined" && location.protocol === "https:" ? "; Secure" : ""}`;
    } else {
      document.cookie = "cm_auth=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; max-age=0; SameSite=Lax";
      document.cookie = "cm_auth=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; max-age=0; SameSite=Lax; Secure";
    }
  } catch { /* ignore */ }
}

function persistSession(s: UserSession) {
  try {
    localStorage.setItem(STORAGE_KEY, s.identifier);
    localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(s));
    setAuthHintCookie(true);
    markActivity();
  } catch { /* storage full */ }
}

function clearStoredSession() {
  try {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(SESSION_CACHE_KEY);
    localStorage.removeItem(ACTIVITY_KEY);
    localStorage.removeItem(PROFILE_OPTIMISTIC_KEY);
    for (let i = localStorage.length - 1; i >= 0; i--) {
      const key = localStorage.key(i);
      if (key && (key.startsWith("cinematch_recs_cache_") || key.startsWith("cinematch_history_cache_"))) {
        localStorage.removeItem(key);
      }
    }
  } catch { /* ignore */ }
  setAuthHintCookie(false);
}

function profileOptimisticActive(): boolean {
  try {
    const raw = localStorage.getItem(PROFILE_OPTIMISTIC_KEY);
    return !!raw && Date.now() < parseInt(raw, 10);
  } catch {
    return false;
  }
}

function readLastActivity(): number {
  try {
    const raw = localStorage.getItem(ACTIVITY_KEY);
    const n = raw ? parseInt(raw, 10) : 0;
    return Number.isFinite(n) ? n : 0;
  } catch {
    return 0;
  }
}

function markActivity() {
  try {
    localStorage.setItem(ACTIVITY_KEY, String(Date.now()));
  } catch { /* ignore */ }
}

function inactivityExpired(): boolean {
  const last = readLastActivity();
  return last > 0 && Date.now() - last > INACTIVITY_LIMIT_MS;
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<UserSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPreferencesOpen, setIsPreferencesOpen] = useState(false);
  const [isTutorialOpen, setIsTutorialOpen] = useState(false);

  const openPreferences = useCallback(() => {
    setIsPreferencesOpen(true);
  }, []);

  const closePreferences = useCallback(() => {
    setIsPreferencesOpen(false);
  }, []);

  const openTutorial = useCallback(() => setIsTutorialOpen(true), []);
  const closeTutorial = useCallback(() => setIsTutorialOpen(false), []);

  const clearSession = useCallback(() => {
    clearStoredSession();
    setSession(null);
  }, []);

  const logout = useCallback(() => {
    const sid = session?.session_id;
    clearSession();
    if (sid) void apiLogout(sid).catch(() => { /* non-fatal */ });
    if (typeof window !== "undefined") {
      window.location.replace("/login");
    }
  }, [clearSession, session]);

  // Global tutorial opener — lets the hamburger menu (or any page) trigger the guide
  useEffect(() => {
    const handleOpenTutorial = () => setIsTutorialOpen(true);
    window.addEventListener("cinematch:open_tutorial", handleOpenTutorial);
    return () => window.removeEventListener("cinematch:open_tutorial", handleOpenTutorial);
  }, []);

  // Listen for global session expiration dispatched by API client (e.g. 404 Session not found)
  useEffect(() => {
    const handleExpired = () => {
      logout();
    };
    window.addEventListener("cinematch:session_expired", handleExpired);
    return () => {
      window.removeEventListener("cinematch:session_expired", handleExpired);
    };
  }, [logout]);

  const restoreSession = useCallback(async () => {
    if (inactivityExpired()) {
      clearStoredSession();
      setSession(null);
      setIsLoading(false);
      return;
    }

    const cached = readCachedSession();
    const savedIdentifier = localStorage.getItem(STORAGE_KEY);

    if (!cached && !savedIdentifier) {
      clearStoredSession();
      setSession(null);
      setIsLoading(false);
      return;
    }

    if (cached) {
      markActivity();
      // Users logged in before the middleware hint existed need the cookie
      // refreshed here, or the server gate would bounce them to /login.
      setAuthHintCookie(true);
      setSession(cached);
      setIsLoading(false);

      if (cached.auth_token) {
        try {
          const fresh = await apiAuthRefresh(cached.auth_token);
          const next = profileOptimisticActive() && cached.profile
            ? { ...fresh, profile: cached.profile, auth_token: fresh.auth_token ?? cached.auth_token }
            : fresh;
          if (isValidUserSession(next)) {
            setSession(next);
            persistSession(next);
          } else {
            // Malformed refresh response — keep the cached session rather
            // than poisoning in-memory state with a null id.
            console.warn("[SessionProvider] Refresh returned an invalid session; keeping cached copy.");
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : "";
          if (
            (err && typeof err === "object" && "isServerSleeping" in err) ||
            msg.includes("500") ||
            msg.includes("ServerSleeping") ||
            msg.includes("exceeded") ||
            msg.includes("SERVER_SLEEPING")
          ) {
            if (typeof window !== "undefined") {
              window.location.href = "/500";
            }
            return;
          }
          if (isSessionExpiredError(err) || msg.includes("401") || msg.includes("Session not found")) {
            clearStoredSession();
            setSession(null);
            if (typeof window !== "undefined") {
              window.location.replace("/login");
            }
          } else {
            console.warn("[SessionProvider] Token refresh failed; keeping cached session.");
          }
        }
      }
      return;
    }


    clearStoredSession();
    setSession(null);
    setIsLoading(false);
  }, []);

  useEffect(() => {
    const t = setTimeout(() => void restoreSession(), 0);
    return () => clearTimeout(t);
  }, [restoreSession]);

  const updateSession = useCallback((newSession: UserSession) => {
    if (!isValidUserSession(newSession)) {
      // Never let a malformed response overwrite a good in-memory session.
      const sid = (newSession as { session_id?: unknown } | null)?.session_id;
      console.warn(
        "[SessionProvider] Ignored invalid session update:",
        JSON.stringify({ has_session_id: Boolean(sid), type_of_session_id: typeof sid })
      );
      return;
    }
    setSession(newSession);
    persistSession(newSession);
  }, []);

  const handlePreferencesUpdate = useCallback(
    (prefs: RecommendationPreferences) => {
      if (session) {
        updateSession({
          ...session,
          profile: {
            ...session.profile,
            preferred_languages: prefs.languages,
            preferred_genres: prefs.genres,
            genre_picks: prefs.genres,
            include_classics: prefs.include_classics,
            age_group: prefs.age_group,
            region: prefs.region,
          },
        });
        try {
          localStorage.setItem("cinematch_profile_optimistic_until", String(Date.now() + 15000));
        } catch { /* ignore */ }
      }

      try {
        const uid = session?.user_id;
        if (uid) localStorage.removeItem(`cinematch_recs_cache_${uid}`);
      } catch { /* ignore */ }

      const sid = session?.session_id;
      if (sid) {
        apiUpdatePreferences(sid, {
          languages: prefs.languages,
          genres: prefs.genres,
          semantic_index: prefs.semantic_index,
          age_group: prefs.age_group,
          region: prefs.region,
          include_classics: prefs.include_classics,
        })
          .then((freshSession) => updateSession(freshSession))
          .catch((err) => console.error("Failed to update preferences on server:", err));
      }
    },
    [session, updateSession]
  );

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

  useEffect(() => {
    if (!session) return;

    markActivity();

    let lastWrite = Date.now();
    const onActivity = () => {
      const now = Date.now();
      if (now - lastWrite > 60 * 1000) {
        lastWrite = now;
        markActivity();
      }
    };

    const events = ["mousemove", "keydown", "wheel", "touchstart", "click"];
    events.forEach((event) => window.addEventListener(event, onActivity, { passive: true }));

    const interval = setInterval(() => {
      if (inactivityExpired()) logout();
    }, 5 * 60 * 1000);

    return () => {
      events.forEach((event) => window.removeEventListener(event, onActivity));
      clearInterval(interval);
    };
  }, [session, logout]);

  return (
    <SessionContext.Provider
      value={{
        session,
        isLoading,
        logout,
        updateSession,
        openPreferences,
        closePreferences,
        isPreferencesOpen,
        isTutorialOpen,
        openTutorial,
        closeTutorial,
      }}
    >
      {children}

      <AnimatePresence>
        {isPreferencesOpen && session && (
          <PreferencesModal
            key="preferences-genie-modal"
            preferences={preferencesFromProfile(session.profile)}
            onUpdate={handlePreferencesUpdate}
            onClose={closePreferences}
            mode="recommendations"
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {isTutorialOpen && (
          <TutorialOverlay
            key="tutorial-overlay"
            isOpen={isTutorialOpen}
            onClose={closeTutorial}
            userId={session?.user_id ?? null}
          />
        )}
      </AnimatePresence>
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
