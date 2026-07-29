"use client";

import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from "react";
import { AnimatePresence } from "framer-motion";
import {
  apiAuthRefresh,
  apiLogout,
  apiUpdatePreferences,
  preferencesFromProfile,
  type UserSession,
  type RecommendationPreferences,
} from "@/lib/api";
import PreferencesModal from "@/components/PreferencesModal";

interface SessionContextType {
  session: UserSession | null;
  isLoading: boolean;
  logout: () => void;
  updateSession: (session: UserSession) => void;
  openPreferences: () => void;
  closePreferences: () => void;
  isPreferencesOpen: boolean;
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
    if (parsed && typeof parsed.session_id === "string" && typeof parsed.identifier === "string") {
      return parsed as UserSession;
    }
    return null;
  } catch {
    return null;
  }
}

function persistSession(s: UserSession) {
  try {
    localStorage.setItem(STORAGE_KEY, s.identifier);
    localStorage.setItem(SESSION_CACHE_KEY, JSON.stringify(s));
    markActivity();
  } catch { /* storage full */ }
}

function clearStoredSession() {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(SESSION_CACHE_KEY);
  localStorage.removeItem(ACTIVITY_KEY);
  localStorage.removeItem(PROFILE_OPTIMISTIC_KEY);
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

  const openPreferences = useCallback(() => {
    setIsPreferencesOpen(true);
  }, []);

  const closePreferences = useCallback(() => {
    setIsPreferencesOpen(false);
  }, []);

  const clearSession = useCallback(() => {
    clearStoredSession();
    setSession(null);
  }, []);

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
      setSession(null);
      setIsLoading(false);
      return;
    }

    if (cached) {
      markActivity();
      setSession(cached);
      setIsLoading(false);

      if (cached.auth_token) {
        try {
          const fresh = await apiAuthRefresh(cached.auth_token);
          const next = profileOptimisticActive() && cached.profile
            ? { ...fresh, profile: cached.profile, auth_token: fresh.auth_token ?? cached.auth_token }
            : fresh;
          setSession(next);
          persistSession(next);
        } catch (err) {
          const msg = err instanceof Error ? err.message : "";
          if (msg.includes("401")) {
            clearStoredSession();
            setSession(null);
          } else {
            console.warn("[SessionProvider] Token refresh failed; keeping cached session.");
          }
        }
      }
      return;
    }

    setSession(null);
    setIsLoading(false);
  }, []);

  useEffect(() => {
    void restoreSession();
  }, [restoreSession]);

  const logout = useCallback(() => {
    const sid = session?.session_id;
    clearSession();
    if (sid) void apiLogout(sid).catch(() => { /* non-fatal */ });
  }, [clearSession, session]);

  const updateSession = useCallback((newSession: UserSession) => {
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
