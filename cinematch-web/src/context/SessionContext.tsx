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

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<UserSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const persistIdentifier = useCallback((identifier?: string) => {
    if (identifier) {
      localStorage.setItem(STORAGE_KEY, identifier);
    }
  }, []);

  const clearSession = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setSession(null);
  }, []);

  const restoreSession = useCallback(async () => {
    const savedIdentifier = localStorage.getItem(STORAGE_KEY);
    if (!savedIdentifier) {
      setSession(null);
      setIsLoading(false);
      return;
    }

    try {
      const restored = await apiLogin(savedIdentifier);
      setSession(restored);
      persistIdentifier(restored.identifier);
    } catch {
      clearSession();
    } finally {
      setIsLoading(false);
    }
  }, [clearSession, persistIdentifier]);

  useEffect(() => {
    void restoreSession();
  }, [restoreSession]);

  const login = useCallback(async (email: string) => {
    const newSession = await apiLogin(email);
    setSession(newSession);
    persistIdentifier(newSession.identifier);
    return newSession;
  }, [persistIdentifier]);

  const logout = useCallback(() => {
    clearSession();
  }, [clearSession]);

  const updateSession = useCallback((newSession: UserSession) => {
    setSession(newSession);
    persistIdentifier(newSession.identifier);
  }, [persistIdentifier]);

  useEffect(() => {
    const handleStorage = (event: StorageEvent) => {
      if (event.key === STORAGE_KEY && event.newValue === null) {
        setSession(null);
      }
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  // Session Inactivity Logout
  useEffect(() => {
    if (!session) return; // Only track when logged in

    const INACTIVITY_TIMEOUT = 30 * 60 * 1000; // 30 minutes

    let timeoutId: ReturnType<typeof setTimeout>;
    const resetTimer = () => {
      if (timeoutId) clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        logout();
      }, INACTIVITY_TIMEOUT);
    };

    resetTimer(); // Start the timer

    const events = ["mousemove", "keydown", "wheel", "touchstart"];
    events.forEach((event) => window.addEventListener(event, resetTimer, { passive: true }));

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
      events.forEach((event) => window.removeEventListener(event, resetTimer));
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
