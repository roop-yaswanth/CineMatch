"use client";

import { createContext, useContext, useEffect, useState, ReactNode, useCallback } from "react";
import { useRouter } from "next/navigation";
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
  const router = useRouter();
  const [session, setSession] = useState<UserSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Restore session on mount
  useEffect(() => {
    const restoreSession = async () => {
      const savedEmail = localStorage.getItem(STORAGE_KEY);
      if (savedEmail) {
        try {
          const restored = await apiLogin(savedEmail);
          setSession(restored);
          // Route based on session state — use replace so back button doesn't escape the app
          if (restored.is_returning && restored.onboarding_complete) {
            router.replace("/dashboard");
          } else {
            router.replace("/onboarding");
          }
        } catch {
          localStorage.removeItem(STORAGE_KEY);
          router.replace("/login");
        }
      } else {
        router.replace("/login");
      }
      setIsLoading(false);
    };
    
    restoreSession();
  }, [router]);

  const login = useCallback(async (email: string) => {
    const newSession = await apiLogin(email);
    setSession(newSession);
    if (newSession.identifier) {
      localStorage.setItem(STORAGE_KEY, newSession.identifier);
    }
    // Route based on session state — use replace to avoid back button issues
    if (newSession.is_returning && newSession.onboarding_complete) {
      router.replace("/dashboard");
    } else {
      router.replace("/onboarding");
    }
    return newSession;
  }, [router]);

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    
    // Flag to tell the Dashboard trap to let us pop the state cleanly
    (window as any).__isLoggingOut = true;
    
    if (window.history.state && window.history.state.isApp) {
      // 1. Pop the fake 'isApp' history state silently
      window.history.back();

      // 2. Wait a tick for the pop to finish, then clear session and replace the underlying state
      setTimeout(() => {
        setSession(null);
        window.location.replace("/login");
        setTimeout(() => { (window as any).__isLoggingOut = false; }, 100);
      }, 50);
    } else {
      setSession(null);
      window.location.replace("/login");
      (window as any).__isLoggingOut = false;
    }
  }, []);

  const updateSession = useCallback((newSession: UserSession) => {
    setSession(newSession);
  }, []);

  // Session Inactivity Logout
  useEffect(() => {
    if (!session) return; // Only track when logged in

    const INACTIVITY_TIMEOUT = 30 * 60 * 1000; // 30 minutes

    let timeoutId: NodeJS.Timeout;
    const resetTimer = () => {
      if (timeoutId) clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        console.log("Session expired due to inactivity.");
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
