"use client";

import { createContext, useContext, useEffect, useState, ReactNode, useCallback, useRef } from "react";
import { useRouter, usePathname } from "next/navigation";
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
  const pathname = usePathname();
  const [session, setSession] = useState<UserSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const isRestoringRef = useRef(false);

  // Restore session on mount
  useEffect(() => {
    if (isRestoringRef.current) return;
    isRestoringRef.current = true;

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
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
    setSession(null);
    // Use router.replace to stay in the SPA — no full-page reload
    router.replace("/login");
  }, [router]);

  const updateSession = useCallback((newSession: UserSession) => {
    setSession(newSession);
  }, []);

  // ── Global back-button guard ──────────────────────────────────────────────
  // We use a single, centralized popstate handler to prevent the browser's
  // back/forward buttons or mobile back gestures from navigating away from
  // the app. This replaces all the per-page popstate hacks that were
  // conflicting with each other.
  //
  // Strategy: On mount, we replace the current state with a sentinel marker
  // and push one dummy entry on top. If the user presses back:
  //   1. The browser pops the dummy entry.
  //   2. Our popstate handler immediately pushes a new dummy entry, keeping
  //      the user on the same page.
  // This creates a "sticky" history that the user can never back out of.
  // Navigating *forward* within the app uses router.replace(), which doesn't
  // add history entries and thus can't be "backed" over either.
  useEffect(() => {
    if (typeof window === "undefined") return;

    const SENTINEL = "__cinematch_guard";

    // Only install once — check if we already have the sentinel
    if (!window.history.state?.[SENTINEL]) {
      window.history.replaceState({ [SENTINEL]: true, path: pathname }, "", window.location.href);
      window.history.pushState({ [SENTINEL]: true, path: pathname }, "", window.location.href);
    }

    const handlePopState = () => {
      // No matter what, push the user back to where they are.
      // This prevents: back to new-tab, back to login after logout, etc.
      window.history.pushState(
        { [SENTINEL]: true, path: window.location.pathname },
        "",
        window.location.href
      );
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [pathname]);

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
