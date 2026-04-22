"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import LoginScreen from "@/components/LoginScreen";
import OnboardingView from "@/components/OnboardingView";
import RecommendationsView from "@/components/RecommendationsView";
import { apiLogin, type UserSession } from "@/lib/api";

type AppPhase = "restoring" | "login" | "onboarding" | "recommendations";

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.3, ease } },
};

const STORAGE_KEY = "cinematch_email";
const INACTIVITY_MS = 30 * 60 * 1000;

export default function AppShell() {
  const [phase, setPhase] = useState<AppPhase>(() => {
    if (typeof window === "undefined") return "restoring";
    return localStorage.getItem(STORAGE_KEY) ? "restoring" : "login";
  });
  const [session, setSession] = useState<UserSession | null>(null);
  const [forcePreferences, setForcePreferences] = useState(false);

  // Restore session from localStorage on mount
  useEffect(() => {
    const savedEmail = localStorage.getItem(STORAGE_KEY);
    if (!savedEmail) return;
    apiLogin(savedEmail)
      .then((restored) => {
        setSession(restored);
        setPhase(
          restored.is_returning && restored.onboarding_complete
            ? "recommendations"
            : "onboarding"
        );
      })
      .catch(() => {
        localStorage.removeItem(STORAGE_KEY);
        setPhase("login");
      });
  }, []);

  const handleLogin = useCallback((nextSession: UserSession) => {
    setSession(nextSession);
    if (nextSession.identifier) {
      localStorage.setItem(STORAGE_KEY, nextSession.identifier);
    }
    if (nextSession.is_returning && nextSession.onboarding_complete) {
      setPhase("recommendations");
      return;
    }
    setPhase("onboarding");
  }, []);

  const handleOnboardingDone = useCallback((nextSession: UserSession) => {
    setSession(nextSession);
    setForcePreferences(false);
    setPhase("recommendations");
  }, []);

  const handleBackToOnboarding = useCallback(() => {
    setForcePreferences(true);
    setPhase("onboarding");
  }, []);

  const handleLogout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setSession(null);
    setForcePreferences(false);
    setPhase("login");
  }, []);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [showTimeoutToast, setShowTimeoutToast] = useState(false);

  const resetInactivityTimer = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      setShowTimeoutToast(true);
      setTimeout(() => {
        setShowTimeoutToast(false);
        handleLogout();
      }, 3000);
    }, INACTIVITY_MS);
  }, [handleLogout]);

  // Block mobile back gesture from navigating away / refreshing the page.
  // Strategy: The SPA does not use the browser back button. We push a dummy
  // state on mount. If the user swipes back, the browser pops the dummy state
  // and fires popstate. We catch it and immediately push the dummy state back.
  // The user is permanently trapped inside the SPA, preventing accidental exits.
  useEffect(() => {
    if (typeof window === "undefined") return;
    
    history.pushState(null, "", window.location.href);
    const handlePopState = () => {
      history.pushState(null, "", window.location.href);
    };
    
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Inactivity auto-logout
  useEffect(() => {
    const events = ["mousedown", "touchstart", "keydown", "scroll"];
    const handler = () => resetInactivityTimer();
    events.forEach((e) => window.addEventListener(e, handler, { passive: true }));
    resetInactivityTimer();
    return () => {
      events.forEach((e) => window.removeEventListener(e, handler));
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [resetInactivityTimer]);

  return (
    <main className="relative min-h-dvh">
      <AnimatePresence mode="wait">
        {phase === "restoring" && (
          <motion.div
            key="restoring"
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              minHeight: "100dvh",
              color: "var(--color-text-muted)",
              fontSize: "14px",
            }}
          >
            Restoring session...
          </motion.div>
        )}

        {phase === "login" && (
          <motion.div key="login" variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <LoginScreen onLogin={handleLogin} />
          </motion.div>
        )}

        {phase === "onboarding" && session && (
          <motion.div key="onboarding" variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <OnboardingView
              session={session}
              onComplete={handleOnboardingDone}
              onLogout={handleLogout}
              forcePreferences={forcePreferences}
            />
          </motion.div>
        )}

        {phase === "recommendations" && session && (
          <motion.div key="recommendations" variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <RecommendationsView
              session={session}
              onSessionUpdate={setSession}
              onBackToOnboarding={handleBackToOnboarding}
              onLogout={handleLogout}
            />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Inactivity timeout toast */}
      <AnimatePresence>
        {showTimeoutToast && (
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 40 }}
            style={{
              position: "fixed",
              bottom: "40px",
              left: "50%",
              transform: "translateX(-50%)",
              zIndex: 9999,
              background: "rgba(16, 16, 20, 0.95)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-pill)",
              padding: "14px 28px",
              color: "var(--color-text-primary)",
              fontSize: "14px",
              fontWeight: 500,
              boxShadow: "0 12px 40px rgba(0,0,0,0.5)",
              whiteSpace: "nowrap",
            }}
          >
            Session expired due to inactivity
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}
