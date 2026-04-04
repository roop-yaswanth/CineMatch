"use client";

import { useCallback, useEffect, useState } from "react";
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

export default function AppShell() {
  const [phase, setPhase] = useState<AppPhase>("restoring");
  const [session, setSession] = useState<UserSession | null>(null);
  const [forcePreferences, setForcePreferences] = useState(false);

  // Restore session from localStorage on mount
  useEffect(() => {
    const savedEmail = localStorage.getItem(STORAGE_KEY);
    if (!savedEmail) {
      setPhase("login");
      return;
    }
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
    </main>
  );
}
