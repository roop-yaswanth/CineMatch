"use client";

import { useState, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import LoginScreen from "@/components/LoginScreen";
import OnboardingView from "@/components/OnboardingView";
import RecommendationsView from "@/components/RecommendationsView";
import type { UserSession } from "@/lib/api";

type AppPhase = "login" | "onboarding" | "recommendations";

const ease = [0.25, 0.1, 0.25, 1] as [number, number, number, number];

const pageVariants = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.3, ease } },
};

export default function Home() {
  const [phase, setPhase] = useState<AppPhase>("login");
  const [session, setSession] = useState<UserSession | null>(null);

  const handleLogin = useCallback((s: UserSession) => {
    setSession(s);
    if (s.is_returning && s.onboarding_complete) {
      setPhase("recommendations");
    } else {
      setPhase("onboarding");
    }
  }, []);

  const handleOnboardingDone = useCallback((s: UserSession) => {
    setSession(s);
    setPhase("recommendations");
  }, []);

  const handleBackToOnboarding = useCallback(() => {
    setPhase("onboarding");
  }, []);

  const handleLogout = useCallback(() => {
    setSession(null);
    setPhase("login");
  }, []);

  return (
    <main className="relative min-h-dvh">
      <AnimatePresence mode="wait">
        {phase === "login" && (
          <motion.div key="login" variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <LoginScreen onLogin={handleLogin} />
          </motion.div>
        )}

        {phase === "onboarding" && session && (
          <motion.div key="onboarding" variants={pageVariants} initial="initial" animate="animate" exit="exit">
            <OnboardingView session={session} onComplete={handleOnboardingDone} onLogout={handleLogout} />
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
