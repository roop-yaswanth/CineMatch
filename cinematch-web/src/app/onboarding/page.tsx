"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";
import { useSession } from "@/context/SessionContext";
import OnboardingView from "@/components/OnboardingView";
import { LoadingScreen } from "@/components/LoadingScreen";

export default function OnboardingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { session, logout, updateSession, isLoading } = useSession();
  const forcePreferences = searchParams.get("reset") === "true";

  // If not authenticated, redirect to login
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  // If user completes onboarding, redirect to dashboard (don't stay on onboarding)
  useEffect(() => {
    if (session && session.onboarding_complete && !forcePreferences) {
      router.replace("/dashboard");
    }
  }, [session, router, forcePreferences]);

  // Block mobile back gesture from navigating away / refreshing the page.
  // Strategy: The SPA does not use the browser back button. We push a dummy
  // state on mount. If the user swipes back, the browser pops the dummy state
  // and fires popstate. We catch it and immediately push the dummy state back.
  useEffect(() => {
    if (typeof window === "undefined") return;

    const currentUrl = window.location.href;
    const guardState = { __cm_onboarding_guard: true };
    // Create a same-document back buffer so browser back never escapes onboarding.
    window.history.replaceState(guardState, "", currentUrl);
    window.history.pushState(guardState, "", currentUrl);
    const handlePopState = (event: PopStateEvent) => {
      event.preventDefault();
      window.history.go(1);
    };

    window.addEventListener("popstate", handlePopState, { capture: true });
    return () => window.removeEventListener("popstate", handlePopState, { capture: true });
  }, []);

  if (!session || isLoading) {
    return <LoadingScreen />;
  }

  const handleComplete = (nextSession: typeof session) => {
    updateSession(nextSession);
    router.replace("/dashboard");
  };

  return (
    <OnboardingView
      session={session}
      onComplete={handleComplete}
      onLogout={logout}
      forcePreferences={forcePreferences}
    />
  );
}
