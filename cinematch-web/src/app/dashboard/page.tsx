"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useSession } from "@/context/SessionContext";
import RecommendationsView from "@/components/RecommendationsView";
import { LoadingScreen } from "@/components/LoadingScreen";

export default function DashboardPage() {
  const router = useRouter();
  const { session, logout, updateSession, isLoading } = useSession();

  // If not authenticated, redirect to login
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  // Block native back button from exiting the app while actively logged in
  useEffect(() => {
    if (session && !isLoading) {
      if (!window.history.state || !window.history.state.isApp) {
        window.history.pushState({ isApp: true }, "", window.location.href);
      }
      
      const handlePopState = (e: PopStateEvent) => {
        // If we are actively logging out, allow the popstate to happen cleanly
        if ((window as any).__isLoggingOut) return;

        // If they navigate backwards, bounce them forward immediately.
        // This avoids infinitely spamming fake pushStates to the history!
        if (!e.state || !e.state.isApp) {
          window.history.forward();
        }
      };
      
      window.addEventListener("popstate", handlePopState);
      return () => window.removeEventListener("popstate", handlePopState);
    }
  }, [session, isLoading]);

  if (!session || isLoading) {
    return <LoadingScreen />;
  }

  const handleSessionUpdate = (nextSession: typeof session) => {
    updateSession(nextSession);
  };

  const handleBackToOnboarding = () => {
    router.replace("/onboarding");
  };

  return (
    <RecommendationsView
      session={session}
      onSessionUpdate={handleSessionUpdate}
      onBackToOnboarding={handleBackToOnboarding}
      onLogout={logout}
    />
  );
}
