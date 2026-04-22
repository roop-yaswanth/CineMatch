"use client";

import { useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import RecommendationsView from "@/components/RecommendationsView";
import { useSession } from "@/context/SessionContext";

export default function DashboardPage() {
  const router = useRouter();
  const { session, isLoading, logout, updateSession } = useSession();

  // Route protection
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    } else if (!isLoading && session && !session.onboarding_complete) {
      router.replace("/onboarding");
    }
  }, [session, isLoading, router]);

  const handleLogout = () => {
    logout();
    router.replace("/login");
  };

  /**
   * Reset handler: immediately marks onboarding_complete=false in localStorage
   * so the route guard redirects correctly, then calls the backend to clear
   * MongoDB (onboarding_feedback, recommendation_pool, etc.).
   */
  const handleBackToOnboarding = useCallback(async () => {
    if (!session) return;
    // 1. Optimistically clear onboarding_complete in cache so route guard works
    updateSession({ ...session, onboarding_complete: false });
    // 2. Ask backend to wipe session state in MongoDB
    try {
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session.session_id }),
      });
    } catch {
      // Non-critical — build_slate will also reset when user re-submits prefs
    }
    router.push("/onboarding");
  }, [session, updateSession, router]);

  if (isLoading || !session) return null;

  return (
    <RecommendationsView
      session={session}
      onSessionUpdate={updateSession}
      onLogout={handleLogout}
      onBackToOnboarding={handleBackToOnboarding}
    />
  );
}
