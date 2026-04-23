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

  const handleBackToOnboarding = useCallback(async () => {
    if (!session) return;
    try {
      sessionStorage.removeItem(`cinematch_recs_cache_${session.session_id}`);
    } catch { /* non-critical */ }
    updateSession({ ...session, onboarding_complete: false });
    try {
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: session.session_id }),
      });
    } catch {
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
