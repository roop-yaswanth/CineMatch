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

  // Lock the *outer* body scroll while the dashboard is mounted. The swipe
  // stack uses `position: fixed` cards, so when the body scrolls the
  // global footer (which lives below children in normal flow) slides up
  // *behind* the fixed cards — that's the "footer on the back of movie
  // posters" leak.
  //
  // We DON'T want to kill scrolling though — rails and the rest of the
  // dashboard content do need vertical scroll. The fix is two-part:
  //   (1) here: lock body overflow so the footer can never enter view; the
  //       footer still renders on every other route normally because we
  //       restore the prior overflow on unmount.
  //   (2) RecommendationsView root carries its own `overflow-y: auto` and
  //       fills the viewport, so all dashboard scrolling happens inside
  //       that container instead of on the body.
  useEffect(() => {
    const html = document.documentElement;
    const body = document.body;
    const prevHtml = html.style.overflow;
    const prevBody = body.style.overflow;
    html.style.overflow = "hidden";
    body.style.overflow = "hidden";
    return () => {
      html.style.overflow = prevHtml;
      body.style.overflow = prevBody;
    };
  }, []);

  const handleLogout = () => {
    logout();
    router.replace("/login");
  };

  const handleBackToOnboarding = useCallback(async () => {
    if (!session) return;
    try {
      localStorage.removeItem(`cinematch_recs_cache_${session.session_id}`);
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
