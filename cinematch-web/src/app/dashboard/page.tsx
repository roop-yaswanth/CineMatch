"use client";

import { useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import RecommendationsView from "@/components/RecommendationsView";
import { useSession } from "@/context/SessionContext";
import { useMounted } from "@/lib/useMounted";

export default function DashboardPage() {
  const router = useRouter();
  const { session, isLoading, logout, updateSession } = useSession();

  const mounted = useMounted();
  const onboardingGateChecked = useRef(false);
  useEffect(() => {
    if (isLoading) return;
    if (!session) {
      if (typeof window !== "undefined") {
        window.location.replace("/login");
      } else {
        router.replace("/login");
      }
      return;
    }
    if (!onboardingGateChecked.current) {
      onboardingGateChecked.current = true;
      if (!session.onboarding_complete) {
        router.replace("/onboarding");
      }
    }
  }, [session, isLoading, router]);

  const handleLogout = () => {
    logout();
  };

  const handleBackToOnboarding = useCallback(async () => {
    if (!session) return;
    try {
      localStorage.removeItem(`cinematch_recs_cache_${session.user_id}`);
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


  if (!mounted || isLoading || !session) {
    return (
      <div
        className="dash-root"
        suppressHydrationWarning
        style={{
          minHeight: "100dvh",
          display: "flex",
          flexDirection: "column",
          fontFamily: "var(--font-sans)",
          background: "var(--color-bg)",
        }}
      >
        <header className="dash-topbar">
          <div className="dash-topbar-inner">
            <h1 className="heading-display dash-brand">CineMatch</h1>
          </div>
        </header>
        <div aria-busy="true" aria-label="Loading recommendations">
          <div className="skeleton-shimmer dash-hero-skel" />
          {[0, 1, 2].map((i) => (
            <section key={i} className="shelf-section">
              <div className="shelf-header">
                <div>
                  <div className="skeleton-shimmer" style={{ height: 11, width: 110, borderRadius: 999, marginBottom: 8 }} />
                  <div className="skeleton-shimmer" style={{ height: 22, width: i === 0 ? 210 : 170, borderRadius: 999 }} />
                </div>
              </div>
              <div className="hide-scrollbar" style={{ display: "flex", gap: "var(--s-card-gap)", overflow: "hidden", padding: "6px var(--rail-x) 16px" }}>
                {Array.from({ length: 9 }).map((_, j) => (
                  <div key={j} className="dash-skel-card" style={{ width: "var(--poster-w)" }}>
                    <div className="skeleton-shimmer skeleton-grain" style={{ aspectRatio: "2 / 3", borderRadius: "var(--radius-poster)" }} />
                    <div style={{ marginTop: 14 }}>
                      <div className="skeleton-shimmer" style={{ height: 14, width: "85%", borderRadius: 4, marginBottom: 6 }} />
                      <div className="skeleton-shimmer" style={{ height: 11, width: "55%", borderRadius: 4 }} />
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    );
  }

  return (
    <RecommendationsView
      session={session}
      onSessionUpdate={updateSession}
      onLogout={handleLogout}
      onBackToOnboarding={handleBackToOnboarding}
    />
  );
}
