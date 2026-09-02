"use client";

/**
 * DashboardPage — composition root for the main recommendation dashboard.
 * SRP: this page only wires session + handlers and decides what to render.
 *   Auth guard: useAuthGuard({ requireOnboarding: true }) — no duplication.
 *   Loading skeleton: DashboardSkeleton — extracted component.
 *   Storage: clearRecsCache — no Demeter violation (no raw localStorage key).
 */

import { useCallback } from "react";
import { useRouter } from "next/navigation";
import RecommendationsView from "@/components/RecommendationsView";
import { useSession } from "@/context/SessionContext";
import { useMounted } from "@/lib/useMounted";
import { useAuthGuard } from "@/hooks/useAuthGuard";
import DashboardSkeleton from "@/components/dashboard/DashboardSkeleton";
import { clearRecsCache } from "@/infrastructure/storage/StorageService";

import { apiResetSession } from "@/lib/api";

export default function DashboardPage() {
  const router = useRouter();
  const { session, isLoading, logout, updateSession } = useSession();
  const mounted = useMounted();

  // Single source of truth for auth + onboarding gate (SRP: no duplicated logic).
  useAuthGuard({ requireOnboarding: true });

  const handleLogout = () => {
    logout();
  };

  const handleBackToOnboarding = useCallback(async () => {
    if (!session) return;
    // Law of Demeter: use domain helper instead of constructing the key here.
    clearRecsCache(session.user_id);
    updateSession({ ...session, onboarding_complete: false });
    try {
      await apiResetSession(session.session_id);
    } catch {
      // Non-critical — redirect proceeds regardless
    }
    router.push("/onboarding");
  }, [session, updateSession, router]);

  if (!mounted || isLoading || !session) {
    // SRP: loading UI is owned by DashboardSkeleton, not this page.
    return <DashboardSkeleton />;
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
