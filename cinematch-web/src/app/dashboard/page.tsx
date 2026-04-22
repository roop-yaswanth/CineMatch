"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import RecommendationsView from "@/components/RecommendationsView";
import { LoadingScreen } from "@/components/LoadingScreen";
import { useSession } from "@/context/SessionContext";

export default function DashboardPage() {
  const router = useRouter();
  const { session, logout, updateSession, isLoading } = useSession();

  useEffect(() => {
    if (isLoading) return;

    if (!session) {
      router.replace("/login");
      return;
    }

    if (!session.onboarding_complete) {
      router.replace("/onboarding");
    }
  }, [session, isLoading, router]);

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!session || !session.onboarding_complete) {
    return null;
  }

  const handleSessionUpdate = (nextSession: typeof session) => {
    updateSession(nextSession);
  };

  const handleBackToOnboarding = () => {
    router.replace("/onboarding?reset=true");
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
