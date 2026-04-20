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
