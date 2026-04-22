"use client";

import { useEffect } from "react";
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

  if (isLoading || !session) return null;

  return (
    <RecommendationsView
      session={session}
      onSessionUpdate={updateSession}
      onLogout={handleLogout}
      onBackToOnboarding={() => router.push("/onboarding")}
    />
  );
}
