"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import OnboardingView from "@/components/OnboardingView";
import { useSession } from "@/context/SessionContext";
import type { UserSession } from "@/lib/api";

export default function OnboardingPage() {
  const router = useRouter();
  const { session, isLoading, logout, updateSession } = useSession();

  useEffect(() => {
    if (!isLoading && !session) {
      if (typeof window !== "undefined") {
        window.location.replace("/login");
      } else {
        router.replace("/login");
      }
    }
  }, [session, isLoading, router]);

  const handleComplete = (updatedSession: UserSession) => {
    updateSession(updatedSession);
    router.replace("/dashboard");
  };

  const handleLogout = () => {
    logout();
  };

  if (isLoading || !session) return null;

  return (
    <OnboardingView
      session={session}
      onComplete={handleComplete}
      onLogout={handleLogout}
      forcePreferences={false}
    />
  );
}
