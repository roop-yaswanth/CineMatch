"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useSession } from "@/context/SessionContext";
import OnboardingView from "@/components/OnboardingView";
import { LoadingScreen } from "@/components/LoadingScreen";

export default function OnboardingPage() {
  const router = useRouter();
  const { session, logout, updateSession, isLoading } = useSession();
  const [forcePreferences, setForcePreferences] = useState(false);

  // If not authenticated, redirect to login
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  // If user completes onboarding, redirect to dashboard (don't stay on onboarding)
  useEffect(() => {
    if (session && session.onboarding_complete) {
      router.replace("/dashboard");
    }
  }, [session, router]);

  if (!session || isLoading) {
    return <LoadingScreen />;
  }

  const handleComplete = (nextSession: typeof session) => {
    updateSession(nextSession);
    router.replace("/dashboard");
  };

  const handleBackToOnboarding = () => {
    setForcePreferences(true);
  };

  return (
    <OnboardingView
      session={session}
      onComplete={handleComplete}
      onLogout={logout}
      forcePreferences={forcePreferences}
    />
  );
}
