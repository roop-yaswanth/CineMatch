"use client";

import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect } from "react";
import { useSession } from "@/context/SessionContext";
import OnboardingView from "@/components/OnboardingView";
import { LoadingScreen } from "@/components/LoadingScreen";

function OnboardingPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { session, logout, updateSession, isLoading } = useSession();
  const forcePreferences = searchParams.get("reset") === "true";

  // If not authenticated, redirect to login
  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  // If user completes onboarding, redirect to dashboard (don't stay on onboarding)
  useEffect(() => {
    if (session && session.onboarding_complete && !forcePreferences) {
      router.replace("/dashboard");
    }
  }, [session, router, forcePreferences]);




  if (!session || isLoading) {
    return <LoadingScreen />;
  }

  const handleComplete = (nextSession: typeof session) => {
    updateSession(nextSession);
    router.replace("/dashboard");
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

export default function OnboardingPage() {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <OnboardingPageContent />
    </Suspense>
  );
}
