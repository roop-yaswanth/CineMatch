"use client";

import { Suspense, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import OnboardingView from "@/components/OnboardingView";
import { LoadingScreen } from "@/components/LoadingScreen";
import { useSession } from "@/context/SessionContext";
import { sessionHomePath } from "@/lib/session-routing";

function OnboardingPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { session, logout, updateSession, isLoading } = useSession();
  const forcePreferences = searchParams.get("reset") === "true";

  useEffect(() => {
    if (!isLoading && !session) {
      router.replace("/login");
    }
  }, [session, isLoading, router]);

  useEffect(() => {
    if (!isLoading && session && session.onboarding_complete && !forcePreferences) {
      router.replace(sessionHomePath(session));
    }
  }, [session, isLoading, router, forcePreferences]);

  if (isLoading) {
    return <LoadingScreen />;
  }

  if (!session) {
    return null;
  }

  const handleComplete = (nextSession: typeof session) => {
    updateSession(nextSession);
    router.replace(sessionHomePath(nextSession));
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
