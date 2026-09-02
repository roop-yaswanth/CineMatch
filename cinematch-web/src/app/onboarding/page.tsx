"use client";

import { useRouter } from "next/navigation";
import OnboardingView from "@/components/OnboardingView";
import { useSession } from "@/context/SessionContext";
import { useAuthGuard } from "@/hooks/useAuthGuard";
import type { UserSession } from "@/lib/api";

export default function OnboardingPage() {
  const router = useRouter();
  const { session, isLoading, logout, updateSession } = useSession();
  const { mounted } = useAuthGuard();

  const handleComplete = (updatedSession: UserSession) => {
    const completedSession: UserSession = {
      ...updatedSession,
      onboarding_complete: true,
    };
    updateSession(completedSession);
    try {
      sessionStorage.setItem("cinematch_just_onboarded", "1");
    } catch {}
    router.replace("/dashboard");
  };

  const handleLogout = () => {
    logout();
  };

  if (!mounted || isLoading || !session) return null;

  return (
    <OnboardingView
      session={session}
      onComplete={handleComplete}
      onLogout={handleLogout}
      forcePreferences={false}
    />
  );
}
