"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import OnboardingView from "@/components/OnboardingView";
import { useSession } from "@/context/SessionContext";
import { useMounted } from "@/lib/useMounted";
import type { UserSession } from "@/lib/api";

export default function OnboardingPage() {
  const router = useRouter();
  const mounted = useMounted();
  const { session, isLoading, logout, updateSession } = useSession();

  useEffect(() => {
    if (mounted && !isLoading && !session) {
      if (typeof window !== "undefined") {
        window.location.replace("/login");
      } else {
        router.replace("/login");
      }
    }
  }, [session, isLoading, router, mounted]);

  const handleComplete = (updatedSession: UserSession) => {
    updateSession(updatedSession);
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
